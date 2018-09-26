from __future__ import print_function
import os
import os.path as op
import time
import sys
import argparse
import json
import logging
import numpy as np
import torch
from torch.optim import Adam
import torch.distributed as dist
from torch.utils.data import DataLoader


try:
    # indirect import of matplotlib (e.g. by caffe) may try to load non-existent X
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    matplotlib = None

try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = op.abspath(this_file)

if __name__ == '__main__':
    # When run as script, modify path assuming absolute import
    sys.path.append(op.join(op.dirname(this_file), '..'))

from mmod.phillylogger import PhillyLogger
from mmod.utils import makedirs, cwd, init_logging, \
    ompi_rank, ompi_size, gpu_indices
from mmod.philly_utils import get_log_parent, get_arg, abspath, get_master_ip, last_log_dir, get_model_path, \
    set_job_id, is_local
from mmod.checkpoint import last_checkpoint
from mmod.filelock import FileLock
from mmod.runcaffe import move_solvers
from mmod.simple_parser import parse_prototxt, read_model


from mtorch.caffenet import CaffeNet
from mtorch.caffesgd import CaffeSGD
from mtorch.multifixed_scheduler import MultiFixedScheduler
from mtorch.imdbdata import ImdbData
from mtorch.tbox_utils import Labeler, DarknetAugmentation


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TorchSession(object):
    def __init__(self, logger=None, gpus=None, restore=True, transfer_weights=True,
                 snapshot_model=None, solver_path=None, 
                 batch_size=None, max_iter=None,
                 opt=None,
                 verbose=False):
        """Reusable PyTorch training session
        :type logger: PhillyLogger
        :param restore: if should restore the model from previous snapshot
        :param gpus: local GPUs (to override automatic assignment)
        :param transfer_weights: if should keep the weights and reuse them in successive training in the session
        :param snapshot_model: initial snapshot (used if no restoring is done), can be a .caffemodel
        :param solver_path: default solver path
        :type solver_path: str
        :param batch_size: optional batch size to override solver(s)
        :type batch_size: int
        :param max_iter: optional max_iter to override solver(s)
        :type max_iter: int
        :param opt: optimizer method (default is caffesgd)
        :type opt: str
        :param verbose: Verbose output
        """
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.opt = opt
        self.verbose = verbose
        self.logger = logger  # type: PhillyLogger
        self.restore = restore
        self.transfer_weights = transfer_weights
        self.snapshot_model = snapshot_model
        self._solver = None  # type: dict
        self._solver_path = None
        self.solver_path = solver_path  # current solver path
        self.gpus = gpus if gpus else list(gpu_indices())
        self.rank = ompi_rank()
        self.world_size = ompi_size()
        self.batch_time = AverageMeter()
        self.iterations = 0  # current number of iterations

        if len(self.gpus) > 1:
            logging.warning("Ran with 1 process/container on {} process(es), performance may degrade".format(
                self.world_size
            ))
        torch.cuda.set_device(self.gpus[0])  # Set current GPU to the first

        if self.world_size > 1:
            dist.init_process_group(
                backend="nccl",
                init_method='tcp://' + get_master_ip() + ':23456',
                world_size=self.world_size,
                rank=self.rank,
                group_name='mtorch'
            )
            # TODO: try a pre-run sync, just to get NCCL initialized early

        self.momentum = None
        self.weight_decay = None

        self.snapshot_prefix = ''  # snapshot_prefix from the previous run
        self.snapshot_interval = 1  # snapshot interval from previous run
        self.display = None  # display iterations

    @property
    def solver_path(self):
        """The solver path
        :rtype: str
        """
        assert self._solver_path and op.isfile(self._solver_path), "Invalid solver path: {}".format(self._solver_path)
        return self._solver_path

    @solver_path.setter
    def solver_path(self, solver_path):
        """The solver path
        :type solver_path: str
        """
        if not solver_path:
            self._solver_path = None
            self._solver = {}
            return
        assert op.isfile(solver_path), "Invalid solver path: {}".format(solver_path)
        self._solver_path = solver_path
        self._solver = parse_prototxt(solver_path)

    def _checkpoint(self):
        """Get the checkpoint to restore from
        Call only once for each run
        """
        snapshot_interval = int(self._solver.get("snapshot") or 0)
        max_iter = int(self.max_iter or self._solver.get("max_iter") or 0)
        snapshot_prefix = self._solver.get("snapshot_prefix", "")

        snapshot_model, self.iterations = '', 0

        if self.restore and snapshot_prefix:
            # see if we have a current weight/snapshot
            snapshot_model, self.iterations, _ = last_checkpoint(
                snapshot_prefix,
                snapshot_interval=snapshot_interval,
                max_iter=max_iter,
                snapshot_ext=".pt",
                weights_ext=None
            )

        # if no current snapshot/weight, see if there is an old weight
        if self.transfer_weights and not snapshot_model and self.snapshot_prefix:
            # if prefix from a previous run, find the last snapshot there
            snapshot_model, _, _ = last_checkpoint(
                self.snapshot_prefix,
                snapshot_interval=snapshot_interval,
                max_iter=max_iter,
                snapshot_ext=".pt",
                weights_ext=None
            )
            if snapshot_model:
                logging.info('Reuse old weights: {} for solver: {}'.format(snapshot_model, self.solver_path))

        if not snapshot_model:
            snapshot_model = self.snapshot_model

        # save for future runs
        self.snapshot_interval = snapshot_interval
        self.snapshot_prefix = snapshot_prefix
        self.max_iter = max_iter

        # make sure snapshot prefix exists
        makedirs(op.dirname(self.snapshot_prefix), exist_ok=True)

        return snapshot_model

    def snapshot(self, model, optimizer=None, with_caffemodel=False):
        """Take a snapshot
        :type model: mtorch.caffenet.CaffeNet
        :type optimizer: torch.optim.SGD
        :param with_caffemodel: if should also snapshot a caffemodel
        """
        if self.rank:
            # only master should snapshot
            return
        snapshot_file = self.snapshot_prefix + "_iter_{}".format(self.iterations)
        snapshot_pt = snapshot_file + '.pt'
        state_dict = model.state_dict()
        state = {
            'iterations': self.iterations,
            'state_dict': state_dict,
        }
        if optimizer:
            state.update({
                'optimizer': optimizer.state_dict(),
            })
        if self.opt:
            state.update({
                'opt': self.opt,
            })
        logging.info("Snapshotting to: {}".format(snapshot_pt))
        torch.save(state, snapshot_pt)

        if not with_caffemodel:
            return

        module = model
        # if model is wrapped in Parallel
        if hasattr(model, 'module'):
            module = model.module
            # remove "module." from state
            state_dict = {
                k[7:]: v for (k, v) in state_dict.iteritems()
            }
        assert isinstance(module, CaffeNet)
        snapshot_caffe = snapshot_file + '.caffemodel'
        logging.info("Snapshotting to caffemodel: {}".format(snapshot_caffe))
        module.save_weights(snapshot_caffe, state=state_dict)

    def train_batch(self, model, inputs, optimizer):
        """Train one batch
        :type model: mtorch.caffenet.CaffeNet
        :type inputs: torch.nn.Module
        :type optimizer: torch.optim.SGD
        """
        end = time.time()

        # compute output
        data, labels = inputs[0].cuda(), inputs[1].cuda().float()
             
        loss = model(data, labels)
        crit = torch.sum(loss)
        assert crit == crit, "Iteration: {} Stop because of NaN loss!".format(
            self.iterations
        )

        # compute gradient and do SGD step
        optimizer.zero_grad()
        crit.backward()
        optimizer.step()

        # measure elapsed time
        self.batch_time.update(time.time() - end)

        if self.display and self.iterations % self.display == 0:
            logging.info("Iteration {} ({:.4f} iter/s, {:.5f}s/{} iter), loss = {:.6f}".format(
                self.iterations,
                1.0 / self.batch_time.avg,
                self.batch_time.sum, self.batch_time.count,
                crit.item())
            )
            self.batch_time.reset()
            if self.logger:
                self.logger.set_iterations(self.iterations, losses=crit.item())

    def solve(self, solver_path=None, batch_size=None, max_iter=None, with_caffemodel=False):
        """caffe solver to solve next
        :type solver_path: str
        :type batch_size: int
        :type max_iter: int
        :param with_caffemodel: if should create intermediate caffemodel
        """
        if not solver_path:
            solver_path = self.solver_path
        if not batch_size:
            batch_size = self.batch_size
        if not max_iter:
            max_iter = self.max_iter

        self.max_iter = max_iter
        self.batch_size = batch_size
        self.solver_path = solver_path
        logging.info('Solving {}'.format(solver_path))

        snapshot_model = self._checkpoint()
        self.display = int(self._solver.get('display') or 0)  # display interval

        if self.logger:
            self.logger.set_iterations(self.iterations)

        if self.max_iter <= self.iterations:
            logging.info(('Ignore training solver file {} '
                          'that is already trained for {} >= {} iterations').format(
                solver_path, self.iterations, self.max_iter)
            )
            return

        self.weight_decay = float(self._solver.get('weight_decay') or 0)
        self.momentum = float(self._solver.get('momentum') or 0)
        lr_policy = self._solver.get('lr_policy', 'fixed')
        if not lr_policy or lr_policy == 'fixed':
            lrs = [float(self._solver['base_lr'])]
            steps = None
        elif lr_policy == "multifixed":
            lrs = [float(lr) for lr in self._solver["stagelr"]]
            steps = [int(ii) for ii in self._solver["stageiter"]]
        else:
            raise NotImplementedError("Learning policy: {} not implemented".format(lr_policy))

        protofile = self._solver.get("train_net") or self._solver.get("net")
        if not op.isfile(protofile) and not op.isabs(protofile):
            # try local train.prototxt besides the solver.prototxt
            for _p in [op.join(op.dirname(self.solver_path), op.basename(protofile)),
                       abspath(protofile, roots=['~', '#'])]:
                if op.isfile(_p):
                    protofile = _p
                    logging.info("training prototxt resolved to: {}".format(protofile))
                    break
        model = CaffeNet(protofile, verbose=self.verbose,
                         local_gpus_size=len(self.gpus),
                         world_size=self.world_size,
                         batch_size=batch_size,
                         use_pytorch_data_layer=True)

        # Load from caffemodel, before cuda()
        if snapshot_model and (not isinstance(snapshot_model, basestring) or snapshot_model.endswith(".caffemodel")):
            logging.info("Finetuning from weights")
            model.load_weights(snapshot_model)
            snapshot_model = None
        params = model.net_info['layers'][0]
        model = model.cuda()

        if self.world_size > 1:
            logging.info("Distributed; gpus: {} devices: {}".format(self.gpus, torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=self.gpus
            )
        elif len(self.gpus) > 1:
            logging.info("Single-Node Multi-GPU; gpus: {} devices: {}".format(self.gpus, torch.cuda.device_count()))
            model = torch.nn.DataParallel(
                model,
                device_ids=self.gpus
            )
        else:
            logging.info("Single-Node Single-GPU; devices: {}".format(torch.cuda.device_count()))

        # Load the checkpoint as a DDP/DP, like what it is saved
        snapshot_iterations = -1
        checkpoint = None
        # TODO: remove inputs form model
        model.inputs = dict()
        model.inputs["data"] = None
        model.inputs["label"] = None
        if snapshot_model:
            snapshot_iterations = self.iterations
            logging.info("Finetuning from snapshot: {}".format(snapshot_model))
            checkpoint = torch.load(snapshot_model, map_location=lambda storage, loc: storage)
            assert isinstance(checkpoint, dict) and 'iterations' in checkpoint, "Invalid snapshot: {}".format(
                snapshot_model
            )
            iterations = checkpoint['iterations']
            assert iterations == self.iterations, "iterations: {} != {} in {}".format(
                self.iterations, iterations, snapshot_model
            )
            model.load_state_dict(checkpoint['state_dict'])

        # TODO: use lr_mult and decay_mult for optimization groups
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)

        scheduler = None
        if self.opt == 'adam':
            initial_lr = np.max(lrs)
            optimizer = Adam(
                [{'params': no_decay, 'weight_decay': 0., 'initial_lr': initial_lr},
                 {'params': decay, 'initial_lr': initial_lr}],
                lr=initial_lr,
                weight_decay=self.weight_decay
            )
        else:
            initial_lr = lrs[0]
            optimizer = CaffeSGD(
                [{'params': no_decay, 'weight_decay': 0., 'initial_lr': initial_lr},
                 {'params': decay, 'initial_lr': initial_lr}],
                lr=initial_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
            if lr_policy == "multifixed":
                scheduler = MultiFixedScheduler(optimizer, steps, lrs, last_iter=self.iterations)

        if checkpoint and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        self.batch_time.reset()

        # switch to train mode
        model.train()
        
        augmenter = DarknetAugmentation()
        labeler = Labeler()
        augmented_dataset = ImdbData(path=params['tsv_data_param']['source'], 
                                transform=augmenter(params), labeler=labeler)

        if self.batch_size is None:
            self.batch_size = int(params['tsv_data_param']['batch_size'])

        self.batch_size *= len(self.gpus)  # TODO: fix on philly

        data_loader = DataLoader(augmented_dataset, batch_size=self.batch_size,
                                 shuffle=True,
                                 # TODO: find way to increase number of workers on 1 GPU
                                 num_workers=8 if len(self.gpus) > 1 else 0,
                                 pin_memory=True)  # TODO: should it be False?
        
        for self.iterations in range(self.max_iter):
            if scheduler:
                scheduler.step()
            for i, batch_inputs in enumerate(data_loader):
                 self.train_batch(model, batch_inputs, optimizer)

            if self.iterations % self.snapshot_interval == 0:
                self.snapshot(model, optimizer=optimizer, with_caffemodel=self.iterations and with_caffemodel)


        # last snapshot
        self.snapshot(model, optimizer=optimizer, with_caffemodel=True)


def main():
    init_logging()
    parser = argparse.ArgumentParser(description='Run PyTorch training with given config file.')

    parser.add_argument('-d', '--datadir', '--dataDir', help='Data directory where the dataset is located',
                        default='data')
    parser.add_argument('-m',  '--outputdir', '--modeldir', '--modelDir',
                        help='Output directory for checkpoints and models',
                        required=False)
    parser.add_argument('-l', '--logdir', '--logDir', help='Log directory', required=False)
    parser.add_argument('--prevmodelpath', help='Previous model path', required=False)
    parser.add_argument('--configfile', '--stdoutdir', '--numgpu', help='Ignored', required=False)
    # ------------------------------------------------------------------------------------------------------------
    parser.add_argument('-s', '--solver', action='append',
                        help='Prototxt solver file for caffe training (specify multiple times to chain SOLVERs)',
                        required=True)
    parser.add_argument('-prev', '--prev', help='Previous model path (override)', required=False)
    parser.add_argument('--skip_weights', help='If should avoid reusing weights across solvers', action='store_true',
                        default=False, required=False)
    parser.add_argument('--skip_snapshot', help='If should skip loading the last valid snapshot', action='store_true',
                        default=False, required=False)
    parser.add_argument('-snapshot', '--snapshot', '-weights', '--weights',
                        help='Initial snapshot or caffemodel to finetune from', required=False)
    parser.add_argument('-t', '--iters', '-max_iter', '--max_iter', dest='max_iters', action='append',
                        help='number of iterations to train (to override --solver file)',  required=False)
    parser.add_argument('-b', '--batch_size', '--batchsize', action='append', type=int,
                        help='per-gpu batch size (to override train.prototxt used by SOLVER file)',
                        required=False)
    parser.add_argument('--verbose', help='Verbose network output', action='store_true',
                        default=False, required=False)
    parser.add_argument('-g', '--gpus', action='append', type=int,
                        help='GPU device ordinals to restrict the training to',
                        required=False)
    parser.add_argument('--opt', default='caffesgd', nargs='?',
                        help='GPU device ordinals to restrict the training to',
                        choices=['caffesgd', 'adam'])
    parser.add_argument('-sc', '--snapshot_caffe', help='Snapshot all models to caffemodel', action='store_true',
                        default=False, required=False)
    parser.add_argument('-e', '--expid', '--jobid', help='The full experiment ID (if local, will be used as job ID)',
                        required=False)

    args = parser.parse_args()
    args = vars(args)
    print("Arguments: %s" % args)
    set_job_id(args['expid'])

    model_path = args['outputdir']
    if not model_path:
        model_path = get_model_path()
    else:
        model_path = abspath(model_path, roots=['#', '.'])
        if not op.isdir(model_path):
            raise RuntimeError("Output directory '{}' does not exist".format(model_path))

    log_dir = args['logdir']
    if not log_dir:
        log_dir = last_log_dir()
    log_dir = abspath(log_dir, roots=['~', '.'])
    if not op.isdir(log_dir):
        raise RuntimeError("Log directory '{}' does not exist".format(log_dir))
    data_path = abspath(args['datadir'], roots=['#', '.'])
    if not op.isdir(data_path):
        raise RuntimeError("Data directory '{}' does not exist".format(data_path))
    solvers = args['solver']
    if not solvers:
        raise RuntimeError("Train solver file(s) must be specified")
    solvers = [abspath(solver, roots=['~', '#', '.']) for solver in solvers]
    skip_weights = args['skip_weights']
    skip_snapshot = args['skip_snapshot']
    snapshot_model = args['snapshot']
    if snapshot_model:
        snapshot_model = abspath(snapshot_model, roots=['#', '~', '.'])
        assert op.isfile(snapshot_model), "{} does not exist".format(snapshot_model)
        if snapshot_model.endswith(".caffemodel"):
            # pre-load the model before the session is loaded, to make the init faster when parallel
            logging.info("Reading {}".format(snapshot_model))
            snapshot_model = read_model(snapshot_model)
    max_iters = args['max_iters'] or None  # type: list
    if max_iters:
        # if given once assume it is for all the solvers
        if len(max_iters) == 1:
            max_iters = max_iters * len(solvers)
        assert len(max_iters) == len(solvers), "[--iters] count: {} != [--solver] count: {}".format(
            len(max_iters), len(solvers)
        )
    batch_sizes = args['batch_size'] or None  # type: list
    if batch_sizes:
        # if given once assume it is for all the solvers
        if len(batch_sizes) == 1:
            batch_sizes = batch_sizes * len(solvers)
        assert len(batch_sizes) == len(solvers), "[--batch_size] count: {} != [--solver] count: {}".format(
            len(batch_sizes), len(solvers)
        )
    verbose = args['verbose']
    gpus = args['gpus']

    prev_model_path = get_arg(args['prev']) or get_arg(args['prevmodelpath'])
    # Find previous log dir, to fill in the progress (used for cloning)
    prev_log_parent = get_log_parent(prev_model_path)

    if is_local() and op.normpath('./data') == op.normpath(data_path):
        work_path = '.'
    else:
        work_path = '/tmp/work'
        # Work-around to use current taxonomies with no change
        with FileLock(work_path):
            sym_data_path = op.join(work_path, 'data')
            if not op.isdir(sym_data_path):
                logging.info("Creating the sym path: {}".format(sym_data_path))
                makedirs(work_path, exist_ok=True)
                try:
                    os.symlink(data_path, sym_data_path)
                except OSError as e:
                    if not op.isdir(sym_data_path):
                        print('Symlink: {} Error: {}'.format(sym_data_path, e))

    # TODO: DistributedDataParallel is presumably better than DataParallel even on a single-node with (N > 1) GPUs
    # If world_size is 1 but we have (N > 1) GPUs, fake a world_size of N and spawn workers

    # Config files may assume paths relative to input data path
    with cwd(work_path):
        new_solvers, metas = move_solvers(solvers, model_path, prev_model_path)
        del solvers

        is_worker = ompi_rank() != 0 and ompi_size() > 1
        if is_worker:
            pl = PhillyLogger(log_dir, is_master=False)
        else:
            pl = PhillyLogger(
                log_dir,
                [m['max_iter'] for m in metas],
                [op.join(op.basename(op.dirname(m['solver_path'])), op.basename(m['solver_path'])) for m in metas],
                prev_log_parent=prev_log_parent
            )
        with pl.redirected() as logger:
            logging.info("Arguments: {}".format(json.dumps(args, indent=2)))
            session = TorchSession(logger,
                                   gpus=gpus,
                                   restore=not skip_snapshot, transfer_weights=not skip_weights,
                                   snapshot_model=snapshot_model,
                                   opt=args['opt'],
                                   verbose=verbose)
            for idx, solver_path in enumerate(new_solvers):
                logger.new_command(idx)
                session.solve(
                    solver_path,
                    batch_size=batch_sizes[idx] if batch_sizes else None,
                    max_iter=max_iters[idx] if max_iters else None,
                    with_caffemodel=args['snapshot_caffe']
                )
        return session


if __name__ == '__main__':
    # The session (for debugging) in ipython %run runtorch.py ...
    sess = main()

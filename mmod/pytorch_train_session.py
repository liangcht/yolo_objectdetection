from __future__ import print_function
import os
import os.path as op
import time
from datetime import datetime
import sys
import argparse
import torch
import torch.distributed as dist
import warnings

try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = op.abspath(this_file)

if __name__ == '__main__':
    # When run as script, modify path assuming absolute import
    sys.path.append(op.join(op.dirname(this_file), '..'))

from mmod.simple_parser import parse_prototxt
from mmod.experimental_utils import *
from mmod.meters import AverageMeter, TimeMeter
from mmod.file_logger import FileLogger

from mtorch.caffesgd import CaffeSGD
from mtorch.multifixed_scheduler import MultiFixedScheduler
from mtorch.dataloaders import yolo_train_data_loader
from mtorch.yolo_v2 import yolo
from mtorch.darknet import darknet_layers
from mtorch.region_target_loss import RegionTargetWithSoftMaxLoss

def to_python_float(t):
    if isinstance(t, (float, int)):
        return t
    if hasattr(t, 'item'):
        return t.item()
    return t[0]


def get_parser():
    parser = argparse.ArgumentParser(description='Run PyTorch training')

    parser.add_argument('-d', '--datadir', '--dataDir', help='Data directory where the dataset is located',
                        default='data')
    parser.add_argument('-m', '--outputdir', '--modeldir', '--modelDir',
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
    parser.add_argument('--restore', help='If should restor the last valid snapshot', action='store_true',
                        default=True, required=False)
    parser.add_argument('-snapshot', '--snapshot', '-weights', '--weights',
                        help='Initial snapshot or caffemodel to finetune from', required=False)
    parser.add_argument('-t', '--iters', '-max_iter', '--max_iter', dest='max_iters', action='append',
                        help='number of iterations to train (to override --solver file)', required=False)
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

    return parser


def get_args_debug():
    args = {}
    args["distributed"] = False
    args["display"] = True
    args["snapshot"] = "/work/fromLei/py_caf_batch_fixed3_70/snapshot/model_iter_10000.pt"
    args["restore"] = False
    args["local_rank"]  = 0
    args["batch_size"] = 1
    args["gpus"] = [1]
    args["workers"] = 0
    args["solver"] = "/work/fromLei/yolo_voc_solver_pytorch_4GPU.prototxt" 
    args["logdir"] = "/work/fromLei/"
    args["train_dataset_path"] = "/work/fromLei/train_yolo_withSoftMaxLoss.prototxt"
    return args



#args = get_parser().parse_args()
#args = vars(args)

args = get_args_debug()
# Only want master rank logging to tensorboard
is_master = (not args["distributed"]) or (dist_utils.env_rank() == 0)
is_rank0 = args["local_rank"] == 0
log = FileLogger(args["logdir"], is_master=is_master, is_rank0=is_rank0)


def snapshot(model, epoch, snapshot_prefix, optimizer=None):
    """Take a snapshot
    :param model: mtorch.caffenet.CaffeNet
    :param epoch:
    :type optimizer: torch.optim.SGD
    """

    snapshot_pt = snapshot_prefix + "_epoch_{}".format(epoch) + '.pt'
    state = {
        'epochs': epoch,
        'state_dict': model.state_dict(),
    }
    if optimizer:
        state.update({
            'optimizer': optimizer.state_dict(),
        })
    log.info("Snapshotting to: {}".format(snapshot_pt))
    torch.save(state, snapshot_pt)


def train(trn_loader, model, criterion, optimizer, scheduler, epoch):
    timer = TimeMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    last_batch = len(trn_loader)
    batch_num = 0
    for inputs in trn_loader:

        batch_num += 1
        data, labels = inputs[0].cuda(), inputs[1].cuda().float()

        scheduler.step()
        timer.batch_start()

        # compute output
        features = model(data)
        optimizer.zero_grad()

        loss = criterion(features, labels) 
       # assert loss == loss, "Batch {} in epoch {}: NaN loss!".format(batch_num, epoch)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # Train batch done. Logging results
        timer.batch_end()
        reduced_loss, batch_total = to_python_float(loss.data), to_python_float(input.size(0))
        if args["distributed"]:  # Must keep track of global batch size, since not all machines are guaranteed equal batches at the end of an epoch
            metrics = torch.tensor([batch_total, reduced_loss]).float().cuda()
            batch_total, reduced_loss = dist_utils.sum_tensor(metrics).cpu().numpy()
            reduced_loss = reduced_loss / dist_utils.env_world_size()

        losses.update(reduced_loss, batch_total)

        should_print = (args["display"] and batch_num % args["display"] == 0) or batch_num == last_batch
        if should_print:
            output = ("{} Epoch {} (Model & Loss: {:.4f} iter/s, {:.5f}s/{} iter), loss = {:.6f}".format(
                batch_num / last_batch, epoch,
                timer.batch_time.val, timer.batch_time.avg,
                losses.val)
            )
            log.verbose(output)


def main():
    log.console(args)

    if args["distributed"]:
        log.console('Distributed initializing process group')
        torch.cuda.set_device(args['local_rank'])  # probably local_rank = 0
        # TODO: ask Ehsan - got to runtorch line 125
        dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=dist_utils.env_world_size())
        assert (dist_utils.env_world_size() == dist.get_world_size())  # check if there are
        log.console("Distributed: success (%d/%d)" % (args.local_rank, dist.get_world_size()))

    log.console("Loading model")

    model = yolo(darknet_layers(), weights_file=args['snapshot'], caffe_format_weights=True).cuda()

    if args["distributed"]:
        model = dist_utils.DDP(model, device_ids=[args['gpus']], output_device=args['gpus'])

    # TODO: the next 3 lines could be put in a Class SOLVER including lr_policy
    solver_params = parse_prototxt(args['solver'])
    lrs = get_lrs(solver_params)
    steps = get_steps(solver_params)

    #optim_params_groups = get_optim_params(model.named_parameters(), lrs[0]),
    
    criterion = RegionTargetWithSoftMaxLoss(model.seen_images)
    criterion = criterion.cuda()#YoloLoss(args['loss_mode']).cuda()  # TODO: implement YoloLoss, add loss_mode to arguments

    decay, no_decay, lr2 = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "last_conv" in name and name.endswith(".bias"):
            lr2.append(param)
        elif "scale" in name:
            decay.append(param)
        elif len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

       
    initial_lr = lrs[0]
    #  TODO: param_groups should be fully inferred from prototxt in a helper method
    param_groups = [{'params': no_decay, 'weight_decay': 0., 'initial_lr': initial_lr, 'lr_mult': 1.},
           {'params': decay,  'initial_lr': initial_lr, 'lr_mult': 1.},
           {'params': lr2, 'weight_decay': 0., 'initial_lr': initial_lr * 2., 'lr_mult': 2.}]



    optimizer = CaffeSGD(param_groups, lr=lrs[0],
                         momentum=float(solver_params['momentum']), 
                         weight_decay=float(solver_params['weight_decay']))

    if args["restore"]:
        checkpoint = torch.load(get_latest_snapshot(solver_params["snapshot_prefix"]),
                                map_location=lambda storage, loc: storage.cuda(args["local_rank"]))  # TODO: ask Ehsan
        model.load_state_dict(checkpoint['state_dict'])
        args["start_epoch"] = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        args["start_epoch"] = 0

    log.console("Creating data loaders (this could take up to 10 minutes if volume needs to be warmed up)")
    if solver_params.get('lr_policy', 'fixed') == "multifixed":
        scheduler = MultiFixedScheduler(optimizer, steps, lrs, last_iter=0)
    else:
        scheduler = None

    if args["distributed"]:
        log.console('Syncing machines before training')
        dist_utils.sum_tensor(torch.tensor([1.0]).float().cuda())
    data_loader = yolo_train_data_loader(args["train_dataset_path"], batch_size=args["batch_size"],
                                        num_workers=args["workers"], distributed=False)

    start_time = datetime.now()  # Loading start to after everything is loaded

    for epoch in range(args["start_epoch"], int(solver_params["max_iter"])):
        train(data_loader, model, criterion, optimizer, scheduler, epoch)
        time_diff = (datetime.now()-start_time).total_seconds() / 3600.0
        log.event('{}\t {.5f}\n'.format(epoch, time_diff))
        if epoch % solver_params["snapshot"] == 0:
            snapshot(model, epoch, optimizer=optimizer)


if __name__ == '__main__':
    # try:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore", category=UserWarning)
    main()

    # except Exception as e:
    #     exc_type, exc_value, exc_traceback = sys.exc_info()
    #     log.event(e)

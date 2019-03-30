from __future__ import print_function
import os
import os.path as op
import time
from datetime import datetime
import sys
import argparse
import re
import torch
import torch.distributed as dist
import dist_utils
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
from mmod.simple_parser import load_labelmap_list

from mtorch.caffesgd import CaffeSGD
from mtorch.multifixed_scheduler import MultiFixedScheduler
from mtorch import dataloaders
from mtorch.dataloaders import yolo_train_data_loader
from mtorch.yolo_v2 import yolo_2extraconv
from mtorch.darknet import darknet_layers
from mtorch.region_target_loss import RegionTargetWithSoftMaxLoss, RegionTargetWithSoftMaxTreeLoss


def to_python_float(t):
    if isinstance(t, (float, int)):
        return t
    if hasattr(t, 'item'):
        return t.item()
    return t[0]


def get_parser():
    """pre pares parser of input parameters"""

    parser = argparse.ArgumentParser(description='Run Yolo training')
    parser.add_argument('-d', '--train', type=str, metavar='TRAINDATA_PATH',
                        help='Path to the training dataset file',
                        required=True)
    parser.add_argument('-m', '--model', type=str, metavar='MODEL_PATH',
                        help='path to latest checkpoint', required=False)
    parser.add_argument('-c', '--is_caffemodel', default=False, action='store_true',
                        help='if provided, assumes model weights are derived from caffemodel, false by default',
                        required=False)
    parser.add_argument('--only_backbone', default=False, action='store_true',
                        help="if provided, only backbone weights are taken from the model, false by default")
    parser.add_argument('-l', '--logdir', help='Log directory', required=True)
    parser.add_argument('-s', '--solver',
                        help='solver file with training parameters',
                        required=True)
    parser.add_argument('--labelmap', type=str, metavar='LABELMAP_PATH',
                        help='path to labelmap', required=True)
    parser.add_argument('-t', '--use_treestructure', default=False, action='store_true',
                        help='if provided, will use tree structure, false by default',
                        required=False)
    parser.add_argument('--tree', type=str, metavar='TREE_PATH',
                        help='path to a tree structure, it prediction based on tree is required',
                        required=False)
    parser.add_argument('--distributed', default=False, action='store_true',
                        help='specify if you want to use distributed training (default=False)',
                        required=False)
    parser.add_argument('--display', default=True, type=bool,
                        help='input False is you do NOT want to print progress (default=True)',
                        required=False)
    parser.add_argument('-r', '--restore', default=False, action='store_true',
                        help='specify if the model should be restored from latest_snapshot, default is false',
                        required=False)
    parser.add_argument('-latest_snapshot', '--latest_snapshot',
                        help='Initial snapshot to finetune from', required=False)
    parser.add_argument('-b', '--batch_size', '--batchsize', default=16, type=int, metavar='PER_GPU_BATCH_SIZE',
                        help='per-gpu batch size (default=16)',
                        required=False)
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='NUM_WORKERS',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--local_rank', help='local_rank', required=False)
    parser.add_argument('--dist_url', default="tcp://127.0.0.1:2345",
                        help='dist_url')
    return parser

MAX_EPOCH = 128

def last_snapshot_path(prefix, max_epoch=None, args=None):
    """Find the last checkpoints inside given prefix paths
    Look at the paths in prefixes, and return once found
    :param prefix: str, paths to check for the snapshot
    :param max_epoch: maximum number of iterations that could be found
    :exception: ValueError if last snapshot is not a valid path or cannot be retried
    :return: snapshot, str
    """
    if args and args["latest_snapshot"] is not None and op.isfile(args["latest_snapshot"]):
        return args["latest_snapshot"]

    last_epoch = -1
    path = op.dirname(prefix)
    if not op.isdir(path):
        raise ValueError("Cannot restore snapshot from invalid path:{}".format(prefix))

    base = op.basename(prefix)
    model_iter_pattern = re.compile(r'epoch_(?P<EPOCH>\d+){}$'.format(re.escape(".pt")))

    epoch = max_epoch
    for fname in os.listdir(path):
        if not fname.startswith(base):
            continue
        model = re.search(model_iter_pattern, fname)
        if model:
            epoch = int(model.group('EPOCH'))
        if epoch == max_epoch:
            return op.join(path, fname)
        if epoch > last_epoch:
            snapshot_path = op.join(path, fname)
            last_epoch = epoch

    if last_epoch == -1:
        raise ValueError("Cannot restore snapshot from provided path:{}".format(prefix))

    return snapshot_path

def get_snapshot(snapshot_prefix, epoch):
    snapshot_pt = snapshot_prefix + "_epoch_{}".format(epoch + 1) + '.pt'
    return snapshot_pt

def snapshot(model, criterion, losses, epoch, snapshot_prefix,
        log, optimizer=None):
    """Takes a snapshot of training procedure
    :param model: model to snapshot
    :param criterion: loss - required for saving seen_images
    :param losses: list of losses
    :param epoch: int, epoch reached so far
    :param snapshot_prefix: str, file to save the current
    :param optimizer: torch.optim.SGD - required to continue training properly,
    is not reqired for testing
    """
    snapshot_dir = op.dirname(snapshot_prefix)
    if not op.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    if op.basename(snapshot_prefix) != "model":
        snapshot_prefix = op.join(snapshot_dir, "model")

    snapshot_pt = get_snapshot(snapshot_prefix, epoch)
    snapshot_losses_pt = snapshot_prefix + "_losses.pt"

    state = {
        'epochs': epoch,
        'state_dict': model.state_dict(),
        'seen_images': criterion.seen_images,
        'region_target.biases': criterion.region_target.biases,
        'region_target.seen_images': criterion.region_target.seen_images
    }
    if optimizer:
        state.update({
            'optimizer': optimizer.state_dict(),
        })
    log.verbose("Snapshotting to: {}".format(snapshot_pt))
    torch.save(state, snapshot_pt)
    torch.save(losses, snapshot_losses_pt)

def restore_state(model, optimizer, latest_snapshot):
    """restores the latest state of training from a snapshot
    :param model: will update model weights from latest state_dict
    :param optimizer:  will update optimizer
    :param latest_snapshot: latest snapshot to use
    :return: seen images and last epoch
    """
    checkpoint = torch.load(latest_snapshot,
                            map_location=lambda storage, loc: storage.cuda(args["local_rank"]))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['seen_images'], checkpoint['epochs']


def train(trn_loader, model, criterion, optimizer, scheduler, epoch, loss_arr,
        args, iterations_left, log):
    """
    :param trn_loader: utils.data.DataLoader, data loader for training
    :param model: torch.nn.Module or nn.Sequential, model
    :param criterion: torch.nn.Module, loss
    :param optimizer: torch.optim, optimizer to update model
    :param scheduler: torch.optim.lr_scheduler, takes care of weight decay and learning rate updates
    :param epoch: int, current epoch
    :param loss_arr: list, stores all the losses
     """
    timer = TimeMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    last_batch = len(trn_loader)
    for i, inputs in enumerate(trn_loader):

        batch_num = i + 1
        data, labels = inputs[0].cuda(), inputs[1].cuda().float()

        scheduler.step()
        timer.batch_start()
        optimizer.zero_grad()

        # compute output
        features = model(data)
        loss = criterion(features, labels)
        assert loss == loss, "Batch {} in epoch {}: NaN loss!".format(batch_num, epoch)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        # Train batch done. Logging results
        timer.batch_end()
        reduced_loss, batch_total = to_python_float(loss.data), to_python_float(data.size(0))
        if args["distributed"]:  # keeping track of all the machines
            metrics = torch.tensor([batch_total, reduced_loss]).float().cuda()
            batch_total, reduced_loss = dist_utils.sum_tensor(metrics).cpu().numpy()
            reduced_loss = reduced_loss / dist_utils.env_world_size()

        losses.update(reduced_loss, batch_total)
        loss_arr.append(torch.tensor(reduced_loss))
        should_print = (args["display"] and batch_num % args["display"] == 0) or batch_num == last_batch
        if should_print:
            output = "{:.2f} Epoch {}: Time per iter = {:.4f}s, Time left = {:.2f}h), loss = {:.4f}, batch_total = {}, lr = {}" \
                .format(float(batch_num) / last_batch, epoch, timer.batch_time.val,
                        (timer.batch_time.val * iterations_left) / 3600.,
                        losses.val, batch_total, scheduler.get_lr())
            log.verbose(output)
        if iterations_left is not None:
            iterations_left = iterations_left - 1
            if iterations_left <= 0:
                break

def main(args, log):
    log.console(args)
    if args["distributed"]:
        log.console('Distributed initializing process group')
        torch.cuda.set_device(args['local_rank'])  # probably local_rank = 0
        dist.init_process_group(backend='nccl', init_method=args["dist_url"], rank=dist_utils.env_rank(),
                                world_size=dist_utils.env_world_size())
        assert (dist_utils.env_world_size() == dist.get_world_size())  # check if there are
        log.console("Distributed: success ({}/{})".format(args["local_rank"], dist.get_world_size()))

    log.console("Loading model")
    cmap = load_labelmap_list(args["labelmap"])

    if args['only_backbone']:
        model = yolo_2extraconv(darknet_layers(weights_file=args['model'],
                                caffe_format_weights=args['is_caffemodel']),
                                num_classes=len(cmap)).cuda()
    else:
        model = yolo_2extraconv(darknet_layers(), weights_file=args['model'],
                                caffe_format_weights=args['is_caffemodel'],
                                num_classes=len(cmap)).cuda()
    seen_images = model.seen_images

    if args["distributed"]:
        model = dist_utils.DDP(model, device_ids=[args['local_rank']], output_device=args['local_rank'])
    if args['solver'].endswith('.prototxt'):
        solver_params = parse_prototxt(args['solver'])
    else:
        assert args['solver'].endswith('.yaml')
        from mmod.io_utils import load_from_yaml_file
        solver_params = load_from_yaml_file(args['solver'])
    lrs = get_lrs(solver_params)
    steps = get_steps(solver_params)

    # code below sets caffe compatible learning rate and weight decay hyper parameters
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

    param_groups = [{'params': no_decay, 'weight_decay': 0., 'initial_lr': lrs[0], 'lr_mult': 1.},
                    {'params': decay, 'initial_lr': lrs[0], 'lr_mult': 1.},
                    {'params': lr2, 'weight_decay': 0., 'initial_lr': lrs[0] * 2., 'lr_mult': 2.}]

    optimizer = CaffeSGD(param_groups, lr=lrs[0],
                         momentum=float(solver_params['momentum']),
                         weight_decay=float(solver_params['weight_decay']))

    log.console("Creating data loaders")
    data_loader = yolo_train_data_loader(args["train"], cmapfile=args["labelmap"], batch_size=args["batch_size"],
                                         num_workers=args["workers"], distributed=args["distributed"])
    log.console("Uses Wrapping Sampler: {}".format(dataloaders.WRAPPING))
    log.console("Uses Random Sampler: {}".format(dataloaders.RANDOM_SAMPLER))

    try:
        num_epochs = int(solver_params["max_epoch"])
    except KeyError:
        try:
            num_epochs = int(round(float(solver_params["max_iter"]) / len(data_loader) + 0.5))
        except KeyError:
            num_epochs = MAX_EPOCH
            log.event("Maximal epoch/iteration not specified in solver params, hence set to {}".format(MAX_EPOCH))
    log.console("Training will have {} epochs".format(num_epochs))

    last_epoch = -1
    if args["restore"]:
        try:
            latest_snapshot = last_snapshot_path(solver_params["snapshot_prefix"],
                    num_epochs, args)
        except ValueError as exp:
            log.event("Cannot restore from latest snapshot " + exp)
        else:
            log.console("Restoring model from latest snapshot {}".format(latest_snapshot))
            seen_images, last_epoch = restore_state(model, optimizer, latest_snapshot)
    start_epoch = last_epoch + 1
    log.console("Training will start from {} epoch".format(start_epoch))

    if args["use_treestructure"]:  # TODO: implement YoloLoss with loss_mode as an argument
        criterion = RegionTargetWithSoftMaxTreeLoss(args["tree"],
                                                    num_classes=len(cmap),
                                                    seen_images=seen_images)
        log.console("Using tree structure with {} softmax normalization".format(criterion.normalization))

    else:
        criterion = RegionTargetWithSoftMaxLoss(num_classes=len(cmap),
                                                seen_images=seen_images)
        log.console("Using plain structure with {} softmax normalization".format(criterion.normalization))

    criterion = criterion.cuda()

    if args["distributed"]:
        log.console('Syncing machines before training')
        dist_utils.sum_tensor(torch.tensor([1.0]).float().cuda())

    if solver_params.get('lr_policy', 'fixed') == "multifixed":
        scheduler = MultiFixedScheduler(optimizer, steps, lrs,
                                        last_iter=start_epoch * len(data_loader))
    else:
        scheduler = None

    start_time = datetime.now()  # Loading start to after everything is loaded

    loss_arr = []
    epoch = start_epoch
    iterations_left = int(float(solver_params["max_iter"]))
    if iterations_left == 0:
        iterations_left = None
    for epoch in range(start_epoch, num_epochs):
        data_loader.sampler.set_epoch(epoch)
        train(data_loader, model, criterion, optimizer, scheduler, epoch,
                loss_arr, args, iterations_left, log)
        time_diff = (datetime.now() - start_time).total_seconds() / 3600.0
        log.event('{}\t {}\n'.format(epoch, time_diff))
        if args["local_rank"] == 0 and epoch % float(solver_params["snapshot"]) == 0:
            snapshot(model, criterion, loss_arr, epoch, solver_params["snapshot_prefix"],
                    optimizer=optimizer, log=log)

    log.console("Snapshoting final model")
    if args["local_rank"] == 0:
        snapshot(model, criterion, loss_arr, epoch, solver_params["snapshot_prefix"],
                optimizer=optimizer, log=log)

    return get_snapshot(solver_params['snapshot_prefix'], epoch)


if __name__ == '__main__':
    args = get_parser().parse_args()
    args = vars(args)
    args["local_rank"] = dist_utils.env_rank()  # this may be different on a different machine

    is_master = (not args["distributed"]) or (dist_utils.env_rank() == 0)
    is_rank0 = args["local_rank"] == 0
    log = FileLogger(args["logdir"], is_master=is_master, is_rank0=is_rank0)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            main(args, log)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        log.event(e)
        raise SystemExit(exc_traceback)

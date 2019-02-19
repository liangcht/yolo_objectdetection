from __future__ import print_function
import os
import os.path as op
import time
from datetime import datetime
import sys
import argparse
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
from mtorch.dataloaders import yolo_train_data_loader
from mtorch.yolo_v2 import yolo
from mtorch.darknet import darknet_layers
from mtorch.region_target_loss import RegionTargetWithSoftMaxLoss, RegionTargetWithSoftMaxTreeLoss


def to_python_float(t):
    if isinstance(t, (float, int)):
        return t
    if hasattr(t, 'item'):
        return t.item()
    return t[0]


def get_parser():
    """prespares parser of input parameters"""

    parser = argparse.ArgumentParser(description='Run Yolo training')
    parser.add_argument('-d', '--train', type=str, metavar='TRAINDATA_PATH', 
                        help='Path to the training dataset file',
                        required=True)
    parser.add_argument('-m', '--model', type=str, metavar='MODEL_PATH',
                        help='path to latest checkpoint', required=False)
    parser.add_argument('-c', '--is_caffemodel', default=False, action='store_true', 
                        help='if provided, assumes model weights are derived from caffemodel, false by default', 
                        required=False)
    parser.add_argument('-l', '--logdir', help='Log directory', required=False)
    parser.add_argument('-s', '--solver',
                        help='solver file with training parameters',
                        required=True)
    parser.add_argument('--labelmap', type=str, metavar='LABELMAP_PATH',
                        help='path to labelmap', required=True)
    parser.add_argument('-t', '--tree', default='', type=str, metavar='TREE_PATH',
                        help='path to a tree structure, it prediction based on tree is required')
    parser.add_argument('--distributed', default=True, type=bool,
                        help='input False is you do NOT want to use distributed training (default=True)',
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


args = get_parser().parse_args()
args = vars(args)
args["local_rank"] = dist_utils.env_rank() # this may be different on a different machine

is_master = (not args["distributed"]) or (dist_utils.env_rank() == 0)
is_rank0 = args["local_rank"] == 0
if "logdir" in args:
    log = FileLogger(args["logdir"], is_master=is_master, is_rank0=is_rank0)
else:
    import logging as log


def snapshot(model, criterion, losses, epoch, snapshot_prefix, optimizer=None):
    """Takes a snapshot of training procedure
    :param model: model to snapshot
    :param criterion: loss - required for saving seen_images
    :param losses: list of losses
    :param epoch: int, epoch reached so far 
    :param snapshot_prefix: str, file to save the current 
    :param optimizer: torch.optim.SGD - required to continue training properly,
    is not reqired for testing
    """

    snapshot_pt = snapshot_prefix + "_epoch_{}".format(epoch) + '.pt'
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


def train(trn_loader, model, criterion, optimizer, scheduler, epoch, loss_arr):
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
            output = "{:.2f} Epoch {}: Time per iter =  {:.4f}s), loss = {:.4f},batch_total = {}, lr = {}"\
                .format(float(batch_num) / last_batch, epoch, timer.batch_time.val,
                        losses.val, batch_total, scheduler.get_lr())
            log.verbose(output)


def main():
    log.console(args)
    if args["distributed"]:
        log.console('Distributed initializing process group')
        torch.cuda.set_device(args['local_rank'])  # probably local_rank = 0
        dist.init_process_group(backend='nccl', init_method=args["dist_url"], rank=dist_utils.env_rank(),
                                world_size=dist_utils.env_world_size())
        assert (dist_utils.env_world_size() == dist.get_world_size())  # check if there are
        log.console("Distributed: success ({}/{})".format(args["local_rank"], dist.get_world_size()))

    log.console("Loading model")

    model = yolo(darknet_layers(), weights_file=args['model'],
                 caffe_format_weights=args['is_caffemodel']).cuda()
    seen_images = model.seen_images

    if args["distributed"]:
        model = dist_utils.DDP(model, device_ids=[args['local_rank']], output_device=args['local_rank'])

    solver_params = parse_prototxt(args['solver'])
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

    if args["restore"]:
        log.console("Restoring model from latest snapshot")
        checkpoint = torch.load(args["latest_snapshot"],
                                map_location=lambda storage, loc: storage.cuda(args["local_rank"]))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        seen_images = checkpoint['seen_images']
        args["start_epoch"] = checkpoint['epochs'] + 1

    cmap = load_labelmap_list(args["labelmap"])
    if "tree" in args:  # TODO: implement YoloLoss with loss_mode as an argument
        criterion = RegionTargetWithSoftMaxTreeLoss(args["tree"], 
                                                    num_classes=len(cmap),
                                                    seen_images=seen_images)
    else:
        criterion = RegionTargetWithSoftMaxLoss(num_classes=len(cmap), seen_images=seen_images)
    criterion = criterion.cuda()

    if args["distributed"]:
        log.console('Syncing machines before training')
        dist_utils.sum_tensor(torch.tensor([1.0]).float().cuda())

    log.console("Creating data loaders")
    data_loader = yolo_train_data_loader(args["train"], batch_size=args["batch_size"],
                                         num_workers=args["workers"], distributed=args["distributed"])

    if solver_params.get('lr_policy', 'fixed') == "multifixed":
        scheduler = MultiFixedScheduler(optimizer, steps, lrs,
                                        last_iter=args["start_epoch"] * len(data_loader))
    else:
        scheduler = None

    start_time = datetime.now()  # Loading start to after everything is loaded
    num_epochs = int(round(float(solver_params["max_iter"]) / len(data_loader) + 0.5))
    loss_arr = []
    for epoch in range(args["start_epoch"], num_epochs):
        train(data_loader, model, criterion, optimizer, scheduler, epoch, loss_arr)
        time_diff = (datetime.now() - start_time).total_seconds() / 3600.0
        log.event('{}\t {}\n'.format(epoch, time_diff))
        if args["local_rank"] == 0 and epoch % float(solver_params["snapshot"]) == 0:
            snapshot(model, criterion, loss_arr, epoch, solver_params["snapshot_prefix"], optimizer=optimizer)

    log.console("Snapshoting final model")
    if args["local_rank"] == 0:
        snapshot(model, criterion, loss_arr, epoch, solver_params["snapshot_prefix"], optimizer=optimizer)


if __name__ == '__main__':
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            main()
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        log.event(e)

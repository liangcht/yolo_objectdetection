from __future__ import print_function
import os
import os.path as op
from datetime import datetime
import sys
import argparse
import re
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

import mmod.dist_utils as dist_utils
from mmod.simple_parser import parse_prototxt
from mmod.experimental_utils import *
from mmod.meters import AverageMeter, TimeMeter
from mmod.file_logger import FileLogger
from mmod.simple_parser import load_labelmap_list
from mmod.io_utils import load_from_yaml_file
from mtorch.caffesgd import CaffeSGD
from mtorch.multifixed_scheduler import MultiFixedScheduler
from mtorch.dataloaders import yolo_train_data_loader
from mtorch.yolo_v2 import yolo_2extraconv
from mtorch.darknet import darknet_layers
from mtorch.yolo_v2_loss import YoloLossForPlainStructure, YoloLossForTreeStructure


def to_python_float(t):
    if isinstance(t, (float, int)):
        return t
    if hasattr(t, 'item'):
        return t.item()
    return t[0]


def get_parser():
    """prepares parser of input parameters"""
    parser = argparse.ArgumentParser(description='Run Yolo training')
    # Required Parameters
    parser.add_argument('-d', '--train', type=str, metavar='TRAINDATA_PATH',
                        help='Path to the training dataset file',
                        required=True)
    parser.add_argument('-l', '--logdir', help='Log directory', 
                        required=True)
    parser.add_argument('-s', '--solver', metavar='SOLVER_FILE',
                        help='solver file with training parameters, either .prototxt or .yaml',
                        required=True)
    parser.add_argument('--labelmap', type=str, metavar='LABELMAP_PATH',
                        help='path to labelmap', 
                        required=True)
    # Model Initialization Parameters
    parser.add_argument('-m', '--model', type=str, metavar='MODEL_PATH',
                        help='path to latest checkpoint', 
                        required=False)
    parser.add_argument('-c', '--is_caffemodel', default=False, action='store_true',
                        help='if provided, assumes model weights are derived from caffemodel, false by default',
                        required=False)
    parser.add_argument('--only_backbone', default=False, action='store_true',
                        help="if provided, only backbone weights are taken from the model, false by default",
                        required=False)
    parser.add_argument('--ignore_mismatch', default=False, action='store_true',
                        help="if provided, only matching part of model will be loaded and mismatched ignored, false by default",
                        required=False)
    # Model Fine-tuning Parameters
    parser.add_argument('--freeze_features', default=False, action='store_true',
                        help="if provided, featurizer will not be updated during training, false by default",
                        required=False)
    parser.add_argument('--freeze_backbone', default=False, action='store_true',
                        help="if provided, backbone weights will not be updated during training, false by default",
                        required=False)
    parser.add_argument('--freeze_extra_convs', default=False, action='store_true',
                        help="if provided, extra_conv weights will not be updated during training, false by default",
                        required=False)
    parser.add_argument('--freeze', default=False, action='store_true',
                        help="if provided, weights will not be updated during training till specified layer, false by default",
                        required=False)
    parser.add_argument('--freeze_till', default=None, type=str, metavar='FREEZE_TILL_THIS_LAYER',
                        help="must be provided is freeze is specified, weights will not be updated till specified layer, false by default",
                        required=False)
    # Loss Parameters
    parser.add_argument('-t', '--use_treestructure', default=False, action='store_true',
                        help='if provided, will use tree structure, false by default',
                        required=False)
    parser.add_argument('--tree', type=str, metavar='PATH_TO_TREE_STRUCTURE_FILE',
                        help='path to a tree structure, it prediction based on tree is required',
                        required=False)
    # Training Parameters
    parser.add_argument('-wrap', '--wrap', default=True, action='store_false',
                        help="specify if you want to use sampler that wrapps around the dataset similar to Caffe (default=True)",
                        required=False)
    parser.add_argument('-random', '--random', default=False, action='store_true',
                        help="specify if you want to use random sampler of the dataset (default=False)",
                        required=False)
    parser.add_argument('--display', default=False, action='store_true',
                        help='input False if you do NOT want to print progress (default=True)',
                        required=False)
    parser.add_argument('--display_freq', default=1, type=int,
                        help='display frequency in batches (default: 1)')
    parser.add_argument('-r', '--restore', default=False, action='store_true',
                        help='specify if the model should be restored from latest_snapshot, default is false',
                        required=False)
    parser.add_argument('-latest_snapshot', '--latest_snapshot', type=str,  metavar='PATH_TO_SNAPSHOT_FILE',
                        help='Initial snapshot to finetune from', 
                        required=False)
    parser.add_argument('-b', '--batch_size', '--batchsize', default=16, type=int, metavar='PER_GPU_BATCH_SIZE',
                        help='per-gpu batch size (default=16)',
                        required=False)
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='NUM_WORKERS_FOR_DATALOADER',
                        help='number of data loading workers (default: 2)',
                        required=False)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)',
                        required=False)
    parser.add_argument('--max_epoch', default=128, type=int, metavar='N',
                        help='maximum number of epoch',
                        required=False)
    parser.add_argument('--min_iters_in_epoch', default=70*4, type=int, metavar='N',
                        help='minimum number of iterations per epoch',
                        required=False)
    # Distributed Training Parameters
    parser.add_argument('--distributed', default=False, action='store_true',
                        help='specify if you want to use distributed training (default=False)',
                        required=False)
    parser.add_argument('--local_rank', help='local_rank', metavar='Local Rank of the GPU',
                        required=False)
    parser.add_argument('--dist_url', default="tcp://127.0.0.1:2345",
                        help='dist_url',
                        required=False)

    return parser


def last_snapshot_path(prefix, max_epoch=None, args=None):
    """Find the last checkpoints inside given prefix paths
    Look at the paths in prefixes, and return once found
    :param prefix: str, paths to check for the snapshot
    :param max_epoch: maximum number of iterations that could be found
    :param args: input arguments
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
    snapshot_path = ""
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
    """concatenates shapshot_prefix and epoch information"""
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
        'region_target.biases': criterion.criterion.region_target.biases,
        'region_target.seen_images': criterion.criterion.seen_images
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
          args, log, iterations_left=None):
    """ training cycle of one epoch
    :param trn_loader: utils.data.DataLoader, data loader for training
    :param model: torch.nn.Module or nn.Sequential, model
    :param criterion: torch.nn.Module, loss
    :param optimizer: torch.optim, optimizer to update model
    :param scheduler: torch.optim.lr_scheduler, takes care of weight decay and learning rate updates
    :param epoch: int, current epoch
    :param loss_arr: list, stores all the losses
    :param args: dict, training parameters
    :param log: for logging epoch information
    :param iterations_left: the remaining number of iterations (added to provide compatibility with Caffe)
     """
    
    if iterations_left and iterations_left < 0:
        return

    timer = TimeMeter()
    losses = AverageMeter()

    last_batch = len(trn_loader)
    print("last_batch {}".format(last_batch))
    print("epoch_number {}".format(epoch))
    for i, inputs in enumerate(trn_loader):
        if iterations_left and iterations_left == 0:
            break
        batch_num = i + 1
        print("batch_num {}".format(batch_num))
        data, labels = inputs[0].cuda(), inputs[1].cuda().float()
        torch.save([data, labels], './test/debug_batch_input')
        scheduler.step()
        timer.batch_start()
        optimizer.zero_grad()

        features = model(data)
        loss = criterion(features, labels)

        if loss != loss:
            # keep telemetry for debugging NaN loss
            torch.save([epoch, data, labels, features, loss], 'crash' + str(args["local_rank"]) + '.pt')
            if args["local_rank"] == 0:
                solver_params = parse_prototxt(args['solver'])
                snapshot(model, criterion, loss_arr, epoch, solver_params["snapshot_prefix"], log, optimizer=optimizer)
            nan_ex = "Batch {} in epoch {}: NaN loss!".format(batch_num, epoch)
            log.event(nan_ex)
            raise Exception(nan_ex)
 
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

        should_print = (args["display"] and batch_num % args["display_freq"] == 0) or batch_num == last_batch
        if should_print:
            if iterations_left:
                output = "{:.2f} Epoch {}: Time per iter = {:.4f}s, Time left = {:.2f}h), loss = {:.4f}, batch_total = {}, lr = {}" \
                    .format(float(batch_num) / last_batch, epoch, timer.batch_time.val,
                            (timer.batch_time.val * iterations_left) / 3600.,
                            losses.val, batch_total, scheduler.get_lr())
            else:
                output = "{:.2f} Epoch {}: Time per iter =  {:.4f}s), loss = {:.4f},batch_total = {}, lr = {}" \
                    .format(float(batch_num) / last_batch, epoch, timer.batch_time.val,
                            losses.val, batch_total, scheduler.get_lr())

            log.verbose(output)

        if iterations_left:
            iterations_left -= 1


def main(args, log):
    log.console(args)
    if args["distributed"]:
        log.console('Distributed initializing process group')
        if not dist.is_initialized():
            torch.cuda.set_device(args['local_rank'])  # probably local_rank = 0
            dist.init_process_group(backend='nccl', init_method=args["dist_url"], rank=dist_utils.env_rank(),
                                    world_size=dist_utils.env_world_size())
        else:
            assert torch.cuda.current_device() == args['local_rank']
        assert (dist_utils.env_world_size() == dist.get_world_size())  # check if there are enough GPUs
        log.console("Distributed: success ({}/{})".format(args["local_rank"], dist.get_world_size()))

    log.console("Loading model")
    cmap = load_labelmap_list(args["labelmap"])

    if args['only_backbone']:
        model = yolo_2extraconv(darknet_layers(weights_file=args['model'],
                                               caffe_format_weights=args['is_caffemodel'],
                                               map_location=lambda storage, loc: storage.cuda(args['local_rank'])),
                                num_classes=len(cmap)).cuda()
        log.console("Only backbone pretrained model was loaded")
    else:
        model = yolo_2extraconv(darknet_layers(),
                                weights_file=args['model'],
                                caffe_format_weights=args['is_caffemodel'],
                                ignore_mismatch=args["ignore_mismatch"],
                                num_classes=len(cmap),
                                map_location=lambda storage, loc: storage.cuda(args['local_rank'])).cuda()
        if model.pretrained_info:
            log.console("Pretrained model was loaded: " + model.pretrained_info)
    seen_images = model.seen_images

    # switch to train mode
    model.train()

    if args["freeze_features"]:
        model.freeze_features()
        log.console("Freezes all the feature layers")
    elif args["freeze"]:
        model.freeze(args["freeze_till"])
        log.console("Freezes till " + args["freeze_till"])
    else:
        if args["freeze_backbone"]:
            model.freeze_backbone(args["freeze_till"])
            log.console("Freezes backbone")
        if args["freeze_extra_convs"]:
            model.freeze_extra_convs(args["freeze_till"])
            log.console("Freezes extra convolutional layers")

    if args["distributed"]:
        model = dist_utils.DDP(model, device_ids=[args['local_rank']], output_device=args['local_rank'])

    if args['solver'].endswith('.prototxt'):
        solver_params = parse_prototxt(args['solver'])
    elif args['solver'].endswith('.yaml'):
        solver_params = load_from_yaml_file(args['solver'])
    else:
        ex = "Solver params cannot be read"
        log.event(ex)
        raise Exception(ex)

    lrs = get_lrs(solver_params)
    steps = get_steps(solver_params)

    # code below sets Caffe compatible learning rate and weight decay hyper parameters
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
    data_loader = yolo_train_data_loader(args)
    if args["wrap"]:
        log.console("Uses Wrapping Sampler")
    if args["random"]:
        log.console("Uses Random Sampler")

    try:
        num_epochs = int(solver_params["max_epoch"])
    except KeyError:
        try:
            num_epochs = int(round(float(solver_params["max_iter"]) / len(data_loader) + 0.5))
        except KeyError:
            num_epochs = args["max_epoch"]
            log.event("Maximal epoch/iteration not specified in solver params, hence set to {}".format(args["max_epoch"]))
    log.console("Training will have {} epochs".format(num_epochs))

    last_epoch = -1
    if args["restore"]:
        try:
            latest_snapshot = last_snapshot_path(solver_params["snapshot_prefix"],
                                                 num_epochs, args)
        except ValueError as exp:
            log.event("Cannot restore from latest snapshot ")
        else:
            log.console("Restoring model from latest snapshot {}".format(latest_snapshot))
            seen_images, last_epoch = restore_state(model, optimizer, latest_snapshot)

    start_epoch = last_epoch + 1
    log.console("Training will start from {} epoch".format(start_epoch))

    if args["use_treestructure"]:  
        criterion = YoloLossForTreeStructure(args["tree"], num_classes=len(cmap), seen_images=seen_images)
    else:
        criterion = YoloLossForPlainStructure(num_classes=len(cmap), seen_images=seen_images)
    log.console(str(criterion))

    criterion = criterion.cuda()

    if args["distributed"]:
        log.console('Syncing machines before training')
        dist_utils.sum_tensor(torch.tensor([1.0]).float().cuda())
    
    if solver_params.get('lr_policy', 'fixed') == "multifixed":
        log.console("Using Multifixed Scheduler")
        scheduler = MultiFixedScheduler(optimizer, steps, lrs,
                                        last_iter=start_epoch * len(data_loader))
    else:
        scheduler = None

    start_time = datetime.now()  # Loading start to after everything is loaded

    loss_arr = []  # Accumulates the losses
    epoch = start_epoch

    try:
        max_iters = solver_params["max_iter"]
    except KeyError:
        iterations_left = None
    else:
        iterations_left = int(float(max_iters))

    for epoch in range(start_epoch, num_epochs):
        data_loader.sampler.set_epoch(epoch)
        train(data_loader, model, criterion, optimizer, scheduler, epoch,
              loss_arr, args, log, iterations_left)
        time_diff = (datetime.now() - start_time).total_seconds() / 3600.0
        log.event('{}\t {}\n'.format(epoch, time_diff))
        if args["local_rank"] == 0 and epoch % float(solver_params["snapshot"]) == 0:
            snapshot(model, criterion, loss_arr, epoch, solver_params["snapshot_prefix"],
                     optimizer=optimizer, log=log)

    log.console("Snapshotting final model")
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


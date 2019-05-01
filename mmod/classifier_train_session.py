import argparse
import os
import os.path as op
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

from mmod import dist_utils as dist_utils
from mmod.logger import Logger, DistributedLogger
from mmod.simple_parser import load_labelmap_list
from mmod.tax_utils import create_inverted
from mmod.imdb import ImageDatabase
from mtorch.weights_init import dense_layers_init
from mtorch.custom_resnet import ResNetWithFeatures
from mtorch.classifier_dataloaders import region_classifier_data_loader
from mtorch.classifier_train_utils import get_criterion, get_optimizer, get_scheduler, train

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def save_checkpoint(state, prefix, epoch, output_dir, is_best=False):
    filename = os.path.join(output_dir, '%s-%04d.pth.tar' % (prefix, epoch))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(output_dir, 'model_best.pth.tar'))


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # necessary inputs
    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--labelmap', metavar='DIR',
                        help='path to dataset')

    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # has default hyper parameter for ResNet
    parser.add_argument('--epochs', default=121, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--is_multi_label', default=False, action='store_true')
    parser.add_argument('--ccs_loss_param', default=0.0, type=float)
    # distributed training
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', help='local_rank', required=False)
    parser.add_argument('--dist_url', default="tcp://127.0.0.1:2345",
                        help='dist_url')
    parser.add_argument('--distributed', default=False, action='store_true',
                        help='specify if you want to use distributed training (default=False)',
                        required=False)
    # need setup output dir
    parser.add_argument('--output-dir', default='./outputs/resnet18', type=str,
                        help='path to save checkpoint and log (default: ./outputs/resnet18)')
    parser.add_argument('--prefix', default=None, type=str,
                        help='model prefix (default: same with model names)')
    # Optimization setting
    parser.add_argument('--balance', default=False, action='store_true',
                        help='balance cross entropy weights')
    parser.add_argument('--lr-policy', default='STEP', type=str,
                        help='learning rate decay policy: STEP, MULTISTEP, EXPONENTIAL, PLATEAU, CONSTANT '
                        '(default: STEP)')
    parser.add_argument('--step-size', default=30, type=int,
                        help='step size for STEP decay policy (default: 30)')
    parser.add_argument('--milestones', default='30,60,90', type=str,
                        help='milestones for MULTISTEP decay policy (default: 30,60,90)')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='gamma for learning rate decay (default: 0.1)')
    parser.add_argument('--neg', dest='neg_weight_file', default=None,
                        help='weights of negative samples used in multi-label training. If specified, balanced loss'
                             ' will be used, otherwise, BCELoss will be used.')
    # force using customized hyper parameter
    parser.add_argument('-f', '--force', dest='force', action='store_true',
                        help='force using customized hyper parameter')
    parser.add_argument('--finetune', dest='finetune', action='store_true',
                        help='finetune last layer by using 0.1x lr for previous layers')
    # display
    parser.add_argument('--display_freq', default=30, type=int,
                        help='display frequency in iterations (default: 30)')

    return parser


def main():

    args = get_parser().parse_args()
    if args.distributed:
        args.local_rank = dist_utils.env_rank()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)  # probably local_rank = 0
        dist.init_process_group(backend='nccl', init_method=args.dist_url, rank=dist_utils.env_rank(),
                                world_size=dist_utils.env_world_size())
        assert (dist_utils.env_world_size() == dist.get_world_size())  # check if there are

    if not args.distributed:
        logger = Logger(args.output_dir, args.prefix)
    else:
        try:
            logger = DistributedLogger(args.output_dir, args.prefix, args.local_rank)
        except:
            logger.info('Cannot create logger, rank:', args.local_rank)
 
    logger.info('distributed? {}'.format(args.distributed))
   
    if args.local_rank == 0:
        logger.info('called with arguments: {}'.format(args))

    imdb = ImageDatabase(args.data)  # this might need change for different application
    if args.balance:
        # creating inverted file (if does not exit)
        # this is useful for histogram calculation
        inverted_file_path = imdb.inverted_path
        if not op.isfile(inverted_file_path):
            create_inverted(imdb, path=inverted_file_path, labelmap=args.labelmap,
                            create_labelmap=not op.isfile(args.labelmap))
    
    if not op.isfile(args.labelmap):
        args.labelmap = imdb.cmapfile
    if not op.isfile(args.labelmap):
        create_inverted(imdb, labelmap=args.labelmap, create_labelmap=True)
 
    cmap = load_labelmap_list(args.labelmap)
    num_classes = len(cmap)
    
    # create model
    if args.pretrained:
        logger.info("=> using pre-trained model '{}'".format(args.arch))
        if args.arch.startswith('resnet'):
            from torchvision.models.resnet import model_urls
        elif args.arch.startswith('alexnet'):
            from torchvision.models.alexnet import model_urls
        elif args.arch.startswith('vgg'):
            from torchvision.models.vgg import model_urls
        model_urls[args.arch] = model_urls[args.arch].replace('https://', 'http://')
        model = models.__dict__[args.arch](pretrained=True)
        if args.arch.startswith('alexnet'):
            classifier = list(model.classifier.children())
            model.classifier = nn.Sequential(*classifier[:-1])
            model.classifier.add_module(
                '6', nn.Linear(classifier[-1].in_features, num_classes))
        else:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            dense_layers_init(model)
        
    else:
        logger.info("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=num_classes)

    if args.ccs_loss_param > 0:
        model = ResNetWithFeatures(model)

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = dist_utils.DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Data loading code
    # this part might need changes per specific application
    data_loader = region_classifier_data_loader(args.data, pos_conf=0.1,
                                                cmapfile=args.labelmap,
                                                batch_size=args.batch_size, num_workers=args.workers,
                                                distributed=True)
    if args.balance:
        raise NotImplementedError("Currently balance is not supported")

    criterion_no_weighted = get_criterion(args.is_multi_label, args.neg_weight_file, cross_entropy_weights=None)

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    if args.distributed:
        dist_utils.sum_tensor(torch.tensor([1.0]).float().cuda())

    args.epochs = 121
    for epoch in range(args.start_epoch, args.epochs):
        epoch_tic = time.time()
        if args.distributed:
            data_loader.sampler.set_epoch(epoch)

        scheduler.step()

        criterion = criterion_no_weighted

        train(args, data_loader, model, criterion, optimizer, epoch, logger)

        if args.local_rank == 0 and epoch % args.display_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'num_classes': num_classes,
                'multi_label': args.is_multi_label,
                'labelmap': cmap,
            }, args.prefix, epoch+1, args.output_dir)
            info_str = 'Epoch: [{0}]\t' \
                       'Time {time:.3f}\t'.format(epoch, time=time.time() - epoch_tic)
            logger.info(info_str)


if __name__ == '__main__':
    torch.manual_seed(2018)
    main()

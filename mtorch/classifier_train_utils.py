import time
from mtorch.classifier_accuracy import get_accuracy_calculator
from mmod.meters import AverageMeter
import torch
import torch.nn as nn
import torch.optim
from torch.optim import lr_scheduler
from bisect import bisect_right
import numpy as np
from mtorch.sigmoid_cross_entropy_loss_with_balancing import SigmoidCrossEntropyLossWithBalancing
from math import exp, ceil
from mtorch.ccs_loss import CCSLoss


def get_criterion(multi_label=False, multi_label_negative_sample_weights_file=None,
                  cross_entropy_weights=None):
    if multi_label:
        if not multi_label_negative_sample_weights_file:
            print("Use BCEWithLogitsLoss")
            criterion = nn.BCEWithLogitsLoss().cuda()
        else:
            print("Use SigmoidCrossEntropyLossWithBalancing")
            with open(multi_label_negative_sample_weights_file, "r") as f:
                weights = [float(line) for line in f]
                criterion = SigmoidCrossEntropyLossWithBalancing(np.array(weights)).cuda()
    else:
        print("Use CrossEntropyLoss")
        if cross_entropy_weights:
            cross_entropy_weights = torch.tensor(cross_entropy_weights)
        criterion = nn.CrossEntropyLoss(weight=cross_entropy_weights).cuda()

    return criterion


def get_init_lr(args):
    if args.start_epoch == 0:
        return args.lr
    if args.lr_policy.lower() == 'step':
        lr = args.lr * args.gamma ** (args.start_epoch // args.step_size)
    elif args.lr_policy.lower() == 'multistep':
        milestones = [int(m) for m in args.milestones.split(',')]
        lr = args.lr * args.gamma ** bisect_right(milestones, args.start_epoch)
    elif args.lr_policy.lower() == 'exponential':
        lr = args.lr * args.gamma ** args.start_epoch
    elif args.lr_policy.lower() == 'plateau':
        assert args.start_epoch == 0, 'cannot resume training for plateau'
        lr = args.lr
    else:
        raise ValueError('Unknown lr policy: {}'.format(args.lr_policy))
    return lr


def set_default_hyper_parameter(args):
    args.epochs = 120
    args.batch_size = 256
    args.lr = 0.1
    args.momentum = 0.9
    args.weight_decay = 1e-4
    args.lr_policy = 'STEP'
    args.step_size = 30
    args.gamma = 0.1


def get_optimizer(model, args):
    # use default parameter for reproducible network
    if not args.force:
        print('Use default hyper parameter')
        set_default_hyper_parameter(args)

    init_lr = get_init_lr(args)
    print('initial learning rate: %f' % init_lr)

    if args.start_epoch > 0:
        groups = [dict(params=list(model.parameters()), initial_lr=init_lr)]
    else:
        groups = model.parameters()
    
    optimizer = torch.optim.SGD(groups, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.finetune:
        group_pretrained = []
        group_new = []
        for name, param in model.named_parameters():
            if 'fc' in name:
                group_new.append(param)
            else:
                group_pretrained.append(param)
        assert len(list(model.parameters())) == len(group_pretrained) + len(group_new)
        groups = [dict(params=group_pretrained, lr=args.lr*0.01, initial_lr=init_lr*0.01),
                    dict(params=group_new,  lr=args.lr, initial_lr=init_lr)]
        optimizer = torch.optim.SGD(groups, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    return optimizer


def get_scheduler(optimizer, args):
    if args.lr_policy.lower() == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma,
                                        last_epoch=args.start_epoch-1)
    elif args.lr_policy.lower() == 'multistep':
        milestones = [int(m) for m in args.milestones.split(',')]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma,
                                             last_epoch=args.start_epoch - 1)
    elif args.lr_policy.lower() == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma,
                                               last_epoch=args.start_epoch - 1)
    elif args.lr_policy.lower() == 'plateau':
        assert args.start_epoch == 0, 'cannot resume training for plateau'
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose=True)
    elif args.lr_policy.lower() == 'constant':
        scheduler = None
    else:
        raise ValueError('Unknown lr policy: {}'.format(args.lr_policy))

    return scheduler


def train(args, train_loader, model, criterion, optimizer, epoch, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    orig_losses = AverageMeter()
    ccs_losses = AverageMeter()
    ccs_loss_layer = CCSLoss()
    ccs_loss_param = args.ccs_loss_param
    
    accuracy = get_accuracy_calculator(multi_label=not isinstance(criterion, nn.CrossEntropyLoss))

    # switch to train mode
    model.train()

    end = time.time()
    tic = time.time()
    for i, (in_data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
   
        # compute output
        all_outputs = model(in_data)

        if ccs_loss_param > 0:
            output, feature = all_outputs[0], all_outputs[1]
            orig_loss = criterion(output, target)
            # NOTE: use detach() to not calculate grad w.r.t. weight in ccs_loss
            weight = model.module.fc.weight
            ccs_loss = ccs_loss_layer(feature, weight, target)
            orig_losses.update(orig_loss.item(), in_data.size(0))
            ccs_losses.update(ccs_loss.item(), in_data.size(0))
            loss = orig_loss + ccs_loss_param * ccs_loss

        else:
            output = all_outputs
            orig_loss = criterion(output, target)
            loss = orig_loss

        # measure accuracy and record loss
        accuracy.calc(output, target)
        losses.update(loss.item(), in_data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            speed = args.print_freq * args.batch_size / float(args.world_size) / (time.time() - tic)
            info_str = 'Epoch: [{0}][{1}/{2}]\t' \
                       'Speed: {speed:.2f} samples/sec\t' \
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        epoch, i, len(train_loader), speed=speed, batch_time=batch_time,
                        data_time=data_time, loss=losses)
            info_str += accuracy.result_str()
            logger.info(info_str)
            tic = time.time()


def get_labels_hist(imdb, normalize=False):
    labels_hist = []
    for i, label in enumerate(imdb.iter_cmap()):
        keys = list(imdb.iter_label(label))
        labels_hist.append(len(keys))
    if normalize:
        total_labels = sum(labels_hist)
        for i in range(len(labels_hist)):         
            labels_hist[i] /= float(total_labels) 
   
    return labels_hist


def get_balance_weights(imdb):
    labels_hist = get_labels_hist(imdb, normalize=True)
    balance_weigths = []
    temperature = 1.0 / len(labels_hist)
    for i, count in enumerate(labels_hist):
        balance_weigths.append(0 if count == 0 else max(temperature, exp(-count ** 2 / (2 * temperature ** 2))))
    balance_weigths[-2] = 0.99
    sum_all = sum(balance_weigths)
    for i, count in enumerate(labels_hist):
        balance_weigths[i] /= sum_all
    return balance_weigths 
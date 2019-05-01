import os.path as op
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from mtorch.classifier_dataloaders import region_classifier_test_data_loader
from mmod.meters import AverageMeter
from mmod.simple_parser import load_labelmap_list
from mmod.utils import open_with_lineidx
from mmod.imdb import ImageDatabase

import json

DEBUG_MODE = False  # TODO:remove in final version


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Classifier Prediction')
    # necessary inputs
    parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
    parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--labelmap', default='', type=str, metavar='PATH',
                    help='path to labelmap')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 16)') 
    parser.add_argument('-k', '--topk', default=1, type=int,
                    metavar='K', help='top k result (default: 1)')
    parser.add_argument('--output', required=False, type=str, metavar='PATH',
                    help='path to save prediction result')
    return parser


def _load_tsv(result_file):
    with open(result_file, 'r') as f:
        lines = f.readlines()
    df_records = {}
    ordered_keys = []
    for line in lines:
        cols = line.split('\t')
        df_records[cols[0]] = cols[1:]
        ordered_keys.append(cols[0])
    return df_records, ordered_keys


def _adjust_dict(ref_dict, attach_word, arch):
    new_dict = {}
    if arch.startswith('alexnet'):
        for key, value in ref_dict.items():
            if key.startswith('features'):
                new_dict[key.replace('features', 'features.' + attach_word)] = value
            elif key.startswith('classifier'):
                new_dict[key]  = value
    elif arch.startswith('resnet'):
        for key, value in ref_dict.items():
            new_dict[attach_word + '.' + key] = value
    return new_dict


def original_format(results, test_file):
    orig_data, orig_keys = _load_tsv(test_file)
    db = ImageDatabase(test_file)
    results_org = []
    prev_image_key = None
    prev_uid = None
    result_org = []
    results_dict = {}
    for (uid, image_key, result) in results:
        if not prev_image_key:
            prev_image_key = image_key
            prev_uid = uid
        if prev_image_key != image_key:
            results_dict[prev_image_key] = (prev_uid, prev_image_key, result_org)
            result_org = []
      
        result_org.append(result)
        prev_image_key = image_key
        prev_uid = uid 
    results_dict[prev_image_key] = (prev_uid, prev_image_key, result_org)
    
    for key in db:
        if db.image_key(key) in results_dict:
            results_org.append(results_dict[db.image_key(key)]) 
        else:
            results_org.append((db.uid(key), db.image_key(key), []))

    return results_org             


def load_model(args):
    checkpoint = torch.load(args.model)
    arch = checkpoint['arch']
    model = models.__dict__[arch](num_classes=checkpoint['num_classes'])
    if arch.startswith('alexnet') or arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(_adjust_dict(checkpoint['state_dict'], 'module', arch))

    print("=> loaded checkpoint '{}' (epoch {})".format(args.model, checkpoint['epoch']))

    cudnn.benchmark = True

    # switch to evaluate mode
    model.eval()

    # load labelmap
    if args.labelmap:
        labelmap = load_labelmap_list(args.labelmap)
    elif 'labelmap' in checkpoint:
        labelmap = model['labelmap']
    else:
        labelmap = [str(i) for i in range(checkpoint['num_classes'])]

    return model, labelmap


def write_predict(outtsv_file, results):
    """Save prediction results to tsv file
    :param outtsv_file: str, file to write results to
    :param results: str, results to write
    """
    with open_with_lineidx(outtsv_file, with_temp=True) as fp, open_with_lineidx(outtsv_file + ".keys", with_temp=True) as kfp:
            for (uid, image_key, result) in results:
                tell = fp.tell()
                fp.write("{}\t{}\n".format(
                    image_key,
                    json.dumps(result, separators=(',', ':'), sort_keys=True),
                ))
                kfp.write("{}\t{}\n".format(
                    uid, tell
                ))


def main():
    if DEBUG_MODE:
        args_list = ['--data', '/work/VIACOM/episodes/796e2e2060/testX.tsv', '--labelmap',
                    '/work/VIACOM/episodes/4074e0c2a0/labelmap.txt', '--model', 
                    '/work/VIACOM/classifier/with_background_v3_resnet_lr_0.01_decay_0.001_resize_256_b64_2-0051.pth.tar']
        args = get_parser().parse_args(args_list)
    else:
        args = get_parser().parse_args()

    add2name = "_diff_eval"
    if args.output is None:
        args.output = op.join(args.model + args.data.replace('/', '_') + add2name + '.predict')

    val_loader = region_classifier_test_data_loader(args.data, args.labelmap,
                                                    batch_size=args.batch_size, num_workers=args.workers)

    model, labelmap = load_model(args)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    tic = time.time()
    results = []
    with torch.no_grad(), open(args.output, 'w') as fout:
        end = time.time()
        for i, inputs in enumerate(val_loader):
            in_data, keys, img_keys, bbox_rects = inputs[0], inputs[1], inputs[2], inputs[3]
            data_time.update(time.time() - end)
    
            # compute output
            output = model(in_data)
            output = output.cpu()

            _, pred_topk = output.topk(args.topk, dim=1, largest=True)

            for n, (key, img_key, bbox_rect) in enumerate(zip(keys, img_keys, bbox_rects)):
                pred = {"class": labelmap[pred_topk[n, 0]], "conf":  output[n, pred_topk[n, 0]].item(),
                        "rect": bbox_rect}
                results.append((key, img_key, pred))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                speed = 100 * args.batch_size / (time.time() - tic)
                info_str = 'Test: [{0}/{1}]\t' \
                            'Speed: {speed:.2f} samples/sec\t' \
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            i, len(val_loader), speed=speed, batch_time=batch_time,
                            data_time=data_time)
                print(info_str)
                tic = time.time()
    
    print("Prediction results will be saved to {}".format(args.output))
    write_predict(args.output, original_format(results, args.data))


if __name__ == '__main__':
    main()

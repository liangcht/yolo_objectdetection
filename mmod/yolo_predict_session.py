import os
import os.path as op
import sys
import time
import json
import argparse
import torch

try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = op.abspath(this_file)

if __name__ == '__main__':
    # When run as script, modify path assuming absolute import
    sys.path.append(op.join(op.dirname(this_file), '..'))

from mmod.file_logger import FileLogger
from mmod.meters import AverageMeter
from mmod.detection import result2bblist
from mmod.simple_parser import load_labelmap_list
from mmod.utils import open_with_lineidx
from mtorch.dataloaders import yolo_test_data_loader
from mtorch.yolo_predict import PlainPredictorSingleClassNMS, PlainPredictorClassSpecificNMS, \
                                TreePredictorSingleClassNMS, TreePredictorClassSpecificNMS
from mtorch.yolo_v2 import yolo_2extraconv
from mtorch.darknet import darknet_layers


def get_parser():
    """prepares and returns argument parser"""
    parser = argparse.ArgumentParser(description='PyTorch Yolo Prediction')
    parser.add_argument('-d', '--test', metavar='TESTSET_PATH',
                        help='full path to dataset tsv file', required=True)
    parser.add_argument('-m', '--model', type=str, metavar='MODEL_PATH',
                        help='path to latest checkpoint', required=True)
    parser.add_argument('-c', '--is_caffemodel', default=False, action='store_true',
                        help='if provided, assumes model weights are derived from caffemodel, false by default',
                        required=False)
    parser.add_argument('--labelmap', type=str, metavar='LABELMAP_PATH',
                        help='path to labelmap', required=True)
    parser.add_argument('--single_class_nms', default=False, action='store_true',
                        help='if provided, will use single class nms (faster but lower mAP), false by default',
                        required=False)
    parser.add_argument('--tree', type=str, metavar='TREE_PATH',
                        help='path to a tree structure, it prediction based on tree is required',
                        required=False)
    parser.add_argument('-t', '--use_treestructure', default=False, action='store_true',
                        help='if provided, will use tree structure, false by default',
                        required=False)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='NUM_WORKERS',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='BATCH_SIZE', help='mini-batch size (default: 8)')
    parser.add_argument('--thresh', type=float, metavar='N',
                        help='confidence threshold for final prediction')
    parser.add_argument('--obj_thresh', type=float, metavar='N',
                        help='objectness threshold for final prediction')
    parser.add_argument('--output', type=str, metavar='PATH',
                        help='path to save prediction result')
    parser.add_argument('-l', '--logdir', help='Log directory, if log info is required', required=False)
    parser.add_argument('--log_interval', default=1, type=int,
                        help='number of itertation to print predict progress')
    return parser


def load_model(num_classes, args, num_extra_convs=2):
    """creates a yolo model for evaluation
    :param num_classes: int, number of classes to detect
    :param num_extra_convs: int, number of extra convolutional layers to add to featurizer (default=3)
    :return model: nn.Sequential or nn.Module
    """
    model = torch.nn.DataParallel(
        yolo_2extraconv(darknet_layers(), weights_file=args['model'],
                        caffe_format_weights=args['is_caffemodel'],
                        num_classes=num_classes).cuda()
    )
    model.eval()
    return model


def write_predict(outtsv_file, results):
    """Save prediction results to tsv file
    :param outtsv_file: str, file to write results to
    :param results: str, results to write
    """
    try:
        with open_with_lineidx(outtsv_file, with_temp=True) as fp, \
                open_with_lineidx(outtsv_file + ".keys", with_temp=True) as kfp:
            for (uid, image_key, result) in results:
                tell = fp.tell()
                fp.write("{}\t{}\n".format(
                    image_key,
                    json.dumps(result, separators=(',', ':'), sort_keys=True),
                ))
                kfp.write("{}\t{}\n".format(
                    uid, tell
                ))
    except Exception as e:
        log.event(e)


def get_predictor(num_classes, args):
    """creates a yolo model for evaluation
    :param num_classes, int, number of classes to detect
    :return model: nn.Sequential or nn.Module
    """
    if args["use_treestructure"]:
        if args["single_class_nms"]:
            return TreePredictorSingleClassNMS(args['tree'], num_classes=num_classes).cuda()
        return TreePredictorClassSpecificNMS(args['tree'], num_classes=num_classes).cuda()
    else:
        if args["single_class_nms"]:
            return PlainPredictorSingleClassNMS(num_classes=num_classes).cuda()
        return PlainPredictorClassSpecificNMS(num_classes=num_classes).cuda()


def main(args, log):
    add2name = ""
    if args["output"] is None:
        add2name += '.single_class_nms' if args["single_class_nms"] else '.class_specific_nms'
        args["output"] = op.join(args["model"] + args["test"].replace('/', '_') + add2name + '.predict')

    log.console("Creating test data loader")
    test_loader = yolo_test_data_loader(args["test"], cmapfile=args["labelmap"],
                                        batch_size=args["batch_size"],
                                        num_workers=args["workers"])

    cmap = load_labelmap_list(args["labelmap"])

    log.console("Loading model")
    model = load_model(len(cmap), args)
    yolo_predictor = get_predictor(len(cmap), args)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    tic = time.time()
    results = list()
    end = time.time()
    for i, inputs in enumerate(test_loader):
        data_time.update(time.time() - end)

        data, keys, image_keys, hs, ws = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]

        # compute output
        for im, key, image_key, h, w in zip(data, keys, image_keys, hs, ws):
            im = im.unsqueeze_(0)
            im = im.float().cuda()
            with torch.no_grad():
                features = model(im)
            prob, bbox = yolo_predictor(features, torch.Tensor((h, w)))

            bbox = bbox.cpu().numpy()
            prob = prob.cpu().numpy()

            assert bbox.shape[-1] == 4
            bbox = bbox.reshape(-1, 4)
            prob = prob.reshape(-1, prob.shape[-1])
            result = result2bblist((h, w), prob, bbox, cmap,
                                   thresh=args["thresh"], obj_thresh=args["obj_thresh"])
            results.append((key, image_key, result))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args["log_interval"] == 0:
            speed = args["log_interval"] * args["batch_size"] / (time.time() - tic)
            info_str = 'Test: [{0}/{1}]\t' \
                       'Speed: {speed:.2f} samples/sec\t' \
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                       'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        i, len(test_loader), speed=speed, batch_time=batch_time, data_time=data_time)
            log.verbose(info_str)
            tic = time.time()

    log.console("Prediction is done, saving results")
    log.console("Prediction results will be saved to {}".format(args["output"]))
    write_predict(args["output"], results)


if __name__ == '__main__':
    args = get_parser().parse_args()
    args = vars(args)

    if "logdir" in args:
        log = FileLogger(args["logdir"], is_master=True, is_rank0=True)
    else:
        import logging as log

    main(args, log)

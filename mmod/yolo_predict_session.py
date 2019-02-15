import torch
import os
import time
import json
import argparse

from mtorch.dataloaders import yolo_test_data_loader
from mtorch.yolo_predict import TreePredictor
from mtorch.yolo_v2 import yolo
from mtorch.darknet import darknet_layers
from mmod.file_logger import FileLogger
from mmod.meters import AverageMeter
from mmod.detection import result2bblist
from mmod.simple_parser import load_labelmap_list


def get_parser():
    """prepares and returns argument parser"""
    parser = argparse.ArgumentParser(description='PyTorch Yolo Prediction')
    parser.add_argument('-d', '--test', metavar='TESTSET_PATH',
                        help='full path to dataset tsv file', required=True)
    parser.add_argument('-m', '--model', type=str, metavar='MODEL_PATH',
                        help='path to latest checkpoint', required=True)
    parser.add_argument('--labelmap', type=str, metavar='LABELMAP_PATH',
                        help='path to labelmap', required=True)
    parser.add_argument('-t', '--tree', default='', type=str, metavar='TREE_PATH',
                        help='path to a tree structure, it prediction based on tree is required')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 1)')
    parser.add_argument('--thresh', default=0.1, type=float, metavar='N',
                        help='threshold for final prediction (default: 0.1)')
    parser.add_argument('--obj_thresh', default=0.02, type=float, metavar='N',
                        help='objectness threshold for final prediction(default: 0.02)')
    parser.add_argument('--output', default='', type=str, metavar='PATH',
                        help='path to save prediction result')
    parser.add_argument('-l', '--logdir', help='Log directory, if log info is required', required=False)
    parser.add_argument('--log_interval', default=1, type=int,
                        help='number of itertation to print predict progress')
    return parser


TO_JSON = True
args = get_parser().parse_args()
args = vars(args)

if "logdir" in args:
    log = FileLogger(args["logdir"], is_master=True, is_rank0=True)
else:
    import logging as log


def load_model():
    """creates a yolo model for evaluation"""
    model = yolo(darknet_layers(), weights_file=args['model'], caffe_format_weights=True).cuda()
    model = torch.nn.DataParallel(model)
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
        log.info("Exception {}".format(e))
        raise


def get_predictor():
    if "tree" in args:
        return TreePredictor(args['tree']).cuda()
    raise NotImplementedError("Currently plain prediction is not supported, please provide tree structure")


def main():
    log.console("Creating test data loader")
    test_loader = yolo_test_data_loader(args["test"], batch_size=args["batch_size"],
                                        num_workers=args["workers"])

    log.console("Loading model")
    model = load_model()

    yolo_predictor = get_predictor()
    cmap = load_labelmap_list(args["labelmap"])

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
            if TO_JSON:
                result = json.dumps(result, separators=(',', ':'), sort_keys=True)
                key = json.dumps(key)
                image_key = json.dumps(image_key)
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
            print(info_str)
            tic = time.time()

    log.console("Prediction is done, saving results")
    write_predict(args["output"], results)


if __name__ == '__main__':
    main()

import logging as log
import torch
import os
import time
import json
from mtorch.dataloaders import yolo_test_data_loader
from mtorch.yolo_predict import TreePredictor
from mtorch.yolo_v2 import yolo
from mtorch.darknet import darknet_layers
from mmod.file_logger import FileLogger
from mmod.meters import AverageMeter
from mmod.detection import result2bblist
from mmod.simple_parser import load_labelmap_list

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # necessary inputs
    parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
    parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--labelmap', default='', type=str, metavar='PATH',
                    help='path to labelmap')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-k', '--topk', default=10, type=int,
                    metavar='K', help='top k result (default: 10)')
    parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to save prediction result')

def get_debug_args():
    args = {}
    args['model'] = '/work/fromLei/test_refactored3/snapshot/model_epoch_120.pt'
    args["test_dataset_path"] = '/work/voc20/test.tsv'
    args["batch_size"] = 2
    args["workers"] = 0
    args["logdir"] = '/work/fromLei/'
    return args

# args = get_parser().parse_args()
# args = vars(args)
args = get_debug_args()
log = FileLogger(args["logdir"], is_master=True, is_rank0=True)
def load_model():
    model = yolo(darknet_layers(),
                 weights_file=args['model'],
                 caffe_format_weights=False).cuda()
    seen_images = model.seen_images  
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    
    return model


def write_predict(outtsv_file, results):
    """Save prediction results to tsv file
    :type outtsv_file: str
    :type in_queue: loky.backend.queues.Queue
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

 
def main():
    to_json = True

    log.console("Creating test data loader")
    test_loader = yolo_test_data_loader(args["test_dataset_path"], batch_size=args["batch_size"],
                                        num_workers=args["workers"])

    log.console("Loading model")
    model = load_model()
    yolo_predictor = TreePredictor('/work/fromLei/tree.txt')
    cmap = load_labelmap_list('/work/fromLei/labelmap.txt')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    tic = time.time()
    results = list()
    end = time.time()
    for i, inputs in enumerate(test_loader):
        data_time.update(time.time() - end)
             
        data, keys, image_keys, hs, ws = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            # compute output
        
        for i, (im, key, image_key, h, w) in enumerate(zip(data, keys, image_keys, hs, ws)):
           # im = torch.load('data.pt')
            im = im.unsqueeze_(0)
            with torch.no_grad():
                features = model(im.cuda().float())
                prob, bbox = yolo_predictor(features, torch.Tensor((h, w)))
            
            bbox = bbox.cpu().numpy()
            prob = prob.cpu().numpy()

            assert bbox.shape[-1] == 4  
            bbox = bbox.reshape(-1, 4)
            prob = prob.reshape(-1, prob.shape[-1])
            result = result2bblist((h,w), prob, bbox, cmap,
                                    thresh=0.52, obj_thresh=0.2)
            if to_json:
                result = json.dumps(result, separators=(',', ':'), sort_keys=True)
            uid = json.dumps(key)
            image_key = json.dumps(image_key)
            results.append((uid, image_key, result))
       
            
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
    
    write_predict(outtsv_file, results)

if __name__ == '__main__':
    main()


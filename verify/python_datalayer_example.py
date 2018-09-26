from __future__ import with_statement

import colorsys
import csv
import logging
import math
import operator
import os
import os.path as op
import sys
import time
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#sys.path.append('/work/objectdetection/')
#import cv2
import torch
from PIL import Image, ImageChops
from torch.utils.data import DataLoader
from torchvision import transforms

from mmod import deteval
#from mmod.utils import init_logging
from mmod.imdb import ImageDatabase
from mtorch import Transforms
from mtorch.imdbdata import ImdbData
from mtorch.tbox_utils import DarkentAugmentation, Labeler

matplotlib.use('TkAgg')


BBOX_DIM = 4

COMPARISON_FOLDER = "data/mazontak/ComparisonCaffePython/rand_py"
VAL1 = "dn_tv"
VAL2 = "tv_dn"

 
PROTO = "/work/mnt/qd_output/Tax1300V10_5_darknet19_coco2017_bb_only/train_data_layer_only.prototxt"
CAFFEMODEL = "/work/mnt/qd_output/Tax1300V10_5_darknet19_coco2017_bb_only/snapshot/model_iter_134000.caffemodel"
TSV_FILENAME = "data/coco2017/train.tsv"



def HSVToRGB(h, s, v):
    return colorsys.hsv_to_rgb(h, s, v)
 
def getDistinctColors(n):
    huePartition = 1.0 / (n + 1)
    return [HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n)]  
    
def get_bounding_boxes(truth):
    length = len(truth)
    bboxs = np.zeros(shape=(length , BBOX_DIM + 1), dtype="float")
    for i, bbox in enumerate(truth):
        bboxs[i,:BBOX_DIM] = [float(val) for val in bbox['rect']]
        bboxs[i,BBOX_DIM] = i
    return bboxs

point_table = ([0] + ([255] * 255))

def visualize_result(result, title="", ax=None, crop_box=None, cols=getDistinctColors(30)):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(result[Transforms.IMAGE])
    #title += "\n Image Size: " + str(result[Transforms.IMAGE].size)
    for i, box in enumerate(result[Transforms.LABEL]):
        xmin, ymin, xmax, ymax = [int(x) for x in box[0:4]]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin - 1,
                             ymax - ymin - 1, fill=False,
                             edgecolor= cols[i] ,
                             linewidth=2.5)
        ax.add_patch(rect)
       # title += "\n BBox" + str(i) + ": " + str((xmin, ymin, xmax, ymax))

    if crop_box is not None:
        rect = plt.Rectangle(crop_box[0:2], crop_box[2] - 1,
                             crop_box[3] - 1, fill=False,
                             edgecolor='g',
                             linewidth=2.5)
        ax.add_patch(rect)

    ax.set_title(title)


def to_bounding_boxes(labels, num_boxes):
    boxes = labels.cpu().reshape((num_boxes, -1))
    nonzero_row_indices =[i for i in range(boxes.shape[0]) if not np.allclose(boxes[i,:],0.0)]
    boxes = boxes[nonzero_row_indices,:] 
    return boxes



#os.environ['GLOG_minloglevel'] = '2'

def main():

    protofile = PROTO
    caffemodel = CAFFEMODEL

    try:
        assert op.isfile(protofile) 
    except:
        logging.info(protofile + "is not a file")

    try:
        assert op.isfile(caffemodel)
    except:
        logging.info(caffemodel + "is not a file")

    
    toPIL = transforms.ToPILImage()

    from mtorch.caffenet import CaffeNet

  #  init_logging()
    
    model = CaffeNet(protofile, keep_diffs=True, verbose=True)
    layer = model.net_info['layers'][0]
    #composed_transform = [random_resizer, place_on_canvas, horizontal_flipper, to_tensor, minus_dc, plus_dc]
    
    augmenter = DarkentAugmentation()
    labeler = Labeler()
    augmented_dataset = ImdbData(path=layer['tsv_data_param']['source'], 
                                transform=augmenter(layer), labeler=labeler)
    cols = getDistinctColors(30)

    data_loader = DataLoader(augmented_dataset, batch_size=32,
                     shuffle=True, num_workers=0)


    plus_dc = Transforms.SubtractMeans((-123.0, -117.0, -104.0))

    device = torch.device('cpu')
    for i, sampled_batch in enumerate(data_loader):
         sz = (sampled_batch[0].to(device)).size()
         for j in range(sz[0]):
            if j % 10 == 0:
                img= torch.squeeze(sampled_batch[0][j,:,:,:]).to(device)
                label = torch.squeeze(sampled_batch[1][j,:])
                bbox = to_bounding_boxes(label, 30)
                sample_to_print = plus_dc({Transforms.IMAGE: img,
                                         Transforms.LABEL: bbox})
                img = toPIL(sample_to_print[Transforms.IMAGE])
                print(bbox.shape)
                fig, ax1 = plt.subplots(1,1)
                visualize_result({labeler.IMG_KEY:img, labeler.LABEL_KEY: bbox}, 
                "Augmented Image, {} image in batch {}".format(j,i), ax1, cols=cols)
            else:
                print("skipping the image " + str(j) + " size: " + str(torch.squeeze(sampled_batch[0][j,:,:,:]).shape))
         if i == 1:
             break
            
      #        

    plt.show()

if __name__ == '__main__':
    main()
    print("Done")

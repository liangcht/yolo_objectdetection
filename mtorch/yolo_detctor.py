import os
import os.path as op
import sys
import time
import json
import argparse
import torch
import numpy as np
from PIL import Image
from torch.utils.data import SequentialSampler

from mtorch.yolo_v2 import yolo_2extraconv
from mtorch.darknet import darknet_layers
from mtorch.yolo_predict import PlainPredictorClassSpecificNMS, TreePredictorClassSpecificNMS
from mtorch.augmentation import TestAugmentation

from mmod.simple_parser import load_labelmap_list
from mmod.detection import result2bblist


def load_model(path_model, num_classes, is_caffemodel=False):
    model = torch.nn.DataParallel(
        yolo_2extraconv(darknet_layers(),
                        weights_file=path_model,
                        caffe_format_weights=is_caffemodel,
                        num_classes=num_classes).cuda()
    )
    model.eval()
    return model

def get_predictor(num_classes, tree=None):
    if tree:
        return TreePredictorClassSpecificNMS(tree, num_classes=num_classes).cuda()
    else:
        return PlainPredictorClassSpecificNMS(num_classes=num_classes).cuda()


class YoloDetector(object):
    def __init__(self, path_model, path_labelmap, thresh, obj_thresh, path_tree=None):
        self.cmap = load_labelmap_list(path_labelmap)
        self.model = load_model(path_model=path_model, num_classes=len(self.cmap))
        self.predictor = get_predictor(num_classes=len(self.cmap), tree=path_tree)

        self.thresh = thresh
        self.obj_thresh = obj_thresh

        self.transform = TestAugmentation()

    def prepare_image(self, image, transform=None):
        img = image
        img = img[:, :, ::-1] # BGR to RGB
        img = Image.fromarray(img.astype('uint8'), mode='RGB')  # save in PIL format

        w, h = img.size
        sample = img

        if transform:
            sample = transform(sample)

        return sample, h, w

    def detect(self, image):
        im = image
        im, h, w = self.prepare_image(im, self.transform())
        im = im.unsqueeze_(0)
        im = im.float().cuda()
        with torch.no_grad():
            features = self.model(im)
        prob, bbox = self.predictor(features, torch.Tensor((h, w)))

        bbox = bbox.cpu().numpy()
        prob = prob.cpu().numpy()


        assert bbox.shape[-1] == 4
        bbox = bbox.reshape(-1, 4)
        prob = prob.reshape(-1, prob.shape[-1])
        result = result2bblist((h, w), prob, bbox, self.cmap,
                                thresh=self.thresh, obj_thresh=self.obj_thresh)
        return result

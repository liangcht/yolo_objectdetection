import os.path as op
from mmod.utils import init_logging

init_logging()

test_protofile = "output/Tax1300V14.4_0.0_0.0_darknet19_448_C_Init.best_model6933_maxIter.10eEffectBatchSize128LR7580_bb_only/test.prototxt"
protofile = "output/Tax1300V14.4_0.0_0.0_darknet19_448_C_Init.best_model6933_maxIter.10eEffectBatchSize128LR7580_bb_only/train.prototxt"
caffemodel = "output/Tax1300V14.4_0.0_0.0_darknet19_448_C_Init.best_model6933_maxIter.10eEffectBatchSize128LR7580_bb_only/snapshot/model_iter_139900.caffemodel"
labelmap = "output/Tax1300V14.4_0.0_0.0_darknet19_448_C_Init.best_model6933_maxIter.10eEffectBatchSize128LR7580_bb_only/deploy/labelmap.txt"

assert op.isfile(protofile) and op.isfile(caffemodel) and op.isfile(test_protofile) and op.isfile(labelmap)

import torch
from mmod.imdb import ImageDatabase
from mmod.detection import resize_for_od, result2bblist
from mtorch.caffenet import CaffeNet
from mmod.simple_parser import load_labelmap_list
model = CaffeNet(test_protofile, verbose=True).eval()
model.load_weights(caffemodel)

db = ImageDatabase('data/voc0712/test.tsv')
im = db.image(0)
blob = resize_for_od(im).transpose(2, 0, 1)
im_info = torch.Tensor(im.shape[:2])
data = torch.from_numpy(blob).unsqueeze(0)

with torch.no_grad():
    prob = model(data, im_info)
bbox = model.blobs['bbox']

bbox = bbox.cpu().numpy()
prob = prob.cpu().numpy()

assert bbox.shape[-1] == 4
bbox = bbox.reshape(-1, 4)
prob = prob.reshape(-1, prob.shape[-1])

cmap = load_labelmap_list(labelmap)
result = result2bblist(im, prob, bbox, cmap, thresh=0.52, obj_thresh=0.2)

from mmod.detection import im_detect
import caffe
net = caffe.Net(test_protofile, caffemodel, caffe.TEST)
scores, boxes = im_detect(net, im)
result2 = result2bblist(im, scores, boxes, cmap, thresh=0.52, obj_thresh=0.2)

model = model.cuda()
data = data.cuda()
im_info = im_info.cuda()

bbox = model.blobs['bbox']
bbox = bbox.cpu().numpy()
prob = prob.cpu().numpy()
assert bbox.shape[-1] == 4
bbox = bbox.reshape(-1, 4)
prob = prob.reshape(-1, prob.shape[-1])
result3 = result2bblist(im, prob, bbox, cmap, thresh=0.52, obj_thresh=0.2)


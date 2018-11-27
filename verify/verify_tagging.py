import caffe  # import caffe before/after PyTorch whichever works
import os.path as op
from mmod.utils import init_logging

init_logging()

classification_protofile = "output/Tagging2K/Classification.prototxt"
classification_caffemodel = "output/Tagging2K/Classification.caffemodel"
feat_protofile = "output/Tagging2K/Featurizer.prototxt"
feat_caffemodel = "output/Tagging2K/Featurizer.caffemodel"

assert op.isfile(classification_protofile) and op.isfile(classification_caffemodel)

import numpy as np
import torch
from mmod.imdb import ImageDatabase
from mmod.detection import resize_for_od
from mtorch.caffenet import CaffeNet

db = ImageDatabase('data/voc0712/test.tsv')
im = db.image(0)
blob = resize_for_od(im, target_size=224, maintain_ratio=False).transpose(2, 0, 1)
data = torch.from_numpy(blob).unsqueeze(0)


model = CaffeNet(feat_protofile, verbose=True).eval()
model.load_weights(feat_caffemodel, ignore_shape_mismatch=False)
with torch.no_grad():
    feat = model(data)

net = caffe.Net(feat_protofile, feat_caffemodel, caffe.TEST)
net.blobs['data'].reshape(1, *blob.shape)
net.blobs['data'].data[...] = blob.reshape(1, *blob.shape)
net.forward()
caffe_feat = net.blobs['inception4d_ConCat'].data[0]
pt_feat = feat.cpu().numpy()

print(np.max(pt_feat - caffe_feat), np.min(pt_feat - caffe_feat))

cmodel = CaffeNet(classification_protofile, verbose=True).eval()
cmodel.load_weights(classification_caffemodel, ignore_shape_mismatch=False)
with torch.no_grad():
    pt_out = cmodel(feat.unsqueeze(0))
sigmoid_coco, sigmoid_taggingv2 = pt_out['sigmoid_coco'], pt_out['sigmoid_taggingv2']
sigmoid_coco, sigmoid_taggingv2 = sigmoid_coco.cpu().numpy(), sigmoid_taggingv2.cpu().numpy()

cnet = caffe.Net(classification_protofile, classification_caffemodel, caffe.TEST)
cnet.blobs['auto_input'].reshape(1, *caffe_feat.shape)
cnet.blobs['auto_input'].data[...] = caffe_feat.reshape(1, *caffe_feat.shape)
caffe_out = cnet.forward()
caffe_sigmoid_coco, caffe_sigmoid_taggingv2 = caffe_out['sigmoid_coco'], caffe_out['sigmoid_taggingv2']

print(np.max(sigmoid_coco - caffe_sigmoid_coco), np.min(sigmoid_coco - caffe_sigmoid_coco))
print(np.max(sigmoid_taggingv2 - caffe_sigmoid_taggingv2), np.min(sigmoid_taggingv2 - caffe_sigmoid_taggingv2))

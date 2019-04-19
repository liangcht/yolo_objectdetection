import os.path as op
import time
import cv2
import numpy as np

from mtorch.caffenet import CaffeNet
from mtorch.deploy.transform_ops import FloatSubtract, CenterCrop, CenterCropFixed, ToFloat
import torch
import torch.nn as nn
from mmod.utils import init_logging
# from torch.onnx.symbolic import *

init_logging()

protofile = "output/gr/celeb/celeb.prototxt"
caffemodel = "output/gr/celeb/celeb.caffemodel"
assert op.isfile(protofile) and op.isfile(caffemodel)

dc = torch.from_numpy(np.array([101.511830898, 120.144889839, 160.999309001], dtype=np.float32).reshape(3, 1, 1))
model = CaffeNet(protofile, verbose=True).eval()
model.load_weights(caffemodel)

dep_model = nn.Sequential(CenterCropFixed(231), FloatSubtract(dc), model)
dep_model.cuda()

im = cv2.resize(cv2.imread('output/gr/celeb/celeb.jpg'), dsize=(257, 257)).transpose(2, 0, 1)[np.newaxis]
blob = torch.from_numpy(im).cuda()
with torch.no_grad():
    prob = dep_model(blob)

dummy_input = torch.randn(1, 3, 257, 257, device='cuda').byte()
torch.onnx.export(dep_model, dummy_input, "output/gr/celeb/celeb.onnx",
                  verbose=True, input_names=['data'], output_names=['prob'])

from mmod.simple_parser import read_blob
meanfile = "output/gr/landmark/mean.binaryproto"
mean_blob = read_blob(meanfile)
pixel_mean = np.array(mean_blob.data).astype('float32')
pixel_mean.resize(mean_blob.channels, mean_blob.height, mean_blob.width)[np.newaxis]
offset = (256 - 227) / 2
pixel_mean = pixel_mean[:, :, offset:offset+227, offset:offset+227]
dc = torch.from_numpy(pixel_mean)
dep_model = nn.Sequential(CenterCropFixed(227), FloatSubtract(dc), model)


blob = torch.from_numpy(im)
with torch.no_grad():
    prob = dep_model(blob)

dummy_input = torch.randn(1, 256, 256, 3, device='cpu').byte()
torch.onnx.export(dep_model, dummy_input, "output/gr/landmark/landmark.onnx",
                  verbose=True, input_names=['data'], output_names=['prob'])

protofile = "output/gr/tagging/Tagging2K/Featurizer.prototxt"
caffemodel = "output/gr/tagging/Tagging2K/Featurizer.caffemodel"
assert op.isfile(protofile) and op.isfile(caffemodel)

model = CaffeNet(protofile, verbose=True).eval()
model.load_weights(caffemodel)

dummy_input = torch.randn(1, 3, 300, 256, device='cpu').byte()
dc = torch.tensor([128]).float()
dep_model = nn.Sequential(FloatSubtract(dc), model)
torch.onnx.export(dep_model, dummy_input, "output/gr/tagging/Tagging2K/Featurizer.onnx",
                  verbose=True, input_names=['data'], output_names=['inception4d_ConCat'])

protofile = "output/gr/tagging/Tagging2K/Classification.prototxt"
caffemodel = "output/gr/tagging/Tagging2K/Classification.caffemodel"
assert op.isfile(protofile) and op.isfile(caffemodel)

targets = ["sigmoid_coco", "sigmoid_taggingv2", "sigmoid_text", "pool5_coco", "pool5_taggingv2"]
model = CaffeNet(protofile, targets=targets, verbose=True).eval()
model.load_weights(caffemodel)

dummy_input = torch.randn(1, 608, 14, 14, device='cpu').float()
torch.onnx.export(model, dummy_input, "output/gr/tagging/Tagging2K/Classification.onnx",
                  verbose=True, input_names=['auto_input'],
                  output_names=targets)

protofile = "output/gr/tagging/Tagging12K/Tagging12K.prototxt"
caffemodel = "output/gr/tagging/Tagging12K/Tagging12K.caffemodel"
assert op.isfile(protofile) and op.isfile(caffemodel)

targets = ['prob-sigmoid@flickr5k@', 'prob-sigmoid@openimage7k@']
model = CaffeNet(protofile, targets=targets, verbose=True).eval()
model.load_weights(caffemodel)

dummy_input = torch.randn(1, 3, 300, 256, device='cpu').byte()
dep_model = nn.Sequential(CenterCrop(224), ToFloat(), model)
torch.onnx.export(dep_model, dummy_input, "output/gr/tagging/Tagging12K/Tagging12K.onnx",
                  verbose=True, input_names=['data'],
                  output_names=targets)

im = cv2.resize(cv2.imread('output/gr/celeb/celeb.jpg'), dsize=(630, 956)).transpose(2, 0, 1)[np.newaxis]
blob = torch.from_numpy(im).cuda()
with torch.no_grad():
    prob = dep_model(blob)


onnxfile = "output/gr/tagging/Tagging2K/Featurizer.onnx"

# make shape dynamic
import onnxruntime as rt
import onnx
model = onnx.load(onnxfile)
model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = '?'
model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = '?'
onnx.save(model, onnxfile)

sess = rt.InferenceSession(onnxfile)
t0 = time.time()
pred = sess.run(['inception4d_ConCat'], {'data': im})[0]
print (time.time() - t0)

import os.path as op
import time
import cv2
import numpy as np
import onnxruntime as rt
import onnx
onnxfile = onnxfile_300x256 = "output/gr/tagging/Tagging2K/Featurizer_300x256.onnx"
onnxfile_256x300 = "output/gr/tagging/Tagging2K/Featurizer_256x300.onnx"
onnxfile_any = "output/gr/tagging/Tagging2K/Featurizer_AnyxAny.onnx"
sess_300x256 = rt.InferenceSession(onnxfile_300x256)
sess_256x300 = rt.InferenceSession(onnxfile_256x300)
sess_any = rt.InferenceSession(onnxfile_any)
im = cv2.resize(cv2.imread('output/gr/celeb/celeb.jpg'), dsize=(330, 256)).transpose(2, 0, 1)[np.newaxis]
pred_300x256 = pred_any = pred_256x300 = None
pred_300x256 = sess_300x256.run(['output'], {'data': im})[0]
pred_any = sess_any.run(['output'], {'data': im})[0]
pred_256x300 = sess_256x300.run(['output'], {'data': im})[0]

pred_300x256 = sess_300x256.run(['output'], {'data': im.astype('float32')})[0]
pred_any = sess_any.run(['output'], {'data': im.astype('float32')})[0]
pred_256x300 = sess_256x300.run(['output'], {'data': im.astype('float32')})[0]




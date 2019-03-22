import os.path as op
import cv2
import numpy as np
from mtorch.caffenet import CaffeNet
from mtorch.deploy.transform_ops import FloatSubtract, CenterCrop
from mtorch.caffetorch import Permute
import torch
import torch.nn as nn

protofile = "output/gr/celeb/celeb.prototxt"
caffemodel = "output/gr/celeb/celeb.caffemodel"
assert op.isfile(protofile) and op.isfile(caffemodel)

dc = torch.from_numpy(np.array([101.511830898, 120.144889839, 160.999309001], dtype=np.float32).reshape(3, 1, 1))
model = CaffeNet(protofile, verbose=True).eval()
model.load_weights(caffemodel)

dep_model = nn.Sequential(CenterCrop(231), FloatSubtract(dc), model)
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
# dep_model = nn.Sequential(Permute(0, 3, 1, 2), CenterCrop(227), FloatSubtract(dc), model) # ONNX issue #678
dep_model = nn.Sequential(CenterCrop(227), FloatSubtract(dc), model)


blob = torch.from_numpy(im)
with torch.no_grad():
    prob = dep_model(blob)

dummy_input = torch.randn(1, 256, 256, 3, device='cpu').byte()
torch.onnx.export(dep_model, dummy_input, "output/gr/landmark/landmark.onnx",
                  verbose=True, input_names=['data'], output_names=['prob'])

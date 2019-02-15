import os.path as op
from mmod.utils import init_logging

init_logging()

test_protofile = "/work/fromLei/test_softmaxtree.prototxt"
caffemodel = "/work/fromLei/test_refactored3/snapshot/model_iter_120.caffemodel"
labelmap = "/work/fromLei/labelmap.txt"

assert op.isfile(caffemodel)
assert op.isfile(test_protofile) 
assert op.isfile(labelmap)

import torch
from mmod.imdb import ImageDatabase
from mmod.detection import resize_for_od, result2bblist
from mtorch.caffenet import CaffeNet
from mtorch.yolo_predict import TreePredictor
from mtorch.yolo_v2 import yolo
from mtorch.darknet import darknet_layers

from mmod.simple_parser import load_labelmap_list

# creating model using PyTorch caffenet"
model = CaffeNet(test_protofile, verbose=True).eval()
model.load_weights(caffemodel)
model.cuda()
# temporary saving the model weights  
snapshot = model.state_dict()
state = {}
state['state_dict'] = snapshot
state['seen_image'] = 0
torch.save(state, 'cur_model.pt')
# loadinf data
db = ImageDatabase('/work/voc20/test.tsv')
im = db.image(0)
blob = resize_for_od(im).transpose(2, 0, 1)
im_info = torch.Tensor(im.shape[:2])
data = torch.from_numpy(blob).unsqueeze(0).cuda()
#propogating the data to get probability and bounding boxes
with torch.no_grad():
    prob = model(data, im_info)
bbox = model.blobs['bbox']

bbox = bbox.cpu().numpy()
prob = prob.cpu().numpy()

assert bbox.shape[-1] == 4
bbox = bbox.reshape(-1, 4)
prob = prob.reshape(-1, prob.shape[-1])

cmap = load_labelmap_list(labelmap)
result = result2bblist(im.shape[:2], prob, bbox, cmap, thresh=0.52, obj_thresh=0.2)

# Propogating data through Caffe 
from mmod.detection import im_detect
import caffe
net = caffe.Net(test_protofile, caffemodel, caffe.TEST)
scores, boxes = im_detect(net, im)
result2 = result2bblist(im.shape[:2], scores, boxes, cmap, thresh=0.52, obj_thresh=0.2)
# model = model.cuda()
# data = data.cuda()
#im_info = im_info.cuda()

# bbox = model.blobs['bbox']
# bbox = bbox.cpu().numpy()
# assert bbox.shape[-1] == 4
# bbox = bbox.reshape(-1, 4)
# prob = prob.reshape(-1, prob.shape[-1])
# result3 = result2bblist(im.shape[:2], prob, bbox, cmap, thresh=0.52, obj_thresh=0.2)



def load_model():
    model = yolo(darknet_layers(),
                 weights_file='cur_model.pt',
                 caffe_format_weights=True).cuda()
    seen_images = model.seen_images  
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    
    return model

new_model = load_model()
yolo_predictor = TreePredictor('/work/fromLei/tree.txt').cuda()
cmap = load_labelmap_list('/work/fromLei/labelmap.txt')

with torch.no_grad():
    features = new_model(data)
    prob, bbox = yolo_predictor(features, im_info)
            
bbox = bbox.cpu().numpy()
prob = prob.cpu().numpy()

assert bbox.shape[-1] == 4  
bbox = bbox.reshape(-1, 4)
prob = prob.reshape(-1, prob.shape[-1])
result3 = result2bblist(im.shape[:2], prob, bbox, cmap,
                                    thresh=0.52, obj_thresh=0.2)
print("PyTorch caffenet result:", result)
print("Pure Caffe result:", result2)
print("New Pytorch result:", result3 )

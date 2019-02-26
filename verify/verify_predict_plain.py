import os.path as op
from mmod.utils import init_logging

init_logging()

test_protofile = "/work/fromLei/test.prototxt"
caffemodel = "/work/fromLei/test_refactored_softmax/snapshot/model_iter_127.caffemodel"
ptmodel = "/work/fromLei/test_refactored_softmax/snapshot/model_epoch_126.pt"

labelmap = "/work/fromLei/labelmap.txt"

assert op.isfile(caffemodel)
assert op.isfile(test_protofile) 
assert op.isfile(labelmap)

import torch
from mmod.imdb import ImageDatabase
from mmod.detection import resize_for_od, result2bblist
from mtorch.caffenet import CaffeNet
from mtorch.yolo_predict import PlainPredictor
from mtorch.yolo_v2 import yolo
from mtorch.darknet import darknet_layers

from mmod.simple_parser import load_labelmap_list

# loading data
db = ImageDatabase('/work/voc20/test.tsv')
im = db.image(0)
blob = resize_for_od(im).transpose(2, 0, 1)
im_info = torch.Tensor(im.shape[:2])
data = torch.from_numpy(blob).unsqueeze(0).cuda()


cmap = load_labelmap_list(labelmap)

# Propogating data through Caffe 
from mmod.detection import im_detect
import caffe
net = caffe.Net(test_protofile, caffemodel, caffe.TEST)
scores, boxes = im_detect(net, im)
result2 = result2bblist(im.shape[:2], scores, boxes, cmap)
softmax_conf =  net.blobs['softmax_conf'].data
conf = net.blobs['conf'].data

def load_model():
    model = yolo(darknet_layers(),
                 weights_file=ptmodel,
                 caffe_format_weights=False).cuda()
    seen_images = model.seen_images  
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    
    return model

new_model = load_model()
yolo_predictor = PlainPredictor().cuda()
cmap = load_labelmap_list('/work/fromLei/labelmap.txt')

with torch.no_grad():
    features = new_model(data)
prob, bbox = yolo_predictor(features, im_info)

prob_caffe = net.blobs["prob"].data[0]

for i in range(21):
    diff = prob_caffe[:, : , :, i] .flatten() - prob.squeeze()[:, :, :, i].detach().cpu().numpy().flatten() 
    print(i ,max(abs(diff)))

bbox = bbox.cpu().numpy()
prob = prob.cpu().numpy()

assert bbox.shape[-1] == 4  
bbox = bbox.reshape(-1, 4)
prob = prob.reshape(-1, prob.shape[-1])


result3 = result2bblist(im.shape[:2], prob, bbox, cmap)
print("Pure Caffe  result:", result2)
print("New Pytorch result:", result3 )

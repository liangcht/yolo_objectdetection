import sys
import os
import os.path as op
from collections import OrderedDict
import numpy as np

from mmod.utils import init_logging

import torch


# The code below verifies that two versions Caffe and PyTorch are compatible 
# You can also compare Caffe with Caffe  - set second_Caffe to true to do so
# NOTE: if you compare Caffe to Caffe  - you need to make sure that all the augmentation is disabled in Caffe


init_logging()

# change information below as needed
protofile = "/work/fromLei/train_yolo.prototxt"

caffemodel = "/work/fromLei/snapshot/model_iter_0.caffemodel"
snapshot_pt = '/work/temp_weights.pt'


assert op.isfile(protofile) and op.isfile(caffemodel)


def get_params_py(model):
    params_py = OrderedDict()
    for name, param in model.named_parameters():
        params_py[name] =  param.data.cpu().numpy()
    return params_py

def get_group_params(model, initial_lr):
    decay, no_decay, lr2 = [], [], []
    for name, param in model.named_parameters():
        if "last_conv" in name and name.endswith(".bias"):
            lr2.append(param)
            print(name ,'lr 2' )
        elif "scale" in name:
            decay.append(param)
            print(name ,'decay 1' )
        elif len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
            print(name ,'decay 0' )
        else:
            decay.append(param)
            print(name ,'decay 1' )    
    return [{'params': no_decay, 'weight_decay': 0., 'initial_lr': initial_lr, 'lr_mult': 1.},
                 {'params': decay, 'initial_lr': initial_lr, 'lr_mult': 1.},
                 {'params': lr2, 'weight_decay': 0., 'initial_lr': initial_lr * 2., 'lr_mult': 2. , 'lr': initial_lr * 2.}]


from mtorch.caffenet import CaffeNet
from mtorch.darknet import darknet_layers
from mtorch.yolo_v2 import yolo
from mtorch.tbox_utils import Labeler
from mtorch.augmentation import DarknetAugmentation
from mtorch.samplers import SequentialWrappingSampler, RandomWrappingSampler
from mtorch.imdbdata import ImdbData
from mtorch.caffeloader import CaffeLoader
from torch.utils.data import DataLoader


model = CaffeNet(protofile, keep_diffs=True, verbose=False)
model.load_weights(caffemodel)
model = model.cuda()

snapshot = model.state_dict()
state = {}
state['state_dict'] = snapshot
state['seen_image'] = 0
torch.save(state, snapshot_pt)

    

augmenter = DarknetAugmentation()
labeler = Labeler()
layer = model.net_info['layers'][model.input_index]
augmented_dataset = ImdbData(path=protofile,
                            transform=augmenter(layer), labeler=labeler)

total_batch_size = 1
sampler = RandomWrappingSampler(
            augmented_dataset, 
            int(np.ceil(float(len(augmented_dataset)) / float(total_batch_size)) * total_batch_size)
            )
data_loader = DataLoader(augmented_dataset, batch_size=total_batch_size,
                        sampler=sampler,
                        num_workers=0,
                        pin_memory=True)

inputs = next(iter(data_loader))

data, labels = inputs[0].cuda(), inputs[1].cuda().float()
        
features, features_darknet = model(data, labels)  # for this line to work  uncomment line 220 in caffenet


dark_layers = darknet_layers(snapshot_pt, True)

model_new = dark_layers.cuda()
features_new = model_new(data) 

features_cpu = features_darknet.cpu().detach().numpy()
features_new_cpu = features_new.cpu().detach().numpy()
diff = np.abs(np.diff(features_cpu.flatten() - features_new_cpu.flatten()))

print("Darknet diff sum:", np.sum(diff))
print("Darknet diff max:", np.max(diff))
print("Darknet diff min:", np.min(diff))



model_new = yolo(dark_layers, weights_file=snapshot_pt, caffe_format_weights=True) 
model_new = model_new.cuda()

features_new = model_new(data) 

features_cpu = features.cpu().detach().numpy()
features_new_cpu = features_new.cpu().detach().numpy()
diff = np.abs(np.diff(features_cpu.flatten() - features_new_cpu.flatten()))

print("Yolo diff sum:", np.sum(diff))
print("Yolo diff max:", np.max(diff))
print("Yolo diff min:", np.min(diff))
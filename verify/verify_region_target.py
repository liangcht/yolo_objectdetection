import sys
import os
import os.path as op
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader

from mtorch.caffenet_weight_converter import prep_dict
from mtorch.caffenet import CaffeNet
from mtorch.darknet import darknet_layers
from mtorch.yolo_v2 import yolo
from mtorch.tbox_utils import Labeler, DarknetAugmentation
from mtorch.samplers import SequentialWrappingSampler, RandomWrappingSampler
from mtorch.imdbdata import ImdbData
from mtorch.caffeloader import CaffeLoader
from mtorch.region_target_loss import RegionTargetWithSoftMaxLoss, RegionTargetWithSoftMaxTreeLoss
from mtorch.caffesgd import CaffeSGD


# change information below as needed
MODE = "SoftMaxLoss"
protofile = "/work/fromLei/train_yolo_with" + MODE + ".prototxt"
caffemodel = "/work/fromLei/snapshot/model_iter_10022.caffemodel"#model_iter_0.caffemodel"
snapshot_pt = '/work/temp_weights.pt'  # this will contain temporary weights for testing only
assert op.isfile(protofile) and op.isfile(caffemodel)

if MODE == "SoftMaxTreeLoss":
    tree =  "/work/fromLei/tree.txt"
    assert op.isfile(tree)


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
        elif "scale" in name:
            decay.append(param)
        elif len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0., 'initial_lr': initial_lr, 'lr_mult': 1.},
                 {'params': decay, 'initial_lr': initial_lr, 'lr_mult': 1.},
                 {'params': lr2, 'weight_decay': 0., 'initial_lr': initial_lr * 2., 'lr_mult': 2. , 'lr': initial_lr * 2.}]


model = CaffeNet(protofile, keep_diffs=True, verbose=False)
model.load_weights(caffemodel)
model = model.cuda()
seen_images_ini = model.seen_images.data.cpu().detach()

state = model.state_dict()
torch.save(state, snapshot_pt)

    

augmenter = DarknetAugmentation()
labeler = Labeler()
layer = model.net_info['layers'][model.input_index]
augmented_dataset = ImdbData(path=protofile,
                            transform=augmenter(layer), labeler=labeler)

total_batch_size = 16
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

last_lr = 0.0001 
weight_decay_ini = 0.0005
optimizer = CaffeSGD(get_group_params(model, last_lr),
        lr=last_lr, momentum=0.9, weight_decay=weight_decay_ini)
optimizer.zero_grad()
loss = model(data, labels)
crit = torch.sum(loss)
crit.backward()
optimizer.step()
params_py_next = get_params_py(model)


dark_layers = darknet_layers(snapshot_pt, True)

model_new = yolo(dark_layers, weights_file=snapshot_pt, caffe_format_weights=True) 
model_new = model_new.cuda()

features_new = model_new(data) 

optimizer_new = CaffeSGD(get_group_params(model_new, last_lr),
        lr=last_lr, momentum=0.9, weight_decay=weight_decay_ini)
optimizer_new.zero_grad()

if MODE == "SoftMaxTreeLoss":
    rt_loss = RegionTargetWithSoftMaxTreeLoss(tree, seen_images=model_new.seen_images)
else:
    rt_loss = RegionTargetWithSoftMaxLoss(seen_images=model_new.seen_images)

rt_loss = rt_loss.cuda()
crit_new = rt_loss(features_new, labels)

crit_new.backward()
optimizer_new.step()
params_py_next_new = get_params_py(model_new)

layers = params_py_next_new.keys()

err_params_next = dict()
update = dict()
update_new = dict()
params_py_next = prep_dict(params_py_next, params_py_next_new)

model_new.seen_images = rt_loss.seen_images

for layer in layers:
    err_params_next[layer] = np.mean(np.abs(params_py_next_new[layer].flatten() - params_py_next[layer].flatten()))    
    print("Difference in weights after update", layer, err_params_next[layer])

print("New loss", crit_new.data.cpu().detach())
print("Old loss", crit.data.cpu().detach())
print("Loss diff", crit_new.data.cpu().detach() - crit.data.cpu().detach())
print("Seen images upon initialization:", seen_images_ini) 
print("Batch size", total_batch_size)
print("Expected number of seen images after loss", seen_images_ini + total_batch_size)
print("Seen images after caffenet loss:", model.seen_images.data.cpu().detach())  
print("Seen images after new loss:", model_new.seen_images.data.cpu().detach())  


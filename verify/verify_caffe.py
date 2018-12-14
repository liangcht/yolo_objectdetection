import sys
import os
import os.path as op
from collections import OrderedDict
import numpy as np

from mmod.utils import init_logging

import caffe
import torch

import matplotlib.pyplot as plt

# The code below verifies that two versions Caffe and PyTorch are compatible 
# You can also compare Caffe with Caffe  - set second_Caffe to true to do so
# NOTE: if you compare Caffe to Caffe  - you need to make sure that all the augmentation is disabled in Caffe


init_logging()

second_Caffe = False
# change information below as needed
protofile = "/work/fromLei/train_softmax_regionloss.prototxt"
solver_prototxt = "/work/fromLei/yolo_voc_solver.prototxt"
solver_prototxt2 = "/work/fromLei/yolo_voc_solver.prototxt"

caffemodel = "/work/fromLei/snapshot/model_iter_0.caffemodel" #darknet19_448.caffemodel"

assert op.isfile(protofile) and op.isfile(caffemodel)


def get_params_py(model):
    params_py = OrderedDict()
    for name, param in model.named_parameters():
        params_py[name] =  param.data.cpu().numpy()
    return params_py

def get_params_caffe(net):    
    params =  OrderedDict()
    for (k,v) in net.params.iteritems():
        if 'bn' in k:
            continue
        if 'conv' in k  or 'scale' in k :
            key  = k + ".weight"
            params[key] = v[0].data.copy()
            if len(v) > 1:
                key  = k + ".bias"
                params[key] = v[1].data.copy()     
    return params

def get_data_caffe(net):
    return OrderedDict((k, v.data.copy()) for (k, v) in net.blobs.iteritems())

def get_diffs_caffe(net):
    return OrderedDict((k, v.diff.copy()) for (k, v) in net.blobs.iteritems())


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

caffe.set_mode_gpu()
solver = caffe.SGDSolver(solver_prototxt)
solver.net.copy_from(caffemodel)
params_caffe_ini = get_params_caffe(solver.net)

if second_Caffe:
    solver2 = caffe.SGDSolver(solver_prototxt2)
    solver2.net.copy_from(caffemodel)
    params_caffe2_ini = get_params_caffe(solver2.net)

else:
    from mtorch.caffenet import CaffeNet
    from mtorch.caffesgd import CaffeSGD
    model = CaffeNet(protofile, keep_diffs=True, verbose=False)
    model.load_weights(caffemodel)
    model = model.cuda()
    

solver.step(1)
data = get_data_caffe(solver.net)
diffs_caffe = get_diffs_caffe(solver.net)
param_caffe_next = get_params_caffe(solver.net)

if second_Caffe:
    solver2.step(1)
    data2 = get_data_caffe(solver2.net)
    diffs_caffe2 = get_diffs_caffe(solver2.net)
    param_caffe2_next = get_params_caffe(solver2.net)
else:
    vdata, labels = torch.from_numpy(data['data']).cuda(), torch.from_numpy(data['label']).cuda()
    params_py = get_params_py(model)
    last_lr = 0.0001 
    weight_decay_ini = 0.0005
    optimizer = CaffeSGD(get_group_params(model, last_lr),
        lr=last_lr, momentum=0.9, weight_decay=weight_decay_ini)
    optimizer.zero_grad()
    new_loss = model(vdata, labels)
    crit = torch.sum(new_loss)
    crit.backward()
    optimizer.step()
    params_py_next = get_params_py(model)


layers = params_caffe_ini.keys()

err_params_ini = dict()
err_params_next = dict()
update = dict()
update_caffe = dict()
for layer in layers:
    if second_Caffe:
        err_params_ini[layer] = np.mean(np.abs(params_caffe_ini[layer].flatten() - params_caffe2_ini[layer].flatten()))
        err_params_next[layer] = np.mean(np.abs(param_caffe_next[layer].flatten() - param_caffe2_next[layer].flatten()))
        update[layer] =  np.mean(np.abs(params_caffe2_ini[layer].flatten() - param_caffe2_next[layer].flatten()))
        update_caffe[layer] =  np.mean(np.abs(params_caffe_ini[layer].flatten() - param_caffe_next[layer].flatten()))
    else:
        err_params_ini[layer] = np.mean(np.abs(params_caffe_ini[layer].flatten() - params_py[layer].flatten()))
        err_params_next[layer] = np.mean(np.abs(param_caffe_next[layer].flatten() - params_py_next[layer].flatten()))
        update[layer] =  np.mean(np.abs(params_py[layer].flatten() - params_py_next[layer].flatten()))
        update_caffe[layer] =  np.mean(np.abs(params_caffe_ini[layer].flatten() - param_caffe_next[layer].flatten()))
    
    print(layer, err_params_ini[layer], err_params_next[layer], update[layer], update_caffe[layer])

layers = model.diffs.keys()

for layer in layers:
    try:
        if second_Caffe:
            err_diff = np.mean(np.abs(diffs_caffe2[layer].flatten() - diffs_caffe[layer].flatten()))
        else:
            err_diff = np.mean(np.abs(model.diffs[layer].cpu().numpy().flatten() - diffs_caffe[layer].flatten()))
        
    except Exception as err:
        print(err)
        continue
    else:
        print(layer, err_diff)
if second_Caffe:
    print("Last Conv", np.mean(np.abs(data['last_conv'].flatten() - data2['last_conv'].flatten())))
print({tname: model.blobs.get(tname).item() for tname in model.targets})

##################################################################################
# The code below tests training uncomment if needed
#############################################################################

# decay, no_decay = [], []
# for name, param in model.named_parameters():
#     if not param.requires_grad:
#         continue
#     if len(param.shape) == 1 or name.endswith(".bias"):
#         no_decay.append(param)
#     else:
#         decay.append(param)




# # params = model.parameters()
# params = [{'params': no_decay, 'weight_decay': 0.}, {'params': decay}]
# steps = [1500, 125036, 225064, 250072]
# lrs = [0.0001, 0.001, 0.0001, 1e-05]
# last_lr = lrs[0]
# optimizer = CaffeSGD(params, lr=last_lr, momentum=0.9, weight_decay=0.0005)
# losses = []
# iterations = 0
# while iterations < 30000:
#     idx = np.searchsorted(steps, iterations)
#     if idx >= len(lrs):
#         idx = len(lrs) - 1
#     lr = lrs[idx]
#     # lr = lrs[0] / (10 ** idx)
#     if lr != last_lr:
#         last_lr = lr
#         print("lr: {}".format(lr))
#         for group in optimizer.param_groups:
#             group['lr'] = lr

#     data, labels = inputs()
#     new_loss = model(data, labels)
#     crit = torch.sum(new_loss)
#     if crit != crit:
#         break
#     losses.append(crit.detach().cpu().numpy())
#     optimizer.zero_grad()
#     crit.backward()
#     optimizer.step()

#     if iterations % 100 == 0:
#         print("{}: loss: {}".format(iterations, crit.item()))
#         print({tname: model.blobs.get(tname).item() for tname in model.targets})

#     if iterations % 500 == 0:
#         torch.save(model.state_dict(), "/tmp/model.pkl")
#     iterations += 1


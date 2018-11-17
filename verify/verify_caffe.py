import sys
import os
import os.path as op
from collections import OrderedDict
import numpy as np

from mmod.utils import init_logging
import torch

import caffe
import matplotlib.pyplot as plt

init_logging()
#os.environ['GLOG_minloglevel'] = '2'


#protofile = "/mnt/qd_output/Tax1300_V2_1_darknet19_p_bb_only/train_small.prototxt"
#caffemodel = "/mnt/qd_output/Tax1300_V2_1_darknet19_p_bb_only/snapshot/model_iter_522264.caffemodel"
protofile = "/work/fromLei/train.prototxt"
protofile_part = "/work/fromLei/train_part.prototxt"
solver_prototxt = "/work/fromLei/yolo_voc_solver.prototxt"
caffemodel = "/work/fromLei/snapshot/model_iter_0.caffemodel" #darknet19_448.caffemodel"

assert op.isfile(protofile) and op.isfile(caffemodel)

# create data symlink
# %cd /tmp/work

def get_params_py(model):
    params_py = OrderedDict()
    for name, param in model.named_parameters():
        params_py[name] =  param.detach().data.cpu().numpy()
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
    return [{'params': no_decay, 'weight_decay': 0.},
            {'params': decay},
            {'params': lr2, 'weight_decay': 0., 'initial_lr': initial_lr * 2.0}]
            



caffe.set_mode_gpu()
solver = caffe.SGDSolver(solver_prototxt)
solver.net.copy_from(caffemodel)
from mtorch.caffenet import CaffeNet
from mtorch.caffesgd import CaffeSGD
model = CaffeNet(protofile, keep_diffs=True, verbose=False)
model.load_weights(caffemodel)
model = model.cuda()

iter = 0
while iter < 110:
#net = caffe.Net(protofile, caffemodel, caffe.TRAIN)
#net.params['region_target'][0].data[0] = 158000  # 0
#loss = net.forward()


    params_caffe_ini = get_params_caffe(solver.net)
#net.backward()
    solver.step(1)
    data = get_data_caffe(solver.net)
    diffs_caffe = get_diffs_caffe(solver.net)
    param_caffe_next = get_params_caffe(solver.net)


# import pickle
# with open("/tmp/data.pkl", "w") as f:
#     pickle.dump([data, data_diff, loss], f)
   

#inputs = model.inputs

#inputs = inputs.cuda()

#import pickle
#with open("/tmp/data.pkl") as f:
#    data, data_diff, loss = pickle.load(f)

    #from torch.autograd import Variable

    vdata, labels = torch.from_numpy(data['data']).cuda(), torch.from_numpy(data['label']).cuda()
#   vdata, labels = Variable(torch.from_numpy(data['extra_conv21'])).cuda(), Variable(torch.from_numpy(data['label'])).cuda()

# num_exp = 17
# if not os.path.isdir("/work/comparison/gradient_differences" + str(num_exp) + "/"):
#     os.mkdir("/work/comparison/gradient_differences" + str(num_exp) + "/")
# else:
#     print("change num_exp")
# torch.save([vdata, labels], "/work/comparison/gradient_differences" + str(num_exp) + "/" + "data.pt" )

    params_py = get_params_py(model)

    new_loss = model(vdata, labels)
    crit = torch.sum(new_loss)

#print(loss)



    lrs = [0.0001, 0.001, 0.0001, 1e-05]
last_lr = lrs[0]
weight_decay_ini = 0.0005
optimizer = CaffeSGD(get_group_params(model, last_lr),
             lr=last_lr, momentum=0.9, weight_decay=weight_decay_ini)


for i, group in enumerate(optimizer.param_groups):
    if i < len(optimizer.param_groups) - 1:
        group['lr'] = last_lr
    else:
        group['lr'] = last_lr * 2.0

optimizer.zero_grad()
crit.backward()

optimizer.step()

params_py_next = get_params_py(model)

layers = params_caffe_ini.keys()

err_params_ini = dict()
err_params_next = dict()
update = dict()
update_caffe = dict()
for layer in layers:
    err_params_ini[layer] = np.mean(np.abs(params_caffe_ini[layer].flatten() - params_py[layer].flatten()))
    err_params_next[layer] = np.mean(np.abs(param_caffe_next[layer].flatten() - params_py_next[layer].flatten()))
    update[layer] =  np.mean(np.abs(params_py[layer].flatten() - params_py_next[layer].flatten()))
    update_caffe[layer] =  np.mean(np.abs(params_caffe_ini[layer].flatten() - param_caffe_next[layer].flatten()))

    print(layer, err_params_ini[layer], err_params_next[layer], update[layer], update_caffe[layer])

layers = model.diffs.keys()

for layer in layers:
    try:
        err_diff = np.mean(np.abs(model.diffs[layer].cpu().numpy().flatten() - diffs_caffe[layer].flatten()))
    except Exception as err:
        print(err)
        continue
    else:
        print(layer, err_diff)

print({tname: model.blobs.get(tname).item() for tname in model.targets})

maxs, means = [], []

#     if len(err_diff.shape) < 1:
#         continue
#     try:
#         err_mean = np.mean(err_diff, axis=0)
#         err_rat = err_diff / (np.squeeze(data_diff[layer]) + 1e-07)
#         err_rat_sum = np.mean(err_rat, axis=0)
#         mean_grad_caffe = np.mean(np.abs(np.squeeze(data_diff[layer])), axis=0)
#         mean_grad_py = np.mean(np.abs(np.squeeze(model.diffs[layer].cpu().numpy())), axis=0)

#         mean_val = np.mean(err_mean.flatten())
#         max_val = np.max(err_rat_sum.flatten())
#         layer = layer.replace("/", "_")

#         fig = plt.figure()
#         plt.imshow(err_rat_sum, vmin=0.0, vmax=max_val)
#         title2 = layer + " " + " max ratio: " + str(max_val)
#         plt.colorbar()
#         plt.title(title2)
#         fig.savefig("/work/comparison/gradient_differences" + str(num_exp) + "/" + layer + "_err_ratio"  ".png")
#         plt.close(fig)

#         fig = plt.figure()
#         plt.imshow(err_mean, vmin=0.0, vmax=max(err_mean.flatten()))
#         title = layer + " "  + " mean: " + str(mean_val)
#         plt.colorbar()
#         plt.title(title)
#         fig.savefig("/work/comparison/gradient_differences" + str(num_exp) + "/" + layer + "_err_mean"  ".png")
#         plt.close(fig)

#         print(title)
#         print(title2)

#         max_abs_grad = max([np.max(mean_grad_caffe.flatten()), np.max(np.max(mean_grad_py.flatten()))])
#         min_abs_grad = min([np.min(mean_grad_caffe.flatten()), np.min(np.min(mean_grad_py.flatten()))])
#         fig = plt.figure()
#         plt.imshow(mean_grad_caffe, vmin=min_abs_grad,vmax=max_abs_grad)
#         title = layer + " "  + " mean grad caffe: " + str(np.mean(mean_grad_caffe.flatten()))
#         plt.colorbar()
#         plt.title(title)
#         fig.savefig("/work/comparison/gradient_differences" + str(num_exp) + "/" + layer + "_caffe_grad"  ".png")
#         plt.close(fig)

#         fig = plt.figure()
#         plt.imshow(mean_grad_py, vmin=min_abs_grad, vmax=max_abs_grad)
#         title = layer + " "  + " mean grad py: " + str(np.mean(mean_grad_py.flatten()))
#         plt.colorbar()
#         plt.title(title)
#         fig.savefig("/work/comparison/gradient_differences" + str(num_exp) + "/" + layer + "_py_grad"  ".png")
#         plt.close(fig)

#     except Exception as err:
#         raise(err)
#     # for ch in range(err_diff.shape[0]):
#     #     maxs.append(np.max(err_diff[ch].flatten() / (np.squeeze(data_diff[layer])[ch].flatten() + 1e-07)))
#     #     means.append(np.mean(err_diff[ch].flatten()))
   
#     # for ch in range(err_diff.shape[0]):
#     #     if maxs[ch] < 1e-02: # maxs[ch] < 100 * means[ch] or maxs[ch] < 1e-07:
#     #         continue 

#     #     layer = layer.replace("/", "_")
#     #     fig = plt.figure()
#     #     plt.imshow(err_diff[ch] / (np.squeeze(data_diff[layer])[ch] + 1e-07), vmin=0.0, vmax=maxs[ch])
#     #     title = layer + " " + str(ch) + " mean: " + str(means[ch])
#     #     title2 = layer + " " + str(ch) + " max: " + str(maxs[ch])
#     #     plt.colorbar()
#     #     plt.title(title)
#     #     fig.savefig("/work/comparison/gradient_differences" + str(num_exp) + "/" + layer + "ch"+ str(ch) + ".png")
#     #     plt.close(fig)
#     #     print(title)
#     #     print(title2)


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


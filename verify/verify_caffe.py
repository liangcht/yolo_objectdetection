import sys
import os
import os.path as op
from collections import OrderedDict
import numpy as np

from mmod.utils import init_logging
import caffe

init_logging()
#os.environ['GLOG_minloglevel'] = '2'


protofile = "/mnt/qd_output/Tax1300_V2_1_darknet19_p_bb_only/train_small.prototxt"
caffemodel = "/mnt/qd_output/Tax1300_V2_1_darknet19_p_bb_only/snapshot/model_iter_522264.caffemodel"

assert op.isfile(protofile) and op.isfile(caffemodel)

# create data symlink
# %cd /tmp/work

caffe.set_mode_gpu()
net = caffe.Net(protofile, caffemodel, caffe.TRAIN)
#net.params['region_target'][0].data[0] = 158000  # 0
loss = net.forward()
data = OrderedDict((k, v.data.copy()) for (k, v) in net.blobs.iteritems())
net.backward()
data_diff = OrderedDict((k, v.diff.copy()) for (k, v) in net.blobs.iteritems())

#import pickle
#with open("/tmp/data.pkl", "w") as f:
#    pickle.dump([data, data_diff, loss], f)

import torch
from mtorch.caffenet import CaffeNet
from mtorch.caffesgd import CaffeSGD

model = CaffeNet(protofile, keep_diffs=True, verbose=True)
inputs = model.inputs

model.load_weights(caffemodel)

model = model.cuda()
inputs = inputs.cuda()

#import pickle
#with open("/tmp/data.pkl") as f:
#    data, data_diff, loss = pickle.load(f)

from torch.autograd import Variable

vdata, labels = Variable(torch.from_numpy(data['data'])).cuda(), Variable(torch.from_numpy(data['label'])).cuda()

new_loss = model(vdata, labels)
crit = torch.sum(new_loss)
crit.backward()

print(loss)
print({tname: model.blobs.get(tname).item() for tname in model.targets})

err_diff = np.abs(model.diffs['extra_conv19'].cpu().numpy().flatten() - data_diff['extra_conv19'].flatten())
print(np.sum(err_diff))


decay, no_decay = [], []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if len(param.shape) == 1 or name.endswith(".bias"):
        no_decay.append(param)
    else:
        decay.append(param)

# params = model.parameters()
params = [{'params': no_decay, 'weight_decay': 0.}, {'params': decay}]
steps = [1500, 125036, 225064, 250072]
lrs = [0.0001, 0.001, 0.0001, 1e-05]
last_lr = lrs[0]
optimizer = CaffeSGD(params, lr=last_lr, momentum=0.9, weight_decay=0.0005)
losses = []
iterations = 0
while iterations < 30000:
    idx = np.searchsorted(steps, iterations)
    if idx >= len(lrs):
        idx = len(lrs) - 1
    lr = lrs[idx]
    # lr = lrs[0] / (10 ** idx)
    if lr != last_lr:
        last_lr = lr
        print("lr: {}".format(lr))
        for group in optimizer.param_groups:
            group['lr'] = lr

    data, labels = inputs()
    new_loss = model(data, labels)
    crit = torch.sum(new_loss)
    if crit != crit:
        break
    losses.append(crit.detach().cpu().numpy())
    optimizer.zero_grad()
    crit.backward()
    optimizer.step()

    if iterations % 100 == 0:
        print("{}: loss: {}".format(iterations, crit.item()))
        print({tname: model.blobs.get(tname).item() for tname in model.targets})

    if iterations % 500 == 0:
        torch.save(model.state_dict(), "/tmp/model.pkl")
    iterations += 1


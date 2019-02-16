import os.path as op
import torch
import torch.nn as nn

from mmod.im_utils import *
from mmod.utils import init_logging


from torchvision.models import resnet34

init_logging()

model_path = 'output/gr/tag/model_best.pth.tar'
assert op.isfile(model_path)

snapshot = torch.load(model_path)
state_dict = snapshot['state_dict']
# remove module
state_dict = {
    k[7:]: v for (k, v) in state_dict.iteritems()
}

features = resnet34()
features.fc = nn.Linear(512, snapshot['num_classes'])
features.load_state_dict(state_dict)
del snapshot, state_dict

model = nn.Sequential(features, nn.Softmax())
model = model.cuda()

dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
torch.onnx.export(model, dummy_input, "output/gr/tag/tag.onnx", verbose=True,
                  input_names=['data'], output_names=['prob'])


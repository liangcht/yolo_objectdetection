import numpy as np
import random
from collections import OrderedDict
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable

from mmod.simple_parser import save_prototxt
from mmod.utils import ompi_size, ompi_rank


class CaffeData(nn.Module):
    def __init__(self, layer, phase, local_gpus_size=1,
                 batch_size=None):
        super(CaffeData, self).__init__()
        self.ltype = layer['type']
        net_info = OrderedDict()
        props = OrderedDict()
        props['name'] = 'temp network'
        net_info['props'] = props
        if 'include' in layer:
            if 'phase' in layer['include']:
                logging.info('CaffeData init phase = %s' % (layer['include']['phase']))
            layer.pop('include')
        if layer['type'] == 'TsvBoxData':
            data_param = 'tsv_data_param'
        elif layer['type'] == 'HDF5Data':
            data_param = 'hdf5_data_param'
        else:
            data_param = 'data_param'

        layer[data_param]['world_size'] = ompi_size()
        layer[data_param]['world_rank'] = ompi_rank()
        if not batch_size:
            batch_size = int(layer[data_param]['batch_size'])
            assert batch_size > 0, "Invalid batch_size: {} in prototxt".format(batch_size)
        self.batch_size = batch_size  # per-GPU batch size

        # batch-size is per-GPU, make it total local effective batch size
        # so that DataParallel would distribute each to a GPU
        batch_size *= local_gpus_size
        layer[data_param]['batch_size'] = batch_size

        self.data_target = None
        self.label_target = None
        tops = layer['top']
        assert len(tops) > 0, "Data layer has no outputs"
        self.data_target = tops[0]
        if len(tops) > 1:
            self.label_target = tops[1]

        net_info['layers'] = [layer]

        rand_val = random.random()
        protofile = '/tmp/.temp_data%f.prototxt' % rand_val
        save_prototxt(net_info, protofile)
        weightfile = '/tmp/.temp_data%f.caffemodel' % rand_val
        open(weightfile, 'w').close()

        import caffe
        caffe.set_mode_cpu()
        if phase == 'TRAIN':
            self.net = caffe.Net(protofile, weightfile, caffe.TRAIN)
        else:
            self.net = caffe.Net(protofile, weightfile, caffe.TEST)
        self.register_buffer('data', torch.zeros(1))
        self.register_buffer('label', torch.zeros(1))

    def extra_repr(self):
        """Extra information
        """
        return 'type={}, batch_size={}'.format(
            self.ltype, self.batch_size
        )

    def forward(self):
        self.net.forward()
        data = self.net.blobs[self.data_target].data
        data = torch.from_numpy(data)
        self.data.resize_(data.size()).copy_(data)
        if self.label_target:
            label = self.net.blobs[self.label_target].data
            label = torch.from_numpy(label)
            self.label.resize_(label.size()).copy_(label)
            return Variable(self.data), Variable(self.label)
        return Variable(self.data)

    @staticmethod
    def to_image(data, mean_value=None):
        """Convert a data to numpy array, correctign the image shape
        :param data: single data returned from forward
            e.g. if forward returns datas batch, data = datas[0]
        :param mean_value: subtracted mean
        """
        if mean_value is None:
            mean_value = [104, 117, 123]
        im = np.moveaxis(data.cpu().numpy(), 0, -1) + mean_value
        return im.astype(np.uint8)[:, :, (2, 1, 0)]

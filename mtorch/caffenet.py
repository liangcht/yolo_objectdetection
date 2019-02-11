from __future__ import print_function
import os
import os.path as op
import argparse
import numpy as np
import math
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
from mmod.utils import init_logging, cwd
from mmod.simple_parser import parse_prototxt, print_prototxt, read_model, read_blob

from mtorch.converter import SUPPORTED_LAYERS, save_caffemodel
from mtorch.caffetorch import FCView, Eltwise, Scale, Crop, Slice, Concat, Permute, SoftmaxWithLoss, \
    Normalize, Flatten, Accuracy, EuclideanLoss, Reorg
from mtorch.reshape import reshape_helper, Reshape
from mtorch.yoloevalcompat import YoloEvalCompat
from mtorch.yolobbs import YoloBBs
from mtorch.region_target import RegionTarget
from mtorch.softmaxtree_loss import SoftmaxTreeWithLoss
from mtorch.softmaxtree import SoftmaxTree
from mtorch.softmaxtree_prediction import SoftmaxTreePrediction
from mtorch.nmsfilter import NMSFilter
from mtorch.caffedata import CaffeData
from mtorch.region_loss import RegionLoss

BOX_DIMS = 5  # TODO: should be defined by parameter
NUM_CHANNELS = 3  # TODO: should be defined by parameter


class CaffeNet(nn.Module):
    def __init__(self, protofile, verbose=False, keep_diffs=False, network_verbose=False,
                 width=None, height=None, channels=None,
                 forward_net_only=True,
                 local_gpus_size=1,
                 world_size=1,
                 seen_images=0, batch_size=None,
                 phase='TRAIN'):
        super(CaffeNet, self).__init__()
        self.phase = phase
        self.blobs = None
        self.output = None
        self.height = None
        self.width = None
        self.local_gpus_size = local_gpus_size  # local number of GPUs
        self.world_size = world_size
        self.gpus_size = local_gpus_size * world_size
        self.batch_size = batch_size
        self.verbose = verbose
        self.network_verbose = network_verbose or self.verbose  # higher verbosity creating the network
        self.keep_diffs = keep_diffs or self.verbose
        self.blob_dims = dict()
        self.input_index = None
        self.input_blobs = ['data', 'label']  # This is the default input blob names, overwritten by protofile

        self.protofile = protofile
        self.net_info = parse_prototxt(protofile)

        # if should have separate data layer (useful for distributed)
        # noinspection PyCallingNonCallable
        self.register_buffer('forward_net_only', torch.tensor(1 if forward_net_only else 0, dtype=torch.uint8))
        # noinspection PyCallingNonCallable
        self.register_buffer('seen_images', torch.tensor(seen_images, dtype=torch.long))
        self.models = self.create_network(
            input_batch_size=self.batch_size, input_width=width, input_height=height, input_channels=channels
        )
        self.diffs = {}
        for name, model in self.models.items():
            assert isinstance(model, nn.Module)
            self.add_module(name, model)

        self.has_mean = False
        if 'mean_file' in self.net_info['props']:
            self.has_mean = True
            self.mean_file = self.net_info['props']['mean_file']
        self.targets = self.get_targets()

    def _blob_shape(self, btname):
        if not isinstance(btname, list):
            btname = [btname]
        shapes = ", ".join([
            "({})".format(
                " x".join([
                    "% 5d" % dim for dim in self.blob_dims[name]
                ]) if name in self.blob_dims else "...") for name in btname
        ])
        return shapes

    def _blob_size(self, btname):
        if not isinstance(btname, list):
            btname = [btname]
        shapes = ", ".join([
            "({})".format(
                " x".join([
                    "% 5d" % dim for dim in self.blobs[name].size()
                ]) if name in self.blobs else "...") for name in btname
        ])
        return shapes

    def set_network_targets(self, targets):
        self.targets = set(targets)

    def set_verbose(self, verbose):
        self.verbose = verbose

    def set_phase(self, phase):
        self.phase = phase
        if phase == 'TRAIN':
            self.train()
        else:
            self.eval()

    def set_mean_file(self, mean_file):
        if mean_file != "":
            self.has_mean = True
            self.mean_file = mean_file
        else:
            self.has_mean = False
            self.mean_file = ""

    def get_outputs(self, outputs):
        blobs = []
        for name in outputs:
            blobs.append(self.blobs[name])
        return blobs

    def forward(self, *inputs):
        if self.training:
            self.set_phase('TRAIN')
        else:
            self.set_phase('TEST')

        # TODO: have an option to delete blobs as soon as they are not needed (not used as bottom), to save memory
        self.blobs = OrderedDict()

        for input_blob, blob in zip(self.input_blobs, inputs):
            self.blobs[input_blob] = blob

        if len(inputs):
            if self.has_mean:
                data = self.blobs['data']
                n_b = data.data.size(0)
                n_c = data.data.size(1)
                n_h = data.data.size(2)
                n_w = data.data.size(3)
                data -= Variable(self.mean_img.view(1, n_c, n_h, n_w).expand(n_b, n_c, n_h, n_w))

            self.seen_images += int(inputs[0].size(0)) * self.gpus_size

        layers = self.net_info['layers']

        layer_num = len(layers)
        i = 0
        self.output = None
        while i < layer_num:
            layer = layers[i]
            lname = layer['name']
            if 'include' in layer and 'phase' in layer['include']:
                phase = layer['include']['phase']
                lname = lname + '.' + phase
                if phase != self.phase:
                    i = i + 1
                    continue
            ltype = layer['type']
            if 'top' not in layer:
                continue
            tname = layer['top']
            tnames = tname if type(tname) == list else [tname]
            if ltype in ['Data', 'AnnotatedData', 'TsvBoxData', 'HDF5Data']:
                if self.forward_net_only.item():
                    assert len(inputs) > 1, "no inputs provided for data"
                else:
                    tdatas = self._modules[lname]()
                    if type(tdatas) != tuple:
                        tdatas = (tdatas,)
                    assert len(tdatas) == len(tnames)
                    assert len(tdatas) > 0
                    for index, tdata in enumerate(tdatas):
                        self.blobs[tnames[index]] = tdata
                    self.seen_images += tdatas[0].size(0) * self.gpus_size
                    if self.verbose:
                        logging.info('forward %-30s produce -> %s' % (
                            lname,
                            self._blob_size(tnames)
                        ))

                i = i + 1
                continue

            bname = layer['bottom']
            bnames = bname if type(bname) == list else [bname]
            bdatas = [self.blobs[name] for name in bnames]
            tdatas = self._modules[lname](*bdatas)
            if type(tdatas) != tuple:
                tdatas = (tdatas,)

            assert (len(tdatas) == len(tnames))
            for index, tdata in enumerate(tdatas):
                name = tnames[index]
                if self.keep_diffs and tdata.requires_grad:
                    def _back_hook(_name):
                        def func(grad):
                            self.diffs[_name] = grad.cpu()

                        return func

                    tdata.register_hook(_back_hook(name))

                self.blobs[name] = tdata

            i = i + 1
            if self.verbose:
                logging.info('forward %-30s %s -> %s' % (
                    lname,
                    self._blob_size(bnames),
                    self._blob_size(tnames)
                ))
        #return self.blobs.get(tname), self.blobs.get('dark6e/conv') #  remove ONLY for debugging yolo and darknet
        # TODO: fix the need in above 
        if self.targets:
            tdatas = [[tname, self.blobs.get(tname)] for tname in self.targets]
            odatas = [d[1] for d in tdatas if d[1] is not None]
            # stack the outputs if all are loss values, otherwise return a dict
            if len(odatas) > 1:
                for d in odatas:
                    if d.numel() != 1:
                        return {
                            d[0]: d[1] for d in tdatas if d[1] is not None
                        }
            return torch.stack(odatas).squeeze()

    def get_targets(self):
        """Automaticlly get the set of network outputs
        :rtype: set
        """
        layers = self.net_info['layers']
        targets = set()
        for layer in layers:
            if 'top' not in layer:
                # Silence layer perhaps
                continue
            tname = layer['top']
            tnames = tname if type(tname) == list else [tname]
            targets.update(tnames)

        bottoms = set()
        for layer in layers:
            if 'bottom' not in layer:
                continue
            bname = layer['bottom']
            bnames = bname if type(bname) == list else [bname]
            bottoms.update(bnames)
        # tops that are not bottoms
        targets -= bottoms
        del bottoms

        if self.verbose:
            logging.info("Network outputs: {}".format(targets))

        return targets

    def print_network(self):
        print(self)
        print_prototxt(self.net_info)

    def save_weights(self, caffemodel, state=None):
        """Save weights to caffemodel
        :param caffemodel: path to save the weights
        :param state: PyTorch model state dict
        """
        if state is None:
            state = self.state_dict()
        save_caffemodel(state, caffemodel, self.protofile,
                        net_info=self.net_info, verbose=self.verbose)

    @staticmethod
    def _set_weights(lname, caffe_weight, target_weight, ignore_shape_mismatch):
        if caffe_weight.numel() == target_weight.numel() or not ignore_shape_mismatch:
            target_weight.data.copy_(caffe_weight.view_as(target_weight))
            return
        logging.info("{} shape mismatch: {} -> {}".format(
            lname, caffe_weight.shape, target_weight.shape
        ))
        copy_count = np.minimum(target_weight.numel(), caffe_weight.numel())
        caffe_weight = caffe_weight.view(caffe_weight.numel())[:copy_count]
        target_weight.data.view(target_weight.numel())[:copy_count] = caffe_weight

    def load_weights(self, caffemodel, ignore_shape_mismatch=True):
        """Load weights from caffemodel
        :param caffemodel: caffemodel file (could be already loaded)
        :param ignore_shape_mismatch: if should ignore shape mismatch and partially load
        """
        if self.has_mean:
            logging.info('mean_file', self.mean_file)
            mean_blob = read_blob(self.mean_file)

            if 'input_shape' in self.net_info['props']:
                channels = int(self.net_info['props']['input_shape']['dim'][1])
                height = int(self.net_info['props']['input_shape']['dim'][2])
                width = int(self.net_info['props']['input_shape']['dim'][3])
            else:
                channels = int(self.net_info['props']['input_dim'][1])
                height = int(self.net_info['props']['input_dim'][2])
                width = int(self.net_info['props']['input_dim'][3])

            mu = np.array(mean_blob.data)
            # noinspection PyTypeChecker
            mu.resize(channels, height, width)
            mu = mu.mean(1).mean(1)
            mean_img = torch.from_numpy(mu).view(channels, 1, 1).expand(channels, height, width).float()

            self.register_buffer('mean_img', torch.zeros(channels, height, width))
            self.mean_img.copy_(mean_img)

        if isinstance(caffemodel, basestring):
            model = read_model(caffemodel)
        else:
            model = caffemodel
        layers = model.layer
        if len(layers) == 0:
            logging.warning('Using V1LayerParameter')
            layers = model.layers

        lmap = {}
        for l in layers:
            lmap[l.name] = l

        layers = self.net_info['layers']
        layer_num = len(layers)
        i = 0
        while i < layer_num:
            layer = layers[i]
            lname = layer['name']
            if 'include' in layer and 'phase' in layer['include']:
                phase = layer['include']['phase']
                lname = lname + '.' + phase
            ltype = layer['type']
            if lname not in lmap:
                i = i + 1
                continue
            
            if ltype in ['Convolution', 'Deconvolution']:
                if self.verbose:
                    logging.info('load weights %s' % lname)
                convolution_param = layer['convolution_param']
                bias = True
                if 'bias_term' in convolution_param and convolution_param['bias_term'] == 'false':
                    bias = False
                self._set_weights(
                    lname,
                    torch.from_numpy(np.array(lmap[lname].blobs[0].data)),
                    self.models[lname].weight,
                    ignore_shape_mismatch
                )
                if bias and len(lmap[lname].blobs) > 1:
                    self._set_weights(
                        lname,
                        torch.from_numpy(np.array(lmap[lname].blobs[1].data)),
                        self.models[lname].bias,
                        ignore_shape_mismatch
                    )
                i = i + 1
            elif ltype == 'BatchNorm':
                if self.verbose:
                    logging.info('load weights %s' % lname)
                if lmap[lname].blobs[2].data[0] > 0.0000001:
                    self.models[lname].running_mean.copy_(
                        torch.from_numpy(np.array(lmap[lname].blobs[0].data) / lmap[lname].blobs[2].data[0]))
                    self.models[lname].running_var.copy_(
                        torch.from_numpy(np.array(lmap[lname].blobs[1].data) / lmap[lname].blobs[2].data[0]))  
                else:
                    self.models[lname].running_mean.copy_(
                        torch.from_numpy(np.array(lmap[lname].blobs[0].data)))
                    self.models[lname].running_var.copy_(
                        torch.from_numpy(np.array(lmap[lname].blobs[1].data)))
                i = i + 1
            elif ltype == 'Scale':
                if self.verbose:
                    logging.info('load weights %s' % lname)
                self._set_weights(
                    lname,
                    torch.from_numpy(np.array(lmap[lname].blobs[0].data)),
                    self.models[lname].weight,
                    ignore_shape_mismatch
                )
                self._set_weights(
                    lname,
                    torch.from_numpy(np.array(lmap[lname].blobs[1].data)),
                    self.models[lname].bias,
                    ignore_shape_mismatch
                )
                i = i + 1

            elif ltype == 'Normalize':
                if self.verbose:
                    logging.info('load weights %s' % lname)
                self.models[lname].weight.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data)))
                i = i + 1
            elif ltype == 'InnerProduct':
                if self.verbose:
                    logging.info('load weights %s' % lname)
                if type(self.models[lname]) == nn.Sequential:
                    self._set_weights(
                        lname,
                        torch.from_numpy(np.array(lmap[lname].blobs[0].data)),
                        self.models[lname][1].weight,
                        ignore_shape_mismatch
                    )
                    if len(lmap[lname].blobs) > 1:
                        self._set_weights(
                            lname,
                            torch.from_numpy(np.array(lmap[lname].blobs[1].data)),
                            self.models[lname][1].bias,
                            ignore_shape_mismatch
                        )
                else:
                    self._set_weights(
                        lname,
                        torch.from_numpy(np.array(lmap[lname].blobs[0].data)),
                        self.models[lname].weight,
                        ignore_shape_mismatch
                    )
                    if len(lmap[lname].blobs) > 1:
                        self._set_weights(
                            lname,
                            torch.from_numpy(np.array(lmap[lname].blobs[1].data)),
                            self.models[lname].bias,
                            ignore_shape_mismatch
                        )
                i = i + 1
            elif ltype == 'RegionTarget' or ltype == 'RegionLoss':
                if self.verbose:
                    logging.info('load weights %s' % lname)
                self.models[lname].seen_images.data.copy_(torch.tensor(lmap[lname].blobs[0].data[0]))
                self.seen_images = self.models[lname].seen_images
                i = i + 1
            else:
                if ltype not in SUPPORTED_LAYERS:
                    logging.error('unknown type %s' % ltype)
                i = i + 1

    def create_network(self,
                       input_batch_size=None, input_width=None, input_height=None, input_channels=None,
                       raise_unknown=True):
        models = OrderedDict()

        layers = self.net_info['layers']
        props = self.net_info['props']
        layer_num = len(layers)

        test_inputs = []
        test_input_shapes = []
        if 'input' in props:
            test_inputs = props['input']
            if not isinstance(test_inputs, list):
                test_inputs = [test_inputs]
        if 'input_shape' in props:
            test_input_shapes = props['input_shape']
            if not isinstance(test_input_shapes, list):
                test_input_shapes = [test_input_shapes]
            if not test_inputs:
                test_inputs = ['data']
        elif 'input_dim' in props:
            # Support old prototxt format
            test_input_shapes = [
                {'dim': dims} for dims in [
                    props['input_dim'][d * 4:d * 4 + 4] for d in range(len(props['input_dim']) / 4)
                ]
            ]

        assert len(test_inputs) == len(test_input_shapes), \
            "input_shape {} and input {} were not of the same size".format(len(test_inputs), len(test_input_shapes))
        if test_inputs:
            self.input_blobs = test_inputs
        for input_blob, shape in zip(test_inputs, test_input_shapes):
            shape = [int(d) for d in shape['dim']]

            shape[0] = input_batch_size or shape[0]
            self.batch_size = self.batch_size or shape[0]
            if len(shape) == 4:
                shape[1] = input_channels or shape[1]
                shape[2] = input_width or shape[2]
                shape[3] = input_height or shape[3]
                self.width = self.width or shape[2]
                self.height = self.height or shape[3]
            self.blob_dims[input_blob] = tuple(shape)

        i = 0
        while i < layer_num:
            layer = layers[i]
            lname = layer['name']
            if 'include' in layer and 'phase' in layer['include']:
                phase = layer['include']['phase']
                lname = lname + '.' + phase
                if phase != self.phase:
                    i = i + 1
                    continue

            ltype = layer['type']
            if 'top' not in layer:
                assert ltype == 'Silence'
                if raise_unknown:
                    raise NotImplementedError("Silence layer not implemented yet")
                if self.network_verbose:
                    logging.info('create %-20s' % lname)
                if self.network_verbose:
                    logging.info("Ignore {}({})".format(ltype, lname))
                continue
            tname = layer['top']
            if ltype == 'TsvBoxData':
                if self.input_index is not None:
                    raise NotImplementedError("Compound data layers not implemented yet for PyTorch data")
                self.input_index = i
                self.height = int(layer['box_data_param']['random_min'])
                self.width = int(layer['box_data_param']['random_min'])
                data_name = tname[0]
                label_name = tname[1]                
                if not self.batch_size:
                    self.batch_size = int(layer['tsv_data_param']['batch_size'])
                    assert self.batch_size > 0, "Invalid batch_size: {} in prototxt".format(self.batch_size)
                n = (self.batch_size or 1) * self.local_gpus_size
                self.blob_dims[label_name] = (n, BOX_DIMS * int(layer['box_data_param']['max_boxes']))
                self.blob_dims[data_name] = (n, NUM_CHANNELS, self.width, self.height)
                i += 1
                if self.network_verbose:
                    logging.info('create %-20s %s' % (
                        lname,
                        self._blob_shape(tname)
                    ))
                continue
            if ltype in ['Data', 'AnnotatedData', 'HDF5Data']:
                # rely on caffe to give us dimensions
                if self.forward_net_only.item() and self.input_index is not None:
                    raise NotImplementedError("Compound data layers with forward_net_only not implemented yet")
                self.input_index = i
                inputs = CaffeData(layer.copy(), self.phase, self.local_gpus_size,
                                   batch_size=self.batch_size)
                if not self.forward_net_only.item():
                    # keep this as a module, so that foward would call it
                    models[lname] = inputs
                label_name = None
                if isinstance(tname, list):
                    data_name = tname[0]
                    if len(tname) > 1:
                        label_name = tname[1]
                else:
                    data_name = tname
                data, label = inputs.forward()  # forward once just to get the size
                dims = [data.size(ii) for ii in range(data.dim())]
                self.height = dims[2]
                self.width = dims[3]
                if label_name:
                    label_dims = [label.size(ii) for ii in range(label.dim())]
                    assert label_dims[0] == dims[0], "Data batch size: {} != Label batch size: {}".format(
                        dims[0],
                        label_dims[0]
                    )
                    self.blob_dims[label_name] = tuple(label_dims)

                self.blob_dims[data_name] = tuple(dims)

                i = i + 1
                if self.network_verbose:
                    logging.info('create %-20s %s' % (
                        lname,
                        self._blob_shape(tname)
                    ))
                del data, label
                continue

            if 'bottom' not in layer:
                # layer without bottom is perhaps data layer
                if raise_unknown:
                    raise NotImplementedError('unknown data layer type #%s#' % ltype)
                logging.error('unknown data layer type #%s#' % ltype)
                i = i + 1
                continue

            bname = layer['bottom']
            if ltype == 'Convolution':
                convolution_param = layer['convolution_param']
                n, c = self.blob_dims[bname][:2]
                out_filters = int(convolution_param['num_output'])
                kernel_size = int(convolution_param['kernel_size'])
                stride = int(convolution_param['stride']) if 'stride' in convolution_param else 1
                pad = int(convolution_param['pad']) if 'pad' in convolution_param else 0
                group = int(convolution_param['group']) if 'group' in convolution_param else 1
                dilation = 1
                if 'dilation' in convolution_param:
                    dilation = int(convolution_param['dilation'])
                bias = True
                if 'bias_term' in convolution_param and convolution_param['bias_term'] == 'false':
                    bias = False
                models[lname] = nn.Conv2d(c, out_filters, kernel_size=kernel_size, stride=stride, padding=pad,
                                          dilation=dilation, groups=group, bias=bias)
                c = out_filters
                w = (self.blob_dims[bname][3] + 2 * pad - kernel_size) / stride + 1
                h = (self.blob_dims[bname][2] + 2 * pad - kernel_size) / stride + 1
                self.blob_dims[tname] = n, c, h, w
                i = i + 1
            elif ltype == 'BatchNorm':
                momentum = 0.1 # this is in fact PyTorch Default
                n, c, h, w = self.blob_dims[bname]
                models[lname] = nn.BatchNorm2d(c, momentum=momentum, affine=False)
                self.blob_dims[tname] = self.blob_dims[bname]
                i = i + 1
            elif ltype == 'Scale':
                n, c, h, w = self.blob_dims[bname]
                models[lname] = Scale(c)
                self.blob_dims[tname] = self.blob_dims[bname]
                i = i + 1
            elif ltype == 'ReLU':
                inplace = (bname == tname)
                if 'relu_param' in layer and 'negative_slope' in layer['relu_param']:
                    negative_slope = float(layer['relu_param']['negative_slope'])
                    models[lname] = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
                else:
                    models[lname] = nn.ReLU(inplace=inplace)
                self.blob_dims[tname] = self.blob_dims[bname]
                i = i + 1
            elif ltype == 'Pooling':
                kernel_size = int(layer['pooling_param']['kernel_size'])
                stride = int(layer['pooling_param'].get('stride', 1))
                padding = int(layer['pooling_param'].get('pad', 0))
                pool_type = layer['pooling_param']['pool']
                if pool_type == 'MAX':
                    models[lname] = nn.MaxPool2d(kernel_size, stride, padding=padding, ceil_mode=True)
                elif pool_type == 'AVE':
                    models[lname] = nn.AvgPool2d(kernel_size, stride, padding=padding, ceil_mode=True)

                n, c, h, w = self.blob_dims[bname]
                new_w = int(math.ceil((w + 2 * padding - kernel_size) / float(stride))) + 1
                new_h = int(math.ceil((h + 2 * padding - kernel_size) / float(stride))) + 1
                self.blob_dims[tname] = n, c, new_h, new_w
                i = i + 1
            elif ltype == 'Eltwise':
                operation = 'SUM'
                if 'eltwise_param' in layer and 'operation' in layer['eltwise_param']:
                    operation = layer['eltwise_param']['operation']
                models[lname] = Eltwise(operation)
                self.blob_dims[tname] = self.blob_dims[bname[0]]
                i = i + 1
            elif ltype == 'InnerProduct':
                filters = int(layer['inner_product_param']['num_output'])
                n, c, h, w = self.blob_dims[bname]
                if w != -1 or h != -1:
                    channels = c * h * w
                    models[lname] = nn.Sequential(FCView(), nn.Linear(channels, filters))
                else:
                    models[lname] = nn.Linear(c, filters)
                self.blob_dims[tname] = n, filters, 1, 1
                i = i + 1
            elif ltype == 'Dropout':
                # channels = self.blob_dims[bname][0]
                dropout_ratio = float(layer['dropout_param']['dropout_ratio'])
                models[lname] = nn.Dropout(dropout_ratio, inplace=True)
                self.blob_dims[tname] = self.blob_dims[bname]
                i = i + 1
            elif ltype == 'Normalize':
                channels = self.blob_dims[bname][1]
                scale = float(layer['norm_param']['scale_filler']['value'])
                models[lname] = Normalize(channels, scale)
                self.blob_dims[tname] = self.blob_dims[bname]
                i = i + 1
            elif ltype == 'Permute':
                orders = [int(order) for order in layer['permute_param']['order']]
                models[lname] = Permute(orders)
                shape = list(self.blob_dims[bname])
                self.blob_dims[tname] = tuple([shape[order] for order in orders])
                i = i + 1
            elif ltype == 'Flatten':
                axis = int(layer['flatten_param'].get('axis', 1))
                assert axis > 0, "Not implemented"
                end_axis = int(layer['flatten_param'].get('end_axis', -1))
                assert end_axis == -1, "Not implemented"
                dims = self.blob_dims[bname]
                assert axis < len(dims)
                models[lname] = Flatten(axis)
                self.blob_dims[tname] = dims[0], np.prod(dims[axis:])
                i = i + 1
            elif ltype == 'Slice':
                axis = int(layer['slice_param'].get('axis', 1))
                assert type(tname == list)
                slice_points = layer['slice_param']['slice_point']
                assert type(slice_points) == list
                assert len(slice_points) == len(tname) - 1
                slice_points = [int(s) for s in slice_points]
                shape = self.blob_dims[bname]
                slice_points.append(shape[axis])
                models[lname] = Slice(axis, slice_points)
                shape = list(shape)
                prev = 0
                for idx, tn in enumerate(tname):
                    shape[axis] = slice_points[idx] - prev
                    self.blob_dims[tn] = tuple(shape)
                    prev = slice_points[idx]
                i = i + 1
            elif ltype == 'Concat':
                axis = int(layer.get('concat_param', {}).get('axis', 1))
                models[lname] = Concat(axis)
                tdims = None
                for bn in bname:
                    dims = list(self.blob_dims[bn])
                    assert axis < len(dims), "btottom: {} axis: {} >= {}".format(bn, axis, len(dims))
                    if tdims is None:
                        tdims = dims
                    else:
                        assert axis < len(tdims), "bottom: {} axis: {} shape changed".format(bn, axis)
                        tdims[axis] += dims[axis]
                self.blob_dims[tname] = tuple(tdims)
                i = i + 1
            elif ltype == 'Crop':
                axis = int(layer['crop_param']['axis'])
                offset = int(layer['crop_param']['offset'])
                models[lname] = Crop(axis, offset)
                self.blob_dims[tname] = self.blob_dims[bname[0]]
                i = i + 1
            elif ltype == 'Deconvolution':
                # models[lname] = nn.UpsamplingBilinear2d(scale_factor=2)
                # models[lname] = nn.Upsample(scale_factor=2, mode='bilinear')
                n, c, h, w = self.blob_dims[bname]
                in_channels = c
                out_channels = int(layer['convolution_param']['num_output'])
                group = int(layer['convolution_param']['group'])
                kernel_w = int(layer['convolution_param']['kernel_w'])
                kernel_h = int(layer['convolution_param']['kernel_h'])
                stride_w = int(layer['convolution_param']['stride_w'])
                stride_h = int(layer['convolution_param']['stride_h'])
                pad_w = int(layer['convolution_param']['pad_w'])
                pad_h = int(layer['convolution_param']['pad_h'])
                kernel_size = (kernel_h, kernel_w)
                stride = (stride_h, stride_w)
                padding = (pad_h, pad_w)
                bias_term = layer['convolution_param']['bias_term'] != 'false'
                models[lname] = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                                   padding=padding, groups=group, bias=bias_term)
                self.blob_dims[tname] = n, out_channels, 2 * h, 2 * w
                i = i + 1
            elif ltype == 'Reshape':
                reshape_dims = layer['reshape_param']['shape']['dim']
                reshape_dims = [int(item) for item in reshape_dims]
                reshape_axis = int(layer['reshape_param'].get('axis', 0))
                reshape_num_axes = int(layer['reshape_param'].get('num_axes', -1))
                assert bname in self.blob_dims, "{}({}) must know the dimensions of {}".format(ltype, lname, bname)
                models[lname] = Reshape(reshape_dims, reshape_axis, reshape_num_axes)
                reshape_dims = reshape_helper(self.blob_dims[bname], reshape_dims,
                                              axis=reshape_axis, num_axes=reshape_num_axes)
                self.blob_dims[tname] = tuple(reshape_dims)
                i = i + 1
            elif ltype == 'Softmax':
                axis = int(layer.get('softmax_param', {}).get('axis', 1))
                models[lname] = nn.Softmax(dim=axis)
                self.blob_dims[tname] = self.blob_dims[bname]
                i = i + 1
            elif ltype == 'Sigmoid':
                models[lname] = nn.Sigmoid()
                self.blob_dims[tname] = self.blob_dims[bname]
                i = i + 1
            elif ltype == 'Accuracy':
                models[lname] = Accuracy()
                self.blob_dims[tname] = 1, 1, 1
                i = i + 1
            elif ltype == 'SoftmaxWithLoss':
                loss_weight = float(layer.get('loss_weight', 1))
                normalization = layer.get('loss_param', {}).get('normalization', 'BATCH_SIZE')
                assert normalization in ['VALID', 'BATCH_SIZE'], "Unsupported normalixation: {}".format(
                    normalization
                )
                ignore_label = layer.get('loss_param', {}).get('ignore_label')
                if ignore_label is not None:
                    ignore_label = int(ignore_label)
                models[lname] = SoftmaxWithLoss(
                    loss_weight=loss_weight, ignore_label=ignore_label,
                    valid_normalization=(normalization == 'VALID')
                )
                self.blob_dims[tname] = 1, 1, 1
                i = i + 1
            elif ltype == 'EuclideanLoss':
                assert isinstance(bname, list)
                assert 2 <= len(bname) <= 3
                loss_weight = float(layer.get('loss_weight', 1))
                models[lname] = EuclideanLoss(loss_weight=loss_weight)
                self.blob_dims[tname] = 1, 1, 1
                i = i + 1
            elif ltype == 'RegionTarget':
                assert isinstance(bname, list) and len(bname) == 4
                assert isinstance(tname, list) and len(tname) == 6
                biases = [float(b) for b in layer['region_target_param']['biases']]
                rescore = layer['region_target_param'].get('rescore', 'true') == 'true'
                anchor_aligned_images = int(layer['region_target_param'].get('anchor_aligned_images', 12800))
                coord_scale = float(layer['region_target_param'].get('coord_scale', 1.0))
                positive_thresh = float(layer['region_target_param'].get('positive_thresh', 0.6))
                models[lname] = RegionTarget(
                    biases,
                    rescore=rescore, anchor_aligned_images=anchor_aligned_images, coord_scale=coord_scale,
                    positive_thresh=positive_thresh,
                    gpus_size=self.gpus_size,
                    seen_images=self.seen_images.item()
                )
                dims = self.blob_dims[bname[0]]
                for ii in range(3):
                    self.blob_dims[tname[ii]] = dims
                for ii in range(3, 6):
                    self.blob_dims[tname[ii]] = self.blob_dims[bname[2]]
                i = i + 1
            elif ltype == 'SoftmaxTree':
                tree = layer['softmaxtree_param']['tree']
                axis = int(layer.get('softmaxtree_param', {}).get('axis', 1))
                models[lname] = SoftmaxTree(tree, axis=axis)
                self.blob_dims[tname] = self.blob_dims[bname]
                i = i + 1
            elif ltype == 'SoftmaxTreeWithLoss':
                assert isinstance(bname, list) and len(bname) == 2
                tree = layer['softmaxtree_param']['tree']
                loss_weight = float(layer.get('loss_weight', 1))
                ignore_label = layer.get('loss_param', {}).get('ignore_label')
                if ignore_label is not None:
                    ignore_label = int(ignore_label)
                normalization = layer.get('loss_param', {}).get('normalization', 'BATCH_SIZE')
                assert normalization in ['VALID', 'BATCH_SIZE'], "Unsupported normalixation: {}".format(
                    normalization
                )
                models[lname] = SoftmaxTreeWithLoss(
                    tree, ignore_label=ignore_label, loss_weight=loss_weight,
                    valid_normalization=(normalization == 'VALID')
                )
                self.blob_dims[tname] = self.blob_dims[bname[0]]
                i = i + 1
            elif ltype == 'SoftmaxTreePrediction':
                assert not isinstance(bname, list) or len(bname) == 2
                tree = layer['softmaxtreeprediction_param']['tree']
                threshold = float(layer.get('softmaxtreeprediction_param', {}).get('threshold', 0.5))
                append_max = layer.get('softmaxtreeprediction_param', {}).get('append_max', 'true') == 'true'
                output_tree_path = layer.get(
                    'softmaxtreeprediction_param', {}
                ).get('output_tree_path', 'false') == 'true'
                models[lname] = SoftmaxTreePrediction(
                    tree, threshold=threshold, append_max=append_max, output_tree_path=output_tree_path
                )
                if isinstance(bname, list):
                    bname = bname[0]
                dims = list(self.blob_dims[bname])
                if append_max:
                    dims[1] += 1
                self.blob_dims[tname] = dims
                i = i + 1
            elif ltype == 'Reorg':
                stride = int(layer.get('reorg_param', {}).get('stride', 2))
                reverse = layer.get('reorg_param', {}).get('reverse', 'false') == 'true'
                if reverse:
                    raise NotImplementedError('Reorg with reverse not implemented')
                flatten = layer.get('reorg_param', {}).get('flatten', 'false') == 'true'
                if flatten:
                    raise NotImplementedError('Reorg with flatten not implemented')
                n, c, h, w = self.blob_dims[bname]
                models[lname] = Reorg(stride)
                self.blob_dims[tname] = n, c * stride * stride, h / stride, w / stride
                i = i + 1
            elif ltype == 'YoloEvalCompat':
                append_max = layer.get('yoloevalcompat_param', {}).get('append_max', 'false') == 'true'
                if append_max:
                    raise NotImplementedError('YoloEvalCompat with append_max not implemented')
                move_axis = layer.get('yoloevalcompat_param', {}).get('move_axis', 'true') == 'true'
                if not move_axis:
                    raise NotImplementedError('YoloEvalCompat without move_axis not implemented')
                if isinstance(bname, list):
                    if len(bname) > 1:
                        raise NotImplementedError('YoloEvalCompat with multi-head not implemented')
                    bname = bname[0]
                models[lname] = YoloEvalCompat()
                self.blob_dims[tname] = tuple(
                    [self.blob_dims[bname][0]] + list(self.blob_dims[bname][2:]) + [self.blob_dims[bname][1]]
                )
                i = i + 1
            elif ltype == 'YoloBBs':
                biases = [float(b) for b in layer['yolobbs_param']['biases']]
                feat_stride = int(layer.get('yolobbs_param', {}).get('feat_stride', 32))
                models[lname] = YoloBBs(biases=biases, feat_stride=feat_stride)
                self.blob_dims[tname] = tuple(list(self.blob_dims[bname[0]]) + [4])
                i = i + 1
            elif ltype == 'NMSFilter':
                assert isinstance(bname, list) and len(bname) == 2
                threshold = float(layer.get('nmsfilter_param', {}).get('threshold', 0.45))
                classes = int(layer.get('nmsfilter_param', {}).get('classes', 1))
                pre_threshold = float(layer.get('nmsfilter_param', {}).get('pre_threshold', -1.0))
                first_class = int(layer.get('nmsfilter_param', {}).get('first_class', 0))

                models[lname] = NMSFilter(
                    nms_threshold=threshold,
                    pre_threshold=pre_threshold,
                    classes=classes,
                    first_class=first_class
                )
                self.blob_dims[tname] = self.blob_dims[bname[1]]
                i = i + 1
            elif ltype == 'RegionLoss':
                models[lname] = RegionLoss() # TODO: add arguments based on prototxt
                self.blob_dims[tname] = self.blob_dims[bname[0]] 
                i = i + 1
            else:
                if raise_unknown:
                    raise NotImplementedError('unknown layer type #%s#' % ltype)
                logging.error('unknown layer type #%s#' % ltype)
                i = i + 1
                continue

            if self.network_verbose:
                logging.info('create %-20s %s -> %s' % (
                    lname,
                    self._blob_shape(bname),
                    self._blob_shape(tname)
                ))

        return models

    @property
    def inputs(self):
        """CaffeData inputs (for compatibility)
        :rtype: CaffeData
        """
        assert self.input_index is not None, "Data layer does not exist or is not created yet"
        layers = self.net_info['layers']
        layer = layers[self.input_index]
        inputs = CaffeData(layer.copy(), self.phase, self.local_gpus_size,
                           batch_size=self.batch_size)
        return inputs


def main():
    init_logging()
    parser = argparse.ArgumentParser(description='Convert Caffe training to PyTorch.')

    parser.add_argument('-s', '--solver',
                        help='Prototxt solver file for caffe training')
    parser.add_argument('--net',
                        help='Caffenet prototxt file for caffe training')
    parser.add_argument('--weights',
                        help='Caffemodel weights file for finetuning')
    parser.add_argument('--work',
                        help='Working path (where "data" folder is located)',
                        default=".")

    args = parser.parse_args()
    args = vars(args)
    os.environ['GLOG_minloglevel'] = '2'

    with cwd(args['work']):
        caffemodel = args['weights']
        protofile = args['net']
        if protofile:
            assert op.isfile(protofile), "{} does not exist".format(protofile)
        solverfile = args['solver']
        if solverfile:
            net = parse_prototxt(solverfile)
            if not protofile:
                protofile = net.get("train_net") or net.get("net")
                assert op.isfile(protofile), "{} does not exist".format(protofile)

        net = CaffeNet(protofile)
        if caffemodel:
            assert op.isfile(caffemodel), "{} does not exist".format(caffemodel)
            net.load_weights(caffemodel)

    return net


if __name__ == '__main__':
    engine = main()

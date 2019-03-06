import os
import os.path as op
import torch
import numpy as np
import logging
import argparse

from mmod.utils import init_logging
from mmod.philly_utils import abspath
from mmod.simple_parser import read_model_proto, array_to_blobproto, parse_prototxt

SUPPORTED_LAYERS = ['Data', 'TsvBoxData', 'AnnotatedData', 'HDF5Data', 'Pooling', 'Eltwise', 'ReLU',
                    'Permute', 'Flatten', 'Slice', 'Concat', 'Softmax', 'SoftmaxWithLoss',
                    'Dropout', 'Reshape', 'Sigmoid', 'EuclideanLoss', 'RegionTarget', 'SoftmaxTreeWithLoss',
                    'Reorg', 'YoloEvalCompat', 'YoloBBs', 'SoftmaxTreePrediction', 'SoftmaxTree', 'NMSFilter',
                    'RegionLoss', 'Input']


try:
    # indirect import of matplotlib (e.g. by caffe) may try to load non-existent X
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    matplotlib = None


def _filter_module(k):
    if k.startswith('module.'):
        return k[7:]
    return k


def save_caffemodel(ptmodel, caffemodel, protofile,
                    net_info=None,
                    verbose=False):
    """Convert the  PyTorch snapshot to caffemodel
    :param ptmodel: PyTorch model [state dict]
    :type ptmodel: dict or str
    :param caffemodel: output caffemodel file path
    :type caffemodel: str
    :param protofile: train.prototxt path
    :type protofile: str
    :param net_info: parsed train.prototxt dictionary info
    :type net_info: dict
    :param verbose: if should print verbose outputs
    """
    state = ptmodel
    if isinstance(ptmodel, basestring):
        state = torch.load(ptmodel)["state_dict"]
        state = {
            _filter_module(k): v for (k, v) in state.iteritems()
        }

    if net_info is None:
        net_info = parse_prototxt(protofile)
    net = read_model_proto(protofile)
    layers = net.layer
    if len(layers) == 0:
        logging.warning('Using V1LayerParameter')
        layers = net.layers

    lmap = {}
    for l in layers:
        lmap[l.name] = l

    layers = net_info['layers']
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
            if verbose:
                logging.info('save weights %s' % lname)
            convolution_param = layer['convolution_param']
            bias = True
            if 'bias_term' in convolution_param and convolution_param['bias_term'] == 'false':
                bias = False

            blobs = [state[lname + '.weight'].data.cpu().numpy()]
            if bias:
                blobs.append(state[lname + '.bias'].data.cpu().numpy())
            i = i + 1
        elif ltype == 'BatchNorm':
            if verbose:
                logging.info('save weights %s' % lname)
            blobs = [
                state[lname + '.running_mean'].data.cpu().numpy(),
                state[lname + '.running_var'].data.cpu().numpy(),
                np.array([1.0], dtype=np.float32)  # TODO: can find a better scale
            ]
            i = i + 1
        elif ltype == 'Scale':
            if verbose:
                logging.info('save weights %s' % lname)
            blobs = [
                state[lname + '.weight'].data.cpu().numpy(),
                state[lname + '.bias'].data.cpu().numpy()
            ]
            i = i + 1
        elif ltype == 'Normalize':
            if verbose:
                logging.info('save weights %s' % lname)
            blobs = [
                state[lname + '.weight'].data.cpu().numpy()
            ]
            i = i + 1
        elif ltype == 'InnerProduct':
            if verbose:
                logging.info('save weights %s' % lname)
            inner_product_param = layer['inner_product_param']
            bias = True
            if 'bias_term' in inner_product_param and inner_product_param['bias_term'] == 'false':
                bias = False
            if (lname + '.1.weight') in state:
                blobs = [state[lname + '.1.weight'].data.cpu().numpy()]
                if bias:
                    blobs.append(state[lname + '.1.bias'].data.cpu().numpy())
            else:
                blobs = [state[lname + '.weight'].data.cpu().numpy()]
                if bias:
                    blobs.append(state[lname + '.bias'].data.cpu().numpy())
            i = i + 1
        elif ltype == 'RegionTarget':
            if verbose:
                logging.info('save weights %s' % lname)
            blobs = [
                np.array([[[[state[lname + '.seen_images'].data.item()]]]], dtype=np.float32)
            ]
            i = i + 1
        else:
            if ltype not in SUPPORTED_LAYERS:
                logging.error('unknown type %s' % ltype)
            i = i + 1
            continue

        lmap[lname].blobs.extend([
            array_to_blobproto(b) for b in blobs
        ])

    with open(caffemodel, 'wb') as f:
        f.write(net.SerializeToString())


def main():
    init_logging()
    parser = argparse.ArgumentParser(description='Convert PyTorch snapshots to caffemodel.')

    parser.add_argument('path',
                        help='snapshot path (where .pt files are located), or a single .pt file',
                        default=".")
    parser.add_argument('--net',
                        help='Caffenet prototxt file for caffe training')
    parser.add_argument('--verbose', help='Verbose output', action='store_true',
                        default=False, required=False)
    args = parser.parse_args()
    protofile = abspath(args.net, roots=['~', '#', '.'])
    net_info = parse_prototxt(protofile)
    path = abspath(args.path, roots=['#', '.'])
    if op.isfile(path):
        fname = op.basename(path)
        name, _ = op.splitext(fname)
        caffemodel = op.join(op.dirname(path), name + '.caffemodel')
        if op.isfile(caffemodel):
            logging.info("{} file already exists".format(caffemodel))
            return
        save_caffemodel(path, caffemodel, protofile, net_info=net_info, verbose=args.verbose)
        return
    for fname in os.listdir(path):
        name, ext = op.splitext(fname)
        if ext != '.pt':
            continue
        caffemodel = op.join(path, name + '.caffemodel')
        if op.isfile(caffemodel):
            logging.info("{} file already exists".format(caffemodel))
            continue
        logging.info("converting {}".format(caffemodel))
        save_caffemodel(op.join(path, fname), caffemodel, protofile, net_info=net_info, verbose=args.verbose)


if __name__ == '__main__':
    main()

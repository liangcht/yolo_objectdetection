from collections import OrderedDict
import logging
import numpy as np
import os

import torch
import torchvision.models
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import caffe

import sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtorch.caffenet import CaffeNet
from mmod.imdb import ImageDatabase
from mmod.detection import resize_for_od
from mmod.utils import init_logging

def prep_dict_resnet_pt2caffe(init_net_dict, net_dict, mapping_file=None, switch_bn2scale=True):

    """fixes the names of layers in initializing state dictionary
    to fit the new network state dictionary
    :param init_net_dict: dictionary to take weights from
    :param net_dict: dictionary to take layers names from
    :param switch_bn2scale: boolean,
    if to switch to BatchNormalization to Scale when searching for a weight
    """
    layer_names = net_dict.keys()
    init_weights = []
    all_mapping = []
    for layer_name in layer_names:
        if layer_name == "forward_net_only" or layer_name == "seen_images":
            init_weights.append((layer_name, net_dict[layer_name]))

        org_layer_name = layer_name
        if '-scale' in org_layer_name:
            layer_name = org_layer_name.replace('-scale', '')
        elif 'fc.1.' in org_layer_name:
            layer_name = org_layer_name.replace('fc.1.', 'fc.')
        layer_name = layer_name.replace('-', '.')

        for init_layer_name, init_weight in init_net_dict.items():
            if layer_name in init_layer_name:
                init_weights.append((org_layer_name, init_weight))
                logging.info("{} to {}".format(init_layer_name, org_layer_name))
                all_mapping.append([org_layer_name, init_layer_name])
                break
    if mapping_file:
        with open(mapping_file, 'w') as fp:
            for row in all_mapping:
                fp.write('\t'.join(row))
                fp.write('\n')
    return OrderedDict(init_weights)

def prep_dict_resnet_pt2caffe_ordered(pt_dict, caffe_dict, mapping_file=None):
    """Gets parameter names from Caffe model and the according weights from
    Pytorch model. Writes the name mapping results to mapping_file

    NOTE: This function assumes that the orders of layers are the same in
    Pytorch and Caffe, in order to generate the name mapping automatically.
    However, this is not always true (especially be careful with the expand
    layers in residual blocks). If the orders do not match, match the names in
    prototxt firstly, then call prep_dict_resnet_pt2caffe

    :param pt_dict: Pytorch state_dict to take weights from
    :param caffe_dict: CaffeNet state_dict to take layers' names from
    :param mapping_file: str, path to file.
    If None, the mapping will not be saved
    """
    caffe_keys = caffe_dict.keys()
    pt_keys = pt_dict.keys()
    ret = []
    all_mapping = []
    assert caffe_keys[0] == "forward_net_only"
    ret.append((caffe_keys[0], caffe_dict[caffe_keys[0]]))
    assert caffe_keys[1] == "seen_images"
    ret.append((caffe_keys[1], caffe_dict[caffe_keys[1]]))

    caffe_idx = 2
    pt_idx = 0

    def map_weights(cur_caffe_keys, cur_pt_keys):
        res = []
        mapping = []
        for k1, k2 in zip(cur_caffe_keys, cur_pt_keys):
            assert is_suffix_match(k1, k2)
            if "expand" in k1:
                assert "downsample" in k2
                if "bn" in k1 or "scale" in k1: assert "downsample.1." in k2
                elif "conv" in k1: assert "downsample.0." in k2
            elif "bn" in k1 or "scale" in k1:
                assert "bn" in k2
            elif "conv" in k1:
                assert "conv" in k2
            elif "ip" in k1:
                assert "fc" in k2
            res.append((k1, pt_dict[k2]))
            logging.info(k1, "=>", k2)
            mapping.append((k1, k2))
        return res, mapping

    while caffe_idx < len(caffe_dict):
        caffe_key = caffe_keys[caffe_idx]
        # batch norm + scale
        if caffe_key.endswith("running_mean"):
            num_params = 5
            cur_caffe_keys = [caffe_keys[caffe_idx+i] for i in range(num_params)]
            cur_pt_keys = [pt_keys[pt_idx+i] for i in [2, 3, 4, 0, 1]]
        # convolution
        elif caffe_key.endswith("weight") and "conv" in caffe_key:
            num_params = 1
            cur_caffe_keys = [caffe_keys[caffe_idx+i] for i in range(num_params)]
            cur_pt_keys = [pt_keys[pt_idx+i] for i in range(num_params)]
        # fully connected
        elif caffe_key.endswith("weight") and "fc" in caffe_key:
            num_params = 2
            cur_caffe_keys = [caffe_keys[caffe_idx+i] for i in range(num_params)]
            cur_pt_keys = [pt_keys[pt_idx+i] for i in range(num_params)]
        else:
            raise Exception("can not recognize caffe layer name: {}".format(caffe_key))

        cur_res, cur_mapping = map_weights(cur_caffe_keys, cur_pt_keys)
        ret.extend(cur_res)
        all_mapping.extend(cur_mapping)
        caffe_idx += num_params
        pt_idx += num_params

    assert caffe_idx == len(caffe_dict) and pt_idx == len(pt_dict)
    with open(mapping_file, 'w') as fp:
        for row in all_mapping:
            fp.write('\t'.join(row))
            fp.write('\n')

    return OrderedDict(ret)

def is_suffix_match(s1, s2):
    suf1 = s1.rsplit('.', 1)[1]
    suf2 = s2.rsplit('.', 1)[1]
    return suf1 == suf2

def load_pt_model(pt_model_path):
    is_cpu_only = not torch.cuda.is_available()
    if is_cpu_only:
        checkpoint = torch.load(pt_model_path, map_location='cpu')
    else:
        checkpoint = torch.load(pt_model_path)
    # checkpoint saves arch and state_dict, not the whole model
    arch = checkpoint['arch']
    pt_model = torchvision.models.__dict__[arch](num_classes=checkpoint['num_classes'])
    pt_model = torch.nn.DataParallel(pt_model)
    pt_model.load_state_dict(checkpoint['state_dict'])

    if not is_cpu_only:
        pt_model = pt_model.cuda()
    return pt_model

def compare_model(caffe_model, is_native_caffe, pt_model, dataset_path, max_imgs=100):
    db = ImageDatabase(dataset_path)
    num_imgs = min(len(db), max_imgs)
    diff_max = 0
    diff_sum = 0
    for i in range(num_imgs):
        im = db.image(i)
        probs_caffe = im_classification(im, caffe_model, is_native_caffe)
        probs_pt = im_classification(im, pt_model)

        diff = compare_prob(probs_caffe, probs_pt)
        diff_max = max(diff_max, diff)
        diff_sum += diff
    print("Dataset: {}, max diff: {}, avg diff: {}"
            .format(dataset_path, diff_max, diff_sum/float(num_imgs)))

def im_classification(im, model, is_native_caffe=False):
    blob = resize_for_od(im, target_size=224, maintain_ratio=False).transpose(2, 0, 1)
    data = torch.from_numpy(blob).unsqueeze(0)

    with torch.no_grad():
        if is_native_caffe:
            model.blobs['data'].reshape(1, *blob.shape)
            model.blobs['data'].data[...] = blob.reshape(1, *blob.shape)
            model.forward()
            outlayer = 'fc' if 'fc' in model.blobs else 'ip'
            out = model.blobs[outlayer].data[0]
            out = out.reshape(-1, out.shape[-1])
            out = torch.from_numpy(out)
        else:
            out = model(data)
            out = out.reshape(-1, out.shape[-1])

        # the out is from fc layer
        probs = F.softmax(out, dim=1)
        probs = probs.cpu().numpy()
    return probs

def compare_prob(arr1, arr2):
    arr1 = [(i, num) for i, num in enumerate(np.squeeze(arr1))]
    arr2 = [(i, num) for i, num in enumerate(np.squeeze(arr2))]
    assert len(arr1) == len(arr2)
    arr1 = sorted(arr1, key=lambda t: -t[1])
    arr2 = sorted(arr2, key=lambda t: -t[1])
    idx = -1
    for t1, t2 in zip(arr1, arr2):
        idx += 1
        if t1[0] != t2[0]:
            # logging.info("diff at index: {}, scale: {}".format(idx, t1[1]-t2[1]))
            return abs(t1[1]-t2[1])
    # logging.info("no diff")
    return 0

def convert_pt2caffe(proto_file, pt_model_path, out_caffemodel, mapping_file=None):
    """ Given Pytorch model file and prototxt, converts to Caffe model, and
    verify the outputs of two models
    """
    caffe_model = CaffeNet(proto_file, phase='TEST')
    pt_model = load_pt_model(pt_model_path)

    pt_dict = pt_model.state_dict()
    caffe_dict = caffe_model.state_dict()

    # If fail, use prep_dict_resnet_pt2caffe instead
    new_caffe_dict = prep_dict_resnet_pt2caffe_ordered(pt_dict, caffe_dict, mapping_file)
    # convert
    caffe_model.load_state_dict(new_caffe_dict)
    caffe_model.save_weights(out_caffemodel)

    # verify
    caffe_model = caffe_model.eval()
    caffe_model.load_weights(out_caffemodel, ignore_shape_mismatch=False)

    pt_model.eval()
    dataset_path = "data/MIT1K-GUID/test.tsv"

    logging.info("CaffeNet V.S. Pytorch")
    compare_model(caffe_model, False, pt_model, dataset_path)

    # NOTE: native Caffe is incompatible with global_pooling=True when
    # kernel_size exists in param
    # logging.info("Native Caffe V.S. Pytorch")
    # cnet = caffe.Net(proto_file, out_caffemodel, caffe.TEST)
    # compare_model(cnet, True, pt_model, dataset_path)


if __name__ == "__main__":
    # change info below if needed
    proto_file = "./resnet18/resnet18.prototxt"
    pt_model_path = "./resnet18/model_best.pth.tar"
    out_caffemodel = "./resnet18/model_best.caffemodel"
    mapping_file = "./resnet18/caffe2pt.mapping"

    init_logging()
    convert_pt2caffe(proto_file, pt_model_path, out_caffemodel, mapping_file)

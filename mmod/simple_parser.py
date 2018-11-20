from __future__ import print_function
from collections import OrderedDict
import numpy as np

from mmod.utils import open_file, is_number


def load_labelmap_list(filename):
    labelmap = []
    with open(filename) as fin:
        labelmap += [unicode(line.rstrip()) for line in fin]
    return labelmap


def parse_key_value(path, key):
    """Parse a prototxt file for a value given to a key
    :param path: Prototxt file path
    :param key: the key to search for, return the first
    :rtype: str
    """
    with open(path, "r") as f_in:
        for line in f_in.readlines():
            line = line.rstrip('\n').strip()
            line_lower = line.lower()
            if ":" not in line:
                continue
            colon_idx = line.index(":")
            if line_lower.startswith(key) and colon_idx >= len(key):
                line = line[colon_idx + 1:].strip().replace('"', '').replace("\\", "/")
                return line


def read_model_proto(proto_file_path):
    """Read prototxt file
    :param proto_file_path: path to .prototxt
    """
    from google.protobuf import text_format
    import caffe
    # noinspection PyUnresolvedReferences
    model = caffe.proto.caffe_pb2.NetParameter()
    with open(proto_file_path) as fp:
        text_format.Parse(fp.read(), model)
    return model


def read_model(caffemodel):
    """Read caffe model file
    :param caffemodel: path to .caffemodel
    """
    import caffe
    # noinspection PyUnresolvedReferences
    model = caffe.proto.caffe_pb2.NetParameter()
    with open(caffemodel, 'rb') as fp:
        model.ParseFromString(fp.read())
    return model


def read_blob(meanmodel):
    """Read blob
    :param meanmodel: path to mean blob
    """
    import caffe
    # noinspection PyUnresolvedReferences
    mean_blob = caffe.proto.caffe_pb2.BlobProto()
    with open(meanmodel, 'rb') as fp:
        mean_blob.ParseFromString(fp.read())
    return mean_blob


def array_to_blobproto(arr, diff=None):
    """Converts a N-dimensional array to blob proto. If diff is given, also
    convert the diff. You need to make sure that arr and diff have the same
    shape, and this function does not do sanity check.
    :type arr: np.ndarray
    :type diff: np.ndarray
    """
    import caffe
    blob = caffe.proto.caffe_pb2.BlobProto()
    blob.shape.dim.extend(arr.shape)
    blob.data.extend(arr.astype(float).flat)
    if diff is not None:
        blob.diff.extend(diff.astype(float).flat)
    return blob


def softmax_tree_path(model):
    """Find the tree path in a Yolov2 model
    :param model: caffe prototxt model
    :rtype: str
    """
    n_layer = len(model.layer)
    for i in reversed(range(n_layer)):
        layer = model.layer[i]
        tree_file = layer.softmaxtree_param.tree
        if tree_file:
            return tree_file
    raise Exception('Could not find the tree file')


def tsv_data_sources(model):
    """Find TSV data sources
    :param model: caffe prototxt model
    :return: list of paths to sources, and optionally list of paths to labels for those sources
    :rtype: (list, list)
    """
    labels = []
    sources = []
    n_layer = len(model.layer)
    for i in range(n_layer):
        layer = model.layer[i]
        if layer.tsv_data_param:
            if layer.tsv_data_param.source:
                sources += layer.tsv_data_param.source
            if layer.tsv_data_param.source_label:
                labels += layer.tsv_data_param.source_label
    return sources, labels or None


def lift_hier(parents):
    """lift each node to include the hiearchy
    :param parents: parent of each node
    """
    def _hier(n):
        yield n
        p = parents[n]
        while p >= 0:
            yield p
            p = parents[p]
    return _hier


def read_softmax_tree(tree_file):
    """Simple parsing of softmax tree with subgroups
    :param tree_file: path to the tree file, or open file object
    :type tree_file: str or file
    """
    group_offsets = []
    group_sizes = []
    cid_groups = []
    parents = []
    child = []  # child group
    child_sizes = []  # number of child groups
    root_size = 0  # number of child sub-groups at root
    last_p = -1
    last_sg = -1
    groups = 0
    sub_groups = 0
    size = 0
    n = 0
    with open(tree_file, 'r') if isinstance(tree_file, basestring) else open_file(tree_file) as f:
        for line in f.readlines():
            tokens = [t for t in line.split(' ') if t]
            assert len(tokens) == 2 or len(tokens) == 3, "invalid tree: {} node: {} line: {}".format(
                tree_file, n, line)
            p = int(tokens[1])
            assert n > p >= -1, "invalid parent: {} node: {} tree: {}".format(p, n, tree_file)
            parents.append(p)
            sg = -1
            if len(tokens) == 3:
                sg = int(tokens[2])
            new_group = new_sub_group = False
            if p != last_p:
                last_p = p
                last_sg = sg
                new_group = True
                sub_groups = 0
            elif sg != last_sg:
                assert sg > last_sg, "invalid sg: {} node: {} tree: {}".format(sg, n, tree_file)
                last_sg = sg
                new_sub_group = True
                sub_groups += 1
            if new_group or new_sub_group:
                group_sizes.append(size)
                group_offsets.append(n - size)
                groups += 1
                size = 0
            child.append(-1)
            child_sizes.append(0)
            if p >= 0:
                if new_group:
                    assert child[p] == -1, "node: {} parent discontinuity in tree: {}".format(n, tree_file)
                    child[p] = groups  # start group of child subgroup
                elif new_sub_group:
                    child_sizes[p] = sub_groups
            else:
                root_size = sub_groups
            n += 1
            size += 1
            cid_groups.append(groups)
    group_sizes.append(size)
    group_offsets.append(n - size)

    assert len(cid_groups) == len(parents) == len(child) == len(child_sizes)
    assert len(group_offsets) == len(group_sizes) == max(cid_groups) + 1
    return group_offsets, group_sizes, cid_groups, parents, child, child_sizes, root_size


def _line_type(_line):
    if _line.find(':') >= 0:
        return 0
    elif _line.find('{') >= 0:
        return 1
    return -1


def _parse_block(fp):
    block = OrderedDict()
    _line = fp.readline().strip()
    while _line != '}':
        ltype = _line_type(_line)
        if ltype == 0:  # key: value
            # print _line
            _line = _line.split('#')[0]
            key, value = _line.split(':')
            key = key.strip()
            value = value.strip().strip('"')
        elif ltype == 1:  # blockname {
            key = _line.split('{')[0].strip()
            value = _parse_block(fp)
        else:
            raise NotImplementedError(ltype)
        if key in block:
            if isinstance(block[key], list):
                block[key].append(value)
            else:
                block[key] = [block[key], value]
        else:
            block[key] = value
        _line = fp.readline().strip()
        _line = _line.split('#')[0]
    return block


def parse_prototxt(protofile):
    """Parse prototxt as dictionary
    :type protofile: str
    :rtype: dict
    """
    with open(protofile, 'r') as fp:
        props = OrderedDict()
        layers = []
        line = fp.readline()
        while line:
            line = line.strip().split('#')[0]
            if not line:
                line = fp.readline()
                continue
            ltype = _line_type(line)
            if ltype == 0:  # key: value
                key, value = line.split(':')
                key = key.strip()
                value = value.strip().strip('"')
                if key in props:
                    if type(props[key]) == list:
                        props[key].append(value)
                    else:
                        props[key] = [props[key], value]
                else:
                    props[key] = value
            elif ltype == 1:  # blockname {
                key = line.split('{')[0].strip()
                if key == 'layer':
                    layer = _parse_block(fp)
                    layers.append(layer)
                else:
                    props[key] = _parse_block(fp)
            line = fp.readline()

    if len(layers) > 0:
        net_info = OrderedDict()
        net_info['props'] = props
        net_info['layers'] = layers
        return net_info
    return props


def format_value(value):
    """See if escaping is needed
    """
    if is_number(value):
        return value
    if value in [
        'true', 'false', 'MAX', 'SUM', 'AVE', 'TRAIN', 'TEST', 'WARP', 'LINEAR', 'AREA', 'NEAREST',
        'CUBIC', 'LANCZOS4', 'CENTER', 'LMDB', 'BATCH_SIZE', 'VALID'
    ]:
        return value
    return '\"%s\"' % value


def print_prototxt(net_info):
    """Print parsed prototxt
    """
    def print_block(block_info, prefix, indent):
        blanks = ''.join([' '] * indent)
        print('%s%s {' % (blanks, prefix))
        for key, value in block_info.items():
            if type(value) == OrderedDict:
                print_block(value, key, indent + 4)
            elif type(value) == list:
                for v in value:
                    print('%s    %s: %s' % (blanks, key, format_value(v)))
            else:
                print('%s    %s: %s' % (blanks, key, format_value(value)))
        print('%s}' % blanks)

    props = net_info['props']
    layers = net_info['layers']
    print('name: \"%s\"' % props['name'])
    if 'input' in props:
        print('input: \"%s\"' % props['input'])
    if 'input_dim' in props:
        print('input_dim: %s' % props['input_dim'][0])
        print('input_dim: %s' % props['input_dim'][1])
        print('input_dim: %s' % props['input_dim'][2])
        print('input_dim: %s' % props['input_dim'][3])
    print('')
    for layer in layers:
        print_block(layer, 'layer', 0)


def save_prototxt(net_info, protofile, region=True):
    """Save parsed prototxt to a file
    """
    with open(protofile, 'w') as fp:
        def print_block(block_info, prefix, indent):
            blanks = ''.join([' '] * indent)
            print('%s%s {' % (blanks, prefix), file=fp)
            for key, value in block_info.items():
                if isinstance(value, OrderedDict):
                    print_block(value, key, indent + 4)
                elif isinstance(value, list):
                    for _v in value:
                        print('%s    %s: %s' % (blanks, key, format_value(_v)), file=fp)
                else:
                    print('%s    %s: %s' % (blanks, key, format_value(value)), file=fp)
            print('%s}' % blanks, file=fp)

        if 'props' in net_info:
            props = net_info['props']
            print('name: \"%s\"' % props['name'], file=fp)
            if 'input' in props:
                print('input: \"%s\"' % props['input'], file=fp)
            if 'input_dim' in props:
                print('input_dim: %s' % props['input_dim'][0], file=fp)
                print('input_dim: %s' % props['input_dim'][1], file=fp)
                print('input_dim: %s' % props['input_dim'][2], file=fp)
                print('input_dim: %s' % props['input_dim'][3], file=fp)
            print('', file=fp)

        if 'layers' in net_info:
            layers = net_info['layers']
            for layer in layers:
                if layer['type'] != 'Region' or region:
                    print_block(layer, 'layer', 0)

        for k, vals in net_info.iteritems():
            if k in ['layers', 'props']:
                continue
            if isinstance(vals, list):
                for v in vals:
                    print('{}: {}'.format(k, format_value(v)), file=fp)
                continue
            print('{}: {}'.format(k, format_value(vals)), file=fp)

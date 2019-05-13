from collections import OrderedDict
import torch


def prep_dict_ignore_mismatch(init_net_dict, model_state_dict):
    """ Adopted from  https://github.com/leizhangcn/maskrcnn-benchmark/blob/c765ab45dedc9f38de96fad23ba013619e4cde0a/maskrcnn_benchmark/utils/model_serialization.py#L10
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    :param init_net_dict: dictionary to take weights from
    :param model_state_dict: dictionary to take layers names from
    :return model_state_dict: updated dict
    :return log_str: string that summarizes
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(init_net_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )

    max_match_size, idxs = match_matrix.max(1)

    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    log_str = ""

    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        if model_state_dict[key].shape != init_net_dict[key_old].shape:
            # if layer weights does not match in size, skip this layer
            log_str += "skipping layer {} because of size mis-match".format(key)
            continue
        model_state_dict[key] = init_net_dict[key_old]
        log_str += log_str_template.format(key, max_size, key_old, max_size_loaded,
                                           tuple(init_net_dict[key_old].shape))
        log_str += '\n'

    return model_state_dict, log_str


def prep_dict(init_net_dict, model_state_dict, switch_bn2scale=True):
    """fixes the names of layers in initializing state dictionary 
    to fit the new network state dictionary
    :param init_net_dict: dictionary to take weights from
    :param model_state_dict: dictionary to take layers names from
    :param switch_bn2scale: boolean, 
           if to switch to BatchNormalization to Scale when searching for a weight
    """
    layer_names = list(model_state_dict.keys())
    init_weights = []
    init_layer_names = list(init_net_dict.keys())
    if "module" in init_layer_names[0]:
        add_token = "module."
    elif init_layer_names[0].startswith("backbone"):
        add_token = "backbone."
    else:
        add_token = ""
    for layer_name in layer_names:
        token = _set_token(layer_name)
        init_layer_name = add_token + token + layer_name.split(token)[-1]
        if switch_bn2scale and "bn." in init_layer_name and ('weight' in init_layer_name
                                                             or 'bias' in init_layer_name):
            init_layer_name = init_layer_name.replace("bn", "scale")
        try:
            init_weight = init_net_dict[init_layer_name]
        except KeyError:
            if "extra_conv" in init_layer_name and "/conv" in init_layer_name:
                init_layer_name = init_layer_name.replace('/conv', '')
                try:
                    init_weight = init_net_dict[init_layer_name]
                except KeyError:
                    raise ValueError("Init dictionary misses initialization for {}".format(layer_name))
            else:
                raise ValueError("Init dictionary misses initialization for {}".format(layer_name))

        init_weights.append((layer_name, init_weight))
    return OrderedDict(init_weights)


def _set_token(layer_name):
    """helper to derive a splitting token for layer name"""
    if 'last_conv' in layer_name:
        return 'last_conv'
    if 'extra_conv' in layer_name:
        return 'extra_conv'
    if 'dark' in layer_name:
        return 'dark'


def prep_dict_pt2caffe(init_net_dict, net_dict):
    """fixes the names of layers in initializing state dictionary 
    to fit the new network state dictionary (good for debugging)
    :param init_net_dict: dictionary to take weights from
    :param net_dict: dictionary to take layers names from 
    if to switch to BatchNormalization to Scale when searching for a weight 
    """
    layer_names = net_dict.keys()
    init_weights = []
    for layer_name in layer_names:
        for init_layer_name, init_weight in init_net_dict['state_dict'].items():
            if layer_name in init_layer_name:
                init_weights.append((layer_name, init_weight))
                print("{} to {}".format(init_layer_name, layer_name))
                break
            if "extra_conv19" in init_layer_name and "/conv.weight" in init_layer_name \
                    and layer_name == "extra_conv19.weight":
                print("{} to {}".format(init_layer_name, layer_name))
                init_weights.append((layer_name, init_net_dict['state_dict'][init_layer_name]))
                break
            if "extra_conv20" in init_layer_name and "/conv.weight" in init_layer_name \
                    and layer_name == "extra_conv20.weight":
                print("{} to {}".format(init_layer_name, layer_name))
                init_weights.append((layer_name, init_net_dict['state_dict'][init_layer_name]))
                break
            if "extra_conv21" in init_layer_name and "/conv.weight" in init_layer_name \
                    and layer_name == "extra_conv21.weight":
                print("{} to {}".format(init_layer_name, layer_name))
                init_weights.append((layer_name, init_net_dict['state_dict'][init_layer_name]))
                break

    init_weights.append(("seen_images", init_net_dict['region_target.seen_images']))
    init_weights.append(("forward_net_only", torch.tensor(1, dtype=torch.uint8)))
    return OrderedDict(init_weights)

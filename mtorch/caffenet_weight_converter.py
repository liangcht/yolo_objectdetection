from collections import OrderedDict
import torch
def prep_dict(init_net_dict, net_dict, switch_bn2scale=True):
    """fixes the names of layers in initializing state dictionary 
    to fit the new network state dictionary
    :param init_net_dict: dictionary to take weights from
    :param net_dict: dictionary to take layers names from 
    :param switch_bn2scale: boolean, 
    if to switch to BatchNormalization to Scale when searching for a weight 
    """
    layer_names = net_dict.keys()
    init_weights = []
    add_token = "module."
    for layer_name in layer_names:            
        token = _set_token(layer_name)
        init_layer_name = add_token + token + layer_name.split(token)[-1]
       
        if switch_bn2scale and "bn." in init_layer_name and ('weight' in init_layer_name 
                                                            or 'scale' in init_layer_name):
            init_layer_name = init_layer_name.replace("bn", "scale")
        try:
            init_weight = init_net_dict[init_layer_name] 
        except KeyError:
            if "extra_conv" in init_layer_name and "/conv" in init_layer_name:
                init_layer_name = init_layer_name.replace('/conv','') 
                try:
                    init_weight = init_net_dict[init_layer_name] 
                except:
                    raise ValueError("Init dictionary misses initialization for {}". format(layer_name))
            else:
                raise ValueError("Init dictionary misses initialization for {}". format(layer_name))

        init_weights.append((layer_name, init_weight))
    return OrderedDict(init_weights)


def _set_token(layer_name):
    """helper to derive a splitting token for layer name"""
    if 'last_conv' in layer_name:
        return 'last_conv'
    if 'extra_conv' in layer_name:
        return 'extra_conv'
    if 'dark' in layer_name:
        return'dark'

def prep_dict_pt2caffe(init_net_dict, net_dict, switch_bn2scale=True):
    """fixes the names of layers in initializing state dictionary 
    to fit the new network state dictionary
    :param init_net_dict: dictionary to take weights from
    :param net_dict: dictionary to take layers names from 
    :param switch_bn2scale: boolean, 
    if to switch to BatchNormalization to Scale when searching for a weight 
    """
    layer_names = net_dict.keys()
    init_weights = []
    add_token = "module."
    for layer_name in layer_names:            
        for init_layer_name, init_weight in init_net_dict.items():
            if layer_name in init_layer_name:
               init_weights.append((layer_name, init_weight))
               print("{} to {}".format(init_layer_name, layer_name))
               break
            if "extra_conv" in init_layer_name and "/conv.weight" in init_layer_name \
            and "extra_conv" in layer_name and "weight" in layer_name:
                print("{} to {}".format(init_layer_name, layer_name))
                init_weights.append((layer_name, init_net_dict[init_layer_name])) 
                break
    init_weights.append(("region_target.biases", torch.tensor([ 1.0800,  1.1900,  3.4200,  4.4100,  6.6300, 11.3800,  9.4200,  5.1100,
        16.6200, 10.5200])))
    init_weights.append(("region_target.seen_images", torch.tensor(15000)))

    init_weights.append(("seen_images", torch.tensor(15000)))
    init_weights.append(("forward_net_only", torch.tensor(1, dtype=torch.uint8)))

    return OrderedDict(init_weights)

    

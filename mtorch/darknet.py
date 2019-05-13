import torch
import torch.nn as nn
from collections import OrderedDict

from mtorch.common_network_blocks import bn_relu_block, conv_bn_relu_block, conv_bn_relu_maxpool_block
from mtorch.caffenet_weight_converter import prep_dict
from mtorch.weights_init import msra_init

__all__ = ['DarknetLayers', 'darknet_layers']

BBOX_DIM = 4 
OBJECTNESS_DIM = 1


def _conv1x1(in_planes, out_planes, stride=1):
    """helper captures 1x1 convolution without padding
    :param in_planes: int, number of input channels
    :param out_planes: int, number of output channels
    :param stride: int, specifies the stride of convolution
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


def conv_1_bn_relu_block(in_planes, out_planes, block_name):
    """captures a block of 1x1 Conv2D followed by batch normalization and leaky ReLU
    :param in_planes: int, number of input channels
    :param out_planes: int, number of output channels
    :param block_name: str, name to attach to the layer name (for initialization from caffemodel and Caffe prototxt)
    :return: sequential model
    """
    return nn.Sequential(nn.Sequential(OrderedDict([
        (block_name + '/conv', _conv1x1(in_planes, out_planes))])),
        bn_relu_block(out_planes, block_name)
    )


class DarknetLayers(nn.Module):
    """captures Darknet layers featurizer
    Parameters:
        in_planes: int, channel dimension of the data
        x: Torch tensor that contains the data
    """

    def __init__(self, in_planes=3):
        """constructor of the DarknetLayer featurizer
        :param in_planes: int, number of channels
        """
        super(DarknetLayers, self).__init__()
        self.net = nn.Sequential(
            conv_bn_relu_maxpool_block(in_planes, 32, 'dark1'),
            conv_bn_relu_maxpool_block(32, 64, 'dark2'),
            conv_bn_relu_block(64, 128, 'dark3a'),
            conv_1_bn_relu_block(128, 64, 'dark3b_1'),
            conv_bn_relu_maxpool_block(64, 128, 'dark3c'),
            conv_bn_relu_block(128, 256, 'dark4a'),
            conv_1_bn_relu_block(256, 128, 'dark4b_1'),
            conv_bn_relu_maxpool_block(128, 256, 'dark4c'),
            conv_bn_relu_block(256, 512, 'dark5a'),
            conv_1_bn_relu_block(512, 256, 'dark5b_1'),
            conv_bn_relu_block(256, 512, 'dark5c'),
            conv_1_bn_relu_block(512, 256, 'dark5d_1'),
            conv_bn_relu_maxpool_block(256, 512, 'dark5e'),
            conv_bn_relu_block(512, 1024, 'dark6a'),
            conv_1_bn_relu_block(1024, 512, 'dark6b_1'),
            conv_bn_relu_block(512, 1024, 'dark6c'),
            conv_1_bn_relu_block(1024, 512, 'dark6d_1'),
            conv_bn_relu_block(512, 1024, 'dark6e')
        )
        msra_init(self.net)
    
    def forward(self, x):
        return self.net(x)

    @property
    def out_planes_dim(self):
        return 1024


def darknet_layers(weights_file=None, caffe_format_weights=False, map_location=None, **kwargs):
    """wrapper to create featurizer network with DarknetLayers
    :param weights_file:  weights to initialize the network
    :param caffe_format_weights: boolean, if the provided weights come from Caffe format
    :param map_location: 
    :param kwargs: any parameters to the network
    :return: nn.Sequential with Darknet layers
    """
    model = DarknetLayers(**kwargs)
    pretrained = weights_file is not None

    if pretrained:
        snapshot = torch.load(weights_file, map_location=map_location)
        orig_dict = snapshot["state_dict"] 

        if caffe_format_weights:
            init_dict = prep_dict(orig_dict, model.state_dict()) 
        else:
            init_dict = orig_dict
        model.load_state_dict(init_dict)
        try:
            model.seen_images = orig_dict["module.seen_images"]
        except KeyError:
            try:
                model.seen_images = orig_dict["seen_images"]
            except KeyError:
                model.seen_images = snapshot["seen_images"]

    return model



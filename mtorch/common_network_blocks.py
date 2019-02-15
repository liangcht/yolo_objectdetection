import torch.nn as nn
from mtorch.caffetorch import Scale
from collections import OrderedDict

__all__ = ['bn_relu_block', 'conv_bn_relu_maxpool_block', 'conv_bn_relu_block']


def _conv3x3(in_planes, out_planes, stride=1):
    """helper: captures 3x3 convolution with padding
    :param in_planes: int, number of input channels
    :param out_planes: int, number of output channels
    :param stride: int, specifies the stride of convolution
    :return: Convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def bn_relu_block(in_planes, block_name, momentum=0.1, negative_slope=0.1, use_scale=True):
    """batch normalization followed by leaky ReLU
    :param in_planes: int, number of input channels
    :param block_name: str, name to attach to the layer name (for initialization from caffemodel and Caffe prototxt)
    :param momentum: float, momentum of the BatchNorm
    :param negative_slope: slope of the leaky ReLu
    :param use_scale: boolean, if to use Scale instead of Affine mode of BatchNormalization
    :return: sequential model
    """
    if use_scale:
        return nn.Sequential(OrderedDict([
            (block_name + '/bn', nn.BatchNorm2d(in_planes, momentum=momentum, affine=False)),
            (block_name + '/scale', Scale(in_planes)),
            (block_name + '/relu', nn.LeakyReLU(negative_slope=negative_slope))
        ])
        )
    else:
        return nn.Sequential(OrderedDict([
            (block_name + '/bn', nn.BatchNorm2d(in_planes, momentum=momentum, affine=True)),
            (block_name + '/relu', nn.LeakyReLU(negative_slope=negative_slope))
        ])
        )
        

def conv_bn_relu_block(in_planes, out_planes, block_name):
    """captures a block of convolutional layer followed by batch normalization and leaky ReLU
    :param in_planes: int, number of input channels
    :param out_planes: int, number of output channels
    :param block_name: str, name to attach to the layer name (for initialization from caffemodel and Caffe prototxt)
    :return: sequential model
    """
    return nn.Sequential(nn.Sequential(OrderedDict([
        (block_name + '/conv', _conv3x3(in_planes, out_planes, stride=1))])),
        bn_relu_block(out_planes, block_name)
    )


def conv_bn_relu_maxpool_block(in_planes, out_planes, block_name):
    """captures a block of convolutional layer followed by batch normalization, leaky ReLU and max pooling
    :param in_planes: int, number of input channels
    :param out_planes: int, number of output channels
    :param block_name: str, name to attach to the layer name (for initialization from caffemodel and Caffe prototxt)
    :return: sequential model
    """
    return nn.Sequential(
        conv_bn_relu_block(in_planes, out_planes, block_name),
        nn.Sequential(OrderedDict([
            (block_name + '/maxpool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
        ]))
    )

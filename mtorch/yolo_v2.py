import torch
import torch.nn as nn
from collections import OrderedDict

from common_network_blocks import conv_bn_relu_block
from weights_init import msra_init
from caffenet_weight_converter import prep_dict

__all__ = ['Yolo', 'yolo']

BBOX_DIM = 4 
OBJECTNESS_DIM = 1


def _extra_conv_layers(in_planes, num_extra_convs):
    """ helper: creates additional
    :param in_planes: int, number of input channels
    :param num_extra_convs: number of extra blocks of Conv2D
        + BtachNorm + LeakyRelu to add to the featurizer
    :return: list, with all the additional layers
    """
    additional_conv_layers = []
    for i in range(num_extra_convs):
        additional_conv_layers.append(conv_bn_relu_block(in_planes, in_planes,
                                                         'extra_conv' + str(19 + i)))
    return additional_conv_layers


class Yolo(nn.Module):
    """captures single-scale Yolo object detection network (aka yolo v2)

    Parameters:
        backbone_model: nn.Module or nn.Sequential, featurizer (typically Darknet)
        num_classes: int, number of classes to detect
        num_extra_convs: int, number of extra blocks of Conv2D
                          + BtachNorm + LeakyRelu to add to the featurizer
        num_anchors: int, number of anchors to use for
        x: Torch tensor that contains the data
    """
    def __init__(self, backbone_model, num_classes=20, num_extra_convs=3, num_anchors=5,
                 backbone_out_planes=None):
        """constructor of YOLO
        :param backbone_model: nn.Module or nn.Sequential, featurizer (typically Darknet)
        :param num_classes: int, number of classes to detect
        :param num_extra_convs: number of extra blocks of Conv2D
                            + BtachNorm + LeakyRelu to add to the featurizer
        :param num_anchors: int, number of anchors to use for bounding box estimation
        :param backbone_out_planes: int, the dimension of the backbone channels

        """
        super(Yolo, self).__init__()
        self._seen_images = 0
        self.backbone = backbone_model

        self.extra_convs = nn.Sequential(*_extra_conv_layers(self._in_planes(backbone_out_planes),
                                                             num_extra_convs))
        msra_init(self.extra_convs)
        regressor_dim = num_anchors * (num_classes + BBOX_DIM + OBJECTNESS_DIM)
        self.regressor = nn.Sequential(OrderedDict([
                    ("last_conv", nn.Conv2d(self._in_planes(backbone_out_planes), regressor_dim, 
                                            kernel_size=1, padding=0, bias=True))
        ]))
        msra_init(self.regressor)

    def forward(self, x):
        return self.regressor(self.extra_convs(self.backbone(x)))
    
    @property
    def seen_images(self):
        """getter for seen_images"""
        return self._seen_images

    @seen_images.setter
    def seen_images(self, val):
        """setter for seen_images"""
        self._seen_images = val

    def _in_planes(self, backbone_out_planes):
        """helper to figure out dimension of the backbone output""" 
        try:
            return self.backbone.out_planes_dim
        except AttributeError:
            if backbone_out_planes is not None:
                return backbone_out_planes
            raise ValueError("Please provide backbone_out_planes argument")        


def yolo(backbone_model, weights_file=None, caffe_format_weights=False, map_location=None, **kwargs):
    """wrapper to create Yolo network
    :param backbone_model: nn.Module or nn.Sequential, featurizer (typically Darknet)
    :param weights_file:  weights to initialize the network
    :param caffe_format_weights: boolean, if the provided weights come from Caffe format
    :param map_location: 
    :param kwargs: any parameters to the network
    :return: model based on Yolo architecture
    """
    model = Yolo(backbone_model, *kwargs)
    pretrained = weights_file is not None

    if pretrained:
        
        snapshot = torch.load(weights_file, map_location=map_location)
        orig_dict = snapshot["state_dict"] 
        #orig_dict = snapshot 
        if caffe_format_weights:
            init_dict = prep_dict(orig_dict, model.state_dict()) 
        else:
            init_dict = orig_dict
        model.load_state_dict(init_dict)
        try:
            model.seen_images = orig_dict["module.seen_images"]
        except:
            model.seen_images = orig_dict["seen_images"]
        
    return model


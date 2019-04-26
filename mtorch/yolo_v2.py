import torch
import torch.nn as nn
from collections import OrderedDict

from mtorch.common_network_blocks import conv_bn_relu_block
from mtorch.weights_init import msra_init
from mtorch.caffenet_weight_converter import prep_dict, prep_dict_ignore_mismatch
from mtorch.custom_layers_ops import freeze_modules_for_training

__all__ = ['Yolo', 'yolo', 'yolo_0extraconv', 'yolo_1extraconv', 'yolo_2extraconv', 'yolo_3extraconv']

BBOX_DIM = 4 
OBJECTNESS_DIM = 1
USE_SCALE = True


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
        self.pretrained_info = None
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

    def freeze_backbone(self, freeze_till=None):
        """entirely freezes the backbone for training:
        :param freeze_till, the layer that stops freezing
        :return boolean, if the layer that stops freezing was found
        """
        return freeze_modules_for_training(self.backbone, freeze_till)

    def freeze_extra_convs(self, freeze_till=None):
        """entirely freezes the backbone for training:
            :param freeze_till, the layer that stops freezing
            :return boolean, if the layer that stops freezing was found
        """
        return freeze_modules_for_training(self.extra_convs, freeze_till)

    def freeze_features(self):
        self.freeze_backbone()
        self.freeze_extra_convs()

    def freeze(self, freeze_till):
        stopped_freeze_in_backbone = self.freeze_backbone(freeze_till)
        if not stopped_freeze_in_backbone:
            stopped_freeze_in_extraconv = self.freeze_extra_convs(freeze_till)
            if not stopped_freeze_in_extraconv:
                raise ValueError("{} is not found in the {}".format(freeze_till, self))


def _get_init_dict(orig_init_dict, model_dict, caffe_format_weights=False, ignore_mismatch=False):
    """helper to adjust if necessary the initializing model to this model"""
    init_info = None
    if ignore_mismatch:
        init_dict, init_info = prep_dict_ignore_mismatch(orig_init_dict, model_dict)
    elif caffe_format_weights:  # TODO - test if possible to replace that with ignore_mismatch
        init_dict = prep_dict(orig_init_dict, model_dict, switch_bn2scale=USE_SCALE)
        init_info = "fully matched caffenet model"
    else:
        init_dict = orig_init_dict
        init_info = "fully matched py model"

    return init_dict, init_info


def yolo(backbone_model, weights_file=None, ignore_mismatch=False,
         caffe_format_weights=False, map_location=None, **kwargs):
    """wrapper to create Yolo network
    :param backbone_model: nn.Module or nn.Sequential, featurizer (typically Darknet)
    :param weights_file:  weights to initialize the network
    :param ignore_mismatch: boolean, ignores any mismatch in layers order/names of layers in initialization model fit,
           be careful when using it, false by default
    :param caffe_format_weights: boolean, if the provided weights come from Caffe format, false by default
    :param map_location: 
    :param kwargs: any parameters to the network (num_classes, num_extra_convs etc.)
    :return: model based on Yolo architecture
    """
    model = Yolo(backbone_model, **kwargs)
    pretrained = weights_file is not None
    if pretrained:
        snapshot = torch.load(weights_file, map_location=map_location)
        orig_dict = snapshot["state_dict"]
        model_dict = model.state_dict()
        init_dict, init_info = _get_init_dict(orig_dict, model_dict, caffe_format_weights, ignore_mismatch )

        model.load_state_dict(init_dict, strict=not ignore_mismatch)
        model.pretrained_info = init_info
        # noinspection PyBroadException
        try:
            model.seen_images = orig_dict["module.seen_images"]
        except Exception:
            # noinspection PyBroadException
            try:
                model.seen_images = orig_dict["seen_images"]
            except Exception:
                model.seen_images = snapshot["seen_images"]
        
    return model


def yolo_0extraconv(*args, **kwargs):
    return yolo(*args, num_extra_convs=0, **kwargs)


def yolo_1extraconv(*args, **kwargs):
    return yolo(*args, num_extra_convs=1, **kwargs)


def yolo_2extraconv(*args, **kwargs):
    return yolo(*args, num_extra_convs=2, **kwargs)


def yolo_3extraconv(*args, **kwargs):
    return yolo(*args, num_extra_convs=3, **kwargs)


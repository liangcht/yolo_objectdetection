import torch
import torch.nn as nn
from mtorch.caffetorch import Slice, SoftmaxWithLoss, EuclideanLoss
from mtorch.reshape import Reshape
from mtorch.region_target import RegionTarget
from mtorch.softmaxtree_loss import SoftmaxTreeWithLoss
from mtorch.softmaxtree import SoftmaxTree

DEFAULT_BIASES = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
SLICE_POINTS = [10, 20, 25]
OBJECTNESS_DIM = 1
DEFAULT_NUM_CLASSES = 20  # for voc20
DEFAULT_NUM_ANCHORS = 5
BBOX_DIM = 4
# Normalization types:
VALID = 'valid'
BATCH = 'batch'
NORM_SOFTMAX = VALID
NORM_SOFTMAXTREE = VALID
# Scales for different losses:
OBJ_SC = 5.0  # objectness loss
NO_OBJ_SC = 1.0  # no objectness loss
XY_SC = 1.0  # xy loss
WH_SC = 1.0  # wh loss
CL_SC_SOFTMAX = 1.0  # classification loss
CL_SC_SOFTMAXTREE = 1.0  # classification loss 
# Other parameters
OBJ_ESC_THRESH = 0.6
NUM_IMAGES_WARM = 12800

__all__ = ['RegionTargetWithSoftMaxLoss', 'RegionTargetWithSoftMaxTreeLoss', 'NUM_IMAGES_WARM']


class RegionTargetLoss(nn.Module):
    """Abstract class for constructing different kinds of RegionTargetLosses
    Parameters:
        num_classes: int, number of classes for classification
        num_anchors: int, number of anchors for bounding box predictions
        biases: list, default anchors
        coords: int, number of coordinates in bounding box to predict
        obj_esc_thresh: int, objectness threshold
        rescore: boolean,
        xy_scale: float, weight of the xy loss
        wh_scale: float, weight of the wh loss
        object_scale: float, weight of the objectness loss
        noobject_scale: float, weight of the no-objectness loss
        coord_scale:float,
        anchor_aligned_images: int, threshold to pass to warm stage
        ngpu: int, number of gpus
        seen_images: int, number of images seen by the model
    """

    def __init__(self, num_classes=DEFAULT_NUM_CLASSES, num_anchors=DEFAULT_NUM_ANCHORS,
                 biases=DEFAULT_BIASES, coords=BBOX_DIM,
                 obj_esc_thresh=OBJ_ESC_THRESH, rescore=True, xy_scale=XY_SC, wh_scale=WH_SC,
                 object_scale=OBJ_SC, noobject_scale=NO_OBJ_SC, coord_scale=1.0,
                 anchor_aligned_images=NUM_IMAGES_WARM, ngpu=1, seen_images=0):
        super(RegionTargetLoss, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        slice_points = [point for point in SLICE_POINTS]
        slice_points.append((num_classes + coords + OBJECTNESS_DIM) * num_anchors)
        self.slice_region = Slice(1, slice_points)
        self.region_target = RegionTarget(
            biases, rescore=rescore,
            anchor_aligned_images=anchor_aligned_images,
            coord_scale=coord_scale,
            positive_thresh=obj_esc_thresh,
            gpus_size=ngpu,
            seen_images=seen_images
        )
        self.xy_loss = EuclideanLoss(loss_weight=xy_scale)
        self.wh_loss = EuclideanLoss(loss_weight=wh_scale)
        self.o_obj_loss = EuclideanLoss(loss_weight=object_scale)
        self.o_noobj_loss = EuclideanLoss(loss_weight=noobject_scale)
        reshape_axis = 1
        reshape_num_axes = 1
        self.reshape_conf = Reshape(self.shape, reshape_axis, reshape_num_axes)

    def forward(self, x, label):
        """
        :param x: torch tensor, the input to the loss layer
        :param label: [N x (coords+1)], expected format: x,y,w,h,cls
        :return: float32 loss scalar 
        """
        xy, wh, obj, conf = self.slice_region(x)
        sig = nn.Sigmoid()
        xy = sig(xy)
        obj = sig(obj)
        conf = self.reshape_conf(conf)
        t_xy, t_wh, t_xywh_weight, t_o_obj, t_o_noobj, t_label = self.region_target(xy, wh, obj, label)
        loss = (
                self.xy_loss(xy, t_xy, t_xywh_weight) +
                self.wh_loss(wh, t_wh, t_xywh_weight) +
                self.o_obj_loss(obj, t_o_obj) +
                self.o_noobj_loss(obj, t_o_noobj) +
                self.classifier_loss(conf, t_label)
        )
        return loss

    @property
    def seen_images(self):
        """getter to the number of images that were evaluated by model"""
        return self.region_target.seen_images

    def classifier_loss(self, x, label):
        """calculates classification loss (SoftMaxTreeLoss)
        after permuting the dimensions of input features (compatibility to Caffe)
        :param x: torch tensor, features
        :param label: ground truth label
        :return torch tensor with loss value
        """
        raise NotImplementedError(
            "Please create an instance of RegionTargetWithSoftMaxLoss or RegionTargetWithSoftTreeMaxLoss")

    @property
    def shape(self):
        """
        :return: list, dimensions for Reshape
        """
        raise NotImplementedError(
            "Please create an instance of RegionTargetWithSoftMaxLoss or RegionTargetWithSoftTreeMaxLoss")


class RegionTargetWithSoftMaxLoss(RegionTargetLoss):
    """Extends RegionTargetLosses by calculating classification loss based on SoftMaxLoss"""

    def __init__(self, ignore_label=-1, class_scale=CL_SC_SOFTMAX, normalization=NORM_SOFTMAX, **kwargs):
        super(RegionTargetWithSoftMaxLoss, self).__init__(**kwargs)
        self.normalization = normalization
        self._classifier_loss = SoftmaxWithLoss(loss_weight=class_scale, ignore_label=ignore_label,
                                                valid_normalization=(normalization == VALID))

    def classifier_loss(self, x, label):
        """calculates classification loss (SoftMaxTreeLoss)
        after permuting the dimensions of input features (compatibility to Caffe)
        :param x: torch tensor, features
        :param label: ground truth label
        :return torch tensor with loss value
        """
        return self._classifier_loss(x.permute([0, 2, 1, 3, 4]), label)

    @property
    def shape(self):
        """
        :return: list, dimensions for Reshape
        """
        return [self.num_anchors, self.num_classes]


class RegionTargetWithSoftMaxTreeLoss(RegionTargetLoss):
    """Extends RegionTargetLosses by calculating classification loss based on SoftMaxTreeLoss"""

    def __init__(self, tree, ignore_label=-1, class_scale=CL_SC_SOFTMAXTREE, normalization=NORM_SOFTMAXTREE, **kwargs):
        super(RegionTargetWithSoftMaxTreeLoss, self).__init__(**kwargs)
        self.normalization = normalization
        self._classifier_loss = SoftmaxTreeWithLoss(
            tree, ignore_label=ignore_label, loss_weight=class_scale,
            valid_normalization=(normalization == VALID)
        )

    def classifier_loss(self, x, label):
        """calculates classification loss (SoftMaxTreeLoss)
        :param label: ground truth label
        :return torch tensor with loss value
        """
        return self._classifier_loss(x, label)

    @property
    def shape(self):
        """
        :return: list, dimensions for Reshape
        """
        return [self.num_classes, self.num_anchors]

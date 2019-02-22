import torch
import torch.nn as nn
from mtorch.caffetorch import Slice, SoftmaxWithLoss, EuclideanLoss
from mtorch.reshape import Reshape
from mtorch.region_target import RegionTarget
from mtorch.softmaxtree_loss import SoftmaxTreeWithLoss
from mtorch.softmaxtree import SoftmaxTree
from mtorch.softmaxtree_prediction import SoftmaxTreePrediction
from mtorch.yolobbs import YoloBBs
from mtorch.nmsfilter import NMSFilter
from mtorch.yoloevalcompat import YoloEvalCompat

DEFAULT_BIASES = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
SLICE_POINTS = [10, 20, 25]
OBJECTNESS_DIM = 1


class YoloPredict(nn.Module):
    """Abstract class for constructing different kinds of RegionTargetLosses
    Parameters:
        num_classes: int, number of classes for classification
        num_anchors: int, number of anchors for bounding box predictions
        biases: list, default anchors
        coords: int, number of coordinates in bounding box to predict
        nms_threshold: float
        pre_threshold: float
    """

    def __init__(self, num_classes=20, num_anchors=5, biases=DEFAULT_BIASES, coords=4,
                 nms_threshold=0.45, pre_threshold=0.005):
        super(YoloPredict, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.class_axis = 1
        slice_points = [point for point in SLICE_POINTS]
        slice_points.append((num_classes + coords + OBJECTNESS_DIM) * num_anchors)
        self.slice_region = Slice(1, slice_points)
        reshape_axis = 1
        reshape_num_axes = 1
        self.reshape_conf = Reshape(self.shape, reshape_axis, reshape_num_axes)
        self.yolo_bboxs = YoloBBs(biases=biases, feat_stride=32)
        self.nms_filter = NMSFilter(nms_threshold=nms_threshold, pre_threshold=pre_threshold,
                                    classes=1, first_class=num_classes)
        self.yolo_evel_compat = YoloEvalCompat()

    def forward(self, x, im_info):
        """
        :param x: torch.Tensor, the input to the prediction
        :param im_info: torch.Tensor, height and width of the original image
        :return: predictions: probability and corresponding bounding boxes
        """
        xy, wh, obj, conf = self.slice_region(x)
        sig = nn.Sigmoid()
        xy = sig(xy)
        obj = sig(obj)
        conf = self.reshape_conf(conf)
        top_preds = self.top_predictions(self.class_probability(conf), obj)
        bboxs = self.yolo_bboxs(xy, wh, im_info)
        prob = self.yolo_evel_compat(self.nms_filter(bboxs, top_preds))
        return prob, bboxs

    def class_probability(self, x):
        """calculates class probabilities
        :param x: torch tensor, features
        :return torch tensor with loss value
        """
        raise NotImplementedError(
            "Please create an instance of TreePredictor")

    def top_predictions(self, class_prob, obj):
        raise NotImplementedError(
            "Please create an instance of TreePredictor")

    @property
    def shape(self):
        """
        :return: list, dimensions for Reshape
        """
        raise NotImplementedError(
            "Please create an instance of TreePredictor")


class PlainPredictor(YoloPredict):
    """Extends RegionTargetLosses by calculating classification loss based on SoftMaxLoss"""

    def __init__(self, num_classes=20, num_anchors=5, biases=DEFAULT_BIASES, coords=4,
                 pred_thresh=0.24, nms_threshold=0.45, pre_threshold=0.005):
        super(PlainPredictor, self).__init__(num_classes, num_anchors, biases, coords,
                                             nms_threshold, pre_threshold)
        self._class_prob = nn.Softmax(dim=self.class_axis)
        self._thresh = pred_thresh
       

        
    def class_probability(self, x):
        """calculates classification probability
        after permuting the dimensions of input features  (compatibility to Caffe)
        :param x: torch tensor, features
        :return torch tensor with loss value
        """
        return self._class_prob(x.permute([0, 2, 1, 3, 4]))
    
    def top_predictions(self, class_prob, obj):
        """predictions
        :param class_prob: classification probabilities
        :param obj: objectness
        :return torch tensor with prediction probabilities
        """
        conf_obj = class_prob * obj.unsqueeze_(self.class_axis)
        conf_obj[conf_obj <= self._thresh] = 0 
        max_conf_obj, max_idxs = torch.max(conf_obj, self.class_axis)
        return torch.cat([conf_obj, max_conf_obj.unsqueeze_(self.class_axis)], self.class_axis)
        
    @property
    def shape(self):
        return [self.num_anchors, self.num_classes]


class TreePredictor(YoloPredict):
    """Extends YoloPredict to perform prediction based on tree structure
    Parameters:
        tree: str, path to the file with tree structure
        obj_esc_thresh: int, objectness threshold
    """
    def __init__(self, tree, num_classes=20, num_anchors=5, biases=DEFAULT_BIASES, coords=4,
                 pred_thresh=0.5, nms_threshold=0.45, pre_threshold=0.005):
        super(TreePredictor, self).__init__(num_classes, num_anchors, biases, coords,
                                            nms_threshold, pre_threshold)
        self._class_prob = SoftmaxTree(tree, axis=self.class_axis)
        self._predictor = SoftmaxTreePrediction(tree, threshold=pred_thresh,
                                                append_max=True, output_tree_path=True)

    def class_probability(self, x):
        """calculates classification probability
        :param x: torch tensor, features
        :return torch tensor with loss value
        """
        return self._class_prob(x)

    def top_predictions(self, class_prob, obj):
        """predictions
        :param class_prob: classification probabilities
        :param obj: objectness
        :return torch tensor with prediction probabilities
        """
        return self._predictor(class_prob, obj)

    @property
    def shape(self):
        """
        :return: list, dimensions for Reshape
        """
        return [self.num_classes, self.num_anchors]

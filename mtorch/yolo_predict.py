import torch.nn as nn
from mtorch.caffetorch import Slice
from mtorch.reshape import Reshape
from mtorch.softmaxtree import SoftmaxTree
from mtorch.softmaxtree_prediction import SoftmaxTreePrediction
from mtorch.yolobbs import YoloBBs
from mtorch.nmsfilter import NMSFilter
from mtorch.yoloevalcompat import YoloEvalCompat
from mtorch.region_prediction import RegionPrediction

__all__ = ['PlainPredictorClassSpecificNMS', 'PlainPredictorSingleClassNMS',
           'TreePredictorClassSpecificNMS', 'TreePredictorSingleClassNMS']

DEFAULT_BIASES = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
SLICE_POINTS = [10, 20, 25]
OBJECTNESS_DIM = 1
DEFAULT_NUM_CLASSES = 20  # for voc20
DEFAULT_NUM_ANCHORS = 5
BBOX_DIM = 4
# thresholds:
# PREDICTION:
PLAIN_PRED_THRESH = 0 #0.005
TREE_PRED_THRESH = 0.1
# NMS filtering
NMS_THRESHOLD = 0.45
PRE_THRESHOLD = 0.005


class YoloPredict(nn.Module):
    """Abstract class for constructing different kinds of Yolo predictions
    Parameters:
        num_classes: int, number of classes for classification
        num_anchors: int, number of anchors for bounding box predictions
        biases: list, default anchors
        coords: int, number of coordinates in bounding box to predict
        nms_threshold: float
        pre_threshold: float
    """

    def __init__(self, num_classes=DEFAULT_NUM_CLASSES, num_anchors=DEFAULT_NUM_ANCHORS,
                 biases=DEFAULT_BIASES, coords=BBOX_DIM,
                 nms_threshold=NMS_THRESHOLD, pre_threshold=0):
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
        first_class, classes = self.nms_params
        self.nms_filter = NMSFilter(nms_threshold=nms_threshold, pre_threshold=pre_threshold,
                                    classes=classes, first_class=first_class)
        self.yolo_eval_compat = YoloEvalCompat()

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
        conf_class = self.class_probability(self.reshape_conf(conf))
        top_preds = self.top_predictions(conf_class, obj)
        bboxs = self.yolo_bboxs(xy, wh, im_info)
        prob = self.yolo_eval_compat(self.nms_filter(bboxs, top_preds))
        return prob, bboxs

    def class_probability(self, x):
        """calculates class probabilities
        :param x: torch tensor, features
        :return torch tensor with loss value
        """
        raise NotImplementedError(
            "This is an abstract class, please use either of" + str(__all__))

    def top_predictions(self, class_prob, obj):
        raise NotImplementedError(
            "This is an abstract class, please use either of" + str(__all__))

    @property
    def shape(self):
        raise NotImplementedError(
            "This is an abstract class, please use either of" + str(__all__))

    @property
    def nms_params(self):
        raise NotImplementedError(
            "This is an abstract class, please use either of" + str(__all__))


class PlainPredictor(YoloPredict):
    """Extends YoloPredict and performs plain prediction"""

    def __init__(self, pred_thresh=PLAIN_PRED_THRESH, **kwargs):
        super(PlainPredictor, self).__init__(**kwargs)
        self._class_prob = nn.Softmax(dim=self.class_axis)
        self._predictor = RegionPrediction(thresh=pred_thresh, class_axis=self.class_axis)

    def class_probability(self, x):
        """calculates classification probability
        after permuting the dimensions of input features  (compatibility to Caffe)
        :param x: torch tensor, features
        :return torch tensor with loss value
        """
        # permutation before softmax b x a x c x spatial dims -->  b x c x a x spatial dims
        # as expected by PyTorch Softmax the class axis = 1 
        return self._class_prob(x.permute([0, 2, 1, 3, 4]))

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
        return [self.num_anchors, self.num_classes]

    @property
    def nms_params(self):
        raise NotImplementedError(
            "This is an abstract class, please use either of" + str(__all__))


class PlainPredictorClassSpecificNMS(PlainPredictor):
    """Extends PlainPredictor by calculating class specific NMS"""

    def __init__(self, **kwargs):
        super(PlainPredictorClassSpecificNMS, self).__init__(**kwargs)

    @property
    def nms_params(self):
        """ which classes to use for nms filtering
        :return: first class, int 
        :return: number of classes, int, starting from the first class
        """
        return [0, self.num_classes]


class PlainPredictorSingleClassNMS(PlainPredictor):
    """Extends PlainPredictor by calculating NMS on max of all classes"""

    def __init__(self, **kwargs):
        super(PlainPredictorSingleClassNMS, self).__init__(**kwargs)

    @property
    def nms_params(self):
        """ which classes to use for nms filtering
        :return: first class, int 
        :return: number of classes, int, starting from the first class
        """
        return [self.num_classes, 1]


class TreePredictor(YoloPredict):
    """Extends YoloPredict to perform prediction based on tree structure
    Parameters:
        tree: str, path to the file with tree structure
        obj_esc_thresh: int, objectness threshold
    """

    def __init__(self, tree, pred_thresh=TREE_PRED_THRESH, **kwargs):
        super(TreePredictor, self).__init__(**kwargs)
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

    @property
    def nms_params(self):
        raise NotImplementedError(
            "Please create an instance of PlainPredictor or TreePredictor")


class TreePredictorClassSpecificNMS(TreePredictor):
    """Extends TreePredictor and performs class specific NMS"""

    def __init__(self, *args, **kwargs):
        super(TreePredictorClassSpecificNMS, self).__init__(*args, **kwargs)

    @property
    def nms_params(self):
        """ which classes to use for nms filtering
        :return: first class, int 
        :return: number of classes, int, starting from the first class
        """
        return [0, self.num_classes]


class TreePredictorSingleClassNMS(TreePredictor):
    """Extends TreePredictor and performs NMS on max of all classes """

    def __init__(self, *args, **kwargs):
        super(TreePredictorSingleClassNMS, self).__init__(*args, **kwargs)

    @property
    def nms_params(self):
        """ which classes to use for nms filtering
        :return: first class, int 
        :return: number of classes, int, starting from the first class
        """
        return [self.num_classes, 1]

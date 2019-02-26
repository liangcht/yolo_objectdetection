import torch
import torch.nn as nn


class RegionPrediction(nn.Module):
    """Caffe RegionPrediction fully compatible class
    Parameters:
        thresh - filtering threshold of confidence = class_conf * obj
        class_axis - axis that corresponds to classes in the torch tensor
        class_prob - class probability (typically output of softmax)
        obj - objectness score (typically after Sigmoid)
    """

    def __init__(self, thresh=0.005, class_axis=1):
        """Constructs the RegionPrediction object
        :param thresh: float, threshold for confidence score (default=0.005)
        :param class_axis: int
        """
        super(RegionPrediction, self).__init__()
        self.thresh = thresh
        self.class_axis = class_axis

    def forward(self, class_prob, obj):
        """
        :param class_prob: class probability (typically output of softmax)
        :param obj: objectness score (typically after Sigmoid)
        :return: confidence score
        """
        conf_obj = class_prob * obj.unsqueeze_(self.class_axis)
        conf_obj[conf_obj <= self.thresh] = 0
        max_conf_obj, _ = torch.max(conf_obj, self.class_axis)
        return torch.cat([conf_obj, max_conf_obj.unsqueeze_(self.class_axis)], self.class_axis)

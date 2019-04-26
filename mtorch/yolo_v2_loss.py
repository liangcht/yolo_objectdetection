import torch.nn as nn
from mtorch import region_target_loss
from mtorch.region_target_loss import RegionTargetWithSoftMaxLoss, RegionTargetWithSoftMaxTreeLoss

__all__ = ['YoloLossForTreeStructure', 'YoloLossForPlainStructure']


class YoloLoss(nn.Module):
    """Abstract class to represent Yolo Loss
    Do NOT instantiate - use YoloLossForTreeStructure  or YoloLossForPlainStructure
    """
    def __init__(self, num_classes, seen_images=region_target_loss.NUM_IMAGES_WARM):
        """Constructor of YoloLoss:
        :param num_classes, int  - number of classes to classify
        :param seen_images, int - number of images that the model saw so far
        """
        super(YoloLoss, self).__init__()
        self._num_classes = num_classes
        self._seen_images = seen_images
        
    @property
    def criterion(self):
        raise NotImplementedError(
            "Please create an instance of YoloLossForTreeStructure or YoloLossForPlainStructure")
    
    @property
    def num_classes(self):
        return self._num_classes

    @property
    def seen_images(self):
        return self._seen_images 

    @property 
    def normalization(self):
        return self.criterion.normalization

    def forward(self, x, label):
        return self.criterion(x, label)
    
    def __repr__(self):
        return "Yolo v2 Loss"
        

class YoloLossForTreeStructure(YoloLoss):
    """Yolo loss that supports tree structure"""
    def __init__(self, tree_structure_file, *args, **kwargs):
        """Constructor of Yolo loss that supports tree structure
        :param tree_structure_file, - file that depicts the tree structure
        """
        super(YoloLossForTreeStructure, self).__init__(*args, **kwargs)
        self._criterion = RegionTargetWithSoftMaxTreeLoss(tree_structure_file,
                                                          num_classes=self.num_classes,
                                                          seen_images=self.seen_images)
    @property
    def criterion(self):
        return self._criterion 

    def __repr__(self):
        return "{} for Tree structure with softmax normalization {} ".format(
            super(YoloLossForTreeStructure,self).__repr__(),
            self.normalization
        )


class YoloLossForPlainStructure(YoloLoss):
    """Yolo loss that supports plain structure"""
    def __init__(self, *args, **kwargs):
        super(YoloLossForPlainStructure, self).__init__(*args, **kwargs)
        self._criterion = RegionTargetWithSoftMaxLoss(num_classes=self.num_classes,
                                                      seen_images=self.seen_images)
    @property
    def criterion(self):
        return self._criterion 

    def __repr__(self):
        return "{} for Plain structure with softmax normalization {} ".format(
            super(YoloLossForPlainStructure, self).__repr__(),
            self.normalization
        )

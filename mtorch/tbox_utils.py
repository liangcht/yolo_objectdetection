import numpy as np
import torch
from torchvision.transforms.functional import crop

BBOX_DIM = 4

__all__ = ['Labeler', 'RegionCropper' ,'ClassLabeler']


def _to_bbox(truth):
    return [float(val) for val in truth['rect']]

def set_box_into_bounds(bbox, bounds):
    """Set the given coordinates within the min/max bounds
    :param bbox, list of coordinates [x1, y1, x2, y2]
    :param bounds, maximum width and height
    :return list of coordinates after setting boundaries
    """
    w, h = bounds
    bbox[0] = max(0, bbox[0])
    bbox[1] = max(0, bbox[1])
    bbox[2] = min(w, bbox[2])
    bbox[3] = min(h, bbox[3])
    return bbox 

def region_crop(img, crop_box):
    """Crops the given image based on the given coordinates
    :param img, numpy arr, image to be transformed
    :param crop_box, list of coordinates [x1, y1, x2, y2]
    :return cropped image
    """
    crop_box = set_box_into_bounds(crop_box, img.size)
        
    upper = int(round(crop_box[1]))
    left = int(round(crop_box[0]))
    height = int(round(crop_box[3])) - upper
    width = int(round(crop_box[2])) - left
        
    return crop(img, i=upper, j=left, h=height, w=width)


class Labeler(object):
    """
    Creates labels in a format of top left bottom right corners of bounding box
    Attaches class per each label
    """

    def __init__(self):
        """Constructor of Labeler Class"""
        pass
    
    def __call__(self, truth_list, cmap, filter_difficult=True):
        """Constructs bounding boxes according to the following format:
        x for left, y for top, x for right, y for bottom, class
        :param truth_list: bounding boxes
        :param cmap: the map the converts between the class string labels
        and corresponding numeric labels
        :param filter_difficult: boolean, if true filters difficult labels,
        if false retains all labels
        :return: number of boxes x 5 numpy array of float32
        """
        return self.create_bounding_boxes(truth_list, cmap, filter_difficult)

    @staticmethod
    def create_bounding_boxes(truth, cmap, filter_difficult):
        """Create bounding boxes
        :param truth: bounding boxes
        :param cmap: class to numeric value conversion map
        :param filter_difficult: boolean, if true filters difficult labels,
        if false retains all labels
        :return: number of boxes x 5 numpy array of float32
        """
        length = len(truth)
        bboxs = np.zeros(shape=(length, BBOX_DIM + 1), dtype="float32")
        last_valid_box = 0
        for bbox in truth:
            if filter_difficult and bbox.get('diff', 0) == 1:
                continue
            bboxs[last_valid_box, :BBOX_DIM] = [float(val) for val in bbox['rect']]
            bboxs[last_valid_box, BBOX_DIM] = cmap.index(bbox['class'])
            last_valid_box += 1
        bboxs = bboxs[:last_valid_box, :]
        return bboxs


class ClassLabeler(object):
    """Creates labels in a format of top left bottom right corners of bounding box
    Attaches class per each label
    """

    def __init__(self, cond=None):
        """Constructor of Labeler Class
        :param cond: any condition that a valid bounding box should follow
        """
        self.condition = cond
    
    def __call__(self, bbox, cmap):
        """Constructs label according to the following format:
        x for left, y for top, x for right, y for bottom, class
        :param bbox: bounding box
        :param cmap: the map the converts between the class string labels
        and corresponding numeric labels
        :return: number of boxes x 5 numpy array of float32
        """
        if not self.condition or self.condition(bbox):
            label = cmap.index(bbox['class']) 
        else:
            label = len(cmap)
        label = torch.from_numpy(np.array(label, dtype=np.int))
  
        return label
     

class RegionCropper(object):
    """Crops regions from provided ground truth based on conditions
    """
    def __init__(self, conds):
        """Constructor of RegionCropper
        :param conds: conditions that valid boxes should follow
        """
        self.conds = conds

    def __call__(self, truths):
        filtered = truths
        for cond in self.conds:
            filtered = [truth for truth in filtered if cond(_to_bbox(truth))]

        return filtered

from mtorch import Transforms
import numpy as np
from torch.utils.data import DataLoader

BBOX_DIM = 4
MEANS = (104.0, 117.0, 123.0)
CANVAS_SIZE = (416, 416)
MAX_BOXES = 30
USE_DARKNET_LIB = True

class Labeler(object):
    """
    Creates labels in a format of top left bottom right corners of bounding box
    Attaches class per each label
    """

    def __init__(self):
        """Constructor of Labeler Class"""
        pass
    
    def __call__(self, truth_list, cmap, filter_difficult=True):
        """
        Constructs bounding boxes according to the following format:
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
        """
        Create bounding boxes
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


class DarknetAugmentation(object):
    """
    Constructs the transform for augmentation
    """
    def __init__(self):
        """
        Constructor does nothing
        """
        pass

    def __call__(self, params):
        """
        composes the transforms
        :param params: parameters for augmentation transforms defined in prototxt
        :return: the composed transform (list of transforms)
        """
        self.hue = float(params['box_data_param']['hue'])
        self.saturation = float(params['box_data_param']['saturation'])
        self.exposure = float(params['box_data_param']['exposure'])
        self.jitter = float(params['box_data_param']['jitter'])
        self.means = [int(float(params['transform_param']['mean_value'][0])),  # B
                      int(float(params['transform_param']['mean_value'][1])),  # G
                      int(float(params['transform_param']['mean_value'][2]))]  # R
        self.max_boxes = int(params['box_data_param']['max_boxes'])
        set_inrange = Transforms.SetBBoxesInRange()
        box_randomizer = Transforms.RandomizeBBoxes(self.max_boxes)
        random_distorter = Transforms.RandomDistort(hue=self.hue, saturation=self.saturation, exposure=self.exposure)
        random_resizer = Transforms.RandomResizeDarknet(self.jitter, library=Transforms.OPENCV)
        darknet_random_resize_place = Transforms.DarknetRandomResizeAndPlaceOnCanvas(jitter=self.jitter)
        horizontal_flipper = Transforms.RandomHorizontalFlip()
        place_on_canvas = Transforms.PlaceOnCanvas()
        minus_dc = Transforms.SubtractMeans(self.means)
        to_tensor = Transforms.ToDarknetTensor(self.max_boxes)

        if USE_DARKNET_LIB:
            self.composed_transforms = Transforms.Compose(
                [set_inrange, box_randomizer, darknet_random_resize_place, random_distorter,
                 horizontal_flipper, to_tensor, minus_dc])
        else:
            self.composed_transforms = Transforms.Compose(
                [set_inrange, box_randomizer, random_resizer, place_on_canvas, random_distorter,
                 horizontal_flipper, to_tensor, minus_dc])
        return self.composed_transforms


class TestAugmentation(object):
    """
    Prepares image for testing
    Currently not used
    """
    def __init__(self):
        self.__call__()

    def __call__(self):
        self.means = MEANS  # TODO: need to be read from params
        fit_to_canvas = Transforms.ResizeToCanvas()
        place_on_canvas = Transforms.PlaceOnCanvas(fixed_offset=True)  
        minus_dc = Transforms.SubtractMeans(MEANS)
        to_tensor = Transforms.ToDarknetTensor(MAX_BOXES)  # TODO: need to be read from params
        self.composed_transforms = Transforms.Compose(
            [fit_to_canvas, place_on_canvas, to_tensor, minus_dc])
        return self.composed_transforms


class DebugAugmentation(object):
    """
    Prepares image for testing
    Currently not used
    """
    def __init__(self):
        self.__call__()

    def __call__(self):
        self.means = MEANS  # TODO: need to be read from params
        crop300 = Transforms.Crop((0, 0, 300, 300), allow_outside_bb_center=False)
        minus_dc = Transforms.SubtractMeans(MEANS)
        to_tensor = Transforms.ToDarknetTensor(MAX_BOXES)  # TODO: need to be read from params
        self.composed_transforms = Transforms.Compose(
            [crop300, to_tensor, minus_dc])
        return self.composed_transforms




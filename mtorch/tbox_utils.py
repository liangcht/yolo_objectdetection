from mtorch import Transforms
import numpy as np
from torch.utils.data import DataLoader

BBOX_DIM = 4
MEANS = (104.0, 117.0, 123.0)
CANVAS_SIZE = (416, 416)
MAX_BOXES = 30


class Labeler(object):
    """
    Creates labels in a format of top left bottom right corners of bounding box
    Attaches class per each label
    """

    def __init__(self):
        """Constructor of Labeler Class"""
        pass
    
    def __call__(self, truth_list, cmap):
        """
        Constructs bounding boxes according to the following format:
        x for left, y for top, x for right, y for bottom, class
        :param truth_list: bounding boxes
        :param cmap: the map the converts between the class string labels
        and corresponding numeric labels
        :return: number of boxes x 5 numpy array of float32
        """
        return self.create_bounding_boxes(truth_list, cmap)

    @staticmethod
    def create_bounding_boxes(truth, cmap):
        """
        Create bounding boxes
        :param truth: bounding boxes
        :param cmap: class to numeric value conversion map
        :return: number of boxes x 5 numpy array of float32
        """
        length = len(truth)
        bboxs = np.zeros(shape=(length, BBOX_DIM + 1), dtype="float32")
        for i, bbox in enumerate(truth):
            bboxs[i, :BBOX_DIM] = [float(val) for val in bbox['rect']]
            bboxs[i, BBOX_DIM] = cmap.index(bbox['class'])
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
        self.means = [int(float(params['transform_param']['mean_value'][0])),  #B
                      int(float(params['transform_param']['mean_value'][1])),  #G
                      int(float(params['transform_param']['mean_value'][2]))]  #R
        random_distorter = Transforms.RandomDistort(self.hue, self.saturation, self.exposure)
        random_resizer = Transforms.RandomResizeDarknet(self.jitter, library=Transforms.TORCHVISION)
        horizontal_flipper = Transforms.RandomHorizontalFlip()
        place_on_canvas = Transforms.PlaceOnCanvas()  
        minus_dc = Transforms.SubtractMeans(self.means)
        to_tensor = Transforms.ToDarknetTensor(int(params['box_data_param']['max_boxes']))

        self.composed_transforms = Transforms.Compose(
            [random_resizer, place_on_canvas, random_distorter, horizontal_flipper, to_tensor, minus_dc])
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
        to_tensor = Transforms.ToTensor(MAX_BOXES) # TODO: need to be read from params
        self.composed_transforms = Transforms.Compose(
            [fit_to_canvas, place_on_canvas, to_tensor, minus_dc])
        return self.composed_transforms





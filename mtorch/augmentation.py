import torch
from torchvision import transforms
from mtorch import Transforms
from mtorch.predict_transforms import ODImResize

MEANS = Transforms.COLOR_MEAN 
TRAIN_CANVAS_SIZE = Transforms.DEF_CANVAS_SIZE  # TODO: read from config
TEST_CANVAS_SIZE = Transforms.DEF_CANVAS_SIZE  # TODO: read from config
MAX_BOXES = 30
NO_FLIP = 0
RANDOM_FLIP = Transforms.FLIP_PROB
SCALE_RANGE = Transforms.DEF_SCALE_RANGE

__all__ = ['BasicDarknetAugmentation', 'DefaultDarknetAugmentation', 
           'DarknetAugmentation', 'TestAugmentation', 'ClassifierTrainAugmentation', 'ClassifierTestAugmentation']


class BasicDarknetAugmentation(object):
    """
    Basic Darknet Augmentation will perform no augmentation (dummy):
     dummy color distortion, dummy flip, dummy cropping
     Resizing to the Darknet Box will be still performed
    """

    def __init__(self):
        """
        Constructor does nothing
        """
        pass

    def __call__(self, params=None):
        """
        composes the transforms for augmentation
        :param params: parameters for augmentation transforms defined in prototxt
        :return: the composed transform (list of transforms)
        """
        self._set_augmentation_params()
        set_inrange = Transforms.SetBBoxesInRange()
        box_randomizer = Transforms.RandomizeBBoxes(self.max_boxes)
        random_distorter = Transforms.RandomDistort(hue=self.hue, saturation=self.saturation, exposure=self.exposure)
        random_resizer = Transforms.RandomResizeDarknet(self.jitter)
        horizontal_flipper = Transforms.RandomHorizontalFlip(self.flip)
        place_on_canvas = Transforms.PlaceOnCanvas()
        to_labels = Transforms.ToDarknetLabels(self.max_boxes)
        to_tensor = Transforms.ToDarknetTensor()
        minus_dc = Transforms.SubtractMeans(self.means)
    
        self.composed_transforms = Transforms.Compose(
            [set_inrange, box_randomizer, random_resizer, place_on_canvas, random_distorter,
            horizontal_flipper, to_labels, to_tensor, minus_dc])
        return self.composed_transforms

    def _set_augmentation_params(self):
        self.hue = 0.0
        self.saturation = 1.0
        self.exposure = 1.0
        self.jitter = 0.0
        self.scale = 1.0
        self.fixed_offset = True
        self.means = [int(mean) for mean in MEANS]  # BGR
        self.max_boxes = MAX_BOXES
        self.flip = NO_FLIP


class DefaultDarknetAugmentation(BasicDarknetAugmentation):
    """
    Constructs the transform for augmentation
    """

    def __init__(self):
        """
        Constructor does nothing
        """
        super(DefaultDarknetAugmentation, self).__init__()

    def __call__(self, params=None):
        """
        composes the transforms
        :param params: parameters for augmentation transforms defined in prototxt
        :return: the composed transform (list of transforms)
        """
        self._set_augmentation_params()
        return super(DefaultDarknetAugmentation, self).__call__()

    def _set_augmentation_params(self):
        self.hue = 0.1
        self.saturation = 1.5
        self.exposure = 1.5
        self.jitter = 0.2
        self.scale = SCALE_RANGE
        self.fixed_offset = False
        self.means = [int(mean) for mean in MEANS]  # BGR
        self.max_boxes = MAX_BOXES
        self.flip = RANDOM_FLIP


class DarknetAugmentation(BasicDarknetAugmentation):
    """
    Constructs the transform for augmentation, compatible with Caffe prototxt
    Parameters:
        params - typically read from Caffe prototxt file
    """

    def __init__(self):
        """
        Constructor does nothing
        """
        super(DarknetAugmentation, self).__init__()

    def __call__(self, params=None):
        """
        composes the transforms
        :param params: parameters for augmentation transforms defined in prototxt
        :return: the composed transform (list of transforms)
        """
        if params is not None:
            self.params = params
            return super(DarknetAugmentation, self).__call__()
        raise ValueError(
            "parameters should be provided for DarknetAugmentation, otherwise use DefaultDarknetAugmentation")

    def _set_augmentation_params(self):
        self.hue = float(self.params['box_data_param']['hue'])
        self.saturation = float(self.params['box_data_param']['saturation'])
        self.exposure = float(self.params['box_data_param']['exposure'])
        self.jitter = float(self.params['box_data_param']['jitter'])
        self.means = [int(float(self.params['transform_param']['mean_value'][0])),  # B
                      int(float(self.params['transform_param']['mean_value'][1])),  # G
                      int(float(self.params['transform_param']['mean_value'][2]))]  # R
        self.max_boxes = int(self.params['box_data_param']['max_boxes'])
        self.flip = RANDOM_FLIP
        self.scale = SCALE_RANGE
        self.fixed_offset = False


class TestAugmentation(object):
    """Prepares image for testing/prediction"""

    def __init__(self):
        pass

    def __call__(self, means=MEANS):
        minus_dc = Transforms.SubtractMeans(means)
        od_resizer = ODImResize(target_size=TEST_CANVAS_SIZE)
        self.composed_transforms = Transforms.Compose(
            [Transforms.ToDarknetTensor(), minus_dc, self._permute_whc, self._to_numpy, od_resizer,
             transforms.functional.to_tensor])
        return self.composed_transforms

    @staticmethod
    def _to_numpy(x):
        return x.numpy()

    @staticmethod
    def _permute_whc(x):
        return x.permute((1, 2, 0))


class ClassifierTrainAugmentation(object):

    def __init__(self):
        self._set_augmentation_params()
    
    def __call__(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=self.scale),
            transforms.RandomAffine(degrees=self.rotation_upto_deg),
            transforms.ColorJitter(brightness=self.exposure, saturation=self.saturation, hue=self.hue),
            transforms.RandomHorizontalFlip(self.flip),
            transforms.ToTensor(),
            normalize,
        ])

    def _set_augmentation_params(self):
        self.hue = 0
        self.saturation = 1
        self.exposure = 1
        self.rotation_upto_deg = 180 
        self.scale = (0.25, 2)
        self.flip = RANDOM_FLIP


class ClassifierTestAugmentation(object):
    def __init__(self):
        pass

    def __call__(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])


class _DebugAugmentation(object):
    """Debug augmentation
    Currently not used
    """

    def __init__(self):
        self.__call__()

    def __call__(self):
        crop300 = Transforms.Crop((0, 0, 300, 300), allow_outside_bb_center=False)
        minus_dc = Transforms.SubtractMeans(MEANS)
        to_labels = Transforms.ToDarknetLabels(MAX_BOXES)
        to_tensor = Transforms.ToDarknetTensor()
        self.composed_transforms = Transforms.Compose(
            [crop300, to_labels, to_tensor, minus_dc])
        return self.composed_transforms


 
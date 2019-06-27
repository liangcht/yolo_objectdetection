"""Classes that represent different transforms for augmentation for object detection
The API is almost identical to respective torchvsion transforms
Important notes - differently from torchvision, the image size is assumed to be a tuple with (WIDTH, HEIGHT).
This is similar to the convention of image size in PIL.Image and also fits bounding box coordinate order (x,y)
"""
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import random
import logging

from mtorch import transform_boxes as tbox

# Dictionary key
IMAGE = "image"  # TODO: change to get that as parameter from prototxt
LABEL = "bbox"  # TODO: change to get that as parameter from prototxt
# Color related constants/ defaults
H = 0
S = 1
V = 2
COLOR_MEAN = (104.0, 117.0, 123.0)
MAX_PIXEL_VAL = 255.0
PADDING_COL = (int(MAX_PIXEL_VAL / 2.0),) * 3
UNIT = (1.0,) * 3
NO_CHANGE_IN_HUE = 0
NO_CHANGE_IN_EXP = 1
NO_CHANGE_IN_SAT = 1
# Randomization
BIG_NUMBER = 10000
# Default canvas
DEF_CANVAS_SIZE = (416, 416)
DEF_SCALE_RANGE = (0.25, 2)
# Resize related constants
INTERP_METHOD = Image.BILINEAR  # TODO: this should be just "bilinear" and based on the library chosen the right naming
TORCHVISION = "torchvision"
#  Number of tries for random augmentation to be left with at least one bounding box
DEF_NUM_TRIES = 1
MIN_NUM_BBOXES = 1
DEF_AREA_RATIO = 0
REQ_VALID_TRANS = False  # if to search for transform with at least one bounding box
# Crop related constants
ORIGIN = (0, 0)
# Flip related constants
FLIP_PROB = 0.5

__all__ = sorted(["Compose", "Resize", "RandomResizeDarknet", "RandomCrop", "RandomHorizontalFlip",
                  "RandomVerticalFlip", "PlaceOnCanvas", "CanvasAdapter", "SubtractMeans",
                  "RandomizeBBoxes", "ToDarknetLabels", "SetBBoxesInRange", "ToDarknetTensor"])


class Compose(object):
    """Composes several transforms together.
    Parameters
    -----------
        transform_list: list of transforms to compose.
    Output
    -----------
        sample: transformed sample with corresponding  IMAGE and LABEL
    Example
    -----------
        >>> Compose([
        >>>     RandomHorizontalFlip(0.5),
        >>>     RandomResize()
        >>>     RandomCrop((200,200))
        >>> ])
    """

    def __init__(self, transform_list):
        """Constructs Compose object
        :param transform_list: list of transforms to concatenate
        """
        self.transforms = transform_list

    def __call__(self, sample):
        """Applies all the transforms on the input sample (left to right reduce)
        :param sample: a dictionary that contains
            a PIL image under key IMAGE and
            a numpy nd array of bounding boxes under key LABEL
        :return: transformed sample - a dictionary with  IMAGE and LABEL
        """
        for t in self.transforms:
            sample = t(sample)
        return sample


class ComposeDebugger(object):
    """Use ONLY for debugging: Composes several transforms together.
    Parameters
    -----------
        transform_list: list of transforms to compose.

    Output
    -----------
        intermediate_results: a list of results of each transform

    Example
    -----------
        >>> Compose([
        >>>     RandomHorizontalFlip(0.5),
        >>>     RandomResize()
        >>>     RandomCrop((200,200))
        >>> ])
    """

    def __init__(self, transform_list):
        """ Constructs ComposeDebugger object
        :param transform_list: list of transforms to compose
        """
        self.transforms = transform_list

    def __call__(self, sample):
        """ Applies transforms on the input sample (left to right reduce)
        :param sample: dictionary with IMAGE and LABEL
        :return: a list of respective results per each transform,
         each result is a dictionary with IMAGE and LABEL
        """
        intermediate_results = [sample]
        for t in self.transforms:
            sample = t(sample)
            intermediate_results.append(sample)
        return intermediate_results

    def __repr__(self):
        """ Creates a string representation of the Compose,
        including all the name of the concatenated transforms.
        :return: string representation of the composed transform
        """
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ImageBoxAugmentation(object):
    """Abstract class for transformation of image and bbox
    Parameters
     ----------
        sample: a dictionary that contains
            a PIL image under key IMAGE and
            a numpy nd array of bounding boxes under key LABEL

    """

    def __init__(self, valid_num_boxes=MIN_NUM_BBOXES):
        """Constructor of ImageBoxAugmentation
        :param valid_num_boxes: required number of bounding boxes
        to be left after augmentation is done, default is 1
        """
        if isinstance(valid_num_boxes, int):
            self.valid_num_bboxes = valid_num_boxes
        else:
            raise ValueError("Please provide an integer for valid_num_boxes")

    def __call__(self, sample):
        """Prepares the sample for other transforms
        :param sample: dictionary with IMAGE and LABEL
        """
        self.image, self.bboxs = sample[IMAGE], sample[LABEL]
        if isinstance(self.image, torch.Tensor):
            self.image = self.image.numpy()
        if isinstance(self.image, (np.ndarray, np.generic)):
            self.h, self.w, self.c = self.image.shape[:3]
            toPIL = transforms.ToPILImage()
            self.image = toPIL(np.uint8(self.image))
        else:
            self.w, self.h = self.image.size
            self.c = len(self.image.getbands())

    def is_valid_transform(self, num_bboxes):
        """Checks if the number of bounding boxes is valid
        :param num_bboxes:
        :exception InsufficientNumOfBBoxes: is thrown if the number
        of bounding boxes is below the valid minimum
        """
        if num_bboxes < self.valid_num_bboxes:
            raise InsufficientNumOfBBoxes(
                "Number {} of bouding boxes is insufficient, {} expected".format(num_bboxes, self.valid_num_bboxes))
        return

    def __str__(self):
        """Representation of the class
        :return: String
        """
        return self.__class__.__name__

    @staticmethod
    def check_correctness(sample):
        """Method to check validity of the input sample
        Validity is defined such that each box is located within the image
        :exception ValueError: is thrown if any bounding box is out of bounds
        """
        sz = sample[IMAGE].size
        for bbox in sample[LABEL]:
            try:
                assert (bbox[0] < bbox[2] and int(bbox[0]) >= 0 and int(bbox[2]) <= int(sz[0]))
                assert (bbox[1] < bbox[3] and int(bbox[1]) >= 0 and int(bbox[3]) <= int(sz[1]))
            except:
                bbox[0] = max(bbox[0], 0)
                bbox[1] = max(bbox[1], 0)
                bbox[2] = min(bbox[2], sz[0])
                bbox[3] = min(bbox[3], sz[1])

    @staticmethod
    def check_dynamic_range(image, max_pixel_val):
        """Checks if all the values of the image are within correct Dynamic Range
        :param image: PIL image
        :param max_pixel_val: maximum value per channel
        :exception ValueError: is thrown if PIL image has values outside of the range
        """
        try:
            extr = image.getextrema()
            max_vals = []
            min_vals = []
            for i in extr:
                if isinstance(i, tuple):
                    min_vals.append(i[0])
                    max_vals.append(i[1])
                else:
                    min_vals.append(extr[0])
                    max_vals.append(extr[1])

            assert (all(max_vals <= max_pixel_val) and all(min_vals >= 0))
        except:
            ValueError("There is out of bounds value: " + str(i))


class Resize(ImageBoxAugmentation):
    """Rescale the PIL image in a sample.

     Parameters
    ----------
        sample: a dictionary that contains
            a PIL image under key IMAGE and
            a numpy nd array of bounding boxes under key LABEL

        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.

        scale_factor (tuple or float): Desired scale factor of the original size. If int,
        aspect ratio is preserved; if tuple, will change aspect ratio.

        If both are provided scale_factor will be used

        interp_method(See PIL.Image options for interpolation): default is bilinear interpolation

    Output
    ---------
        resized sample: dictionary with two fields defined by IMAGE and LABEL with their content resized
    """

    def __init__(self, output_size=None, scale_factor=None, interp_method=INTERP_METHOD,
                 library=TORCHVISION, valid_num_bboxes=MIN_NUM_BBOXES):
        """Constructor of Resize object
        :param output_size: a tuple or integer that specifies the size of the image after resizing
        :param scale: a tuple or integer that specifies the scaling factor to determine the output size,
        overrides output_size.
        :param interp_method: name of interpolation method (bilinear is a default)
        :param library: name of library to use for image resizing,
        :param valid_num_bboxes: int, minimal number of bounding boxes to remain after transform
        """
        super(Resize, self).__init__(valid_num_bboxes)
        self.interp_method = interp_method
        self.library = library
        if output_size is None and scale_factor is None:
            raise ValueError("Please provide output_size or scale_factor")
        if output_size is not None and scale_factor is not None:
            logging.warning("Output_size is ignored since scale_factor is provided")
        self.set_output_size(output_size)
        self.set_scale_factor(scale_factor)

    def __call__(self, sample):
        """Applies the resizing operation on the sample
        :param sample: dictionary with IMAGE and LABEL
        :return: resized sample with IMAGE and LABEL transformed accordingly
        """

        super(Resize, self).__call__(sample)
        self.__get_output_size() if self.__output_size_is_given else self.__calc_output_size()

        if self.__new_h == self.h and self.__new_w == self.w:
            return sample

        resized_bboxes = tbox.resize(self.bboxs, (self.w, self.h), (self.__new_w, self.__new_h))

        if REQ_VALID_TRANS:
            self.is_valid_transform(resized_bboxes.shape[0])

        # NOTE: expected size format:
        # Torchvision - (height, width)
        if self.library is not None and self.library == TORCHVISION:
            resized_image = transforms.functional.resize(self.image,
                                                         size=(int(self.__new_h), int(self.__new_w)),
                                                         interpolation=self.interp_method)

        result = {IMAGE: resized_image, LABEL: resized_bboxes}
        ImageBoxAugmentation.check_correctness(result)
        return result

    def __get_output_size(self):
        """ Gets the output height and width from the output size is provided"""
        if isinstance(self.output_size, int):
            if self.h > self.w:
                self.__new_h, self.__new_w = self.output_size * self.h / self.w, self.output_size
            else:
                self.__new_h, self.__new_w = self.output_size, self.output_size * self.w / self.h
        else:
            self.__new_w, self.__new_h = self.output_size

    def set_output_size(self, output_size):
        """Setter for output_size attribute
        :param output_size: value to assign
        :exception ValueError: if output_size is None
        """
        if output_size is not None:
            assert isinstance(output_size, (int, tuple))
            self.output_size = output_size
            self.__output_size_is_given = True
        else:
            ValueError("Please provide valid output size")

    def set_scale_factor(self, scale_factor):
        """Setter for output_size attribute
        :param scale_factor: value to assign
        :exception ValueError: if scale_factor is None
        """
        if scale_factor is not None:
            assert isinstance(scale_factor, (float, tuple))
            self.scale_factor = scale_factor
            self.__output_size_is_given = False
        else:
            ValueError("Please provide valid output size")

    def __calc_output_size(self):
        """ Calculates the output size if scale factor is provided"""
        if isinstance(self.scale_factor, tuple):
            self.__new_h, self.__new_w = self.scale_factor[0] * self.h, self.scale_factor[1] * self.w
        else:
            self.__new_h, self.__new_w = self.scale_factor * self.h, self.scale_factor * self.w

    def __repr__(self):
        """ Representation of the class
        :return: String representation of the Resize object, including th original and new sizes
        """
        format_string = self.__class__.__name__ + '('
        format_string += "initial size (width, height): {}, {}".format(self.w, self.h)
        format_string += '\n'
        format_string += "output size (width, height): {}, {} ".format(*self.resized_to())
        format_string += ')\n'
        return format_string

    @property
    def resized_to(self):
        """Remembers the last output size used for resizing
        :return: the new size (width, height) the sample was resized to
        """
        return self.__new_w, self.__new_h


class RandomResizeDarknet(ImageBoxAugmentation):
    """Randomly resize the PIL image in a sample to a given size,
    according to Darknet strategy:
    ***jitter the original aspect ratio
    ***fit inside the provided output_size
    ***rescale within the bounds of output_size

    Parameters
    ----------
        sample: a dictionary that contains
            a PIL image under key IMAGE and
            a numpy nd array of bounding boxes under key LABEL

        jitter(float): random perturbation to be added to sample IMAGE size to alter its aspect ration

        scale (tuple or float): scale range to choose from random resizing the output size
        (1.0 may be provided if scaling not needed)

        output_size(tuple or None): output size to rescale image to,
         this size will be randomly altered by random scale

        interp_method: default is bilinear interpolation for all library options,
        please see your library for other choice

        library: name of library to use upon resizing

        tries: number of tries to ensure minimal number of bounding boxes within the resized image

        valid_num_bboxes: minimum number of bboxes to be left after transform
    
    Output
    ---------
        resized sample: dictionary with two fields defined by IMAGE and LABEL with their content resized
    """

    def __init__(self, jitter=0, scale=DEF_SCALE_RANGE, output_size_limit=DEF_CANVAS_SIZE,
                 interp_method=INTERP_METHOD, library=TORCHVISION, tries=DEF_NUM_TRIES,
                 valid_num_bboxes=MIN_NUM_BBOXES):
        """Constructor of RandomResizeDarknet object
        :param jitter: random perturbation of the sample IMAGE size
        :param scale: scale range to randomly resize the output size
        :param output_size_limit: the output size to resize to, if none sample image size will be used
        :param interp_method: name of interpolation to use upon resizing
        :param library: name of library to use upon resizing
        :param tries: number of tries to ensure minimal number of bounding boxes within the resized image
        :param valid_num_bboxes: minimum number of bboxes to be left after transform
        """
        super(RandomResizeDarknet, self).__init__(valid_num_bboxes)
        self.output_size_limit = output_size_limit
        if isinstance(scale, (tuple, float)):
            self.scale = scale
        else:
            raise ValueError("Scale should be 2-D tuple of floats or float")
        self.jitter = jitter
        self.library = library
        self.tries = tries
        self.interp_method = interp_method

    def __call__(self, sample):
        """Resizes the provided sample
        :param sample: dictionary with IMAGE and LABEL
        :return: dictionary with two fields defined by IMAGE and LABEL with their content resized
        """
        super(RandomResizeDarknet, self).__call__(sample)

        self.resizer = Resize(output_size=self.__get_rand_output_size(), interp_method=self.interp_method,
                              library=self.library, valid_num_bboxes=self.valid_num_bboxes)

        for i in range(self.tries):
            try:
                result = self.resizer(sample)
            except InsufficientNumOfBBoxes as err:
                self.resizer.set_output_size(self.__get_rand_output_size())
            else:
                return result
        return sample

    def __get_random_scale(self):
        """Helper to get random scaling factor"""
        if isinstance(self.scale, tuple):
            try:
                sc = random.uniform(*self.scale)
            except Exception as err:
                raise ValueError(err)
        elif isinstance(self.scale, float):
            sc = np.float32(self.scale)
        return sc

    def __get_rand_output_size(self):
        """Helper to randomly jitter aspect ratio and randomly choose output_size"""
        dw = self.jitter * self.w
        dh = self.jitter * self.h
        scale_factor = self.__get_random_scale()
        new_aspect_ratio = np.float32(self.w + random.uniform(-dw, dw)) / np.float32(self.h + random.uniform(-dh, dh))
        if new_aspect_ratio < 1:
            new_h = scale_factor * self.output_size_limit[1]
            new_w = new_h * new_aspect_ratio
        else:
            new_w = scale_factor * self.output_size_limit[0]
            new_h = np.float32(new_w) / new_aspect_ratio
        return new_w, new_h

    @property
    def resized_to(self):
        """Encapsulates the last size used for resizing
        :return: a width and a height after resizing
        """
        return self.resizer.resized_to

    def __repr__(self):
        """Representation of the object
        :return: String
        """
        return self.resizer.__repr__()


class Crop(ImageBoxAugmentation):
    """Crop the sample in a sample based on crop_box.

     Parameters
    ----------
        sample: dictionary with two fields defined by IMAGE and LABEL
        crop_box : tuple of length 4. :math:`(x_{min}, y_{min}, width, height)`.
                  this argument can be supplied both upon initialization and call
        allow_outside_bb_center: True if to keep bounding box with center outside of the crop, False otherwise

    Output
    -----------
        cropped sample: dictionary with two fields defined by IMAGE and LABEL with their content cropped
    """

    def __init__(self, crop_box=None, allow_outside_bb_center=True):
        """Constructs a Crop object
        :param crop_box: region to crop ((x, y) for top left corner, (width height) for size)
        :param allow_outside_bb_center: true if to keep bounding boxes with center out of the image,
        otherwise false
        """
        super(Crop, self).__init__()
        if crop_box is not None:
            tbox.check_crop_box_dim(crop_box)
        self.crop_box = crop_box
        self.allow_outside_bb_center = allow_outside_bb_center

    def __call__(self, sample, crop_box=None, allow_outside_bb_center=None):
        """Performs cropping on the provided sample
        :param sample: dictionary with IMAGE and LABEL
        :param crop_box:
        :param allow_outside_bb_center:
        :return: cropped sample
        """
        super(Crop, self).__call__(sample)
        if self.crop_box is None:
            if crop_box is None:
                return sample
            else:
                tbox.check_crop_box_dim(crop_box)
                self.crop_box = crop_box
        if allow_outside_bb_center is not None:
            self.allow_outside_bb_center = allow_outside_bb_center

        self.crop_box = self.__into_bounds()
        upper = int(round(self.crop_box[1]))
        left = int(round(self.crop_box[0]))
        height = int(round(self.crop_box[3]))
        width = int(round(self.crop_box[2]))
        cropped_bboxes = tbox.crop(self.bboxs, self.crop_box,
                                   allow_outside_center=self.allow_outside_bb_center)

        cropped_image = transforms.functional.crop(self.image, i=upper, j=left, h=height, w=width)

        result = {IMAGE: cropped_image, LABEL: cropped_bboxes}
        ImageBoxAugmentation.check_correctness(result)
        return result

    def __into_bounds(self):
        """Helper that fits the crop box fts into image bounds"""
        return (self.crop_box[0], self.crop_box[1],
                min(self.crop_box[2], self.w - self.crop_box[0]),
                min(self.crop_box[3], self.h - self.crop_box[1]))

    def __repr__(self):
        """Representation of a Crop object (including the latest crop region)
        :return: string
        """
        format_string = self.__class__.__name__ + '('
        format_string += "initial size (width, height): {}, {} ".format(self.w, self.h)
        format_string += '\n'
        if self.crop_box is not None:
            format_string += "crop box (left, top, width, height): {}, {}, {}, {} ".format(*self.crop_box)
        else:
            format_string += str(self.crop_box)
        format_string += ')\n'
        return format_string


class RandomCrop(Crop):
    """Crop randomly the sample
     Parameters
    ----------
        sample: dictionary with two fields defined by IMAGE and LABEL
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
        offsets (tuple): offset coordinate (x,y) from the origin (top left corner of the image)
        allow_outside_bb_center: True if to keep bounding box with center outside of the crop, False otherwise
    Output
    -----------
         cropped sample - dictionary with two fields defined by IMAGE and LABEL with their content cropped
    """

    def __init__(self, output_size, offsets=ORIGIN, allow_outside_bb_center=True):
        """Constructs a RandomCrop object
        :param output_size: the output size of the cropped region
        :param offsets: top left corner of the cropped region inside the image
        :param allow_outside_bb_center: keep bounding boxes with center out of the image
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            tbox.check_size_dim(output_size, 'output_size')
            self.output_size = output_size
        self.offset_w, self.offset_h = offsets
        super(RandomCrop, self).__init__(crop_box=None, allow_outside_bb_center=allow_outside_bb_center)

    def __call__(self, sample, crop_box=None, allow_outside_bb_center=None):
        """Crops the provided sample
        :param sample: dictionary with two fields defined by IMAGE and LABEL
        :return: cropped sample with IMAGE and LABEL altered accordingly
        """
        super(Crop, self).__call__(sample)
        self.output_size = tuple([min(self.output_size[0], self.w),
                                  min(self.output_size[1], self.h)])
        if self.output_size[0] == self.w and self.output_size[1] == self.h:
            self.__crop = None
            return sample
        self.__random_placement()
        self.__crop = (self.left, self.top, self.output_size[0], self.output_size[1])
        result = super(RandomCrop, self).__call__(sample, self.__crop)
        ImageBoxAugmentation.check_correctness(result)
        return result

    def __random_placement(self):
        """Helper to randomly generate cropped region"""
        new_w, new_h = self.output_size
        if self.h - (new_h - self.offset_h) < self.offset_h:
            raise ValueError("Height: New {}, old {}".format(new_h, self.h))
        if self.w - (new_w - self.offset_w) < self.offset_w:
            raise ValueError("Width: New {}, old {}".format(new_w, self.w))

        self.top = np.random.randint(self.offset_h, self.h - (new_h - self.offset_h) - 1)
        self.left = np.random.randint(self.offset_w, self.w - (new_w - self.offset_w) - 1)

    @property
    def recent_crop(self):
        """Remembers the crop region applied in the recent crop
        :return: the cropped region top left coordinate and size
        """
        return self.__crop

    def __repr__(self):
        """Representation of the random crop
        :return: string
        """
        format_string = super(RandomCrop, self).__repr__()
        return format_string


class RandomHorizontalFlip(ImageBoxAugmentation):
    """Randomly flips the sample horizontally with probability 0.5
    Parameters
    ----------
        sample: dictionary with two fields defined by IMAGE and LABEL
        prob (float): probability of the image being flipped. Default value is 0.5
    
    Output
    -----------
         randomly flipped sample - dictionary with two fields defined by IMAGE and LABEL
    """

    def __init__(self, prob=FLIP_PROB):
        """Constructs the HorizontalFlip object
        :param prob: probability to flip the sample
        """
        super(RandomHorizontalFlip, self).__init__()
        self.prob = prob
        self.__is_flipped = False

    def __call__(self, sample):
        """Flips/ does not flip the sample (according to random draw with defined probability)
        :param sample: dictionary with two fields defined by IMAGE and LABEL
        :return: flipped/ original sample
        """
        super(RandomHorizontalFlip, self).__call__(sample)
        self.__is_flipped = False

        if random.random() < self.prob:
            flipped_image = transforms.functional.hflip(self.image)
            flipped_bboxes = tbox.flip(self.bboxs, (self.w, self.h), flip_x=True)
            self.__is_flipped = True
            result = {IMAGE: flipped_image, LABEL: flipped_bboxes}
            ImageBoxAugmentation.check_correctness(result)
            return result

        return sample

    @property
    def has_flipped(self):
        """Remembers if the sample was flipped or not
        :return: boolean - true if sample has been flipped, false otherwise
        """
        return self.__is_flipped

    def __repr__(self):
        """Representation of Flip Object, including info on recent flip
        :return: string
        """
        format_string = self.__class__.__name__ + '('
        format_string += ("Flipped Horizontally" if self.has_flipped else "Not Flipped")
        format_string += ') \n'
        return format_string


class RandomVerticalFlip(ImageBoxAugmentation):
    """Randomly flips the sample horizontally with probability 0.5

     Parameters
    ----------
        sample: dictionary with two fields defined by IMAGE and LABEL
        prob (float): probability of the image being flipped. Default value is 0.5

    Output
    -----------
         randomly flipped sample - dictionary with two fields defined by IMAGE and LABEL

    """

    def __init__(self, prob=FLIP_PROB):
        """Constructs the VerticalFlip object
        :param prob: probability to flip the sample
        """
        super(RandomVerticalFlip, self).__init__()
        self.prob = prob
        self.__is_flipped = False

    def __call__(self, sample):
        """Flips/ does not flip the sample (according to random draw with defined probability)
        :param sample: dictionary with two fields defined by IMAGE and LABEL
        :return: flipped/ original sample
        """
        super(RandomVerticalFlip, self).__call__(sample)
        self.__is_flipped = False

        if random.random() < self.prob:
            flipped_image = transforms.functional.vflip(self.image)
            flipped_bboxes = tbox.flip(self.bboxs, (self.w, self.h), flip_y=True)
            self.__is_flipped = True
            result = {IMAGE: flipped_image, LABEL: flipped_bboxes}
            ImageBoxAugmentation.check_correctness(result)
            return result

        return sample

    @property
    def has_flipped(self):
        """Remembers if the sample was flipped or not
        :return: true if sample has been flipped, false otherwise
        """
        # TODO: common super class should be added to remove code duplication of has_flipped in Vertical/Horizontal flip
        return self.__is_flipped

    def __repr__(self):
        """ Representation of Flip Object, including info on recent flip
        :return: string
        """
        format_string = self.__class__.__name__ + '('
        format_string += ("Flipped Vertically" if self.has_flipped else "Not Flipped")
        format_string += ') \n'
        return format_string


class DistortColor(ImageBoxAugmentation):
    """Distorts Color, manipulating hue, saturation and brightness(aka exposure in Darknet)
     Note that this class delegates to torchvision API to perform those operations
     Compatibility with Darknet:
        Hue and  Exposure have compatible performance with Darknet,
        Saturation does not

     Parameters
    ----------
        sample: dictionary with two fields defined by IMAGE and LABEL
        hue: amount of hue to add
        saturation: factor to increase/decrease saturation by
        exposure: factor to increase/ decrease exposure by

    Output
    -----------
         color distorted sample - dictionary with two fields defined by IMAGE and LABEL

    """

    def __init__(self, hue=None, saturation=None, exposure=None, pixel_max_value=int(MAX_PIXEL_VAL)):
        """Constructs DistortColor object
        :param hue: amount of hue to add
        :param saturation: the factor by which to decrease/ increase the saturation
        :param exposure: the factor by which to decrease/ increase the exposure
        :param pixel_max_value: the maximal value of a single RGB channel
        """
        super(DistortColor, self).__init__()
        self.saturation = saturation
        self.hue = hue
        self.exposure = exposure
        self.pixel_max_val = pixel_max_value

    def __call__(self, sample):
        """Performs color distortion operations
        :param sample: dictionary with two fields defined by IMAGE and LABEL
        :return: sample with color distorted IMAGE (LABEL is unaltered)
        """
        if self.hue == NO_CHANGE_IN_HUE and \
                self.saturation == NO_CHANGE_IN_SAT and \
                self.exposure == NO_CHANGE_IN_EXP:
            return sample

        super(DistortColor, self).__call__(sample)
        im = self.image
        if self.hue is not None and self.hue != NO_CHANGE_IN_HUE:
            im = transforms.functional.adjust_hue(im, self.hue)

        if self.saturation is not None and self.saturation != NO_CHANGE_IN_SAT:
            im = transforms.functional.adjust_saturation(im, self.saturation)

        if self.exposure is not None and self.exposure != NO_CHANGE_IN_EXP:
            im = transforms.functional.adjust_brightness(im, self.exposure)

        ImageBoxAugmentation.check_dynamic_range(im, self.pixel_max_val)
        return {IMAGE: im, LABEL: self.bboxs}


class DarknetDistortColor(ImageBoxAugmentation):
    """Distorts Color, manipulating hue, saturation and brightness(aka exposure in Darknet)
     Parameters
    ----------
        sample: dictionary with two fields defined by IMAGE and LABEL
        hue: amount of hue to add
        saturation: factor to increase/decrease saturation by
        exposure: factor to increase/ decrease exposure by

    Output
    -----------
         color distorted sample - dictionary with two fields defined by IMAGE and LABEL
    """

    def __init__(self, hue=None, saturation=None, exposure=None, pixel_max_value=int(MAX_PIXEL_VAL)):
        """Constructs DistortColor object
        :param hue: amount of hue to add
        :param saturation: the factor by which to decrease/ increase the saturation
        :param exposure: the factor by which to decrease/ increase the exposure
        :param pixel_max_value: the maximal value of a single RGB channel
        """
        super(DarknetDistortColor, self).__init__()
        self.saturation = saturation
        self.hue = hue
        self.exposure = exposure
        self.pixel_max_val = pixel_max_value

    def __call__(self, sample):
        """Performs color distortion operations
        :param sample: dictionary with two fields defined by IMAGE and LABEL
        :return: sample with color distorted IMAGE (LABEL is unaltered)
        """
        if self.hue == NO_CHANGE_IN_HUE and \
                self.saturation == NO_CHANGE_IN_SAT and \
                self.exposure == NO_CHANGE_IN_EXP:
            return sample

        super(DarknetDistortColor, self).__call__(sample)
        hsv_im = self.image.convert('HSV')
        im_channels = list(hsv_im.split())

        def change_hue(x):
            x += self.hue * self.pixel_max_val
            if x > self.pixel_max_val:
                x -= self.pixel_max_val
            if x < 0:
                x += self.pixel_max_val
            return x

        if self.hue is not None and self.hue != NO_CHANGE_IN_HUE:
            im_channels[H] = Image.eval(im_channels[H], change_hue)
        if self.saturation is not None and self.saturation != NO_CHANGE_IN_SAT:
            im_channels[S] = Image.eval(im_channels[S],
                                        lambda pixel: pixel * self.saturation)
        if self.exposure is not None and self.exposure != NO_CHANGE_IN_EXP:
            im_channels[V] = Image.eval(im_channels[V],
                                        lambda pixel: pixel * self.exposure)

        im = Image.merge(hsv_im.mode, tuple(im_channels))
        im = im.convert('RGB')
        ImageBoxAugmentation.check_dynamic_range(im, self.pixel_max_val)
        return {IMAGE: im, LABEL: self.bboxs}


class RandomDistort(ImageBoxAugmentation):
    """Randomly generates color distortion values and delegates the distortion
    to Distort objects
     Parameters
    ----------
        sample: dictionary with two fields defined by IMAGE and LABEL
        hue: the maximal possible hue
        saturation: maximum factor to increase/decrease saturation by
        exposure: maximum factor to increase/ decrease exposure by

    Output
    -----------
         color distorted sample - dictionary with two fields defined by IMAGE and LABEL
    """

    def __init__(self, hue=NO_CHANGE_IN_HUE, saturation=NO_CHANGE_IN_SAT,
                 exposure=NO_CHANGE_IN_EXP, pixel_max_value=int(MAX_PIXEL_VAL)):
        """Constructs RandomDistort object
        :param hue: the maximal possible hue (no limit on value)
        :param saturation: maximum factor to increase/decrease saturation by (no limit on value)
        :param exposure: maximum factor to increase/ decrease exposure by (no limit on value)
        :param pixel_max_value: maximum possible value per color channel
        """

        super(RandomDistort, self).__init__()
        if isinstance(hue, float):
            self.hue = hue
        else:
            raise ValueError("Hue should be float")
        if isinstance(saturation, float):
            self.saturation = saturation
        else:
            raise ValueError("Saturation should be float")
        if isinstance(exposure, float):
            self.exposure = exposure
        else:
            raise ValueError("Exposure should be float")

        self.pixel_max_value = pixel_max_value

    def __call__(self, sample):
        """Performs color distortion on the provided sample
        :param sample: dictionary with two fields defined by IMAGE and LABEL
        :return: sample with color distorted IMAGE (LABEL is unaltered)
        """

        def rand_scale(s):
            if s is None:
                return None
            scale = random.uniform(1, s)
            if random.randint(1, BIG_NUMBER) % 2:
                return scale
            return 1. / scale

        self.__rand_hue = random.uniform(-self.hue, self.hue)
        self.__rand_sat = rand_scale(self.saturation)
        self.__rand_exp = rand_scale(self.exposure)
        tv_distorter = DistortColor(self.__rand_hue,
                                    NO_CHANGE_IN_SAT,  # is not compatible with Darknet, hence omitted
                                    self.__rand_exp,
                                    pixel_max_value=self.pixel_max_value)
        dn_distorter = DarknetDistortColor(NO_CHANGE_IN_HUE,
                                           self.__rand_sat,
                                           NO_CHANGE_IN_EXP,
                                           pixel_max_value=self.pixel_max_value)

        return dn_distorter(tv_distorter(sample))

    @property
    def distortion(self):
        """ Remembers what was the last distortion applied
        :return: distortion parameter
        (amount of hue, factor of saturation, factor of exposure)
        """
        return self.__rand_hue, self.__rand_sat, self.__rand_exp


class CanvasAdapter(ImageBoxAugmentation):
    """Adapts the sample to the network Canvas/input image

     Parameters
    ----------
        size: the size of the output image
        default_pixel_val: color value to pad image
        dx: delta x from the origin (left of the image)
        dy: delta y from the origin (top of the image)

    Output
    -----------
         adapted sample - dictionary with two fields defined by IMAGE and LABEL
    """

    def __init__(self, size=DEF_CANVAS_SIZE, default_pixel_val=PADDING_COL, dx=0.0, dy=0.0):
        """Constructs CanvasAdapter object
        :param size: the size of the output image
        :param default_pixel_val: color value to pad image
        :param dx: delta x from the origin (left of the image)
        :param dy: delta y from the origin (top of the image)
        """
        super(CanvasAdapter, self).__init__()
        self.cw = size[0]
        self.ch = size[1]
        self.dx = dx
        self.dy = dy
        self.canvas = Image.new('RGB', size, default_pixel_val)

    def __call__(self, sample):
        """Performs crop of the image and pastes the cropped region on Canvas
        :param sample: dictionary with two fields defined by IMAGE and LABEL
        :return: sample adapted to the canvas
        """
        super(CanvasAdapter, self).__call__(sample)

        new_left = max(-self.dx, 0.0)
        new_top = max(-self.dy, 0.0)
        left = max(self.dx, 0.0)
        top = max(self.dy, 0.0)

        cropper1 = Crop([new_left, new_top, self.w - new_left, self.h - new_top], allow_outside_bb_center=True)
        cropper2 = Crop([0, 0, self.cw, self.ch], allow_outside_bb_center=True)
        cropped1 = cropper1(sample)
        cropped = cropper2(cropped1)
        self.canvas.paste(cropped[IMAGE].copy(), (int(left), int(top)))

        set_bboxes, mask = tbox.translate(self.bboxs, self.dx, self.dy, self.canvas.size)

        result = {IMAGE: self.canvas, LABEL: set_bboxes}
        ImageBoxAugmentation.check_correctness(result)
        return result


class ResizeToCanvas(ImageBoxAugmentation):
    """Resizes sample to fit the network Canvas/input image

     Parameters
    ----------
        size: the size of the output image
    Output
    -----------
         adapted sample - dictionary with two fields defined by IMAGE and LABEL
    """

    def __init__(self, size=DEF_CANVAS_SIZE):
        """Constructs the ResizeToCanvas object
        :param size: the size of the output Canvas/image (input to network)
        """
        super(ResizeToCanvas, self).__init__()
        self.cw = size[0]
        self.ch = size[1]

    def __call__(self, sample):
        """Resizes the image to fit the Canvas
        :param sample: dictionary with two fields defined by IMAGE and LABEL
        :return: sample resized to fit Canvas coordinates
        """
        super(ResizeToCanvas, self).__call__(sample)
        im_scale = float(min(self.cw, self.ch)) / float(max(self.h, self.w))
        resizer = Resize(scale_factor=im_scale)
        result = resizer(sample)
        ImageBoxAugmentation.check_correctness(result)
        return result


class PlaceOnCanvas(ImageBoxAugmentation):
    """Generates the location where to place the image on network Canvas/ input image

     Parameters
    ----------
        fixed_offset: true if the image should be centered on Canvas,
                    false if the image should be randomly placed on Canvas
        canvas_size: the size of the output image
        default_pixel_value: default padding value for padding the cropped region on Canvas

    Output
    -----------
        sample placed on Canvas - dictionary with two fields defined by IMAGE and LABEL
    """

    def __init__(self, fixed_offset=False, canvas_size=DEF_CANVAS_SIZE,
                 default_pixel_value=(int(MAX_PIXEL_VAL / 2.0),) * 3,
                 valid_num_bboxes=MIN_NUM_BBOXES):
        """Constructs PlaceOnCavas object
        :param fixed_offset: true if the image should be centered on Canvas,
                    false if the image should be randomly placed on Canvas
        :param canvas_size: the size of the output image
        :param default_pixel_value: default padding value for padding the cropped region on Canvas
        """
        super(PlaceOnCanvas, self).__init__(valid_num_bboxes)
        self.fixed_offset = fixed_offset
        self.canvas_size = canvas_size
        self.default_pixel_value = default_pixel_value

    def __call__(self, sample):
        """Places the sample within the limits of Canvas (NO resizing, just cropping)
        :param sample:  dictionary with two fields defined by IMAGE and LABEL
        :return: the sample placed on Canvas
        """
        super(PlaceOnCanvas, self).__call__(sample)
        dx, dy = self.__get_offset_on_canvas()
        adapter = CanvasAdapter(self.canvas_size, self.default_pixel_value, dx, dy)
        return adapter(sample)

    def __get_offset_on_canvas(self):
        """Helper to calculate the offset from origin for cropped region placement"""
        canvas_w, canvas_h = self.canvas_size
        if self.fixed_offset:
            return float(canvas_w - self.w) / 2.0, float(canvas_h - self.h) / 2.0
        return random.uniform(0, canvas_w - self.w), random.uniform(0, canvas_h - self.h)


class RandomizeBBoxes(ImageBoxAugmentation):
    """Randomizes bounding boxes (Caffe compatibility) and omits invlalid boxes
     if max_num_bboxes is provided, only LABEL with number of bounding boxes greater than
     max_num_bboxes will be randomized

     Parameters
    ----------
        min_num_bboxes_to_randomize (optional): integer number of maximum boxes to allow

    Output
    -----------
        sample  dictionary with IMAGE unaltered  and LABEL randomized,
    """

    def __init__(self, min_num_bboxes_to_randomize=None):
        """Constructor of RandomizeBBoxes object
        :param min_num_bboxes_to_randomize: minimum number of bounding boxes that requires
        randomization
        """
        super(RandomizeBBoxes, self).__init__()
        self.min_num_bboxes_to_randomize = min_num_bboxes_to_randomize

    def __call__(self, sample):
        """ Executes the randomization of boxes in sample
        :param sample:  dictionary with two fields defined by IMAGE and LABEL
        :return: the sample with bounding boxes randomized and invalid boxes dropped
        """
        super(RandomizeBBoxes, self).__call__(sample)
        self.__keep_only_valid_boxs()
        if self.min_num_bboxes_to_randomize is None or self.bboxs.shape[0] > self.min_num_bboxes_to_randomize:
            self.bboxs = np.random.permutation(self.bboxs)
        return {IMAGE: self.image, LABEL: self.bboxs}

    def __keep_only_valid_boxs(self):
        """ helper - validation of bbounding boxes, taken from Caffe"""
        mask = np.ones(self.bboxs.shape[0], dtype=bool)

        bboxs_xywh = tbox.to_xy_wh(self.bboxs, self.w, self.h)
        x_cent = bboxs_xywh[:, 0]
        y_cent = bboxs_xywh[:, 1]
        box_w = bboxs_xywh[:, 2]
        box_h = bboxs_xywh[:, 3]

        mask = np.logical_and(mask, x_cent <= 0.999)
        mask = np.logical_and(mask, x_cent > 0)
        mask = np.logical_and(mask, y_cent <= 0.999)
        mask = np.logical_and(mask, y_cent > 0)
        mask = np.logical_and(mask, box_w > -0.001)
        mask = np.logical_and(mask, box_h > -0.001)

        self.bboxs = self.bboxs[mask]


class SetBBoxesInRange(ImageBoxAugmentation):
    """Validates bounding boxes, sets bounding boxes in range of image and omits invlalid boxes
    
    Output
    -----------
        sample  dictionary with IMAGE unaltered  and LABEL validated,
    """

    def __init__(self):
        """Constructor of SetBBoxesInRange object
        """
        super(SetBBoxesInRange, self).__init__()

    def __call__(self, sample):
        """ Executes the validation of boxes in sample
        :param sample:  dictionary with two fields defined by IMAGE and LABEL
        :return: the sample with bounding boxes validated and invalid boxes dropped
        """
        super(SetBBoxesInRange, self).__call__(sample)
        self.__set_bboxs_inrange()
        return {IMAGE: self.image, LABEL: self.bboxs}

    def __set_bboxs_inrange(self):
        """ helper - validation of bbounding boxes, taken from Caffe"""
        self.bboxs[:, 0] = np.maximum(np.minimum(self.bboxs[:, 0], self.w), 0)
        self.bboxs[:, 2] = np.maximum(np.minimum(self.bboxs[:, 2], self.w), 0)
        self.bboxs[:, 1] = np.maximum(np.minimum(self.bboxs[:, 1], self.h), 0)
        self.bboxs[:, 3] = np.maximum(np.minimum(self.bboxs[:, 3], self.h), 0)


class InsufficientNumOfBBoxes(Exception):
    """Abstraction of Exception to signal Insufficient Number of Boxes in Sample"""
    pass


class SubtractMeans(object):
    """ Subtracts mean value from torch tensor image
    Delegates to torchvision API for normalization.
    May normalize the color value after mean removal (aka normalization)
    Note that the change is done IN PLACE

    Parameters
    ----------
        mean : mean value to subtract
        max_pixel_val: maximum value per color channel
        norm: normalization of color values

    Output
    -----------
        sample with IMAGE after subtraction and normalization (LABEL is unaltered)
    """

    def __init__(self, mean=COLOR_MEAN, max_pixel_val=MAX_PIXEL_VAL, norm=UNIT):
        """Constructs the SubtractMeans object
        :param mean: mean value to subtract
        :param max_pixel_val:  maximum value per color channel
        :param norm: normalization of color values
        """
        self.mean = tuple([mean_c / max_pixel_val for mean_c in mean])
        self.norm = tuple([n_c / max_pixel_val for n_c in norm])

    def __call__(self, sample):
        """Performs the subtraction of mean and possible normalization on sample
        :param sample: dictionary with two fields defined by IMAGE and LABEL
        :return: sample with IMAGE after subtraction and normalization (LABEL is unaltered)
        """
        if isinstance(sample, dict):
            sample[IMAGE] = transforms.functional.normalize(sample[IMAGE], self.mean, self.norm)
        else:
            sample = transforms.functional.normalize(sample, self.mean, self.norm)
        return sample


class ToDarknetLabels(object):
    """Convert  bounding boxes in sample to Darknet labels by flattening 
       Discards boxes beyond maximum allowed
    
    Parameters:
        num_bboxes: int, maximum boxes allowed

    Output
    -----------
        a dictionary with torch tensors for IMAGE and LABEL
        For example: IMAGE is torchvision tensor of 3 X 416 X 416
                    LABEL is torchvision tensor of 1 x 150
    """

    def __init__(self, num_bboxes=None):
        """Constructor of ToDarknetTensor object
        :param num_bboxes: number of boxes to include in label
        """
        self.num_bboxes = num_bboxes

    def __call__(self, sample):
        """Prepares sample to fit Darknet format
        :param sample: dictionary with PIL IMAGE and NUMPY.ARRAY LABEL
        :return: dictionary with torchvision tensors for IMAGE and LABEL
        """
        bboxes = tbox.to_xy_wh(sample[LABEL], sample[IMAGE].size[0], sample[IMAGE].size[1])
        labels = self._keep_max_num_bboxes(bboxes).flatten()
        return {IMAGE: sample[IMAGE],
                LABEL: labels}

    def _keep_max_num_bboxes(self, bboxes):
        """Discards boxes beyond num_bboxes"""
        if self.num_bboxes is not None:
            cur_num = bboxes.shape[0]
            diff_to_max = self.num_bboxes - cur_num
            if diff_to_max > 0:
                bboxes = np.lib.pad(bboxes, ((0, diff_to_max), (0, 0)),
                                    "constant", constant_values=(0.0,))
            elif diff_to_max < 0:
                bboxes = bboxes[:self.num_bboxes, :]
        return bboxes

class ToDarknetTensor(object):
    """Convert PIL image and numpy array of bounding boxes (if present) in sample
    to network Tensors.

    Output
    -----------
        a dictionary with torch tensors for IMAGE and LABEL
        For example: IMAGE is torchvision tensor of 3 X 416 X 416
                    LABEL is torchvision tensor of 1 x 150
    """

    def __init__(self):
        """Constructor of ToDarknetTensor object"""
        pass

    def __call__(self, sample):
        """Prepares sample to fit Darknet format
        :param sample: dictionary with PIL IMAGE and NUMPY.ARRAY LABEL
        :return: dictionary with torchvision tensors for IMAGE and LABEL
        """
        if isinstance(sample, dict):
            return {IMAGE: self.to_BGR(transforms.functional.to_tensor(sample[IMAGE])),
                    LABEL: torch.from_numpy(sample[LABEL])}

        return self.to_BGR(transforms.functional.to_tensor(sample))

    @staticmethod
    def to_BGR(img):
        """Transforms RGB into BGR
        :param img: torch tensor or numpy array image with 3 channels
        :return: image with the first and last channels swapped
        """
        assert (img.shape[0] == 3)
        return img[(2, 1, 0), :, :]

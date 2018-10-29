"""Bounding boxes transformation functions."""
from __future__ import division
import numpy as np
import math

__all__ = ['crop', 'flip', 'resize', 'translate']
CROP_BOX_DIM = 4
IMAGE_SPATIAL_DIM = 2


def __check_dim(value, dim, error_msg):
    """helper to check dimension validity """
    if not len(value) == dim:
        raise ValueError(
            "Invalid {} parameter, requires length {}, given {}".format(error_msg, dim, len(value)))


def check_crop_box_dim(crop_box):
    """helper to check crop region dimension validity """
    __check_dim(crop_box, CROP_BOX_DIM, "crop_box")
    if not len(crop_box) == CROP_BOX_DIM:
        raise ValueError(
            "Invalid crop_box parameter, requires length {}, given {}".format(CROP_BOX_DIM, len(crop_box)))


def check_size_dim(size, error_msg):
    """helper to check dimension validity """

def crop(bbox, crop_box=None, allow_outside_center=True):
    """Crop bounding boxes according to slice area.
    This method is mainly used with image cropping to ensure bonding boxes fit
    within the cropped image.
    Parameters
   ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    crop_box : tuple
        Tuple of length 4. :math:`(x_{min}, y_{min}, width, height)`
    allow_outside_center : bool
        If `False`, remove bounding boxes which have centers outside cropping area.
    Returns
    -------
    numpy.ndarray
        Cropped bounding boxes with shape (M, 4+) where M <= N.
    """
    
    if crop_box is None:
        return bbox

    check_crop_box_dim(crop_box)
    if sum([int(c is None) for c in crop_box]) == CROP_BOX_DIM:
        return bbox
    
    area_old = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    bbox = bbox.copy()

    l, t, w, h = crop_box
    left = l if l else 0
    top = t if t else 0
    right = left + (w if w else np.inf)
    bottom = top + (h if h else np.inf)
    crop_bbox = np.array((left, top, right, bottom))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        centers = (bbox[:, :2] + bbox[:, 2:4]) / 2
        mask = np.logical_and(crop_bbox[:2] <= centers, centers < crop_bbox[2:]).all(axis=1)

    # transform borders
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bbox[:2])
    bbox[:, 2:4] = np.minimum(bbox[:, 2:4], crop_bbox[2:4])
    bbox[:, :2] -= crop_bbox[:2]
    bbox[:, 2:4] -= crop_bbox[:2]
   
    area_new = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    mask = np.logical_and(mask, area_old * 0.5 <= area_new)
    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:4]).all(axis=1))
    bbox = bbox[mask]
    return bbox


def flip(bbox, size, flip_x=False, flip_y=False):
    """Flip bounding boxes according to image flipping directions.
    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    size : tuple
        Tuple of length 2: (width, height) to ensure compatibility with PIL images .
    flip_x : bool
        Whether flip horizontally.
    flip_y : bool
        Whether flip vertically.
    Returns
    -------
    numpy.ndarray
        Flipped bounding boxes with original shape.
    """

    check_size_dim(size, "size")

    width, height = size
    bbox = bbox.copy()

    if flip_y:
        ymax = height - bbox[:, 1]
        ymin = height - bbox[:, 3]
        bbox[:, 1] = ymin
        bbox[:, 3] = ymax
    if flip_x:
        xmax = width - bbox[:, 0]
        xmin = width - bbox[:, 2]
        bbox[:, 0] = xmin
        bbox[:, 2] = xmax

    return bbox


def resize(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize operation.
    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    in_size : tuple
        Tuple of length 2: (width, height) for input.
    out_size : tuple
        Tuple of length 2: (width, height) for output.
    Returns
    -------
    numpy.ndarray
        Resized bounding boxes with original shape.
    """
    check_size_dim(in_size, "in_size")
    check_size_dim(out_size, "out_size")
    bbox = bbox.astype("float")
    x_scale = out_size[0] / in_size[0]
    y_scale = out_size[1] / in_size[1]

    bbox[:, 0] = x_scale * bbox[:, 0]
    bbox[:, 2] = x_scale * bbox[:, 2]
    bbox[:, 1] = y_scale * bbox[:, 1]
    bbox[:, 3] = y_scale * bbox[:, 3]

    return bbox


def translate(bbox, x_offset=0, y_offset=0, size=None):
    """Translate bounding boxes by offsets.
    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    x_offset : int or float
        Offset along x axis.
    y_offset : int or float
        Offset along y axis.
    size: optional size limits that will contain the translated box
    Returns
    -------
    numpy.ndarray
        Translated bounding boxes with original shape.
    """

    bbox = bbox.copy()
    bbox[:, :2] += (x_offset, y_offset)
    bbox[:, 2:4] += (x_offset, y_offset)

    if size is not None:
        bbox[:, :2] = np.maximum(np.minimum(bbox[:, :2], np.array(size)), 0)
        bbox[:, 2:4] = np.maximum(np.minimum(bbox[:, 2:4], np.array(size)), 0)
    return bbox

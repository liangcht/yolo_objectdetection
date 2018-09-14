import cv2
import base64
import os
import os.path as op
import numpy as np


def img_from_file(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def img_to_base64(path):
    with open(path, 'r') as fp:
        return base64.b64encode(fp.read())


def img_from_base64(imagestring):
    if not imagestring:
        return None
    jpgbytestring = base64.b64decode(imagestring)
    nparr = np.frombuffer(jpgbytestring, np.uint8)
    # noinspection PyBroadException
    try:
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def im_rescale(im, target_size):
    im_size_max = max(im.shape[0:2])
    if target_size == im_size_max:
        return im
    im_scale = float(target_size) / float(im_size_max)
    im_resized = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
    return im_resized


VALID_IMAGE_TYPES = ['.jpg', '.jpeg', '.tiff', '.bmp', '.png']


def recursive_files_list(path, ext_list=None, ignore_prefixes=None):
    """Recursively search the path and create the dictionary
    :param path: Starting path to seach
    :param ext_list: list of file extensions we want
    :param ignore_prefixes: list of subdirectory prefixes to ignore descending into
    :rtype: list
    """
    assert op.isdir(path), "{} is not a directory".format(path)
    if isinstance(ignore_prefixes, basestring):
        ignore_prefixes = [ignore_prefixes]
    if ext_list is None:
        ext_list = VALID_IMAGE_TYPES
    dir_list = []
    file_list = []
    for fname in os.listdir(path):
        # Ignore all hidden files and folders
        if fname.startswith("."):
            continue
        fpath = op.join(path, fname)
        if op.isdir(fpath):
            if ignore_prefixes:
                ignore = False
                for prefix in ignore_prefixes:
                    if fname.startswith(prefix):
                        ignore = True
                        break
                if ignore:
                    continue
            dir_list.append(fpath)
            continue
        if op.isfile(fpath) and op.splitext(fpath)[1].lower() in ext_list:
            file_list.append(fpath)

    for fname in dir_list:
        file_list += recursive_files_list(
            fname,
            ext_list=ext_list, ignore_prefixes=ignore_prefixes
        )
    return file_list

import logging
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


def int_rect(rect):
    left, top, right, bot = rect
    return int(np.floor(left)), int(np.floor(top)), int(np.ceil(right)), int(np.ceil(bot))


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


def tile_rects(db, keys, key_rects, target_size, label, jpg_path):
    """Create single inverted file for a db
    :param db: the imdb to create
    :type db: ImageDatabase
    :param keys: keys in db to draw images from
    :param key_rects: rects associated with above keys
    :param target_size: patch size to align maximum dimension to
    :param label: current label
    :param jpg_path: output jpg path
    """
    
    rows = np.ceil(np.sqrt(len(keys)))
    cols = np.ceil(len(keys) / rows)
    collage = np.zeros((int(rows) * target_size, int(cols) * target_size, 3))

    target_size = target_size - 8
    # try to pack them in one pass
    max_h = 0  # maximum height in the current row
    max_x = 0  # maximum x seen
    x, y = 0, 0
    for key, rect in zip(keys, key_rects):
        im = db.image(key)
        left, top, right, bot = rect
        if bot <= top or right <= left:
            logging.error("Ignore Invalid ROI: {} for label: {} key: {}".format(rect, label, key))
            continue
        roi = im_rescale(im[top:bot, left:right], target_size)
        h, w = roi.shape[:2]
        x2 = x + w
        y2 = y + h
        if x2 > collage.shape[1]:
            # next row
            x = 0
            y += max_h + 4
            y2 = y + h
            x2 = x + w
            max_h = 0
        if x2 > max_x:
            max_x = x2
        if h > max_h:
            max_h = h
        
        collage[y:y2, x:x2] = roi

        x = x2 + 4
        

    # clip the collage
    collage = collage[:y + max_h, :max_x, :]
    if jpg_path:
        logging.info("Writing collage {}".format(jpg_path))
        cv2.imwrite(jpg_path, collage)

    return collage

import torch.utils.data as data
import multiprocessing as mp
from PIL import Image

try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

from mmod.imdb import ImageDatabase

READ_WITH_OPENCV = True
if READ_WITH_OPENCV:
    import numpy as np
    import cv2

# Dictionary key
IMAGE = "image"  # TODO: change to get that as parameter from prototxt
LABEL = "bbox"   # TODO: change to get that as parameter from prototxt

_cur_data = None


class ImdbData(data.Dataset):
    """ This class encapsulates access to the database,
    retrieves of a sample that corresponds to an input index from the database,
    applies necessary transforms
    and returns the sample
    """

    def __init__(self, path,
                 transform=None, labeler=None, predict_phase=False):
        """Constructor of ImdbData
        :param path: string - path to the dataset with images and labels
        :param transform: see Transforms.py - transform to be applied on sample
        :param labeler: see tbox_utils.py - object responsible for creating labels for sample
        """
        self._path = path
        self.transform = transform
        self.labeler = labeler
        self.predict_phase = predict_phase

    def __repr__(self):
        return 'ImdbData({}, size={})'.format(self._path, len(self))

    def __getitem__(self, index):
        imdb = self.imdb
        key = imdb.normkey(index)
        if READ_WITH_OPENCV:
            img = imdb.image(key)
            correct_img = img[:, :, (2, 1, 0)] # BGR to RGB 
            img = Image.fromarray(correct_img, mode='RGB')  # save in PIL format
        else:
            raw = imdb.raw_image(index)
            img = Image.open(StringIO(raw)).convert('RGB')
        if self.predict_phase:
            w, h = img.size
        if self.labeler is not None:
            label = self.labeler(imdb.truth_list(key), imdb.cmap)
            sample = {
                IMAGE: img,
                LABEL: label
            }
        else:
            sample = img
        if self.transform is not None:
            sample = self.transform(sample)
        if isinstance(sample, dict):  # typical for training
            return sample[IMAGE], sample[LABEL]
        if self.predict_phase:  # typical for prediction
            return sample, imdb.uid(key), imdb.image_key(key), h, w
        return sample

    def __len__(self):
        return len(self.imdb)
    
    @property
    def imdb(self):
        proc = mp.current_process()
        pid = proc.pid
        opid = None
        global _cur_data
        if _cur_data:
            opid = _cur_data[0]
            if opid == pid:
                return _cur_data[1]
            _cur_data = None
        # this has to be print because this could be in another process
        print("ImdbData process: {} ({}{})".format(
            proc.name,
            "" if not opid else "{}->".format(opid),
            pid
        ))
        imdb = ImageDatabase(self._path)
        assert len(imdb), "No images found in: {}".format(imdb)
        imdb.open_db()
        _cur_data = [pid, imdb]
        return imdb


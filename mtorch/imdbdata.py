import torch.utils.data as data
from PIL import Image
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

from mmod.imdb import ImageDatabase
from torchvision import transforms
import threading

# Dictionary key
IMAGE = "image" # TODO: change to get that as parameter from prototxt
LABEL = "bbox"   # TODO: change to get that as parameter from prototxt

cur_data = threading.local()


class ImdbData(data.Dataset):
    """ This class encapsulates access to the database,
    retrieves of a sample that corresponds to an input index from the database,
    applies necessary transforms
    and returns the sample
    """

    def __init__(self, path,
                 transform=None, labeler=None):
        """Constructor of ImdbData
        :param path: string - path to the dataset with images and labels
        :param transform: see Transforms.py - transform to be applied on sample
        :param labeler: see tbox_utils.py - object responsible for creating labels for sample
        """
        self._path = path
        self.transform = transform
        self.labeler = labeler
        self.path = path
        assert len(self), "No images found in: {}".format(self.imdb)

    def __repr__(self):
        return 'ImdbData({}, size={})'.format(self._path, len(self))

    def __getitem__(self, index):
        imdb = self.imdb
        key = imdb.normkey(index)
        raw = imdb.raw_image(index)
        img = Image.open(StringIO(raw)).convert('RGB')
        if self.labeler is not None:
            label = self.labeler(imdb.truth_list(key), imdb.cmap)
            sample = {}
            sample[IMAGE] = img
            sample[LABEL] = label
            self.img = img
            self.label = label
        else:
            sample = img
        if self.transform is not None:
            sample = self.transform(sample)
        if isinstance(sample, dict): 
            return sample[IMAGE], sample[LABEL]
        return sample

    def __len__(self):
        return len(self.imdb)
    
    @property
    def imdb(self):
        global cur_data
        imdb = getattr(cur_data, 'imdb', None)
        if imdb is None:
            imdb = cur_data.imdb = ImageDatabase(self.path)
            print(threading.current_thread().name)
        return imdb
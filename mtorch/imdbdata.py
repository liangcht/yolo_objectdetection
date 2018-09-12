import torch.utils.data as data
from PIL import Image
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

from mmod.imdb import ImageDatabase


class ImdbData(data.Dataset):
    def __init__(self, path,
                 transform=None, target_transform=None):
        self._path = path
        self.transform = transform
        self.target_transform = target_transform
        self.imdb = ImageDatabase(path)
        self.imdb.open_db()  # open the db to keep the file objects open once
        assert len(self), "No images found in: {}".format(self.imdb)

    def __repr__(self):
        return 'ImdbData({}, size={})'.format(self._path, len(self))

    def __getitem__(self, index):
        key = self.imdb.normkey(index)
        raw = self.imdb.raw_image(index)
        target = self.imdb.truth_list(key)
        sample = Image.open(StringIO(raw)).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.imdb)

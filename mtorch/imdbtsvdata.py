from PIL import Image
import json
from qd.qd_common import img_from_base64
from qd.qd_pytorch import TSVSplit

# Dictionary key
IMAGE = "image"  # TODO: change to get that as parameter from prototxt
LABEL = "bbox"   # TODO: change to get that as parameter from prototxt


class ImdbTSVData(TSVSplit):
    """ This class encapsulates access to the database,
    retrieves of a sample that corresponds to an input index from the database,
    applies necessary transforms
    and returns the sample
    """

    def __init__(self, path, cmapfile=None,
                 transform=None, labeler=None, predict_phase=False):
        """Constructor of ImdbData
        :param path: string - colon seperated data:split:version. if version is
        if version is missing, it is 0
        :param transform: see Transforms.py - transform to be applied on sample
        :param labeler: see tbox_utils.py - object responsible for creating labels for sample
        """
        path_splits = path.split('$')
        self.data = path_splits[0]
        self.split = path_splits[1]
        self.version = int(path_splits[2]) if len(path_splits) == 3 else 0

        super(ImdbTSVData, self).__init__(data=self.data,
                split=self.split, version=self.version, cache_policy=None)

        self.transform = transform
        self.labeler = labeler
        assert labeler is None
        self.predict_phase = predict_phase
        self.cmapfile=cmapfile
        self.cmap = None

    def get_cmap(self):
        if self.cmap is None:
            from qd.qd_common import load_list_file
            self.cmap = load_list_file(self.cmapfile)
        return self.cmap

    def __repr__(self):
        return '{}, {}, version = {}'.format(self.data, self.split,
                self.version)

    def __getitem__(self, index):
        key, str_label, str_image = super(ImdbTSVData, self).__getitem__(index)

        img = img_from_base64(str_image)
        correct_img = img[:, :, (2, 1, 0)] # BGR to RGB
        img = Image.fromarray(correct_img, mode='RGB')  # save in PIL format
        if self.predict_phase:
            w, h = img.size
        if self.labeler is not None:
            label = self.labeler(json.loads(str_label), self.cmap)
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
            return sample, key, key, h, w
        return sample

    def __len__(self):
        return super(ImdbTSVData, self).__len__()


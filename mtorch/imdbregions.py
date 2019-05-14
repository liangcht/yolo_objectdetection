import torch.utils.data as data
import multiprocessing as mp
from PIL import Image

from mtorch.tbox_utils import region_crop
from mmod.imdb import ImageDatabase


READ_WITH_OPENCV = True
if READ_WITH_OPENCV:
    import numpy as np

_cur_data = None
_cur_regions = None


def _get_labels_hist(imdb, normalize=False):
    labels_hist = []
    for i, label in enumerate(imdb.iter_cmap()):
        keys = list(imdb.iter_label(label))
        labels_hist.append(len(keys))
    if normalize:
        total_labels = sum(labels_hist)
        for i in range(len(labels_hist)):
            labels_hist[i] /= float(total_labels)

    return labels_hist


def _get_upsampling_factor(imdb):
    labels_hist = _get_labels_hist(imdb)
    max_num = max(labels_hist)
    upsample_factor = []
    for count in labels_hist:
        upsample_factor.append(int(np.ceil(float(max_num) / float(count))) if count > 0 else 1)

    return upsample_factor


class ImdbRegions(data.Dataset):
    """ This class encapsulates access to the database of image regions,
    retrieves of a sample that corresponds to an input index from the database,
    applies necessary transforms
    and returns the sample
    # TODO: abstract out common code with IMDBData
    """

    def __init__(self, path, region_cropper, cmapfile=None,
                 transform=None, labeler=None, predict_phase=False):
        """Constructor of ImdbData
        :param path: string - path to the dataset with images and labels
        :param transform: see Transforms.py - transform to be applied on sample
        :param labeler: see tbox_utils.py - object responsible for creating labels for sample
        """
        self._path = path
        self.region_cropper = region_cropper
        self.transform = transform
        self.labeler = labeler
        self.predict_phase = predict_phase
        self.cmapfile = cmapfile

    def __repr__(self):
        return 'ImdbRegions({}, size={})'.format(self._path, len(self.regions))

    def __getitem__(self, index):
        imdb = self.imdb
      
        regions = self.regions
        key = regions[index]["key"]
        bbox = regions[index]["region"]
        crop_box = [float(val) for val in bbox['rect']]
        if READ_WITH_OPENCV:
            img = imdb.image(key)
            correct_img = img[:, :, (2, 1, 0)]  # BGR to RGB
            img = Image.fromarray(correct_img, mode='RGB')  # save in PIL format            
        else:
            raw = imdb.raw_image(index)
            img = Image.open(StringIO(raw)).convert('RGB')
        w, h = img.size

        region = region_crop(img, crop_box)

        if self.transform is not None:
            region = self.transform(region)

        if self.labeler:
            label = int(self.labeler(bbox, self.imdb.cmap))
            sample = (region, label)
        else:
            sample = region
        
        if self.predict_phase:  # typical for prediction
            return sample, imdb.uid(key), imdb.image_key(key), bbox['rect']
        return sample

    def __len__(self):
        return len(self.regions)
    
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
        imdb = ImageDatabase(self._path, cmapfile=self.cmapfile)
        assert len(imdb), "No images found in: {}".format(imdb)
        imdb.open_db()
        _cur_data = [pid, imdb]
        return imdb

    @property
    def regions(self):
        proc = mp.current_process()
        pid = proc.pid
        opid = None
        global _cur_regions
        if _cur_regions:
            opid = _cur_regions[0]
            if opid == pid:
                return _cur_regions[1]
            _cur_regions = None
        # this has to be print because this could be in another process
        if not self.predict_phase:
            up_factors = _get_upsampling_factor(self.imdb)
        else:
            up_factors = None
        regions, labels_hist = self._get_region_proposals(up_factors)
        _cur_regions = [pid, regions]
        return regions

    def _get_region_proposals(self, up_factors):
        region_proposals = []
        cmap = self.imdb.cmap
        label_count = [0] * len(cmap)
       
        for key in self.imdb:
            regions = self.region_cropper(self.imdb.truth_list(key))
            for region in regions:
                if up_factors and region["class"] != 'Unknown' and region["class"] != 'Background':        
                    up_factor = up_factors[cmap.index(region["class"])]
                else:
                    up_factor = 1
                label_count[cmap.index(region["class"])] += 1
                for i in range(up_factor):
                    region_proposals.extend([{"key": key, "region": region}])
            
        return region_proposals, label_count

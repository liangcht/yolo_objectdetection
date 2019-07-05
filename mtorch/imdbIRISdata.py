import io
import os
import sys
import zipfile
import struct
import threading
import torch
import torch.utils.data
from PIL import Image
from PIL import ImageFile
import numpy as np
import pdb

# Dictionary key
IMAGE = "image"  # TODO: change to get that as parameter from prototxt
LABEL = "bbox"   # TODO: change to get that as parameter from prototxt


class SimpleZipFile(object):
    def __init__(self, filename):
        self.zip_filename = filename
        self.name_to_info = {}
        self.fp = None

        with zipfile.ZipFile(filename, 'r', zipfile.ZIP_STORED) as zip_file:
            # Verify that the zip file is supported.
            for z in zip_file.infolist():
                # All files must not be compressed.
                # All files must not be encrypted.
                if z.compress_type != zipfile.ZIP_STORED or z.flag_bits & 0x01:
                    raise NotImplementedError('this zip file type is not supported')

                assert z.file_size == z.compress_size
                self.name_to_info[z.filename] = (z.header_offset, z.compress_size)

    def open(self, filename):
        if filename not in self.name_to_info:
            raise KeyError('There is no item named {} in the archive'.format(filename))

        if not self.fp:
            self.fp = open(self.zip_filename, 'rb')

        header_offset, compress_size = self.name_to_info[filename]

        self.fp.seek(header_offset)
        fheader = self.fp.read(zipfile.sizeFileHeader)
        if len(fheader) != zipfile.sizeFileHeader:
            raise zipfile.BadZipFile('Truncated file header')
        fheader = struct.unpack(zipfile.structFileHeader, fheader)
        if fheader[zipfile._FH_SIGNATURE] != zipfile.stringFileHeader:
            raise zipfile.BadZipFile('Bad magic number for file header')
        self.fp.read(fheader[zipfile._FH_FILENAME_LENGTH])
        if fheader[zipfile._FH_EXTRA_FIELD_LENGTH]:
            self.fp.read(fheader[zipfile._FH_EXTRA_FIELD_LENGTH])

        return io.BytesIO(self.fp.read(compress_size))

    def close(self):
        if self.fp:
            self.fp.close()
            self.fp = None


class ImageDataset(torch.utils.data.Dataset):
    """Load images which are listed in a txt file.
    This dataset supports ZIP container. example: dataset.zip@4.jpg

    The dataset txt file should have 2 columns. The 1st column is the filename, and the 2nd column is the label id.

    Supported label types:
      - Single integer
      - OD file labels
    """
    def __init__(self, images_list_filepath, transform=None, predict_phase=False):
        self.images_filepaths = []
        self.base_dir = os.path.dirname(images_list_filepath)
        self.zip_objects = {}
        self.transform = transform
        self.predict_phase=predict_phase
        print("IRIS imdb phase {}".format(predict_phase))

        with open(images_list_filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                columns = line.strip().split()
                assert len(columns) == 2

                # Load Zip file. Use the normal ZipFile class for labels.
                if '@' in columns[1].strip():
                    zip_filepath, _ = columns[1].split('@')
                    if not os.path.isabs(zip_filepath):
                        zip_filepath = os.path.join(self.base_dir, zip_filepath)
                    if zip_filepath not in self.zip_objects:
                        self.zip_objects[zip_filepath] = zipfile.ZipFile(zip_filepath)

                target = self._load_target(columns[1].strip())
                self.images_filepaths.append((columns[0].strip(), target))

        for filepath, label in self.images_filepaths:
            if '@' in filepath:
                zip_filepath, image_filepath = filepath.split('@')
                if not os.path.isabs(zip_filepath):
                    zip_filepath = os.path.join(self.base_dir, zip_filepath)
                if zip_filepath not in self.zip_objects:
                    self.zip_objects[zip_filepath] = SimpleZipFile(zip_filepath)

    def __getitem__(self, index):
        image_filepath, target = self.images_filepaths[index]
        image = None
        try:
            image = self._load_image(image_filepath)
        except Exception as e:
            print("Failed to load an image: {}".format(image_filepath))
            sys.stdout.flush()
            raise e

        if self.predict_phase:
            sample=image
            sample = self.transform(sample)
            w, h = image.size
            import pdb
            pdb.set_trace()
            return sample, index, h, w, target
        else:
            # Convert absolute coordinates to (x1, y1, x2, y2)
            abs_target = [None] * len(target)
            for i, t in enumerate(target):
                abs_target[i] = [t[1], t[2], t[3], t[4], t[0]]
            target = np.array(abs_target)
            sample = {IMAGE: image, LABEL:target}
            sample = self.transform(sample)
            return sample[IMAGE], sample[LABEL]

    def __len__(self):
        return len(self.images_filepaths)

    def _load_target(self, target):
        path = os.path.join(self.base_dir, target)
        return self._load_od_labels(path)

    def _load_od_labels(self, filepath):
        if '@' in filepath:
            zip_filepath, label_filepath = filepath.split('@')
            with self.zip_objects[zip_filepath].open(label_filepath) as f:
                return self._load_od_label_file(f)
        else:
            with open(filepath, 'r') as f:
                return self._load_od_label_file(f)

    def _load_od_label_file(self, f):
        labels = []
        for line in f:
            label, x0, y0, x1, y1 = line.split()
            label = int(label)
            x0 = float(x0)
            y0 = float(y0)
            x1 = float(x1)
            y1 = float(y1)
            labels.append((label, x0, y0, x1, y1))
        return labels

    def _load_image(self, filepath):
        if "@" in filepath:
            zip_filepath, image_filepath = filepath.split("@")
            if not os.path.isabs(zip_filepath):
                zip_filepath = os.path.join(self.base_dir, zip_filepath)

            # Work around for corrupted files in datasets
            ImageFile.LOAD_TRUNCATED_IMAGES = True

            with self.zip_objects[zip_filepath].open(image_filepath) as f:
                image = Image.open(f)
                return image.convert('RGB')
        else:
            if not os.path.isabs(filepath):
                filepath = os.path.join(self.base_dir, filepath)

            with open(filepath, 'rb') as f:
                # Work around for corrupted files in datasets
                ImageFile.LOAD_TRUNCATED_IMAGES = True

                image = Image.open(f)
                return image.convert('RGB')

import os.path as op
import logging
import base64
import json
import numpy as np
from contextlib import contextmanager

from mmod.simple_parser import tsv_data_sources, read_model_proto, softmax_tree_path, parse_key_value
from mmod.tsv_file import TsvFile
from mmod.im_utils import img_from_base64, img_from_file, recursive_files_list, VALID_IMAGE_TYPES
from mmod.utils import tsv_read, file_cache, FileCache, splitfilename


class ImageDatabase(object):
    _IGNORE_ATTRS = {'_cache', '_cached_truth', '_image_keys'}

    _DB_TYPE_DIRECTORY = 1  # a directory with jpg files
    _DB_TYPE_PROTOTXT = 2   # a prototxt with data
    _DB_TYPE_TSV = 3        # a TSV file
    _DB_TYPE_IMAGE = 4      # a single image file

    def __init__(self, path, raise_error=False, name=None):
        self._index = None  # type: TsvFile or list
        self._cache = None  # type: FileCache
        self._type = None
        self._raise_error_invalid_image = raise_error
        self._name = name  # type: str
        self._tax_path = None  # type: str
        self._cmapfile = None  # type: str
        self._cached_truth = None  # type: dict
        self._image_keys = {}
        if op.isdir(path):
            self._type = self._DB_TYPE_DIRECTORY
        else:
            ext = op.splitext(path)[1].lower()
            if ext == ".prototxt":
                self._type = self._DB_TYPE_PROTOTXT
            elif ext == ".tsv":
                self._type = self._DB_TYPE_TSV
            elif ext in VALID_IMAGE_TYPES:
                self._type = self._DB_TYPE_IMAGE
            assert op.isfile(path), "file does not exit at: {}".format(path)

        self._path = op.normpath(path).replace("\\", "/")  # type: str
        assert self._type, "Invalid db at {}".format(path)
        if self._type == self._DB_TYPE_IMAGE:
            self._index = [path]
        elif self._type == self._DB_TYPE_DIRECTORY:
            self._index = recursive_files_list(path, ignore_prefixes=["vis"])
        elif self._type == self._DB_TYPE_PROTOTXT:
            model = read_model_proto(path)
            self._tax_path = softmax_tree_path(model)
            sources, labels = tsv_data_sources(model)
            self._cmapfile = parse_key_value(path, 'labelmap')

            self._index = TsvFile(sources,
                                  labels=labels,
                                  cmapfiles=self._cmapfile)
        elif self._type == self._DB_TYPE_TSV:
            self._index = TsvFile(path)

        assert self._index, "No images found for {}".format(self)

    def __repr__(self):
        index_str = ", size: {}".format(len(self))
        if isinstance(self._index, TsvFile):
            index_str = ", index: {}".format(self._index)
        return 'ImageDatabase(type: {}{})'.format(
            self.type, index_str
        )

    def __getstate__(self):
        return {
            key: val if key not in self._IGNORE_ATTRS else None
            for key, val in self.__dict__.iteritems()
        }

    def __getitem__(self, key):
        """return image as an array
        :rtype: numpy.ndarray
        """
        if self.is_directory or self.is_image:
            key = self.normkey(key)
            im = img_from_file(key)
            if self._raise_error_invalid_image:
                if im is None:
                    raise RuntimeError("file: {} is invalid".format(key))
            return im
        if self._type == self._DB_TYPE_PROTOTXT or self._type == self._DB_TYPE_TSV:
            _, b64str = self._read_tsv(key)
            return img_from_base64(b64str)
        assert False, "Invalid type: {}".format(self._type)

    def __contains__(self, key):
        if (self.is_directory or self.is_image) and isinstance(key, (int, long, np.integer)):
            return -len(self._index) <= key < len(self._index)
        return key in self._index

    def __iter__(self):
        """Iterate over all the keys in the index
        :rtype: tuple[str, int, int]
        """
        for key in self._index:
            yield key

    def __len__(self):
        return len(self._index)

    def __eq__(self, other):
        if not isinstance(other, ImageDatabase):
            return NotImplemented

        return self._path == other._path

    def __ne__(self, other):
        return not self == other

    def iteritems(self):
        for key in self:
            yield key, self[key]

    def iter_cmap(self, source=None):
        """Iterate through all the classes
        """
        if self.is_directory or self.is_image:
            raise NotImplementedError("truth not implemented for db: '{}'".format(self.type))

        # if there is no cmapfile nor inverted file
        if not self._index.has_cmapfile and not self._index.has_inverted:
            logging.warn("Using truth to iterate cmap for {}".format(self))
            # use the truth
            for label in self.all_truths(cache_truth=True):
                yield label
            return
        for label in self._index.iter_cmap(source=source):
            yield label

    @property
    def cmap(self):
        """Class-mapping list
        :rtype: list
        """
        return list(self.iter_cmap())

    @property
    def cmapfile(self):
        """Path to the *single* labelmap file (may or may not exist)
        :rtype: str
        """
        if self._cmapfile:
            return self._cmapfile
        if self.is_directory:
            return op.join(self.path, 'labelmap.txt')
        return op.join(op.dirname(self.path), 'labelmap.txt')

    @property
    def inverted_path(self):
        """Path to the *single* inverted index file (may or may not exist)
        :rtype: str
        """
        if self.is_directory or self.is_image:
            raise NotImplementedError("not implemented for db: '{}'".format(self.type))
        return splitfilename(self.path, 'inverted.label', is_composite=self._index.is_composite)

    @property
    def shuffle_path(self):
        """Path to the shuffle file (may or may not exist)
        :rtype: str
        """
        if self.is_directory or self.is_image:
            raise NotImplementedError("not implemented for db: '{}'".format(self.type))
        return splitfilename(self.path, 'shuffle.txt', is_composite=self._index.is_composite, keep_ext=False)

    @property
    def is_inverted(self):
        """If we have inverted file for label iteration
        """
        if self.is_directory or self.is_image:
            raise NotImplementedError("not implemented for db: '{}'".format(self.type))
        if self._index.composite_non_inverted:
            return False
        return self._index.has_inverted

    def source_index(self, source):
        """Find the source index
        :param source: source data source to limit the labels from
        :type source: str
        :rtype: int
        """
        if self.is_directory or self.is_image:
            raise NotImplementedError("not implemented for db: '{}'".format(self.type))
        return self._index.source_index(source)

    def source(self, source_idx):
        """Find the source
        :param source_idx: source index
        :type source_idx: int
        :rtype: str
        """
        if self.is_directory or self.is_image:
            raise NotImplementedError("not implemented for db: '{}'".format(self.type))
        return self._index.source(source_idx)

    def iterate_inverted(self, source=None):
        """Iterate inverted label (without any existing inverted file)
        :param source: source data source to limit iterations
        :type source: str
        :rtype: (str, str, list[int])
        """
        for label, truth in self.all_truths(cache_truth=True).iteritems():
            source_lines = {}
            for uid in truth:
                src, line = json.loads(uid)
                if source and src != source:
                    continue
                if src not in source_lines:
                    source_lines[src] = []
                source_lines[src].append(line)
            for src, lines in source_lines.iteritems():
                yield label, src, lines

    def iter_label(self, class_label, source=None, tax=None):
        """Iterate over keys for a class label (truth)
        :param class_label: class label to find in the index
        :type class_label: str
        :param source: source data source to limit the labels from
        :type source: str
        :param tax: taxonomy to use for on-the-fly translation
        :type tax: mmod.taxonomy.Taxonomy
        :rtype: tuple[str, int, int]
        """
        if self.is_directory or self.is_image:
            raise NotImplementedError("truth not implemented for db: '{}'".format(self.type))

        if (self._index.composite_non_inverted and not tax) or (not self._index.has_inverted):
            class_label = class_label.lower()  # we cache and search the lower form
            logging.warn("Iterate label {} using the truth without any inverted file".format(class_label))
            for label, source, lines in self.iterate_inverted(source=source):
                if label.lower() != class_label:
                    # TODO: try using tax (to translate) if given
                    continue
                for line in lines:
                    key = source, line
                    key = self._index[key]
                    yield key
            return
        for key in self._index.iter_label(class_label, source=source, tax=tax):
            yield key

    def iter_label_items(self, class_label, source=None, tax=None):
        """Iterate over keys and values for a class label (truth)
        :param class_label: class label to find in the index
        :type class_label: str
        :param source: source data source to limit the labels from
        :type source: str
        :param tax: taxonomy to use for on-the-fly translation
        :type tax: mmod.taxonomy.Taxonomy
        :rtype: tuple[tuple[str, int, int], str]
        """
        with file_cache() as cache:
            for key in self.iter_label(class_label, source=source, tax=tax):
                label_file, label_offset = self._index.label_offset(key)
                fp = cache.open(label_file)
                cols = tsv_read(fp, 2, seek_offset=label_offset)
                if len(cols) != 2:
                    logging.warning("label for {} is ignored".format(key))
                    continue
                yield key, cols[1]

    def iter_filtered_label_items(self, class_label, source=None, tax=None):
        """Iterate over keys and values for a class label (truth)
        Filter (and translate) the truth to those in the class-mapping
        :param class_label: class label to find in the index
        :type class_label: str
        :param source: source data source to limit the labels from
        :type source: str
        :param tax: taxonomy to use for on-the-fly translation
        :type tax: mmod.taxonomy.Taxonomy
        :return: key, filtered_in_rects, filtered_out_rects
        :rtype: tuple[tuple[str, int, int], list[dict], list[dict]]
        """
        if self.is_directory or self.is_image:
            raise NotImplementedError("truth not implemented for db: '{}'".format(self.type))
        for key, truth in self.iter_label_items(class_label, source=source, tax=tax):
            source, _, _ = key
            datasource = op.basename(op.dirname(source))
            rects = json.loads(truth)
            in_rects = []
            out_rects = []
            for rect in rects:
                class_label = rect['class']
                for label in tax.translate_from(class_label, datasource=datasource):
                    if not label:
                        out_rects.append(rect)
                        break
                    rect['class'] = label  # use the translated class
                    if label != class_label:
                        rect['source_class'] = class_label
                    in_rects.append(rect)
                    break
            yield key, in_rects, out_rects

    def iter_sources(self):
        """Iterate data sources
        :rtype: str
        """
        if self.is_directory or self.is_image:
            yield self.path
            return

        for source in self._index.iter_sources():
            yield source

    def iter_source_range(self):
        """Iterate data sources
        :rtype: str
        """
        if self.is_directory or self.is_image:
            raise NotImplementedError("not implemented for db: '{}'".format(self.type))

        for source, lrng in self._index.iter_source_range():
            yield source, lrng

    @property
    def type(self):
        if self._type == self._DB_TYPE_DIRECTORY:
            return 'directory'
        if self._type == self._DB_TYPE_PROTOTXT:
            return 'prototxt'
        if self._type == self._DB_TYPE_TSV:
            return 'tsv'
        if self._type == self._DB_TYPE_IMAGE:
            return 'image'
        assert False, "Invalid type: {}".format(self._type)

    @property
    def is_image(self):
        return self._type == self._DB_TYPE_IMAGE

    @property
    def is_directory(self):
        return self._type == self._DB_TYPE_DIRECTORY

    @property
    def is_prototxt(self):
        return self._type == self._DB_TYPE_PROTOTXT

    @property
    def is_tsv(self):
        return self._type == self._DB_TYPE_TSV

    @property
    def name(self):
        """Image database name
        """
        if self._name:
            return self._name
        base = op.basename(self.path)
        self._name = op.join(op.basename(op.dirname(self.path)), base).replace('\\', '_').replace('/', '_')
        return self._name

    @property
    def path(self):
        """Full path of the db
        """
        return self._path

    @property
    def tax_path(self):
        """Path to where thetaxonomy may reside
        """
        return self._tax_path or self._path

    @contextmanager
    def open(self):
        if self.is_directory or self.is_image:
            yield
            return

        try:
            self._index.open()
            yield
        finally:
            self._index.close()

    def open_db(self):
        if self.is_directory or self.is_image:
            return
        self._index.open()

    def close(self):
        if self.is_directory or self.is_image:
            return
        self._index.close()

    def is_open(self):
        if self.is_directory or self.is_image:
            return True
        return self._index.is_open()

    def _read_tsv(self, key):
        assert isinstance(self._index, TsvFile)
        tsv_path, offset = self._index.offset(key)
        tsv_in = self._index.open(tsv_path)
        tsv_in.seek(offset)
        cols = [x.strip() for x in tsv_in.readline().split("\t")]
        if len(cols) < 1:
            return None, None
        if len(cols) == 2:
            return cols[0], cols[1]
        if len(cols) < 2:
            return cols[0], None
        return cols[0], cols[2]

    def normkey(self, key):
        """Convert any key to the normalized key
        When data is external always use this to make consequitive access faster
        :type key: tuple[str, int, int] | tuple[str, int] | int | str
        :rtype: tuple[str, int, int] | str
        """
        if self.is_directory or self.is_image:
            if key not in self:
                raise KeyError("{} file not in the db".format(key))
            if isinstance(key, basestring):
                return key
        return self._index[key]

    def base64(self, key):
        """return image in base64 format
        If the format is already in base64 avoids conversion
        :rtype: str
        """
        if self.is_directory or self.is_image:
            key = self.normkey(key)
            with open(key, 'r') as fp:
                return base64.b64encode(fp.read())
        _, b64str = self._read_tsv(key)
        return b64str

    def image(self, key):
        """return image as an array
        :rtype: numpy.ndarray
        """
        return self[key]

    def raw_image(self, key):
        """return raw image file content
        """
        if self.is_directory or self.is_image:
            key = self.normkey(key)
            with open(key, 'r') as fp:
                return fp.read()
        _, b64str = self._read_tsv(key)
        return base64.b64decode(b64str)

    def uid(self, key):
        """Return the unique id of the key
        :rtype: str
        """
        if self.is_directory or self.is_image:
            return self.normkey(key)
        return self._index.uid(key)

    def uid_of_image_key(self, image_key):
        """Return the unique id of the key
        :rtype: str
        """
        if self.is_directory or self.is_image:
            return self.normkey(image_key)
        if self._image_keys:
            return self._image_keys[image_key]
        logging.info("Create UID to Image Key mapping in {}".format(self))
        for key in self:
            self._image_keys[self.image_key(key)] = self.uid(key)
        return self.uid_of_image_key(image_key)

    def image_key(self, key):
        """Return the image key for a key
        :rtype: str
        """
        if self.is_directory or self.is_image:
            return self.normkey(key)
        tsv_path, offset = self._index.label_offset(key)
        if not self._cache:
            self._cache = FileCache()

        tsv_in = self._cache.open(tsv_path)
        cols = tsv_read(tsv_in, 1, seek_offset=offset)
        return cols[0]

    @staticmethod
    def _read_rects(fp, label_offset):
        cols = tsv_read(fp, 2, seek_offset=label_offset)
        if len(cols) != 2:
            return
        rects = json.loads(cols[1]) if cols[1] else []
        if not isinstance(rects, list):
            rects = [{'class': cols[1]}]
        return rects

    def _read_label(self, key, fp, label_offset, retdict, class_label=None):
        rects = self._read_rects(fp, label_offset)
        uuid = self.uid(key)
        for rect in rects:
            label = rect['class'].strip()
            if class_label and class_label != label:
                continue
            if label not in retdict:
                retdict[label] = dict()
            if uuid not in retdict[label]:
                retdict[label][uuid] = []
            bbox = [x + 1 for x in rect['rect']] if 'rect' in rect else []
            retdict[label][uuid] += [(rect['diff'] if 'diff' in rect else 0, bbox)]

    def truth_list(self, key, class_label=None, cache=True):
        """Load the truth for a given key
        :type key: tuple[str, int, int] | tuple[str, int] | int | str
        :param class_label: if specified filter truth to only this label
        :param cache: if should keep the file handle in a cache for future access
        :rtype: list
        """
        if self.is_directory or self.is_image:
            raise NotImplementedError("truth not implemented for db: '{}'".format(self.type))
        if cache and not self._cache:
            self._cache = FileCache()
        label_file, label_offset = self._index.label_offset(key)
        with self._cache.open_file(label_file) if cache else open(label_file, 'r') as fp:
            rects = self._read_rects(fp, label_offset)
        # filter the list
        if class_label:
            rects = [rect for rect in rects if rect['class'].strip() == class_label]
        return rects

    def truth(self, key, class_label=None, cache=True):
        """Load the truth for a given key, group by labels
        :type key: tuple[str, int, int] | tuple[str, int] | int | str
        :param class_label: if specified filter truth to only this label
        :param cache: if should keep the file handle in a cache for future access
        :rtype: dict
        """
        if self.is_directory or self.is_image:
            raise NotImplementedError("truth not implemented for db: '{}'".format(self.type))
        if cache and not self._cache:
            self._cache = FileCache()
        label_file, label_offset = self._index.label_offset(key)
        retdict = {}
        with self._cache.open_file(label_file) if cache else open(label_file, 'r') as fp:
            self._read_label(key, fp, label_offset, retdict, class_label=class_label)
        return retdict

    def all_truths(self, class_label=None, cache_truth=False):
        """Load the truth
        :param class_label: if specified filter truth to only this label
        :param cache_truth: if should cache the result for future calls
        :rtype: dict
        """
        if self.is_directory or self.is_image:
            raise NotImplementedError("truth not implemented for db: '{}'".format(self.type))

        if self._cached_truth is not None:
            return self._cached_truth

        retdict = dict()
        with file_cache() as cache:
            for key in self:
                label_file, label_offset = self._index.label_offset(key)
                fp = cache.open(label_file)
                self._read_label(key, fp, label_offset, retdict, class_label=class_label)
        if cache_truth:
            self._cached_truth = retdict
        return retdict

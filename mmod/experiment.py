import logging
import os.path as op
import re

from mmod.utils import makedirs
from mmod.simple_parser import parse_key_value, load_labelmap_list


class Experiment(object):
    def __init__(self, imdb, caffenet=None, caffemodel=None,
                 caffemodel_clone=None,
                 name=None, vis_path=None, cmapfile=None,
                 input_range=None, root=None, data=None, reset=False, expid=None):
        """
        :type imdb: mmod.imdb.ImageDatabase
        :param caffenet: test.prototxt file path
        :type caffenet: str
        :param caffemodel: base visualization path
        :type caffemodel: str
        :param caffemodel_clone: this can be clone of caffemodel to use (original one could be on slow fs)
        :type caffemodel_clone: str
        :param name: experiment name (if given will be used as the fixed name)
        :type name: str
        :type vis_path: str
        :type cmapfile: str
        :param input_range: range of inputs to in this experiment
        :type input_range: xrange
        :param root: root folder of the experiment output
        :type root: str
        :param data: root folder of the experiment data
        :type data: str
        :param reset: if should automatically set caffenet and caffemodel when possible
            and ignore given caffemodel,caffenet
        :param expid: extra ID for auto-naming the experiment
        """
        assert imdb, "No ImageDatabase"
        self.imdb = imdb
        self._caffenet = caffenet
        self._caffemodel = caffemodel
        self._caffemodel_clone = caffemodel_clone
        self._orig_name = self._name = name
        self._vis_path = vis_path
        self._cmapfile = cmapfile
        self._cmap = None  # cached cmap
        self._input_range = input_range
        self._root = root
        self._data = data
        self._expid = expid

        self._tree_file = None
        self._auto_test(reset)

    def __repr__(self):
        return 'Experiment(name: {}, imdb: {})'.format(
            self.name,
            self.imdb
        )

    def __len__(self):
        if self._input_range is None:
            return len(self.imdb)
        return len(self._input_range)

    def __eq__(self, other):
        if not isinstance(other, Experiment):
            return NotImplemented

        return (
            self.path == other.path and
            self._caffenet == other._caffenet and
            self._caffemodel == other._caffemodel and
            self.imdb == other.imdb
        )

    def __ne__(self, other):
        return not self == other

    @property
    def name(self):
        """Experiment name
        :rtype: str
        """
        if self._orig_name:
            return self._orig_name
        if self._name:
            return self._name
        self._name = ''
        if self._caffemodel:
            m = re.match('^(?P<MODEL>.+\.caffemodel)', op.basename(self._caffemodel))
            if m:
                model = m.group("MODEL")
                self._name = model
        if self._input_range:
            if self._name:
                self._name += "."
            self._name += "{}-{}".format(self._input_range[0], self._input_range[-1] + 1)
        if self._caffenet:
            base = op.basename(op.dirname(self._caffenet))
            if base not in self._name and base not in self.imdb.name:
                if self._name:
                    self._name += "."
                self._name += base
        if self._name:
            self._name += "."
        self._name += self.imdb.name
        if self._expid:
            self._name += ".{}".format(self._expid)
        return self._name

    @property
    def input_range(self):
        if self._input_range is None:
            return xrange(0, len(self.imdb))
        return self._input_range

    @input_range.setter
    def input_range(self, value):
        """Experiment root
        :type value: xrange
        """
        self._input_range = value
        # invalidate the automatic name
        self._name = None

    @property
    def root(self):
        """Experiment root
        :rtype: str
        """
        if self._root is None:
            if self.imdb.is_directory:
                self._root = self.imdb.path
            elif self.imdb.is_image:
                self._root = op.dirname(self.imdb.path) or '.'
            else:
                assert self._caffemodel, "Experiment: {} has no caffemodel".format(self)
                self._root = op.dirname(self._caffemodel) or '.'
                makedirs(self._root, exist_ok=True)
        assert self._root and op.isdir(self._root), "Root does not exist: {}".format(self._root)
        return self._root

    @root.setter
    def root(self, value):
        """Experiment root
        :type value: str
        """
        makedirs(value, exist_ok=True)
        self._root = value

    @property
    def data(self):
        """Experiment data
        :rtype: str
        """
        assert self._data and op.isdir(self._data), "Data root does not exist: {}".format(self._data)
        return self._data

    @data.setter
    def data(self, value):
        """Experiment data
        :type value: str
        """
        makedirs(value, exist_ok=True)
        self._data = value

    @property
    def path(self):
        """Path to create experiment
        :rtype: str
        """
        return op.join(self.root, self.name)

    @property
    def predict_path(self):
        """Path to predictions of the experiment
        :rtype: str
        """
        return self.path + ".predict"

    @property
    def caffenet(self):
        """caffenet file path
        :rtype: str
        """
        assert self._caffenet and op.isfile(self._caffenet), "Cannot access caffenet: {}".format(self._caffenet)
        return self._caffenet

    @caffenet.setter
    def caffenet(self, value):
        """caffenet file path
        """
        self._name = None
        self._caffenet = value

    @property
    def caffemodel(self):
        """caffemodel file path
        :rtype: str
        """
        path = self._caffemodel
        if self._caffemodel_clone:
            path = self._caffemodel_clone
        assert path and op.isfile(path), "Cannot access caffemodel: {}".format(path)
        return path

    @caffemodel.setter
    def caffemodel(self, value):
        """caffemodel file path
        """
        self._name = None
        self._caffemodel = value

    @property
    def cmapfile(self):
        """Classmap file path
        :rtype: str or list
        """
        if self._cmapfile:
            return self._cmapfile

        if self.imdb.is_prototxt:
            self._cmapfile = parse_key_value(self.imdb.path, "labelmap")
            return self._cmapfile

        # if we have caffenet use it first to find the labelmap
        if self._caffenet:
            # need caffenet to set the cmap file
            caffenet = self.caffenet
            trainnet = op.join(op.dirname(caffenet), "train.prototxt")
            if op.isfile(trainnet):
                cmapfile = parse_key_value(trainnet, "labelmap")
                if op.isfile(cmapfile):
                    self._cmapfile = cmapfile
                    return self._cmapfile
            cmapfile = op.join(op.dirname(self._caffenet), 'labelmap.txt')
            if op.isfile(cmapfile):
                self._cmapfile = cmapfile
                return self._cmapfile

        if self.imdb.is_directory:
            cmapfile = op.join(self.imdb.path, 'labelmap.txt')
            if op.isfile(cmapfile):
                self._cmapfile = cmapfile
                return self._cmapfile
        if self.imdb.is_image:
            cmapfile = op.join(op.dirname(self.imdb.path), 'labelmap.txt')
            if op.isfile(cmapfile):
                self._cmapfile = cmapfile
                return self._cmapfile

        # try to get the implicit cmap file from imdb
        self._cmapfile = self.imdb.cmap
        return self._cmapfile

    @property
    def cmap(self):
        """Class-mapping list
        :rtype: list
        """
        if self._cmap:
            return self._cmap
        cmap = self.cmapfile
        if isinstance(cmap, list):
            self._cmap = cmap
            return self._cmap
        self._cmap = load_labelmap_list(cmap)
        return self._cmap

    @property
    def tree_file(self):
        """Tree file path
        :rtype: str
        """
        if self._tree_file:
            return self._tree_file
        # need caffenet to set the tree file
        caffenet = self.caffenet
        self._tree_file = op.abspath(parse_key_value(caffenet, "tree"))
        if not op.isfile(self._tree_file):
            self._tree_file = op.join(op.dirname(caffenet), 'tree.txt')
        assert self._tree_file and op.isfile(self._tree_file), "Could not find the tree: {} in: {}".format(
            self._tree_file,
            caffenet
        )
        return self._tree_file

    def _auto_test(self, reset):
        """Automatically set caffenet and caffemodel, based on imdb
        :param reset: if should automatically set caffenet and caffemodel when possible
        """
        if reset or not self._caffemodel or not self._caffenet:
            if self.imdb.is_prototxt:
                # train and test are usually in the same location
                base_path = op.dirname(self.imdb.path)
                caffenet = op.join(base_path, "test.prototxt").replace(
                    "\\", "/"
                )
                if op.isfile(caffenet):
                    self._caffenet = caffenet
                solver = op.join(base_path, "solver.prototxt")
                if op.isfile(solver):
                    max_iter = parse_key_value(solver, "max_iter")
                    if max_iter:
                        caffemodel = op.join(base_path, "snapshot/model_iter_{}.caffemodel".format(max_iter)).replace(
                            "\\", "/"
                        )
                        logging.info("Auto loading caffemodel: {}".format(caffemodel))
                        if op.isfile(caffemodel):
                            self._caffemodel = caffemodel

    def vis_path(self, key=None):
        """The visualization path for a key
        :param key: the image key in db
        :rtype: str
        """
        if self._vis_path is None:
            self._vis_path = op.join(self.root, "vis", self.name)

        if key is None:
            makedirs(self._vis_path, exist_ok=True)
            return self._vis_path
        key = self.imdb.normkey(key)
        if self.imdb.is_directory or self.imdb.is_image:
            path = op.dirname(key)
            relpath = op.relpath(path, self.imdb.path)
            assert not relpath.startswith(".."), "{} must be in {}".format(path, self.imdb.path)
            _vis_path = op.join(self._vis_path, relpath)
            makedirs(_vis_path, exist_ok=True)
            return op.join(_vis_path, op.basename(key))

        source, lidx, idx = key
        _vis_path = op.join(self._vis_path, source.replace("/", "_"))
        makedirs(_vis_path, exist_ok=True)
        relpath = "{}_{}.jpg".format(lidx, idx)
        return op.join(_vis_path, relpath)

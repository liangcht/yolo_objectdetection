import os
import os.path as op
from ete2 import Tree
from mmod.io_utils import load_from_yaml_file


class Taxonomy(object):
    _IGNORE_ATTRS = {
        '_translation', '_inverted_translation', '_global_translation', '_inverted_global_translation',
        '_cmap', '_data_sources'
    }

    def __init__(self, path, trans_path=None, param_path=None):
        self._len = None
        self._leaf_count = 0
        self._path = path
        self._trans_path = trans_path
        if op.splitext(path)[1].lower() == ".yaml":
            self._yaml_file = path
        else:
            self._yaml_file = op.join(op.dirname(self._path), "root.yaml")
        if not param_path:
            # parameters used to generate this taxonomy
            param_path = op.join(op.dirname(self._yaml_file), "generate_parameters.yaml")
            if not op.isfile(param_path):
                param_path = None
        self._param_file = param_path
        self._data_sources = None  # data sources
        self._root = Tree(name=self._yaml_file)
        self._translation = None  # type: dict
        self._inverted_translation = None  # type: dict
        self._global_translation = None  # type: dict
        self._inverted_global_translation = None  # type: dict
        self._cmap = None

    def __len__(self):
        if self._len is None:
            self._load_yaml()
        return self._len

    def __repr__(self):
        return 'Taxonomy(size: {}, leafs: {}, path: {})'.format(
            len(self), self._leaf_count, self._path
        )

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self._path == other._path

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getstate__(self):
        return {
            key: val if key not in self._IGNORE_ATTRS else None
            for key, val in self.__dict__.iteritems()
        }

    def __contains__(self, label):
        if label in self.cmap:
            return True
        # if the term exists in another form, also the taxonomy contains it
        for l in self.translate_from(label):
            if l in self.cmap:
                return True
        return False

    @property
    def translation(self):
        """Use the root.yaml file to find the translation dictionary
        :rtype: dict
        """
        if self._translation is not None:
            return self._translation

        self._load_yaml()
        return self._translation

    @property
    def inverted_translation(self):
        """invert the translation
        :rtype: dict
        """
        if self._inverted_translation is not None:
            return self._inverted_translation

        trans = self.translation
        self._inverted_translation = {
            datasource: {} for datasource in trans
        }
        for datasource in self._inverted_translation:
            for label, values in trans[datasource].iteritems():
                if not values:
                    continue
                for v in values:
                    self._inverted_translation[datasource][v] = label
        return self._inverted_translation

    @property
    def inverted_global_translation(self):
        """invert the global translation
        :rtype: dict
        """
        if self._inverted_global_translation is not None:
            return self._inverted_global_translation

        trans = self.global_translation
        self._inverted_global_translation = {}
        for k, vals in trans.iteritems():
            for v in vals:
                if v not in self._inverted_global_translation:
                    self._inverted_global_translation[v] = []
                self._inverted_global_translation[v].append(k)
        return self._inverted_global_translation

    @property
    def translation_path(self):
        """Translation yaml files path
        :rtype: str
        """
        return self._trans_path or ""

    @translation_path.setter
    def translation_path(self, value):
        """Translation yaml files path
        :type value: str
        """
        # reset the current global translations, to re-read
        self._global_translation = None
        self._inverted_global_translation = None
        self._trans_path = value

    @property
    def global_translation(self):
        """Use the root.yaml file to find the translation dictionary
        :rtype: dict
        """
        if self._global_translation is not None:
            return self._global_translation
        if not self._trans_path:
            return {}
        self._global_translation = {}
        for fname in os.listdir(self._trans_path):
            if op.splitext(fname)[1] == ".yaml":
                self._add_global_trans(op.join(self._trans_path, fname))
        return self._global_translation

    def _add_global_trans(self, path):
        trans = load_from_yaml_file(path)
        for item in trans:
            assert isinstance(item, dict) and "name" in item and "noffset" in item, \
                "invalid translation: {} in: {}".format(item, path)
            name = item["name"].strip()
            if name not in self._global_translation:
                self._global_translation[name] = []
            if "definitions" not in item:
                noffset = item["noffset"]
                if noffset not in self._global_translation[name]:
                    self._global_translation[name].append(noffset)
                continue
            definitions = item["definitions"]
            assert isinstance(definitions, list), \
                "invalid translation: {} in: {}".format(definitions, path)
            for val in definitions:
                assert isinstance(val, dict) and "noffset" in val, \
                    "invalid translation: {} inside: {} in: {}".format(
                        val, definitions, path
                    )
                noffset = val["noffset"]
                if noffset not in self._global_translation[name]:
                    self._global_translation[name].append(noffset)

    def iter_search_nodes(self, name):
        """Iterate over Taxonomy node(s) with given name
        """
        if self._len is None:
            self._load_yaml()
        for node in self._root.iter_search_nodes(name=name):
            yield node

    def translate_from_global(self, class_label):
        """Translate a class label globally regardless of datasource
        :param class_label: the original label
        :type class_label: str
        :rtype: str
        """
        for label in [class_label, class_label.lower()]:
            if label in self.cmap:
                return label

        inv_trans = self.inverted_global_translation
        for label in inv_trans.get(class_label) or []:
            for l in [label, label.lower()]:
                if l in self.cmap:
                    # return the first label that is in the taxonomy
                    return label

    def translate_from(self, class_label, datasource=None):
        """Translate a label to taxonomy label
        :param class_label: the original label (maybe not in the taxonomy)
        :param datasource: the data source for translation
        :return: translated label (that exists in the taxonomy)
        If the label is specifically set to null in the whitelist of a datasource, None will be returned
        :rtype: str
        """
        inv_trans = self.inverted_translation
        if datasource is None:
            labels_set = set()
            for datasource in inv_trans:
                labels = inv_trans.get(datasource) or {}
                lb = labels.get(class_label, self.translate_from_global(class_label))
                if lb is None:
                    continue
                if lb in labels_set:
                    continue
                labels_set.add(lb)
                yield lb
            return

        # for one data source, only one translation in taxonomy exists
        labels = inv_trans.get(datasource) or {}
        yield labels.get(class_label, self.translate_from_global(class_label))

    def translate_to(self, class_label, datasource=None):
        """Iterate all the labels that translate to given class_label for a data source
        Assume a white-list, terms that are not mentioned are returned with no translation
        Returned labels will not repeat
        :param class_label: the label (in the taxonomy)
        :param datasource: the data source for translation
        :rtype: str
        """
        labels_set = set()
        trans = self.translation
        all_data = []
        if datasource is None:
            for datasource in trans:
                data_trans = trans.get(datasource, {}).get(class_label, [class_label])
                if data_trans:
                    all_data.append(data_trans)
        else:
            data_trans = trans.get(datasource, {}).get(class_label, [class_label])
            if data_trans is None:
                # if label is specifically set to null, ignore it
                return
            all_data.append(data_trans)
        global_trans = self.global_translation
        if global_trans:
            for label in global_trans.get(class_label) or []:
                for l in [label, label.lower()]:
                    if l in labels_set:
                        continue
                    labels_set.add(l)
                    yield l
        if not trans:
            # in the absence of any translation, we just combine lower case
            label = class_label
            for l in [label, label.lower()]:
                if l in labels_set:
                    continue
                labels_set.add(l)
                yield l
            return
        for data_trans in all_data:
            for label in data_trans or []:
                for l in [label, label.lower()]:
                    if l in labels_set:
                        continue
                    labels_set.add(l)
                    yield l

    def iter_cmap(self):
        """Iterate through the class map specified by this taxonomy
        """
        if self._translation is None:
            self._load_yaml()
        for t in self._root.iter_descendants():
            yield t.name

    @property
    def cmap(self):
        """class mapping set
        :rtype: set
        """
        if self._cmap:
            return self._cmap
        self._cmap = set(list(self.iter_cmap()))
        return self._cmap

    def _add_current_as_child(self, one, root):
        """Add one to root
        """
        if type(one) is dict:
            list_value_keys = [k for k in one if type(one[k]) is list]
            if len(list_value_keys) == 1:
                name = list_value_keys[0]
            else:
                assert 'name' in one, one
                name = one['name']
            assert isinstance(name, basestring), "{} in {} not string".format(name, root)
            child_subgroups = getattr(root, 'child_subgroups', -1)
            if name.startswith('__'):
                # just increase the subgroups count of the root
                setattr(root, 'child_subgroups', child_subgroups + 1)
                return
            sub_root = root.add_child(name=name)
            feats = {'sub_group': child_subgroups}
            for k in one:
                if self._data_sources and k not in self._data_sources:
                    # if this is certainly not a data source
                    continue
                v = one[k]
                if type(v) is not list and k != 'name':
                    feats[k] = v
                    if v and (not isinstance(v, basestring) or '/' in v):
                        continue
                    if self._translation is None:
                        self._translation = {}
                    if k not in self._translation:
                        self._translation[k] = {}
                    if not v:
                        self._translation[k][name] = None
                        continue
                    self._translation[k][name] = [l.strip() for l in v.split(",")]
            sub_root.add_features(**feats)
            if len(list_value_keys) == 1:
                for sub_one in one[list_value_keys[0]]:
                    self._add_current_as_child(sub_one, sub_root)
        else:
            if one is None:
                one = 'None'
            name = one
            assert type(name) is str or type(name) is unicode
            child_subgroups = getattr(root, 'child_subgroups', -1)
            if name.startswith('__'):
                # just increase the subgroups count of the root
                setattr(root, 'child_subgroups', child_subgroups + 1)
                return
            sub_root = root.add_child(name=name)
            sub_root.add_features(sub_group=child_subgroups)

    def _load_yaml(self):
        self._len = 0
        self._translation = {}
        config_tax = load_from_yaml_file(self._yaml_file)
        assert isinstance(config_tax, list), "invalid {}".format(self._yaml_file)
        for one in config_tax:
            self._add_current_as_child(one, self._root)
        _cached_content = self._root.get_cached_content()
        self._len = len(_cached_content) - 1  # size of taxonomy, not including the root
        self._leaf_count = len(_cached_content[self._root])

import os.path as op
import numpy as np
import json
import logging
from contextlib import contextmanager
from collections import OrderedDict

from mmod.utils import is_in_sorted, search_in_sorted, search_both_sorted, tsv_read, splitfilename, tsv_multi_column, \
    FileCache, file_cache, tail_path
from mmod.simple_parser import load_labelmap_list, is_number
from mmod.simple_tsv import SimpleTsv


class TsvFile(object):
    _IGNORE_DICTS = {'_offsets', '_label_offsets', '_inverted_offsets', '_cache',
                     '_is_open', '_shuffle'}

    def __init__(self, sources, labels=None, cmapfiles=None):
        self._len = 0
        self._is_open = False  # if the index is full open
        self._sources = {}
        self._sources_idx = {}
        self._idx_sources = {}
        self._labels = {}
        self._cmaps = {}

        self._offsets = {}
        self._label_offsets = {}
        self._inverted_offsets = {}
        self._cache = None  # type: FileCache
        self._sources_type = {}
        self._inverted_file = None
        self._shuffle_file = None
        self._is_composite = False
        self._shuffle = None
        self._cmapfile = None
        self._has_inverted = {}

        self._check_keys = False

        if isinstance(sources, basestring):
            sources, labels = self._composite_sources(sources, labels)

        if labels:
            if isinstance(labels, basestring):
                labels = [labels]
        else:
            # find the labels for sources
            labels = []
            for source in sources:
                label_file = splitfilename(source, 'label')
                if not op.isfile(label_file):
                    label_file = None
                labels.append(label_file)
        assert len(labels) == len(sources)

        self._load_sources(sources, labels=labels)
        assert self._len, "Index file is empty"

        if cmapfiles:
            if isinstance(cmapfiles, basestring):
                cmapfiles = [cmapfiles] * len(sources)
        else:
            cmapfiles = []
            for source in sources:
                if self.class_type(source) == 'index':
                    # no_bb dbs only keep the names in labelmap.txt
                    cmap_file = op.join(op.dirname(source), 'labelmap.txt')
                    if not op.isfile(cmap_file):
                        cmap_file = splitfilename(source, 'labelmap')
                else:
                    cmap_file = splitfilename(source, 'labelmap')
                    if not op.isfile(cmap_file):
                        cmap_file = op.join(op.dirname(source), 'labelmap.txt')
                if not op.isfile(cmap_file):
                    cmap_file = None
                cmapfiles.append(cmap_file)
        assert len(cmapfiles) == len(sources)
        self._cmaps = {
            op.normpath(source).replace("\\", "/"): cmapfile
            for (source, cmapfile) in zip(sources, cmapfiles)
        }

    def __repr__(self):
        if len(self._sources) == 1:
            source = self._sources.keys()[0]
            stype = self.label_type(source)
            return 'TsvFile(size: {}, source: {}, type: {})'.format(
                len(self), tail_path(source), stype
            )
        labeled = len(self._labels) == len(self._sources)
        return 'TsvFile(size: {}, {}: {}{})'.format(
            len(self), "labeled sources" if labeled else "sources", len(self._sources),
            ", shuffle: {}".format(tail_path(self._shuffle_file)) if self._shuffle_file else ""
        )

    def __del__(self):
        self.close()

    def __getstate__(self):
        return {
            key: val if key not in self._IGNORE_DICTS else {}
            for key, val in self.__dict__.iteritems()
        }

    def __getitem__(self, key):
        """Convert to the normalized tuple key to be used to look up in index
        :param key: The key to look up in the index
        :type key: tuple[str, int, int] | tuple[str, int] | int | str
        :return: source: source file name
                 line_index: line index of key in the source file
                 index: index of key in the source file
        :rtype: tuple[str, int, int]
        """
        if isinstance(key, tuple):
            if len(key) == 2:
                key_source, lidx = key
                if not isinstance(key_source, basestring) or not isinstance(lidx, (int, long, np.integer)) or lidx < 0:
                    raise ValueError("{} is not valid".format(key))
                if key_source not in self._sources:
                    raise KeyError("{} is not in the index".format(key))
                lrng, rng = self._sources[key_source]
                idx = search_in_sorted(lrng, lidx)
                if idx is None:
                    raise KeyError("{} is not in the index (could be deleted)".format(key))
                return key_source, lidx, idx
            if self._check_keys and key not in self:
                raise KeyError("{} is not in the index".format(key))
            return key
        if isinstance(key, basestring):
            key = key.strip()
            if not key.startswith('['):  # fast check
                raise ValueError("{} is not valid".format(key))
            # key must be a json encoded list
            try:
                norm_key = json.loads(key)
                if not isinstance(norm_key, list) or len(norm_key) < 2 or len(norm_key) > 3:
                    raise ValueError("Type is is not a tuple")
                norm_key = tuple(norm_key)
            except ValueError as e:
                raise ValueError("{} is not valid err: {}".format(key, e))
            if len(norm_key) == 2:
                # find the current value
                return self[norm_key]
            if len(norm_key) != 3:
                raise ValueError("{} is not valid".format(key))
            return norm_key
        # key is flat index
        if key < 0:
            key += self._len
        for source, (lrng, rng) in self._sources.iteritems():
            if key in rng:
                key -= rng[0]
                # source file, line number, non-deleted item number
                return source, int(lrng[key]), key
        raise KeyError("{} is not in the index".format(key))

    def __contains__(self, key):
        """Convert to the normalized tuple key to be used to look up in index
        :param key: The key to look up in the index
        :type key: tuple[str, int, int] | tuple[str, int] | int | str
        :rtype: bool
        """
        if isinstance(key, tuple):
            if len(key) < 2 or len(key) > 3:
                return False
            if len(key) == 2:
                source, lidx = key
                if source not in self._sources:
                    return False
                lrng, rng = self._sources[source]
                if not rng:
                    return False
                return is_in_sorted(lrng, lidx)
            source, lidx, idx = key
            if source not in self._sources:
                return False
            lrng, rng = self._sources[source]
            if not rng:
                return False
            return idx + rng[0] in rng and is_in_sorted(lrng, lidx)
        if isinstance(key, basestring):
            key = key.strip()
            if not key.startswith('['):  # fast check
                # key is a list
                return False
            try:
                key = self[key]
            except (ValueError, KeyError):
                return False
            if len(key) == 3:
                return key in self
            return True
        return -self._len <= key < self._len

    def __iter__(self):
        for source, (lrng, rng) in self._sources.iteritems():
            for idx in rng:
                idx -= rng[0]
                yield source, int(lrng[idx]), idx

    def __len__(self):
        return self._len

    def _load_sources(self, sources, labels=None):
        """Load source ranges
        :param sources: list of source files
        :param labels: list of label files
        """
        assert not labels or len(labels) == len(sources), "Number of labels {} != number of sources {}".format(
            len(labels), len(sources)
        )
        label_file = None
        last_idx = 0
        self._sources = OrderedDict()
        self._labels = OrderedDict()
        for source_idx, source in enumerate(sources):
            tsv = SimpleTsv(source)
            count = len(tsv)
            assert count < 0xFFFFFFFF, "File too large: {}".format(tsv)
            deleted_lines = 0
            lrng = []
            if labels:
                # Find and remove deleted labels
                label_file = labels[source_idx]
                if label_file:
                    with open(label_file, 'r') as fp:
                        label_count = 0
                        for line in fp:
                            if line.startswith('d\t'):
                                deleted_lines += 1
                            else:
                                lrng.append(np.uint32(label_count))
                            label_count += 1
                    assert label_count == count, "label file: {} length {} != {} length of {}".format(
                        label_file, label_count, count, tsv
                    )
                    if deleted_lines:
                        count -= deleted_lines
                        lrng = np.array(lrng)

            rng = xrange(last_idx, last_idx + count)
            if not deleted_lines:
                lrng = xrange(count)
            source = op.normpath(source).replace("\\", "/")
            self._sources[source] = lrng, rng  # Line range, and Index range both of count length
            self._sources_idx[source] = source_idx
            self._idx_sources[source_idx] = source
            if labels:
                self._labels[source] = label_file
            last_idx += count
        self._len = last_idx

    def _composite_sources(self, source, labels):
        """Return sources from potentially composite source (list of other sources)
        :type source: basestring
        :type labels: list
        :rtype: (list, list)
        """
        if not tsv_multi_column(source):
            with open(source, 'r') as fp:
                sources = [l for l in fp.read().splitlines() if l]
            if not labels:
                labelfile = splitfilename(source, "label")
                if op.isfile(labelfile):
                    with open(labelfile, 'r') as fp:
                        labels = [l for l in fp.read().splitlines() if l]
            if not labels:
                assert source.endswith('X.tsv')
                labels = [
                    source.replace('X.tsv', '{}.label.tsv'.format(idx)) for idx in range(len(sources))
                ]
            self._is_composite = True
            # composite sources also could have a single inverted file
            inverted_file = splitfilename(source, 'inverted.label', is_composite=True)
            shuffle_file = splitfilename(source, 'shuffle.txt', is_composite=True, keep_ext=False)
            if op.isfile(inverted_file) and op.isfile(shuffle_file):
                self._inverted_file = inverted_file
                self._shuffle_file = shuffle_file
            cmapfile = op.join(op.dirname(source), "labelmap.txt")
            if op.isfile(cmapfile):
                self._cmapfile = cmapfile

            return sources, labels

        return [source], labels

    def iter_sources(self):
        """Iterate data sources
        :rtype: str
        """
        for source in self._sources:
            yield source

    def iter_source_range(self):
        """Iterate data sources and their line range
        :rtype: str
        """
        for source, (lrng, _) in self._sources.iteritems():
            yield source, lrng

    @property
    def is_composite(self):
        """Is this a composite TSV (trainX, testX)
        """
        return self._is_composite

    @property
    def composite_non_inverted(self):
        """Composite TSV without single inverted file
        """
        return self.is_composite and not self._inverted_file

    @property
    def has_inverted(self):
        """If all sources have inverted file
        :rtype: bool
        """
        if self._inverted_file:
            # if composite we use it
            return True
        if not isinstance(self._has_inverted, dict):
            return self._has_inverted
        for source in self._sources:
            inverted_file = splitfilename(source, 'inverted.label')
            if not op.isfile(inverted_file):
                self._has_inverted = False
                return self._has_inverted
        self._has_inverted = True
        return self._has_inverted

    @property
    def has_cmapfile(self):
        """If all sourced have cmapfile
        :rtype: bool
        """
        if self._cmapfile:
            # if composite we use it
            return True
        for cmapfile in self._cmaps.values():
            if cmapfile is None:
                return False
        return True

    def _cached_inverted_offsets(self, inverted_file, cmap_list=None, cache=None):
        offsets = self._inverted_offsets.get(inverted_file)
        if offsets is not None:
            return offsets

        offsets = {}
        assert op.isfile(inverted_file), "{} inverted file does not exist".format(inverted_file)
        with open(inverted_file, "r") if not cache else cache.open_file(inverted_file) as fp:
            offset = 0
            for line in iter(fp.readline, ""):
                idx = line.find('\t')
                if idx < 1:
                    break
                label = line[:idx].lower()
                # For index sources see if these labels are class numbers
                if cmap_list and is_number(label):
                    label_idx = int(label)
                    if label_idx < 0 or label_idx >= len(cmap_list):
                        logging.error("Ignore invalid index: {} in: {}".format(
                            label_idx, inverted_file
                        ))
                    else:
                        label = cmap_list[label_idx].lower()
                # go where the lines are
                offsets[label] = offset + idx + 1
                offset += len(line)
        self._inverted_offsets[inverted_file] = offsets
        return offsets

    @staticmethod
    def _filter_shuffle_lines(source_idx, lines, shuffle):
        return [shuffle[idx][1] for idx in lines if shuffle[idx][0] == source_idx]

    def _iter_inverted(self, class_label, source=None, tax=None, cache=None):
        """Iterate over inverted file and its offset for a class label
        :param class_label: class label to find in the index
        :type class_label: str
        :param source: source data source to limit the labels from
        :type source: str
        :param tax: taxonomy to use for on-the-fly translation
        :type tax: mmod.taxonomy.Taxonomy
        :type cache: FileCache
        :rtype: (str, list[int])
        """
        sources = self._sources
        if source is not None:
            source = op.normpath(source).replace("\\", "/")
            assert source in self._sources, "{} is not a valid data source".format(source)
            sources = [source]
        # with shuffle file we have to read the entire list because we cannot seek to a file
        if self._shuffle is None:
            with open(self._shuffle_file) as fp:
                self._shuffle = [[int(l) for l in line.split()] for line in fp.readlines()]

        inverted_file = self._inverted_file
        offsets = self._cached_inverted_offsets(inverted_file, cache=cache)
        if not tax:
            offset = offsets.get(class_label)
            if offset is None:
                # label not found in this source
                return
            with open(inverted_file, "r") if not cache else cache.open_file(inverted_file) as fp:
                lines = tsv_read(fp, 1, seek_offset=offset, sep='\n')
                if not lines:
                    return
                lines = [int(line) for line in lines[0].split()]
            for source in sources:
                source_idx = self._sources_idx[source]
                filtered_lines = self._filter_shuffle_lines(source_idx, lines, self._shuffle)
                yield source, filtered_lines
            return

        with open(inverted_file, "r") if not cache else cache.open_file(inverted_file) as fp:
            for label in tax.translate_to(class_label, datasource=op.basename(op.dirname(source))):
                offset = offsets.get(label)
                if offset is None:
                    # label not found in this source
                    continue
                lines = tsv_read(fp, 1, seek_offset=offset, sep='\n')
                if not lines:
                    return
                lines = [int(line) for line in lines[0].split()]
                for source in sources:
                    source_idx = self._sources_idx[source]
                    filtered_lines = self._filter_shuffle_lines(source_idx, lines, self._shuffle)
                    yield source, filtered_lines

    def _iter_inverted_offset(self, class_label, source=None, tax=None, cache=None):
        """Iterate over inverted file and its offset for a class label
        :param class_label: class label to find in the index
        :type class_label: str
        :param source: source data source to limit the labels from
        :type source: str
        :param tax: taxonomy to use for on-the-fly translation
        :type tax: mmod.taxonomy.Taxonomy
        :type cache: FileCache
        :rtype: (str, str, int)
        """
        sources = self._sources
        if source is not None:
            source = op.normpath(source).replace("\\", "/")
            assert source in self._sources, "{} is not a valid data source".format(source)
            sources = [source]
        for source in sources:
            inverted_file = splitfilename(source, 'inverted.label')
            cmap_list = self._cached_cmap(source) if self.class_type(source) == 'index' else None
            offsets = self._cached_inverted_offsets(inverted_file, cmap_list=cmap_list, cache=cache)
            if not tax:
                offset = offsets.get(class_label)
                if offset is None:
                    # label not found in this source
                    continue
                yield source, inverted_file, offset
                return

            for label in tax.translate_to(class_label, datasource=op.basename(op.dirname(source))):
                offset = offsets.get(label)
                if offset is None:
                    # label not found in this source
                    continue
                yield source, inverted_file, offset

    def iter_label(self, class_label, source=None, tax=None):
        """Iterate over keys for a class label
        :param class_label: class label to find in the index
        :type class_label: str
        :param source: source data source to limit the labels from
        :type source: str
        :param tax: taxonomy to use for on-the-fly translation
        :type tax: mmod.taxonomy.Taxonomy
        :rtype: tuple[str, int, int]
        """
        class_label = class_label.lower()  # we cache and search the lower form
        if self._shuffle_file:
            with file_cache() as cache:
                for source, lines in self._iter_inverted(
                        class_label, source=source, tax=tax, cache=cache
                ):
                    # have to sort the lines because they come from a shuffled file
                    for lidx, idx in self._filter_valid_lines(source, np.sort(lines)):
                        yield source, lidx, idx
            return
        with file_cache() as cache:
            for source, inverted_file, offset in self._iter_inverted_offset(
                    class_label, source=source, tax=tax, cache=cache
            ):
                with cache.open_file(inverted_file) as fp:
                    # read from offset to the end
                    lines = tsv_read(fp, 1, seek_offset=offset, sep='\n')
                    if lines:
                        lines = lines[0].split()
                    lines = [int(line) for line in lines]
                    for lidx, idx in self._filter_valid_lines(source, lines):
                        yield source, lidx, idx

    def _cached_cmap(self, source):
        cmap_list = self._cmaps[source]
        if not cmap_list:
            return
        if not isinstance(cmap_list, list):
            cmap_list = load_labelmap_list(cmap_list)
            self._cmaps[source] = cmap_list

        return cmap_list

    def iter_cmap(self, source=None):
        """Iterate through all the classes
        :param source: source data source to limit the labels from
        :type source: str
        :rtype: str
        """
        if self._cmapfile:
            # if there is a single labelmap for the composite db, use it
            for label in load_labelmap_list(self._cmapfile):
                yield label
            return

        cmap = set()

        sources = self._sources
        if source is not None:
            source = op.normpath(source).replace("\\", "/")
            assert source in self._sources, "{} is not a valid data source".format(source)
            sources = [source]
        for source in sources:
            cmap_list = self._cached_cmap(source)
            if cmap_list:
                for label in cmap_list:
                    if label in cmap:
                        continue
                    cmap.add(label)
                    yield label
                return
            # use inverted file as cmap
            inverted_file = self._inverted_file or splitfilename(source, 'inverted.label')
            offsets = self._cached_inverted_offsets(inverted_file)

            for label in offsets:
                if label in cmap:
                    continue
                cmap.add(label)
                yield label

    def _type(self, source):
        source = op.normpath(source).replace("\\", "/")

        stype, ctype = self._sources_type.get(source) or (None, None)
        if stype and ctype:
            return stype, ctype
        assert source in self._sources, "{} is not a valid data source".format(source)
        # read the first non-deleted label
        lrng, _ = self._sources[source]
        key = source, lrng[0], 0
        label_file, label_offset = self.label_offset(key)
        with open(label_file, 'r') as fp:
            row = tsv_read(fp, 2, seek_offset=label_offset)
            assert len(row) == 2, "Invalid label file: {}".format(label_file)
            try:
                rects = json.loads(row[1])
            except ValueError:
                rects = []
            ctype = 'name'
            stype = 'no_bb'
            if not isinstance(rects, list) or len(rects) == 0 or not isinstance(rects[0], dict):
                ctype = 'index'
                stype = 'no_bb'
            elif any(np.sum(r.get('rect', [])) > 1 for r in rects):
                ctype = 'name'
                stype = 'with_bb'

        self._sources_type[source] = stype, ctype
        return stype, ctype

    def class_type(self, source):
        """Find class label type (name, or index)
        :param source: source data source to limit the labels from
        :type source: str
        :return: "name", "index"
        :rtype: str
        """
        _, ctype = self._type(source)
        return ctype

    def label_type(self, source):
        """Find tsv source type (with or without bounding box)
        :param source: source data source to limit the labels from
        :type source: str
        :return: "with_bb", "no_bb"
        :rtype: str
        """
        stype, _ = self._type(source)
        return stype

    def source_index(self, source):
        """Find the source index
        :param source: source data source to limit the labels from
        :type source: str
        :rtype: int
        """
        return self._sources_idx[source]

    def source(self, idx):
        """Find the source
        :param idx: index of a source
        :type idx: int
        :rtype: str
        """
        return self._idx_sources[idx]

    def _filter_valid_lines(self, source, lines):
        """Filter lines to those included from the source
        :param source: source tsv file
        :type source: str
        :param lines: sorted non-recurrent list of lines to check
        :type lines: list[int] | numpy.ndarray
        :rtype: (int, int)
        """
        if len(lines) == 0:
            return
        lrng, rng = self._sources[source]
        if not rng:
            return
        for idx, _ in search_both_sorted(lrng, lines):
            yield int(lrng[idx]), idx

    @contextmanager
    def check_keys(self, check=True):
        """Context to check or not check validity of keys
        :param check: if shoudl check the validity
        """
        saved = self._check_keys
        self._check_keys = check
        try:
            yield
        finally:
            self._check_keys = saved

    def open(self, path=None):
        """Open the index or a file in the index
        :param path: source path
        """
        if not self._cache:
            self._cache = FileCache()
        if path is None:
            for source in self._sources:
                self._cache.open(source)
            self._is_open = True
            return

        assert path in self._sources, "Source not in the index: {}".format(path)
        return self._cache.open(path)

    def close(self):
        """Close the index
        """
        if self._cache:
            self._cache.close()
            self._cache = None
        self._is_open = False

    def is_open(self):
        """If index is open for fast access
        """
        return bool(self._is_open)

    def offset(self, key):
        """Find source and its offset for a key
        :param key: The key to look up in the index
        :type key: tuple[str, int, int] | tuple[str, int] | int | str
        :return: source: source file name
                 offset: file offset of key in the source file
        :rtype: (str, int)
        """
        source, _, idx = self[key]
        offsets = self._offsets.get(source)
        if offsets is None:
            label_file = self._labels.get(source)
            tsv = SimpleTsv(source, label_file)
            offsets = np.array(tsv.offsets(), dtype=np.int64)
            self._offsets[source] = offsets
        return source, offsets[idx]

    def label_offset(self, key):
        """Find label_file and its offset for a key
        :param key: The key to look up in the index
        :type key: tuple[str, int, int] | tuple[str, int] | int | str
        :return: label: file name to read label from
                 offset: file offset of key in the above file
        :rtype: (str, int)
        """
        source, _, idx = self[key]
        label_file = self._labels.get(source)
        if not label_file:
            # If no label file is available, read label from source itself
            return self.offset(key)

        offsets = self._label_offsets.get(label_file)
        if offsets is None:
            lrng, _ = self._sources[source]
            tsv = SimpleTsv(label_file, label_file)
            offsets = np.array(tsv.offsets(), dtype=np.int64)
            self._label_offsets[label_file] = offsets
        return label_file, offsets[idx]

    def uid(self, key):
        """Return the unique id of the key
        :param key: The key to look up in the index
        :type key: tuple[str, int, int] | tuple[str, int] | int | str
        :rtype: str
        """
        key = self[key]
        return json.dumps(key[:2], separators=(',', ':'))

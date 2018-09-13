import logging
import os.path as op
from itertools import izip
from contextlib import contextmanager
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO


class SimpleTsv(object):
    def __init__(self, path, label_path=None):
        self._path = path
        self._label_path = label_path
        self._index_path = op.splitext(path)[0] + '.lineidx'
        self._index_content = None  # type: StringIO
        self._len = None

    def __repr__(self):
        return 'SimpleTsv(size: {} path: {})'.format(
            len(self), self._path
        )

    def __len__(self):
        if self._len is not None:
            return self._len

        if op.isfile(self._index_path):
            self._len = 0
            with open(self._index_path, 'r') as fp:
                for _ in fp:
                    self._len += 1
        else:
            self._create()

        return self._len

    def _create(self):
        if op.isfile(self._index_path):
            return self._index_path
        logging.error("Try creating TSV index: {}".format(self._index_path))
        # see if we can create the index
        self._len = 0
        offset = 0
        try:
            with open(self._path, 'r') as fp, open(self._index_path, 'wb') as wfp:
                for _ in iter(fp.readline, ""):
                    self._len += 1
                    wfp.write("{}\n".format(offset))
                    offset = fp.tell()
            return self._index_path
        except (IOError, OSError) as e:
            self._index_content = StringIO()
            logging.error("Cannot create index file: {} error: {}".format(self._index_path, e))
            with open(self._path, 'r') as fp:
                for _ in iter(fp.readline, ""):
                    self._len += 1
                    self._index_content.write("{}\n".format(offset))
                    offset = fp.tell()

    @contextmanager
    def open_index(self, mode="r"):
        """Open index associated with the tsv
        :return:
        """
        if self._index_content is not None:
            self._index_content.seek(0)
            yield self._index_content
            return
        index_path = self._create()
        if index_path:
            with open(index_path, mode) as fp:
                yield fp
            return
        self._index_content.seek(0)
        yield self._index_content

    def offsets(self):
        """File offsets
        :rtype: list[int]
        """
        if self._label_path:
            # read indices while excluding the deleted ones
            with self.open_index() as fp, open(self._label_path, 'r') as fp_label:
                offsets = []
                for index, label in izip(fp, fp_label):
                    if label.startswith('d\t'):
                        continue
                    offsets.append(int(index))
        else:
            with self.open_index() as fp:
                offsets = [int(index) for index in fp.readlines()]

        return offsets

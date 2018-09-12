import logging
import numpy as np
import os.path as op
from mmod.imdb import ImageDatabase
from mmod.taxonomy import Taxonomy
from mmod.utils import open_with_lineidx, splitfilename


def iterate_tsv_imdb(path, valid_splits=None):
    """Iterate TSV files in a path as image databases
    :param path: root path to find the data sources
    :rtype: ImageDatabase
    :param valid_splits: valid splits to try in the given path
    :type valid_splits: list[str]
    """
    if valid_splits is None:
        valid_splits = ['train', 'trainval', 'test']
    # check the type of the dataset
    for split in valid_splits:
        label_tsv = op.join(path, "{}.label.tsv".format(split))
        tsv_file = op.join(path, "{}.tsv".format(split))
        if not op.isfile(label_tsv) or not op.isfile(tsv_file):
            continue
        db = ImageDatabase(tsv_file)
        yield db


def sample_tax(tax, dbs, max_label):
    """Sample the taxonomy from dbs for max_label samples
    :type tax: Taxonomy
    :param dbs: image dbs already set for the taxonomy
    :type dbs: list[ImageDatabase]
    :param max_label: maximum number of samples per-label
    :rtype: tuple[str, list[dict], str]
    """
    assert len(dbs), "No data sources are given"
    assert max_label > 0, "max_label is not positive"
    label_count = {
        label: 0 for label in tax.iter_cmap()
    }
    for label in tax.iter_cmap():
        for db in dbs:
            assert isinstance(db, ImageDatabase)
            for key, rects, _ in db.iter_filtered_label_items(label, tax=tax):
                uid = db.uid(key)
                b64 = db.base64(key)
                labels = set()  # labels in this image
                for rect in rects:
                    cls = rect['class']
                    if cls in labels:
                        # if same class is seen multiple times, count it just once
                        continue
                    labels.add(cls)
                    label_count[cls] += 1
                yield uid, rects, b64
                if label_count[label] >= max_label:
                    break
            if label_count[label] >= max_label:
                break


def create_inverted(db, path=None, shuffle=None, labelmap=None, only_inverted=False):
    """Create single inverted file for a db
    :param db: the imdb to create
    :type db: ImageDatabase
    :param path: output path for inverted file
    :type path: str
    :param shuffle: output path for shuffle file
    :type shuffle: str
    :param labelmap: labelmap path to create
    :type labelmap: str
    :param only_inverted: if should only create inverted file (using existing shuffle file)
    """
    if path is None:
        path = splitfilename(db.path, 'inverted.label')
    if shuffle is None:
        shuffle = op.splitext(db.path)[0] + '.shuffle.txt'
    if labelmap is None:
        labelmap = op.join(op.dirname(db.path), 'labelmap.txt')
    if only_inverted:
        with open(shuffle) as fp:
            shuffles = [[int(l) for l in line.split()] for line in fp.readlines()]

        shuffle_line = 0
        source_lines = {}
        for source_idx, line in shuffles:
            source = db.source(source_idx)
            source_lines[(source, line)] = shuffle_line
            shuffle_line += 1
        with open_with_lineidx(path) as fp:
            prev_label = None
            label_shuffle_lines = []
            for label, source, lines in db.iterate_inverted():
                shuffle_lines = [source_lines[(source, line)] for line in lines]
                if not shuffle_lines:
                    continue
                if label == prev_label:
                    label_shuffle_lines += shuffle_lines
                else:
                    if label_shuffle_lines:
                        assert prev_label
                        label_shuffle_lines = [str(l) for l in np.sort(label_shuffle_lines)]
                        fp.write("{}\t{}\n".format(
                            prev_label, " ".join(label_shuffle_lines)
                        ))
                    label_shuffle_lines = shuffle_lines
                    prev_label = label
            if label_shuffle_lines:
                assert prev_label
                label_shuffle_lines = [str(l) for l in np.sort(label_shuffle_lines)]
                fp.write("{}\t{}\n".format(
                    prev_label, " ".join(label_shuffle_lines)
                ))

        return
    line_idx = 0
    with open_with_lineidx(path) as fp, open_with_lineidx(shuffle) as fps, open(labelmap, "w") as fpl:
        for label, source, lines in db.iterate_inverted():
            source_idx = db.source_index(source)
            shuffle_lines = []
            for line in lines:
                fps.write("{}\t{}\n".format(
                    source_idx, line
                ))
                shuffle_lines.append(str(line_idx))
                line_idx += 1
            fp.write("{}\t{}\n".format(
                label, " ".join(shuffle_lines)
            ))
            fpl.write("{}\n".format(label))

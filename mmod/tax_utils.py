import logging
import cv2
import numpy as np
import os.path as op
from mmod.imdb import ImageDatabase
from mmod.taxonomy import Taxonomy
from mmod.utils import open_with_lineidx, splitfilename, open_file, makedirs
from mmod.im_utils import im_rescale


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
    if only_inverted and op.isfile(shuffle):
        with open(shuffle) as fp:
            shuffles = [[int(l) for l in line.split()] for line in fp.readlines()]
    else:
        shuffles = []
        with open_file(None) if only_inverted else open_with_lineidx(shuffle) as fps:
            for source, lrng in db.iter_source_range():
                source_idx = db.source_index(source)
                for line in lrng:
                    shuffles.append([source_idx, line])
                    if fps:
                        fps.write("{}\t{}\n".format(
                            source_idx, line
                        ))

    shuffle_line = 0
    source_lines = {}
    for source_idx, line in shuffles:
        source = db.source(source_idx)
        source_lines[(source, line)] = shuffle_line
        shuffle_line += 1
    with open_with_lineidx(path) as fp, \
            open_file(None) if only_inverted else open(labelmap, "w") as fpl:
        prev_label = None
        label_shuffle_lines = []
        for label, source, lines in db.iterate_inverted():
            if fpl:
                fpl.write("{}\n".format(label))
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


def create_collage(db, path=None, max_label=100, target_size=100):
    """Create single inverted file for a db
    :param db: the imdb to create
    :type db: ImageDatabase
    :param path: output directory path for collages
    :type path: str
    :param max_label: maximum number of samples per-label
    :param target_size: patch size to align maximum dimension to
    """
    if path is None:
        path = op.join(op.dirname(db.path), "collage_{}".format(max_label))
        makedirs(path, exist_ok=True)
    assert op.isdir(path), "{} directory is not accessable".format(path)
    for label in db.iter_cmap():
        logging.info("Sampling collage for {}".format(label))
        keys = list(db.iter_label(label))
        total = len(keys)
        if total > max_label:
            keys = [
                keys[idx]
                for idx in np.sort(np.random.choice(max_label, replace=False, size=(max_label,)))
            ]
            # take first/random rect from each key frame
            key_rects = [
                np.array(db.truth_list(key, label)[0]['rect'], dtype=int)
                for key in keys
            ]
        else:
            # some keys (with multiple rects) could be duplicated
            all_rects = [
                [np.array(rect['rect'], dtype=int) for rect in db.truth_list(key, label)]
                for key in keys
            ]
            all_counts = {key: len(rects) for key, rects in zip(keys, all_rects)}
            key_indices = {key: idx for idx, key in enumerate(keys)}
            new_keys = []
            key_rects = []
            while all_counts:
                to_remove = []
                for key, count in all_counts.iteritems():
                    new_keys.append(key)
                    key_rects.append(all_rects[key_indices[key]][count - 1])
                    if len(new_keys) == max_label:
                        break
                if len(new_keys) == max_label:
                    break
                for key in all_counts:
                    all_counts[key] -= 1
                    if not all_counts[key]:
                        to_remove.append(key)
                for key in to_remove:
                    all_counts.pop(key, None)
            keys = new_keys
            del all_rects, all_counts, key_indices, new_keys

        rows = np.ceil(np.sqrt(len(keys)))
        cols = np.ceil(len(keys) / rows)
        jpg_path = op.join(path, "{}_{}.jpg".format(label.replace(" ", "_"), total))
        collage = np.zeros((int(rows) * target_size, int(cols) * target_size, 3))

        # TODO: re-arrange based on landscape, vertical patches to better fit
        # try to pack them in one pass
        max_h = 0  # maximum height in the current row
        x, y = 0, 0
        y2 = 0
        for key, rect in zip(keys, key_rects):
            im = db.image(key)
            left, top, right, bot = rect
            roi = im_rescale(im[top:bot, left:right], target_size)
            h, w = roi.shape[:2]
            x2 = x + w
            y2 = y + h
            if x2 > collage.shape[1]:
                # next row
                x = 0
                y += max_h
                y2 = y + h
                x2 = x + w
                max_h = 0
            if h > max_h:
                max_h = h
            collage[y:y2, x:x2] = roi

            x = x2

        # clip the collage
        collage = collage[:y2, :, :]
        logging.info("Writing collage {}".format(jpg_path))
        cv2.imwrite(jpg_path, collage)

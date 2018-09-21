import logging
import numpy as np
import os.path as op
from mmod.imdb import ImageDatabase
from mmod.taxonomy import Taxonomy
from mmod.utils import open_with_lineidx, splitfilename, open_file, makedirs
from mmod.im_utils import int_rect, tile_rects


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
    :type tax: mmod.taxonomy.Taxonomy
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


def create_inverted(db, path=None, shuffle=None, labelmap=None,
                    create_shuffle=True, create_labelmap=True):
    """Create single inverted file for a db
    :param db: the imdb to create
    :type db: ImageDatabase
    :param path: output path for inverted file
    :type path: str
    :param shuffle: output path for shuffle file
    :type shuffle: str
    :param labelmap: labelmap path to create
    :type labelmap: str
    :param create_shuffle: if should [re]create shuffle file
    :param create_labelmap: if should [re]create labelmap file
    """
    if path is None:
        path = db.inverted_path
    if shuffle is None:
        shuffle = db.shuffle_path
    if labelmap is None:
        labelmap = db.cmapfile
    if not create_shuffle and op.isfile(shuffle):
        with open(shuffle) as fp:
            shuffles = [[int(l) for l in line.split()] for line in fp.readlines()]
    else:
        shuffles = []
        with open_file(None) if not create_shuffle else open_with_lineidx(shuffle) as fps:
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
            open_file(None) if not create_labelmap else open(labelmap, "w") as fpl:
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


def _sample_rects(db, keys, labels, max_label):
    if isinstance(labels, basestring):
        labels = [labels] * len(keys)
        multi_label = False
    else:
        multi_label = len(np.unique(labels)) > 1
    assert len(keys) == len(labels)

    # re-arrange so that we first sample each label, then each key and then each rect for the (label, key)
    label_keys = {}
    for key, label in zip(keys, labels):
        rects = [np.array(int_rect(rect['rect'])) for rect in db.truth_list(key, label)]
        key_rects = label_keys.get(label)
        if not key_rects:
            label_keys[label] = {key: rects}
            continue
        old_rects = key_rects.get(key)
        if not old_rects:
            key_rects[key] = rects
            continue
        old_rects += rects

    new_keys = []
    new_rects = []
    while label_keys and len(new_keys) < max_label:
        to_remove_label = []
        for label, key_rects in label_keys.iteritems():
            if not multi_label:
                keys = key_rects
            else:
                # each time we see the same label, shuffle the keys
                keys = key_rects.keys()
                np.random.shuffle(keys)
            to_remove = []
            for key in keys:
                rects = key_rects[key]
                new_keys.append(key)
                idx = np.random.randint(len(rects))
                new_rects.append(rects.pop(idx))
                if len(new_keys) == max_label:
                    break
                if not rects:
                    to_remove.append(key)
                if multi_label:
                    # break to process the next label before next key of the same label
                    break
            if len(new_keys) == max_label:
                break
            for key in to_remove:
                key_rects.pop(key, None)
            if not key_rects:
                to_remove_label.append(label)
        for label in to_remove_label:
            label_keys.pop(label, None)
    return new_keys, new_rects


def create_collage(db, tax=None, path=None, max_label=100, max_children=200, target_size=100,
                   start_label=None, end_label=None, seed=None):
    """Create single inverted file for a db
    :param db: the imdb to create
    :type db: ImageDatabase
    :param tax: taxonomy to sample leafs
    :type tax: mmod.taxonomy.Taxonomy
    :param path: output directory path for collages
    :type path: str
    :param max_label: maximum number of samples per-label
    :param max_children: maximum number of samples per leaf-labels
    :param target_size: patch size to align maximum dimension to
    :param start_label: starting label to start creating the collage for
    :param end_label: final label to create collage for
    :param seed: random seed
    """
    if seed is None:
        seed = np.random.randint(low=0, high=0xffffffff)
    np.random.seed(seed)
    if path is None:
        path = op.join(op.dirname(db.path), "collage_{}_{}".format(max_label, seed))
        makedirs(path, exist_ok=True)
    if tax is None and max_children:
        for tree_file in ["tree.txt", "root.yaml"]:
            tree = op.join(op.dirname(db.path), tree_file)
            if op.isfile(tree):
                tax = Taxonomy(tree)
                break
    assert op.isdir(path), "{} directory is not accessable".format(path)
    first_seen = False
    last_seen = False

    for label in db.iter_cmap():
        if last_seen:
            break
        if end_label and label == end_label:
            last_seen = True
        if start_label:
            if label == start_label:
                first_seen = True
            if not first_seen:
                continue
        logging.info("Sampling collage for {}".format(label))
        keys = list(db.iter_label(label))
        total = len(keys)
        if total > max_label:
            keys = [
                keys[idx]
                for idx in np.sort(np.random.choice(total, replace=False, size=(max_label,)))
            ]
            # take first/random rect from each key frame
            key_rects = [
                np.array(int_rect(db.truth_list(key, label)[0]['rect']))
                for key in keys
            ]
        else:
            keys, key_rects = _sample_rects(db, keys, label, max_label)

        if keys:
            jpg_path = op.join(path, "{}_{}.jpg".format(label.replace(" ", "_"), total))
            tile_rects(db, keys, key_rects, target_size, label, jpg_path)
            del keys, key_rects
        else:
            logging.error("Ignore label: {} for lack of data".format(label))

        if not tax or not max_children:
            continue
        for node in tax.iter_search_nodes(label):
            if node.is_leaf():
                continue
            logging.info("Sampling children of {}".format(label))
            children = list(node.iter_leaf_names())
            keys = []
            labels = []
            for child_label in children:
                new_keys = list(db.iter_label(child_label))
                keys += new_keys
                labels += [child_label] * len(new_keys)
                del new_keys
            total = len(keys)
            keys, key_rects = _sample_rects(db, keys, labels, max_children)
            jpg_path = op.join(path, "{}_children_{}_{}.jpg".format(label.replace(" ", "_"), len(children), total))
            tile_rects(db, keys, key_rects, target_size, label, jpg_path)

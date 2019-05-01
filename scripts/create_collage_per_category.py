import sys
import numpy as np
import os.path as op
import logging
import os as os
import json
import six

try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]

this_file = op.abspath(this_file)
sys.path.append(op.join(op.dirname(this_file), '..'))
sys.path.append(op.join(op.dirname(this_file), 'mmod'))

from mmod.imdb import ImageDatabase
from mmod.im_utils import int_rect, tile_rects

# add_path("mmod")
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

def _sample_rects(db, keys, labels, max_label):
    if isinstance(labels, six.string_types):
        labels = [labels] * len(keys)
        multi_label = False
    else:
        multi_label = len(np.unique(labels)) > 1
    assert len(keys) == len(labels)

    # re-arrange so that we first sample each label, then each key and then each rect for the (label, key)
    label_keys = {}
    for key, label in zip(keys, labels):
        rects = [np.array(int_rect(rect['rect']))
                 for rect in db.truth_list(key, label)]
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
        for label, key_rects in six.iteritems(label_keys):
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

def parse_ds_name(a):
    return a[a.find("/")+1: a.rfind("/")]

def load_list_file(fname):
    with open(fname, 'r') as fp:
        lines = fp.readlines()
    result = [line.strip() for line in lines]
    if len(result) > 0 and result[-1] == '':
        result = result[:-1]
    return result

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def process_term_list(term_list):
    new_term_list = []
    for full_term in term_list:
        terms = full_term.split('_')
        term = terms[0]

        i = 1
        while (not isfloat(terms[i])) and (terms[i] !="children"):
            term = term + " " + terms[i]
            i = i+1

        new_term_list.append(term)

    return new_term_list

def main():
    path_root = "/home/chunzhao/brand1048Train"
    db_filename = 'data/brand1048Clean/train.tsv'
    term_list_filename = '/raid/data/brand1048Clean/labelmap.txt'

    logging.info("db:{}".format(db_filename))
    logging.info("term_list:{}".format(term_list_filename)

    db = ImageDatabase(db_filename)
    term_list = load_list_file(term_list_filename)
    
    labels = term_list

    max_label = 100 
    target_size = 512


    if not os.path.exists(path_root):
        os.mkdir(path_root)

    for i, label in enumerate(labels):
        
        if label == "CrackerJack":
            continue
            
        # if len(labels) > 200:
        # path = op.join(path_root, label[0].lower())
        path = path_root
        if not os.path.exists(path):
            os.mkdir(path)
        # else:
        #     path = path_root

        for source_link in list(db.iter_sources()):
            # for key in db.iter_label(term , source='data/mturk700_url_as_keyClean/trainval.tsv'):
            keys = list(db.iter_label(label, source=source_link))

            total = len(keys)            
            
            if total == 0:
                continue
            
            if total > max_label:
                keys = [
                    keys[idx]
                    for idx in np.sort(np.random.choice(total, replace=False, size=(max_label,)))
                ]

                if label == "CrackerJack":
                    for key in keys:
                        logging.info("len(db.truth_list({}, {}))=".format(key, label, len(db.truth_list(key, label)) ))

                # take first/random rect from each key frame
                key_rects = [
                    np.array(int_rect(db.truth_list(key, label)[0]['rect']))
                    for key in keys
                ]

            else:
                keys, key_rects = _sample_rects(db, keys, label, max_label)
            
            dsName = parse_ds_name(source_link)

            if keys:
                jpg_path = op.join(path, "{}_{}.jpg".format(label.replace(" ", "_"), dsName))
                txt_path = op.join(path, "{}_{}.tsv".format(label.replace(" ", "_"), dsName))
                
                if not op.isfile(jpg_path) or not op.isfile(txt_path):
                    tile_rects(db, keys, key_rects, target_size, label, jpg_path)
                    with open(txt_path, 'w') as fw:
                        for i, key in enumerate(keys):
                            image_key = db.image_key(key)
                            rect = key_rects[i].tolist()
                            fw.write("{}\t{}\t{}\n".format(json.dumps(key), json.dumps(image_key), json.dumps(rect)))

                del keys, key_rects
            else:
                logging.error("Ignore label: {} for lack of data".format(label))

    sys.stdout.write("\n")

if __name__ == '__main__':
    # When run as script, modify path assuming absolute import
    main()

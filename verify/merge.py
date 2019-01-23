from __future__ import print_function
import sys
import logging
import os.path as op
import requests
import argparse
import json

from mmod.utils import init_logging, open_with_lineidx, makedirs
from mmod.imdb import ImageDatabase
from mmod.experiment import Experiment
from mmod.philly_utils import fix_winpath
from mmod.tax_utils import create_db_from_predict
from mmod.deteval import eval_one, print_reports

#in_path = "Tagger/groundtruth/mit1k.images.tsv"
in_path = "Tagger/groundtruth/instagram.images.tsv"
db = ImageDatabase(in_path)
exp = Experiment(db)

#det_2k = exp.load_detections("Tagger/groundtruth/mit1k.msft.tsv", group_by_label=False)
#det_5k = exp.load_detections("Tagger/groundtruth/mit1k.msft5K.tsv", group_by_label=False, thresh=0.0297)
det_2k = exp.load_detections("Tagger/groundtruth/instagram.msft.tsv", group_by_label=False)
det_5k = exp.load_detections("Tagger/groundtruth/instagram.msft5K.tsv", group_by_label=False, thresh=0.0297)

with open(op.expanduser("~/Desktop/bozak/instagram2+5.tsv"), 'wb') as fp:
    for key in db:
        uid = db.uid(key)
        tags = det_2k.get(uid, [])
        tags_2k = set(t['class'] for t in tags)
        tags += [d for d in det_5k.get(uid, []) if d['class'] not in tags_2k]
        results = ";".join([":".join([t['class'], "{}".format(t['conf'])]) for t in tags])
        fp.write("{}\t{}\n".format(
            db.image_key(key),
            results
        ))

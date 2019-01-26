from __future__ import print_function
import os.path as op

from mmod.utils import init_logging, open_with_lineidx
from mmod.imdb import ImageDatabase
from mmod.tax_utils import create_db_from_predict, create_inverted

init_logging()

in_path = "iris/Celeb/ImageRecogntion/2.selfie"
db = ImageDatabase(in_path)

with open_with_lineidx("iris/Celeb/Video/train.tsv") as fp:
    for k in db:
        im_id = op.basename(k)
        label = op.basename(op.dirname(k))
        fp.write("{}\t{}\t{}\n".format(
            im_id, label, db.base64(k)
        ))

all_im_id = {}
all_label = {}
with open("iris/Celeb/ImageRecogntion/2.selfie_SatoriIdMKeyMUrl.tsv") as fp:
    for line in fp:
        label, name, im_id = line.split("\t")
        all_im_id[name] = im_id.strip()
        all_label[name] = label

with open_with_lineidx("iris/Celeb/Selfie/train.tsv") as fp:
    for k in db:
        name = op.splitext(op.basename(k))[0]
        im_id = all_im_id[name]
        label = all_label[name]
        if im_id == 'Unknown':
            im_id = op.basename(k)
        fp.write("{}\t{}\t{}\n".format(
            im_id, label, db.base64(k)
        ))

db = ImageDatabase("iris/Celeb/Selfie/train.tsv")
create_inverted(db, create_shuffle=False)

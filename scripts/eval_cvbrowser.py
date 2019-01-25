from __future__ import print_function
import sys
import logging
import os.path as op
import json
import numpy as np
import requests
import argparse

try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = op.abspath(this_file)

if __name__ == '__main__':
    # When run as script, modify path assuming absolute import
    sys.path.append(op.join(op.dirname(this_file), '..'))

from mmod.utils import makedirs, is_number, init_logging, open_with_lineidx
from mmod.imdb import ImageDatabase
from mmod.experiment import Experiment
from mmod.api_utils import convert_api_od
from mmod.runeval import run_eval
from mmod.tax_utils import create_db_from_predict, resample_db
from mmod.philly_utils import fix_winpath
from mmod.im_utils import img_from_bytes, im_rescale, img_to_bytes


class Logger(object):
    @staticmethod
    def set_iterations(iterations):
        print("Iterations: {}    ".format(iterations), end='\r')
        sys.stdout.flush()


def main():
    init_logging()
    parser = argparse.ArgumentParser(description='Downsample a db and use the downsampled db to evaluate cvbrowser.')
    parser.add_argument('dbpath', metavar='IN_DB_PATH', help='Path to an image db (to downsample and evaluate)')
    parser.add_argument('outdbpath', metavar='OUT_DB_DIR',
                        help='Directory path to create the downsampled db with predictions as ground truth.')
    parser.add_argument('--size', help='Number of random samples to take from IN_DB_PATH database',
                        type=int, default=-1)
    parser.add_argument('--seed', help='Random seed', type=int, default=7889)
    parser.add_argument('--thresh', '--class_thresh',
                        help='Class threshold file to use in sampling from predictions (or a single threshod)',
                        default=0.6)
    parser.add_argument('--obj_thresh', '--objthresh',
                        help='Objectness threshold to sample from predictions',
                        default=0.2)
    parser.add_argument('--predict',
                        help='Prediction file to use (to avoid prediction step)')
    parser.add_argument('--auth',
                        help='Authentication file path (or key)')
    parser.add_argument('--subscription', '--subs',
                        help='Authentication is subscription key', action='store_true')
    parser.add_argument('--api',
                        help='CV API endpoint url (pass "" to avoid evaluation)',
                        default="http://tax1300v14x3.westus.azurecontainer.io/api/detect")
    args = vars(parser.parse_args())

    api_endpoint = args['api']
    in_path = args['dbpath']
    assert in_path and op.isfile(in_path), "'{}' does not exists, or is not a file".format(
        in_path
    )

    obj_thresh = args['obj_thresh']
    assert obj_thresh >= 0, "obj_thresh: {} < 0".format(obj_thresh)
    thresh = args['thresh']
    if is_number(thresh, float):
        class_thresh = float(thresh)
        logging.info("No per-class threshold file given")
    else:
        assert op.isfile(thresh), 'threshold file: {} does not exist'.format(thresh)
        with open(thresh) as f:
            class_thresh = {l[0]: float(l[1].rstrip()) for l in
                            [line.split('\t') for line in f.readlines()]}

    db = ImageDatabase(in_path)
    predict_file = fix_winpath(args['predict'])
    out_path = args['outdbpath']
    outtsv_file = op.join(out_path, "cvbrowser.predict")
    makedirs(out_path, exist_ok=True)
    if predict_file:
        assert predict_file and op.isfile(predict_file), "{} does not exist".format(predict_file)
        logging.info("Creating db in {} from {}".format(out_path, predict_file))
        db = create_db_from_predict(db, predict_file, class_thresh, out_path)

    if not api_endpoint:
        raise SystemExit("No endpoint is given to evaluate {}".format(db))

    np.random.seed(args['seed'])
    size = args['size']
    if size <= 0 or size > len(db):
        size = len(db)
    if size != len(db):
        input_range = np.random.choice(len(db) - 1, replace=False, size=(size,))
        logging.info("Creating size {} db in {} from {}".format(size, out_path, db))
        db = resample_db(db, input_range, out_path)

    session = requests.Session()
    auth = args['auth']
    if auth:
        auth_path = op.expanduser(args['auth'])
        if op.exists(auth_path):
            with open(auth_path) as fp:
                auth = fp.readline().strip()
        if args['subscription']:
            session.headers.update({'Ocp-Apim-Subscription-Key': auth})
        else:
            session.auth = ("Basic", auth)
    session.headers.update({'Content-Type': 'application/octet-stream'})

    logging.info("cvapi detection for {}".format(db))
    processed = 0
    total_count = len(db)
    with open_with_lineidx(outtsv_file, "w") as fp, \
            open_with_lineidx(outtsv_file + ".keys") as kfp:
        for key in db:
            uid = db.uid(key)
            data = db.raw_image(uid)
            # work around service size limit
            if len(data) > 1024 * 1024 * 4:
                data = img_from_bytes(data)
                logging.info("Resize uid: {} from {}x{}".format(uid, data.shape[0], data.shape[1]))
                data = img_to_bytes(im_rescale(data, 2048))
            res = session.post(api_endpoint, data=data)
            if res.status_code == 400:
                # show parsable errors
                try:
                    err = json.loads(res.content)
                    logging.error("Error: {} uid: {}".format(err, uid))
                    continue
                except ValueError:
                    pass
            assert res.status_code == 200, "cvapi post failed uid: {}".format(uid)
            result = convert_api_od(json.loads(res.content)['objects'])
            result = [
                crect for crect in result
                if crect['conf'] >= (class_thresh[crect['class']] if class_thresh else thresh)
            ]
            tell = fp.tell()
            fp.write("{}\t{}\n".format(
                db.image_key(key),
                json.dumps(result, separators=(',', ':'), sort_keys=True),
            ))
            kfp.write("{}\t{}\n".format(
                uid, tell
            ))
            processed += 1
            print("{} of {}    ".format(processed, total_count), end='\r')
            sys.stdout.flush()

    exp = Experiment(db, predict_path=outtsv_file)
    logging.info("cvapi evaluate od {}".format(exp))
    run_eval(exp)


if __name__ == '__main__':
    main()

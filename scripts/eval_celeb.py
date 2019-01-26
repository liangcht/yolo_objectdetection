from __future__ import print_function
import sys
import logging
import os.path as op
import requests
import argparse
import json

try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = op.abspath(this_file)

if __name__ == '__main__':
    # When run as script, modify path assuming absolute import
    sys.path.append(op.join(op.dirname(this_file), '..'))

from mmod.utils import init_logging, open_with_lineidx, makedirs
from mmod.imdb import ImageDatabase
from mmod.experiment import Experiment
from mmod.philly_utils import fix_winpath
from mmod.tax_utils import create_db_from_predict
from mmod.deteval import eval_one, print_reports
from mmod.api_utils import convert_api_celeb
from mmod.im_utils import img_from_bytes, im_rescale, img_to_bytes


def main():
    init_logging()
    parser = argparse.ArgumentParser(description='Evaluate a db against a celebrity endpoint.')
    parser.add_argument('dbpath', metavar='IN_DB_PATH', help='Path to an image db to evaluate')
    parser.add_argument('outdbpath', metavar='OUT_DB_DIR',
                        help='Directory path to create intermediate db with given predictions as ground truth.')
    parser.add_argument('--predict',
                        help='Prediction file to use (to create intermediate OUT_DB_DIR)')
    parser.add_argument('--auth',
                        help='Authentication file path (or key)')
    parser.add_argument('--subscription', '--subs',
                        help='Authentication is subscription key', action='store_true')
    parser.add_argument('--api',
                        help='CV API endpoint url (pass "" to avoid evaluation)',
                        default="http://localhost:9000/api/models/celebrities/analyze")
    parser.add_argument('--thresh', default=0, type=float,
                        help='Threshold to apply')
    args = vars(parser.parse_args())

    api_endpoint = args['api']
    in_path = args['dbpath']
    assert in_path and op.isfile(in_path), "'{}' does not exists, or is not a file".format(
        in_path
    )
    thresh = args['thresh'] or 0

    db = ImageDatabase(in_path)
    predict_file = fix_winpath(args['predict'])
    out_path = args['outdbpath']
    outtsv_file = op.join(out_path, "tagger.predict")
    makedirs(out_path, exist_ok=True)
    if predict_file:
        assert predict_file and op.isfile(predict_file), "{} does not exist".format(predict_file)
        logging.info("Creating db in {} from {}".format(out_path, predict_file))
        db = create_db_from_predict(db, predict_file, thresh, out_path)

    logging.info("cvapi tagging {}".format(db))

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
            result = convert_api_celeb(json.loads(res.content)['result']['celebrities'])
            tell = fp.tell()
            fp.write("{}\t{}\n".format(
                db.image_key(key),
                json.dumps(result, separators=(',', ':'), sort_keys=True)
            ))
            kfp.write("{}\t{}\n".format(
                uid, tell
            ))
            processed += 1
            print("{} of {}    ".format(processed, total_count), end='\r')
            sys.stdout.flush()

    exp = Experiment(db)
    logging.info("cvapi evaluate tagging {}".format(exp))
    detresults = exp.load_detections(outtsv_file, thresh=thresh)
    truths = db.all_truths()
    reports = {'all': {-1: eval_one(truths, detresults)}}
    print_reports(reports, report_file_table=op.join(out_path, "report.table"))


if __name__ == '__main__':
    main()

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


def main():
    init_logging()
    parser = argparse.ArgumentParser(description='Evaluate a db against a tagger endpoint.')
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
                        default="http://localhost:9000/api/tag")
    parser.add_argument('--thresh', '--class_thresh',
                        action='append',
                        help='Class threshold file(s) to apply to predictions')
    args = vars(parser.parse_args())

    api_endpoint = args['api']
    in_path = args['dbpath']
    assert in_path and op.isfile(in_path), "'{}' does not exists, or is not a file".format(
        in_path
    )
    threshs = args['thresh'] or []
    class_thresh = {}
    for thresh in threshs:
        with open(thresh) as f:
            th = {}
            is_scaled = False
            for l in [line.split('\t') for line in f.readlines()]:
                val = float(l[1].rstrip())
                if not is_scaled and val >= 100:
                    is_scaled = True
                    logging.info("Threshld file {} is scaled by 1000".format(thresh))
                    for k in th:
                        th[k] /= 1000
                if is_scaled:
                    val /= 1000
                th[l[0]] = val
            class_thresh.update(th)

    db = ImageDatabase(in_path)
    predict_file = fix_winpath(args['predict'])
    out_path = args['outdbpath']
    outtsv_file = op.join(out_path, "tagger.predict")
    makedirs(out_path, exist_ok=True)
    if predict_file:
        assert predict_file and op.isfile(predict_file), "{} does not exist".format(predict_file)
        logging.info("Creating db in {} from {}".format(out_path, predict_file))
        db = create_db_from_predict(db, predict_file, class_thresh, out_path)

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
            tags = json.loads(res.content)['tags']
            results = ";".join([":".join([t['name'], "{}".format(t['confidence'])]) for t in tags])
            tell = fp.tell()
            fp.write("{}\t{}\n".format(
                db.image_key(key),
                results
            ))
            kfp.write("{}\t{}\n".format(
                uid, tell
            ))
            processed += 1
            print("{} of {}    ".format(processed, total_count), end='\r')
            sys.stdout.flush()

    exp = Experiment(db)
    logging.info("cvapi evaluate tagging {}".format(exp))
    detresults = exp.load_detections(outtsv_file, class_thresh or 0)
    truths = db.all_truths()
    reports = {'all': {-1: eval_one(truths, detresults)}}
    print_reports(reports, report_file_table=op.join(out_path, "report.table"))


if __name__ == '__main__':
    main()

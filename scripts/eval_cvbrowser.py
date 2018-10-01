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

from mmod.utils import makedirs, is_number, init_logging, open_with_lineidx, tsv_read
from mmod.imdb import ImageDatabase
from mmod.experiment import Experiment
from mmod.api_utils import convert_api
from mmod.runeval import run_eval, run_detect
from mmod.simple_parser import parse_prototxt, parse_key_value


def _conv_label(lb, cmap):
    if lb in cmap:
        return lb
    lb = lb.replace('_', ' ')
    assert lb in cmap, 'label: {} not valid'.format(lb)
    return lb


class Logger(object):
    @staticmethod
    def set_iterations(iterations):
        print("Iterations: {}    ".format(iterations), end='\r')
        sys.stdout.flush()


if __name__ == '__main__':
    init_logging()
    parser = argparse.ArgumentParser(description='Downsample a db and use the downsampled db to evaluate cvbrowser.')
    parser.add_argument('dbpath', metavar='IN_DB_PATH', help='Path to an image db (to downsample)')
    parser.add_argument('outdbpath', metavar='OUT_DB_DIR',
                        help='Directory path to create the downsampled db with predictions as ground truth')
    parser.add_argument('--overwrite', help='If should overwrite exsisting OUT_DB_DIR', action='store_true')
    parser.add_argument('--clean', help='If should re-do all predictions', action='store_true')
    parser.add_argument('--size', help='Number of random samples to take from IN_DB_PATH database',
                        type=int, default=100)
    parser.add_argument('--seed', help='Random seed', type=int, default=7889)
    parser.add_argument('--thresh', '--class_thresh',
                        help='Class threshold file to use in sampling from predictions (or a single threshod)',
                        default=0.6)
    parser.add_argument('--obj_thresh', '--objthresh',
                        help='Objectness threshold to sample from predictions',
                        default=0.2)
    parser.add_argument('-s', '--solver', metavar='SOLVER',
                        help='Caffe solver prototxt (for testing)')
    parser.add_argument('-weights', '--weights', '--caffemodel',
                        help='caffemodel weights (will use SOLVER to find if not given)')
    parser.add_argument('--net', '--caffenet',
                        help='test caffenet prototxt (will use SOLVER to find if not given)')
    parser.add_argument('--cmap', '--labelmap',
                        help='labelmap file (will use SOLVER to find if not given)')
    parser.add_argument('--auth',
                        help='Authentication file path',
                        default="~/auth/cvbrowser.txt")
    parser.add_argument('--api',
                        help='CV API endpoint url',
                        default="http://tax1300v14x1.westus.azurecontainer.io/api/detect")
    parser.add_argument('-e', '--expid', '--jobid',
                        help='The full experiment ID',
                        required=True)
    args = vars(parser.parse_args())

    exp_id = args['expid']
    assert exp_id, "experiemnt ID is required"

    out_path = args['outdbpath']
    assert out_path and not op.isfile(out_path), "OUT_DB_DIR: '{}' must be a directory".format(out_path)
    assert args['overwrite'] or not op.isdir(out_path), "{} exists, use --overwrite to overwrite it".format(out_path)
    makedirs(out_path, exist_ok=True)
    assert op.isdir(out_path), "{} is not an accessable directory path".format(out_path)

    path = args['dbpath']
    assert op.isfile(path), "{} is not a valid file path".format(path)
    imdb = ImageDatabase(path)
    size = args['size']
    assert len(imdb) >= size > 0, "Invalid sample size: {} for db: {}".format(size, imdb)

    caffenet = args['net']
    caffemodel = args['weights']
    solverfile = args['solver']
    cmapfile = args['cmap']
    train_protofile = None
    if solverfile:
        if caffemodel and caffenet and cmapfile:
            logging.info("caffenet: {} and caffemodel: {} and cmapfile: {} are given, ignoring solver: {}".format(
                caffenet, caffemodel, cmapfile, solverfile
            ))
        else:
            net = parse_prototxt(solverfile)
            if not caffemodel:
                caffemodel = net["snapshot_prefix"] + "_iter_" + net["max_iter"] + ".caffemodel"
                assert op.isfile(caffemodel), "caffemodel: {} in {} does not exist".format(caffemodel, solverfile)
            if not caffenet or not cmapfile:
                train_protofile = net.get("train_net") or net.get("net")
            if not cmapfile:
                cmapfile = parse_key_value(train_protofile, "labelmap")
                assert op.isfile(cmapfile), "cmapfile: {} in {} does not exist".format(cmapfile, solverfile)
            if not caffenet:
                caffenet = op.join(op.dirname(train_protofile), "test.prototxt")
                assert op.isfile(caffenet), "caffenet: {} in {} does not exist".format(caffenet, solverfile)
    assert caffemodel and caffemodel and op.isfile(caffemodel) and op.isfile(caffenet), \
        "caffemodel: {} or caffenet: {} do not exist".format(caffenet, caffemodel)

    if not cmapfile and caffenet:
        cmapfile = parse_key_value(caffenet, "labelmap")
        if not cmapfile:
            cmapfile = op.join(op.dirname(parse_key_value(caffenet, "tree")), "labelmap.txt")
    assert cmapfile and op.isfile(cmapfile), "cmapfile: {} does not exist".format(cmapfile)

    name = op.basename(caffemodel) + "." + exp_id
    exp = Experiment(imdb, caffenet, caffemodel, name=name, cmapfile=cmapfile)
    logging.info("Input experiment: {}".format(exp))
    outtsv_file = exp.predict_path
    new_label_file = op.join(out_path, "test0.tsv")

    obj_thresh = args['obj_thresh']
    assert obj_thresh >= 0, "obj_thresh: {} < 0".format(obj_thresh)
    thresh = args['thresh']
    class_thresh = None
    if is_number(thresh, float):
        thresh = float(thresh)
        logging.info("No per-class threshold file given")
    else:
        assert op.isfile(thresh), 'threshold file: {} does not exist'.format(thresh)
        with open(thresh) as f:
            class_thresh = {_conv_label(l[0], exp.cmap): float(l[1].rstrip()) for l in
                            [line.split('\t') for line in f.readlines()]}
        thresh = None

    np.random.seed(args['seed'])
    input_range = np.random.choice(len(imdb) - 1, replace=False, size=(size,))

    if not op.isfile(outtsv_file) or args['clean']:
        run_detect(exp=exp, logger=Logger(),
                   thresh=thresh, class_thresh=class_thresh, obj_thresh=obj_thresh, input_range=input_range)
    else:
        logging.info("Using previous prediction: {}, use --clean to re-do".format(outtsv_file))

    all_det = {}
    with open(outtsv_file) as fp, open(outtsv_file + ".keys") as fpk:
        for line in fpk:
            uid, offset = line.split("\t")
            offset = int(offset)
            cols = tsv_read(fp, 2, seek_offset=offset)
            all_det[uid] = json.loads(cols[1])

    with open_with_lineidx(new_label_file) as fp:
        with imdb.open():
            for idx in range(len(imdb)):
                key = exp.imdb.normkey(idx)
                uid = exp.imdb.uid(key)
                if uid not in all_det:
                    fp.write("d\td\n")
                    continue
                result = all_det[uid]
                fp.write("{}\t{}\n".format(
                    uid,
                    json.dumps(result, separators=(',', ':'), sort_keys=True),
                ))

    path = op.join(out_path, "testX.tsv")
    logging.info("Creating the down-sampled db at `{}` from db: {}".format(path, imdb.path))
    with open(path, "w") as fp:
        fp.write("{}\n".format(imdb.path))
    with open(op.join(out_path, "testX.label.tsv"), "w") as fp:
        fp.write("{}\n".format(new_label_file))
    # now load the newly created db
    evaldb = ImageDatabase(path)
    name = op.basename(caffemodel) + "." + exp_id + ".cvbrowser"
    evalexp = Experiment(evaldb, caffenet, caffemodel, name=name, cmapfile=cmapfile)
    outtsv_file = evalexp.predict_path

    session = requests.Session()
    with open(op.expanduser(args['auth'])) as fp:
        session.auth = ("Basic", fp.readline().strip())
    session.headers.update({'Content-Type': 'application/octet-stream'})

    logging.info("cvapi detection for {}".format(evalexp))
    processed = 0
    total_count = len(evaldb)
    with open_with_lineidx(outtsv_file, "w") as fp:
        for key in evaldb:
            uid = evaldb.uid(key)
            data = evaldb.raw_image(uid)
            res = session.post(args['api'], data=data)
            assert res.status_code == 200, "cvapi post failed uid: {}".format(uid)
            result = convert_api(json.loads(res.content)['objects'])
            result = [
                crect for crect in result
                if crect['conf'] >= (class_thresh[crect['class']] if class_thresh else thresh)
            ]
            fp.write("{}\t{}\n".format(
                uid,
                json.dumps(result, separators=(',', ':'), sort_keys=True),
            ))
            processed += 1
            print("{} of {}    ".format(processed, total_count), end='\r')
            sys.stdout.flush()

    logging.info("Evaluating {}".format(exp))
    run_eval(evalexp)

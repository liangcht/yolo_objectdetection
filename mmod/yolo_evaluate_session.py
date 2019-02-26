import os
import os.path as op
import argparse
from mmod.experiment import Experiment
from mmod.deteval import eval_one, print_reports
from mmod.imdb import ImageDatabase
from mmod.deteval import deteval
from mmod.runeval import run_eval


def get_parser():
    """prepares and returns argument parser"""
    parser = argparse.ArgumentParser(description='PyTorch Yolo Evaluation')
    parser.add_argument('-d', '--dbpath', metavar='dataset_path',
                        help='full path to dataset tsv file', required=True)
    parser.add_argument('--predict',
                        help='Prediction file to evaluate')
    return parser


def main():
    # parsing arguments
    args = get_parser().parse_args()
    args = vars(args)

    # dataset
    in_path = args['dbpath']
    assert in_path and op.isfile(in_path), "'{}' does not exists, or is not a file".format(
        in_path
    )
    db = ImageDatabase(in_path)

    predict_file = args["predict"]
    if predict_file:
        assert predict_file and op.isfile(predict_file), "{} does not exist".format(predict_file)

    # evaluation
    exp = Experiment(db, input_range=xrange(len(db)), predict_path=predict_file)
    err_map = run_eval(exp, ovthresh=None)
    print("mAP@0.3", err_map)


if __name__ == '__main__':
    main()

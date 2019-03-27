import os.path as op
import argparse
from mmod.experiment import Experiment
from mmod.imdb import ImageDatabase
from mmod.runeval import run_eval


def get_parser():
    """prepares and returns argument parser"""
    parser = argparse.ArgumentParser(description='PyTorch Yolo Evaluation')
    parser.add_argument('-d', '--dbpath', metavar='dataset_path',
                        help='full path to dataset tsv file', required=True)
    parser.add_argument('--predict',
                        help='Prediction file to evaluate')
    return parser


def evaluate_prediction(predict_filename, dataset_filename):
    """function for evaluation of prediction results:
    :param: predict_filename, str - full path to the file with prediction results
    :param: dataset_filename, str - full path to the dataset file
     """
    in_path = dataset_filename
    assert in_path and op.isfile(in_path), "'{}' does not exists, or is not a file".format(
        in_path
    )
    db = ImageDatabase(in_path)

    predict_file = predict_filename
    if predict_file:
        assert predict_file and op.isfile(predict_file), "{} does not exist".format(predict_file)

    # evaluation
    exp = Experiment(db, input_range=xrange(len(db)), predict_path=predict_file)
    err_map = run_eval(exp, ovthresh=None)

    if not err_map:
        print("Evaluation result already exists in ", op.dirname(predict_filename))
    else:
        print("mAP@0.5", err_map)


def main():
    # parsing arguments
    args = get_parser().parse_args()
    args = vars(args)

    evaluate_prediction(args["predict"], args['dbpath'])


if __name__ == '__main__':
    main()

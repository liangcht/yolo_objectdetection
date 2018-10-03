from __future__ import print_function
import os
import time
import os.path as op
import sys
import argparse
import logging
import json
import shutil
from loky.backend.queues import Queue
from loky.backend.process import LokyProcess as Process

try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = op.abspath(this_file)

if __name__ == '__main__':
    # When run as script, modify path assuming absolute import
    sys.path.append(op.join(op.dirname(this_file), '..'))

from mmod.phillylogger import PhillyLogger
from mmod.utils import makedirs, cwd, init_logging, gpu_indices, ompi_rank
from mmod.runcaffe import moved_solvers
from mmod.philly_utils import abspath, get_log_parent, get_arg, last_log_dir, set_job_id
from mmod.checkpoint import iterate_weights
from mmod.io_utils import write_to_file, read_to_buffer
from mmod.deteval import deteval
from mmod.detector import Detector, write_predict, read_image
from mmod.imdb import ImageDatabase
from mmod.experiment import Experiment
from mmod.filelock import FileLock


def run_detect(gpus=None, caffenet=None, caffemodel=None,
               exp=None, thresh=None, class_thresh=None, obj_thresh=None,
               test_data=None,
               logger=None, interval=1, iterations=0, input_range=None, root=None, expid=None):
    """Run detection detection and evaluation
    :param gpus: list of gpus to use
    :type gpus: list
    :param caffenet: The caffe test prototxt file path
    :type caffenet: str
    :param caffemodel: The caffemodel weights file path
    :type caffemodel: str
    :param exp: The experiment to initialize with
    :type exp: Experiment
    :param thresh: global class threshold
    :param class_thresh: per-class threshold
    :param obj_thresh: objectness threshold
    :param test_data: test_data for evaluation
            it can be a subdirectory name inside the 'data' folder
            it can be an constructed as ImageDatabase
    :type test_data: str
    :param logger: philly logger instance
    :type logger: PhillyLogger
    :param interval: snapshot interval to use for normalization of progress
    :param iterations: total iterations so far
    :param input_range: range of inputs to evaluate (default is to evaluate all)
    :param root: experiment roo
    :param expid: Extra experiment ID
    :rtype: mmod.experiment.Experiment
    """
    if gpus is None:
        gpus = list(gpu_indices())
    num_gpu = len(gpus)
    logging.info("Detecting {} for {} on {} GPUs".format(caffemodel, test_data, num_gpu))

    if exp is not None:
        imdb = exp.imdb
    else:
        assert test_data
        assert caffenet and caffemodel and op.isfile(caffenet) and op.isfile(caffemodel)
        name = None
        if op.isabs(test_data) and (op.isdir(test_data) or op.isfile(test_data)):
            imdb = ImageDatabase(test_data)
        else:
            # when test_data is a subdirectory inside 'data', quickdetection generated
            for fname in ['testX.tsv', 'test.tsv']:
                intsv_file = op.join('data', test_data, fname)
                if op.isfile(intsv_file):
                    break
            name = op.basename(caffemodel) + "." + test_data
            imdb = ImageDatabase(intsv_file, name=test_data)
            if expid:
                name += ".{}".format(expid)

        caffemodel_clone = None
        if op.isdir("/tmp"):
            caffemodel_clone = op.join("/tmp", "{}.caffemodel".format(ompi_rank()))
            shutil.copy(caffemodel, caffemodel_clone)
            if os.stat(caffemodel_clone).st_size < 1:
                logging.error("caffemodel: {} is not ready yet".format(caffemodel))
                return
        exp = Experiment(imdb, caffenet, caffemodel, caffemodel_clone=caffemodel_clone,
                         input_range=input_range, name=name, root=root, expid=expid)

    outtsv_file = exp.predict_path
    if op.isfile(outtsv_file):
        logging.info("Ignore already computed prediction: {} Experiment: {}".format(outtsv_file, exp))
        return exp

    # create one detector for each GPU
    detectors = [
        Detector(exp, num_gpu=num_gpu, gpu=gpu) for gpu in gpus
    ]

    logging.info("Detection Experiment {}".format(exp))

    if input_range is None:
        input_range = xrange(len(imdb))
    else:
        assert input_range[0] >= 0, "Invalid range: {} in {}".format(input_range, imdb)
        if input_range[-1] >= len(imdb):
            logging.info("Last range corrected: {} in {}".format(input_range, imdb))
            input_range = range(input_range[0], len(imdb))
        if len(input_range) == 0:
            logging.warning("Empty range: {} Experiment: {}".format(input_range, exp))
            return exp
    total_count = len(input_range)
    assert total_count, "No data to evaluate in experiment: {}".format(exp)
    assert total_count < 0xFFFFFFFF, "Too many images to evaluate"
    processed = 0
    in_queue = Queue(2000 * len(gpus))

    def result_done(res):
        in_queue.put(res.result())

    writer = None
    reader = None
    try:
        # noinspection PyBroadException
        try:
            writer = Process(name="writer", target=write_predict, args=(outtsv_file, in_queue,))
            writer.daemon = True
            writer.start()

            out_queue = Queue(400 * len(gpus))
            reader = Process(name="reader", target=read_image, args=(imdb, input_range, out_queue,))
            reader.daemon = True
            reader.start()

            idx = 0
            while True:
                idx += 1
                out = out_queue.get()
                if not out:
                    break
                key, im = out
                det_idx = idx % len(detectors)
                detector = detectors[det_idx]
                result = detector.detect_async(
                    key, im=im,
                    thresh=thresh, class_thresh=class_thresh, obj_thresh=obj_thresh
                )
                result.add_done_callback(result_done)  # call when future is done to averlap
                processed += 1
                if logger and processed % 100 == 0:
                    logger.set_iterations(iterations + interval * float(processed) / total_count)
        except Exception as e:
            logging.error("Exception thrown: {}".format(e))
            raise
    finally:
        logging.info("Joining reader")
        if reader:
            reader.join()
        logging.info("Shutting down the detectors")
        for detector in detectors:
            detector.shutdown()
        if writer:
            in_queue.put(None)
            writer.join()
    return exp


def run_eval(exp, ovthresh=None):
    """Run evaluation of the predictions
    :param exp: The experiment to initialize with
    :type exp: Experiment
    :param ovthresh: overlap thresholds
    :returns: err_map: mean average precision
    """
    report_file, result = deteval(exp, ovthresh=ovthresh)
    simple_file = report_file + '.map.json'
    simple_class_file = report_file + '.class_ap.json'
    if op.isfile(simple_class_file) and op.isfile(simple_file):
        logging.info("Ignore already computed evaluation: {}".format(simple_file))
        return
    if result is None:
        logging.info('read report from: {}'.format(report_file))
        result = read_to_buffer(report_file)
        logging.info('json parsing...')
        result = json.loads(result)

    overall = result['overall']
    err = None
    for key in ['0.3', 0.3, '-1', -1]:
        if key in overall:
            err = overall[key]['map']
            break
    assert err is not None, "Could not find the loss"

    if not op.isfile(simple_file):
        s = {}
        for size_type in result:
            if size_type not in s:
                s[size_type] = {}
            for thresh in result[size_type]:
                if thresh not in s[size_type]:
                    s[size_type][thresh] = {}
                s[size_type][thresh]['map'] = \
                    result[size_type][thresh]['map']
        write_to_file(json.dumps(s), simple_file)
    if not op.isfile(simple_class_file):
        s = {}
        for size_type in result:
            if size_type not in s:
                s[size_type] = {}
            for thresh in result[size_type]:
                if thresh not in s[size_type]:
                    s[size_type][thresh] = {}
                s[size_type][thresh]['class_ap'] = \
                    result[size_type][thresh]['class_ap']
        write_to_file(json.dumps(s), simple_class_file)

    return err


def main():
    init_logging()
    parser = argparse.ArgumentParser(description='Evaluate a Caffe training (while it is running).')

    parser.add_argument('-d', '--datadir', '--dataDir', help='Data directory where the dataset is located',
                        required=True)
    parser.add_argument('-m',  '--outputdir', '--modeldir', '--modelDir',
                        help='Ignored',
                        required=False)
    parser.add_argument('-l', '--logdir', '--logDir', help='Log directory', required=False)
    parser.add_argument('--prevmodelpath', help='Previous model path', required=False)
    parser.add_argument('--configfile', '--stdoutdir', '--numgpu', help='Ignored', required=False)
    # ------------------------------------------------------------------------------------------------------------
    parser.add_argument('-s', '--solver', action='append',
                        help='Prototxt solver file for caffe training (specify multiple times to chain -weights)',
                        required=True)
    parser.add_argument('-prev', '--prev', help='Previous model path (override)', required=False)
    parser.add_argument('-g', '--gpus', action='append', type=int,
                        help='GPU device ordinals to restrict the evaluation to',
                        required=False)
    parser.add_argument('--use_lock', help='Use file locks (needed when multiple process write to logs)',
                        action='store_true')
    parser.add_argument('-p', '--path', '-weights', '--weights',
                        help=('Job model diretory to evaluate (default is snapshot/ besides solver)'
                              ' or a single caffemodel file path to evaluate'),
                        required=False)
    parser.add_argument('-range', '--range', help='Range of inputs to evaluate', required=False,
                        type=int, nargs=2)
    parser.add_argument('-test', '--test_data', '--test', help='Test data to evaluate (overrides default)',
                        required=False,
                        action='append')
    parser.add_argument('-e', '--expid', '--jobid', help='The full experiment ID (if local, will be used as job ID)',
                        required=False)

    args = parser.parse_args()
    args = vars(args)
    print("Arguments: {}".format(args))
    set_job_id(args['expid'])

    log_dir = args['logdir']
    if not log_dir:
        log_dir = last_log_dir()
    log_dir = abspath(log_dir, roots=['~', '.'])
    if not op.isdir(log_dir):
        raise RuntimeError("Log directory '{}' does not exist".format(log_dir))

    data_path = abspath(args['datadir'], roots=['#', '.'])
    if not op.isdir(data_path):
        raise RuntimeError("Data directory '{}' does not exist".format(data_path))
    solvers = args['solver']
    if not solvers:
        raise RuntimeError("Train solver file(s) must be specified")
    solvers = [abspath(solver, roots=['~', '#', '.']) for solver in solvers]

    # We want to monitor a previous run, so we do not care about current output directory
    eval_path = get_arg(args['path'])
    if eval_path:
        eval_path = abspath(eval_path, roots=['#', '.'])
        if not op.isdir(eval_path) and not op.isfile(eval_path):
            raise RuntimeError("Eval path {} does not exist".format(eval_path))
    input_range = args['range']
    if input_range:
        input_range = xrange(input_range[0], input_range[1])

    use_lock = not args["use_lock"]
    prev_log_parent = get_log_parent(get_arg(args['prev']) or get_arg(args['prevmodelpath']))

    # local GPU devices
    gpus = args['gpus']
    if not gpus:
        gpus = list(gpu_indices())

    work_path = '/tmp/work'
    # Work-around to use current taxonomies with no change
    with FileLock(work_path):
        sym_data_path = op.join(work_path, 'data')
        if not op.isdir(sym_data_path):
            logging.info("Creating the sym path: {}".format(sym_data_path))
            makedirs(work_path, exist_ok=True)
            try:
                os.symlink(data_path, sym_data_path)
            except OSError as e:
                if not op.isdir(sym_data_path):
                    print('Symlink: {} Error: {}'.format(sym_data_path, e))

    with cwd(work_path):
        caffemodel_path = None
        model_dir = eval_path
        if model_dir and op.isfile(eval_path):
            model_dir = None
            caffemodel_path = eval_path

        metas = moved_solvers(solvers, model_dir=model_dir)
        prefixes = [m['snapshot_prefix'] for m in metas]
        max_iters = [m['max_iter'] for m in metas]
        test_data = args['test_data']
        if test_data:
            all_test_data = [test_data] * len(solvers)
        else:
            all_test_data = [[ops['name'] for ops in m['dataset_ops'] if 'name' in ops] for m in metas]
        intervals = [m['snapshot'] for m in metas]
        # Run secondary test data as sub-commands
        subcommand_names = [[test_data for test_data in test_data_list[1:]]
                            for test_data_list in all_test_data]
        with PhillyLogger(
            log_dir,
            max_iters,
            [op.join(op.basename(op.dirname(solver_path)), op.basename(solver_path))
             for solver_path in solvers],
            subcommand_names=subcommand_names,
            use_lock=use_lock,
            prev_log_parent=prev_log_parent
        ).redirected() as logger:
            logging.info("Arguments: {}".format(json.dumps(args, indent=2)))
            logging.info("gpus: {}->{}".format(gpus[0], gpus[-1]))
            done = False
            more = False
            _done_set = set()
            _last_time = None
            while more or not done:
                for idx, iterations, cur_iter, caffemodel, done, more in iterate_weights(
                        prefixes,
                        intervals,
                        max_iters,
                        caffemodel=caffemodel_path
                ):
                    # if same model is returned again
                    if cur_iter in _done_set:
                        if not _last_time:
                            _last_time = time.time()
                        if time.time() - _last_time > 30:
                            # if last model is done, previous models will not appear too, so break
                            if done:
                                more = False
                                break
                        time.sleep(5)
                    _last_time = None
                    _done_set.add(cur_iter)
                    logger.new_command(idx)
                    logger.set_iterations(iterations, cur_iter=cur_iter)
                    solver_path = solvers[idx]
                    caffenet = abspath(op.join(op.dirname(solver_path), 'test.prototxt'),
                                       roots=['~', '#'])

                    interval = intervals[idx]
                    errs = []
                    all_tests = all_test_data[idx]
                    if not all_tests:
                        logging.info("No test found for {}".format(solver_path))
                    for test_data in all_tests:
                        assert isinstance(test_data, basestring)
                        exp = run_detect(gpus, caffenet, caffemodel,
                                         test_data=test_data,
                                         logger=logger, interval=interval, iterations=iterations,
                                         input_range=input_range,
                                         expid=args['expid'])
                        if exp is None:
                            errs = []
                            break

                        err_map = run_eval(exp)
                        del exp
                        if err_map is None:
                            continue
                        errs.append(err_map * 100)
                    iterations += interval
                    if errs:
                        logger.set_iterations(iterations, losses=errs, cur_iter=cur_iter)
            if more:
                logging.info("There are more models to be evaluated, but ignored because last one is evaluated")


if __name__ == '__main__':
    main()

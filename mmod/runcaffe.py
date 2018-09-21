from __future__ import print_function
import os
import os.path as op
import sys
import argparse
import re
import subprocess
import json
import logging
import time
import numpy as np

try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = op.abspath(this_file)

if __name__ == '__main__':
    # When run as script, modify path assuming absolute import
    sys.path.append(op.join(op.dirname(this_file), '..'))

from mmod.phillylogger import PhillyLogger
from mmod.utils import makedirs, cwd, run_and_terminate_process, kill_after, init_logging, \
    ompi_rank, ompi_size, gpu_indices
from mmod.philly_utils import expand_vc_user, abspath, mirror_paths, get_log_parent, get_arg, last_log_dir, \
    get_model_path, set_job_id, job_id
from mmod.checkpoint import last_checkpoint
from mmod.simple_parser import parse_key_value, parse_prototxt, save_prototxt
from mmod.io_utils import load_from_yaml_file
from mmod.filelock import FileLock


def read_meta_file(path):
    """Read meta file from a path
    :param path: path to look for parameters file
    :rtype: dict
    """
    meta_path = op.join(path, 'parameters.json')
    if op.isfile(meta_path):
        with open(meta_path) as metaFile:
            return json.load(metaFile)

    meta_path = op.join(path, 'parameters.txt')
    if op.isfile(meta_path):
        import ast
        logging.info('Converting {}'.format(meta_path))
        with open(meta_path) as metaFile:
            return ast.literal_eval(metaFile.read())

    meta_path = op.join(path, 'parameters.yaml')
    if op.isfile(meta_path):
        return load_from_yaml_file(meta_path)

    import glob
    from datetime import datetime
    yaml_pattern = op.join(path, 'parameters_*.yaml')
    yaml_files = glob.glob(yaml_pattern)
    if len(yaml_files) > 0:
        def parse_time(_f):
            m = re.search('.*parameters_(.*)\.yaml', _f)
            _t = datetime.strptime(m.group(1), '%Y_%m_%d_%H_%M_%S')
            return _t

        times = [parse_time(f) for f in yaml_files]
        fts = [(f, t) for f, t in zip(yaml_files, times)]
        fts.sort(key=lambda x: x[1], reverse=True)
        meta_path = fts[0][0]
        meta = load_from_yaml_file(meta_path)
        meta['datetime'] = str(fts[0][1])
        return meta

    logging.error("Could not find the meta parameters in {}".format(path))
    return {}


def move_solver(solver_path, out_solver_path, solver_relpath, prev_model_path, overwrite=False):
    """ Copy the solver prototxt file, and make adjustments
    :param solver_path: input solver file path
    :param out_solver_path: output solver path
    :param solver_relpath: input solver relative path (to preserve)
    :param prev_model_path: where previously models where output
    :param overwrite: if should overwite existing solver
    :returns meta dictionary augmented with information parsed from the solver
    :rtype: dict
    """
    rank = ompi_rank()

    base_out_solver = op.dirname(out_solver_path)
    if not rank:
        makedirs(base_out_solver, exist_ok=True)
    solver = parse_prototxt(solver_path)
    snapshot_interval = int(solver.get("snapshot") or 0) or None
    display = int(solver.get("display") or 0)
    max_iter = int(solver["max_iter"])

    snapshot = op.join(base_out_solver, "snapshot/model")
    solver["snapshot_prefix"] = snapshot
    logging.info("Snapshot prefix changed to: '{}' solver: {}".format(snapshot, solver_path))
    if not rank:
        # make sure new snapshot path is there
        makedirs(op.dirname(snapshot), exist_ok=True)

    for net in ["train_net", "test_net", "net"]:
        if net not in solver:
            continue
        target_net = solver[net].strip().replace('"', '').replace("\\", "/")
        target_net = abspath(target_net, roots=[op.dirname(solver_path), expand_vc_user('~')])
        if not op.isfile(target_net):
            target_net = op.join(op.dirname(solver_path), op.basename(target_net))
        if not op.isfile(target_net):
            target_net = op.join(op.dirname(solver_path), "train.prototxt")
        if not op.isfile(target_net):
            raise RuntimeError("train_net at {} could not be found".format(target_net))
        solver[net] = target_net
        logging.info("{} changed to: '{}' solver: {}".format(net, target_net, solver_path))

    # only master node should create the output, it is a shared location to all nodes
    no_output = rank or (op.isfile(out_solver_path) and not overwrite)
    if not no_output:
        save_prototxt(solver, out_solver_path)

    meta = read_meta_file(op.dirname(solver_path))
    meta['solver_path'] = solver_path
    meta['out_solver_path'] = out_solver_path
    param_max_iter = meta.get('max_iter')
    if isinstance(param_max_iter, basestring) and param_max_iter.endswith('e'):
        meta['epoch_size'] = int(param_max_iter[:-1])
    meta['max_iter'] = max_iter
    meta['snapshot_prefix'] = snapshot
    meta['display'] = display
    if snapshot_interval:
        meta['snapshot'] = snapshot_interval
    prevrun_snapshot_prefix = ''
    if prev_model_path:
        try:
            prevrun_snapshot_prefix = parse_key_value(op.join(prev_model_path, solver_relpath),
                                                      "snapshot_prefix")
        except IOError:
            prevrun_snapshot_prefix = meta.get('prevrun_snapshot_prefix', '')
    # snapshot prefix from previous run (of the same solver)
    meta['prevrun_snapshot_prefix'] = prevrun_snapshot_prefix

    out_meta_path = op.join(op.dirname(out_solver_path), 'parameters.json')
    if rank:
        while not op.isfile(out_meta_path):
            logging.info("Waiting for master solver path: {}".format(out_solver_path))
            time.sleep(0.5)
            continue
    else:
        # Save the updated meta file
        logging.info("Save meta file: {}".format(out_meta_path))
        if overwrite or not op.isfile(out_meta_path):
            with open(out_meta_path, 'w') as metaFile:
                json.dump(meta, metaFile)

    return meta


def moved_solvers(solvers, model_dir=None):
    """New Solvers metas
    :param solvers: list of solvers to move for philly
    :param model_dir: where the models should be output
    :type solvers: list
    :rtype: list[dict]
    """

    metas = []

    solvers, relpaths = mirror_paths(solvers)
    for idx, (solver_path, solver_relpath) in enumerate(zip(solvers, relpaths)):
        if not op.exists(solver_path):
            raise RuntimeError("Train solver file %s does not exist" % solver_path)

        if model_dir:
            out_solver_path = op.join(model_dir, solver_relpath)
        else:
            out_solver_path = solver_path

        meta = read_meta_file(op.dirname(solver_path))
        base_out_solver = op.dirname(out_solver_path)
        meta['solver_path'] = solver_path
        meta['out_solver_path'] = out_solver_path
        meta['snapshot_prefix'] = op.join(base_out_solver, "snapshot/model")
        solver = parse_prototxt(solver_path)
        snapshot_interval = int(solver.get("snapshot") or 0) or None
        display = int(solver.get("display") or 0)
        max_iter = int(solver["max_iter"])
        meta['max_iter'] = max_iter
        meta['display'] = display
        if snapshot_interval:
            meta['snapshot'] = snapshot_interval

        metas.append(meta)

    assert len(metas) == len(solvers)
    return metas


def move_solvers(solvers, model_path=None, prev_model_path=None):
    """Move multiple solvers to model_path for philly
    :param solvers: list of solvers to move for philly
    :type solvers: list
    :param model_path: where the models should be output
    :param prev_model_path: where previously models where output
    :rtype: (list[str], list[dict])
    """
    new_solvers = []
    metas = []

    solvers, relpaths = mirror_paths(solvers)
    for idx, (solver, solver_relpath) in enumerate(zip(solvers, relpaths)):
        if not op.exists(solver):
            raise RuntimeError("Train solver file %s does not exist" % solver)

        if model_path:
            out_solver_path = op.join(model_path, solver_relpath)
        else:
            out_solver_path = op.join(op.dirname(solver), job_id() or '0', op.basename(solver))

        meta = move_solver(solver, out_solver_path, solver_relpath, prev_model_path)
        new_solvers.append(out_solver_path)
        metas.append(meta)

    assert len(metas) == len(solvers)
    return new_solvers, metas


def run_caffe(solver_path, logger, snapshot=None, weights=None, gpus=None,
              ignore_shape_mismatch=True, retries=0):
    """Run caffe training against a solver
    :param solver_path: path to the solver prototxt config file
    :param logger: philly logger instance
    :type logger: PhillyLogger
    :param snapshot: absolute path to the previous solver snapshot (.solverstate)
    :param weights: absolute path to the previous weights (.caffemodel)
    :param gpus: GPUs to run the caffe
    :type gpus: list or str or np.ndarray
    :param ignore_shape_mismatch: if should load weights of different size
    :param retries: how many times to retry on Caffe-process failures
    """
    iter_loss_pattern = re.compile(
        r'^.*\sIteration\s+(?P<ITERATION>\d+)\s.*loss\s=\s(?P<LOSS>[-+]?\d*.?\d+(e[-+]\d+)?)(\s.*)?$')
    abborted_pattern = re.compile(r'^\*\*\* (Aborted at|SIGSEGV)')

    if not gpus:
        gpus = "all"
    elif not isinstance(gpus, basestring):
        gpus = ",".join([str(gpu) for gpu in gpus])

    cmds = [
        "caffe", "train",
        "-solver", solver_path,
        "-gpu", gpus,
        "-log_dir", logger.log_dir,
        "-sighup_effect", "stop",   # do not snapshot while being killed
        "-sigint_effect", "stop",   # do not snapshot while being killed
        "-logtostderr",  # no need for extra log files
    ]

    # if there is a recent snapshot
    if snapshot:
        cmds += [
            "-snapshot", snapshot,
        ]
    elif weights:
        cmds += [
            "-weights", weights,
        ]
    if ignore_shape_mismatch:
        cmds += ['-ignore_shape_mismatch']

    while True:
        aborted_return = 0
        logging.info("Executing: {}".format(' '.join(cmds)))
        with run_and_terminate_process(cmds,
                                       preexec_fn=os.setsid,  # send signals to forked processes
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       bufsize=1) as process:
            for line in iter(process.stdout.readline, ""):
                line = line.strip()
                print(line)  # this must be a print() not logging
                if aborted_return:
                    continue
                m = re.match(abborted_pattern, line)
                if m:
                    logging.error("Process aborted unexpectedly")
                    aborted_return = -1
                    kill_after(process, 10)
                m = re.match(iter_loss_pattern, line)
                if m:
                    logger.set_iterations(int(m.group('ITERATION')), losses=float(m.group('LOSS')))

            process.wait()
            ret = process.returncode or aborted_return
            if ret and retries > 0:
                logging.error("Retry: {} in 5 seconds. Exit code: {}".format(retries, ret))
                retries -= 1
                time.sleep(5)
                continue
            return ret or aborted_return


def main():
    init_logging()
    parser = argparse.ArgumentParser(description='Run Caffe training with given config file.')

    parser.add_argument('-d', '--datadir', '--dataDir', help='Data directory where the dataset is located',
                        required=True)
    parser.add_argument('-m',  '--outputdir', '--modeldir', '--modelDir',
                        help='Output directory for checkpoints and models',
                        required=False)
    parser.add_argument('-l', '--logdir', '--logDir',
                        help='Log directory', required=False)
    parser.add_argument('--prevmodelpath', help='Previous model path', required=False)
    parser.add_argument('--configfile', '--stdoutdir', '--numgpu', help='Ignored', required=False)
    # ------------------------------------------------------------------------------------------------------------
    parser.add_argument('-s', '--solver', action='append',
                        help='Prototxt solver file for caffe training (specify multiple times to chain -weights)',
                        required=True)
    parser.add_argument('-prev', '--prev', help='Previous model path (override)', required=False)
    parser.add_argument('--skip_weights', help='If should avoid reusing weights across solvers', action='store_true',
                        default=False, required=False)
    parser.add_argument('--skip_snapshot', help='If should skip loading the last valid snapshot', action='store_true',
                        default=False, required=False)
    parser.add_argument('--use_lock', help='Use file locks (needed when multiple process write to logs)',
                        action='store_true')
    parser.add_argument('--retries', help='How many times to retry on Caffe failure', type=int, default=0,
                        required=False)
    parser.add_argument('-g', '--gpus', action='append', type=int,
                        help='GPU device ordinals to restrict the training to',
                        required=False)
    parser.add_argument('-snapshot', '--snapshot',
                        help='Initial snapshot to finetune from', required=False)
    parser.add_argument('-weights', '--weights',
                        help='Initial caffemodel to finetune from', required=False)
    parser.add_argument('-e', '--expid', '--jobid', help='The full experiment ID (if local, will be used as job ID)',
                        required=False)

    args = parser.parse_args()
    args = vars(args)
    print("Arguments: %s" % args)
    set_job_id(args['expid'])

    model_path = args['outputdir']
    if not model_path:
        model_path = get_model_path()
    else:
        model_path = abspath(model_path, roots=['#', '.'])
        if not op.isdir(model_path):
            raise RuntimeError("Output directory {} does not exist".format(model_path))

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

    skip_weights = args['skip_weights']
    skip_snapshot = args['skip_snapshot']
    use_lock = args["use_lock"]
    retries = args["retries"]
    init_snapshot = args['snapshot']
    if init_snapshot:
        init_snapshot = abspath(init_snapshot, roots=['#', '~', '.'])
        assert op.isfile(init_snapshot), "{} does not exist".format(init_snapshot)
    init_weights = args['weights']
    if init_weights:
        init_weights = abspath(init_weights, roots=['#', '~', '.'])
        assert op.isfile(init_weights), "{} does not exist".format(init_weights)

    prev_model_path = get_arg(args['prev']) or get_arg(args['prevmodelpath'])
    # Find previous log dir, to fill in the progress (used for cloning)
    prev_log_parent = get_log_parent(prev_model_path)
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

    # Config files may assume paths relative to input data path
    with cwd(work_path):
        new_solvers, metas = move_solvers(solvers, model_path, prev_model_path)

        is_worker = ompi_rank() != 0 and ompi_size() > 1
        if is_worker:
            pl = PhillyLogger(log_dir, use_lock=use_lock, is_master=False)
        else:
            pl = PhillyLogger(
                log_dir,
                [m['max_iter'] for m in metas],
                [op.join(op.basename(op.dirname(m['solver_path'])), op.basename(m['solver_path'])) for m in metas],
                use_lock=use_lock,
                prev_log_parent=prev_log_parent
            )
        with pl.redirected() as logger:
            logging.info("Arguments: {}".format(json.dumps(args, indent=2)))
            prevsolver_snapshot_prefix = ''  # snapshot_prefix from the previous solver
            prevsolver_max_iter = None
            prevsolver_snapshot_interval = 1
            for idx, (solver_path, meta) in enumerate(zip(new_solvers, metas)):
                logging.info('Solving {} with {}'.format(solver_path, meta))
                max_iter = meta['max_iter']

                logger.new_command(idx)
                if skip_snapshot:
                    snapshot, iterations, weights = '', 0, ''
                else:
                    # find the previous snapshot and weights from the same solver
                    snapshot, iterations, weights = last_checkpoint(
                        [meta['snapshot_prefix'],
                         meta['prevrun_snapshot_prefix']],
                        snapshot_interval=meta.get('snapshot', 1), max_iter=max_iter)
                # Reuse the previous weights (even from another solver)
                if not skip_weights and prevsolver_snapshot_prefix and not snapshot and not weights:
                    _, _, weights = last_checkpoint(
                        [prevsolver_snapshot_prefix],
                        snapshot_interval=prevsolver_snapshot_interval, max_iter=prevsolver_max_iter
                    )
                    if weights:
                        logging.info('Reuse weights: {} for solver: {}'.format(weights, solver_path))
                if not snapshot:
                    # if init snapshot use it
                    if idx == 0 and init_snapshot:
                        snapshot = init_snapshot
                    elif not weights:
                        weights = init_weights
                logger.set_iterations(iterations)
                if max_iter <= iterations:
                    logging.info(('Ignore training solver file {} '
                                  'that is already trained for {} >= {} iterations').format(
                        solver_path, iterations, max_iter)
                    )
                else:
                    returncode = run_caffe(solver_path, logger,
                                           snapshot=snapshot, weights=weights, gpus=gpus,
                                           retries=retries)
                    if returncode:
                        raise Exception("Caffe process exit code: {}".format(returncode))

                prevsolver_snapshot_prefix = meta['snapshot_prefix']
                prevsolver_snapshot_interval = meta.get('snapshot', 1)
                prevsolver_max_iter = max_iter
                logger.set_iterations(max_iter)


if __name__ == '__main__':
    main()

from __future__ import print_function
import os
import sys
import argparse
import re
import subprocess
from contextlib import contextmanager

if sys.version_info >= (3, 0):
    from os import makedirs
else:
    import errno

    def makedirs(name, mode=511, exist_ok=False):
        try:
            os.makedirs(name, mode=mode)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            if not exist_ok:
                raise


def convert_paths(solver_path, model_path):
    """ Copy the prototxt file, and make adjustments for philly
    :param solver_path: input file path
    :param model_path: output model file path
    :returns out_solver_path: converted file path
             max_iter: config value for max_iter
    :rtype (string, int)
    """
    max_iter = None
    with open(solver_path, "r") as f_in:
        out_solver_path = os.path.join(model_path, os.path.basename(solver_path))
        with open(out_solver_path, "w") as f_out:
            for line in f_in.readlines():
                line = line.rstrip('\n')
                line_lower = line.lower().strip()
                if line_lower.startswith("snapshot_prefix"):
                    snapshot = os.path.join(model_path, line[line.index(":") + 1:].strip().replace('"', ''))\
                        .replace("\\", "/")
                    line = 'snapshot_prefix: "{}"'.format(snapshot)
                    print("Snapshot changed to: '{}'".format(line))
                    # make sure new snapshot path is there
                    makedirs(os.path.dirname(snapshot), exist_ok=True)
                elif line_lower.startswith("max_iter"):
                    max_iter = int(line[line.index(":") + 1:].strip())
                f_out.write("{}\n".format(line))

    return out_solver_path, max_iter


def run_caffe(solver_path, log_dir, extra_args, max_iter):
    """Run caffe training against a solver
    :param solver_path: path to the solver prototxt config file
    :param log_dir: path to save the log files
    :param extra_args: extra parameters passed
    :param max_iter: maximum number of iterations
    """
    max_iter_pattern = None
    if not max_iter:
        max_iter_pattern = re.compile(r'^\s*max_iter:\s(?P<MAX_ITER>\d+)\s*$')
    iter_loss_pattern = re.compile(
        r'^.*\sIteration\s+(?P<ITERATION>\d+)\s.*loss\s=\s(?P<LOSS>[-+]?\d*.?\d+(e[-+]\d+)?)(\s.*)?$')

    # Caffe expects the leading slash
    log_dir = os.path.join(log_dir, '')

    # TODO: add option for "caffe test"

    cmds = ["caffe", "train",
            "-solver", solver_path,
            "-gpu", "all",
            "-log_dir", log_dir,
            "-log", log_dir]

    cmds += extra_args
    print("Executing: {}".format(cmds))
    process = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in process.stdout:
        line = line.strip()
        print(line)
        if not max_iter:
            # Another chance to find max_iter
            m = re.match(max_iter_pattern, line)
            if m:
                max_iter = int(m.group('MAX_ITER'))
        if not max_iter:
            continue
        m = re.match(iter_loss_pattern, line)
        if m:
            progress = float(m.group('ITERATION')) / max_iter * 100
            loss = float(m.group('LOSS'))
            print("PROGRESS: {}%".format(progress))
            print("EVALERR: {}%".format(loss))

    process.wait()
    return process.returncode


@contextmanager
def cwd(path):
    """Change directory to the given path and back
    """
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


def main():
    parser = argparse.ArgumentParser(description='Run Caffe training with given config file.')

    parser.add_argument('-configfile', '--configfile', help='Config file currently running',
                        required=False)
    parser.add_argument('-solver', '--solver', help='Prototxt solver file for caffe training',
                        required=True)
    parser.add_argument('-datadir', '--datadir', help='Data directory where the dataset is located',
                        required=True)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=True)
    parser.add_argument('-logdir', '--logdir', help='Log directory', required=True)

    args, extra = parser.parse_known_args()
    args = vars(args)
    print("Arguments: %s" % args)
    if extra:
        print('Extra arguments: "%s" (will be ignored)' % " ".join(extra))

    model_path = args['outputdir']
    if not model_path or not os.path.isdir(model_path):
        raise RuntimeError("Output directory %s does not exist" % model_path)
    log_dir = args['logdir']
    if not log_dir or not os.path.isdir(log_dir):
        raise RuntimeError("Log directory %s does not exist" % log_dir)
    data_path = args['datadir']
    if not data_path or not os.path.isdir(data_path):
        raise RuntimeError("Data directory %s does not exist" % data_path)
    solver = args['solver']
    if not solver or not os.path.exists(solver):
        raise RuntimeError("Train solver file %s does not exist" % solver)

    # prepare solver file, and parse it
    solver_path, max_iter = convert_paths(solver, model_path)

    # Config files may assume paths relative to input data path
    with cwd(data_path):
        returncode = run_caffe(solver_path, log_dir, extra, max_iter)

    if returncode:
        raise Exception("Caffe process exit code: {}".format(returncode))

if __name__ == '__main__':
    main()

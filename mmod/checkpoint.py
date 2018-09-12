import re
import glob
import os
import logging
import os.path as op
import numpy as np


def last_checkpoint(prefixes, snapshot_interval=None, max_iter=None,
                    snapshot_ext=".solverstate", weights_ext=".caffemodel"):
    """Find the last checkpoints inside given prefix paths
    Look at the paths in prefixes, and return once found
    :param prefixes: paths to check for the solverstate and weights
    :type prefixes: list or str
    :param snapshot_interval: valid snapshot intervals
    :param max_iter: maximum number of iterations that could be found
    :param snapshot_ext: file extension of snapshot (pass None to ignore snapshots)
    :param weights_ext: file extension of saved weights  (pass None to ignore weights)
    :return: snapshot, iterations, weights
    :rtype: (str, int, str)
    """
    if not isinstance(prefixes, list):
        prefixes = [prefixes]
    if not snapshot_interval:
        snapshot_interval = 1
    model_iter_pattern = None
    solver_iter_pattern = None
    if snapshot_ext:
        solver_iter_pattern = re.compile(r'iter_(?P<ITER>\d+){}$'.format(re.escape(snapshot_ext)))
    if weights_ext:
        model_iter_pattern = re.compile(r'iter_(?P<ITER>\d+){}$'.format(re.escape(weights_ext)))
    snapshot = ''
    iterations = -1
    model_iterations = 0
    weights = ''
    found_solver = False
    found_model = False
    for prefix in prefixes:
        path = op.dirname(prefix)
        base = op.basename(prefix)
        if not op.isdir(path):
            if path:
                logging.info('last_checkpoint ignored invalid path: "{}"'.format(path))
            continue

        if iterations >= 0:
            found_solver = True
        if model_iterations:
            found_model = True
        if found_solver and found_model:
            break
        solver = model = None
        for fname in os.listdir(path):
            if not fname.startswith(base):
                continue
            if solver_iter_pattern:
                solver = re.search(solver_iter_pattern, fname) if not found_solver else None
            if model_iter_pattern:
                model = re.search(model_iter_pattern, fname) if not found_model else None
            if solver:
                iters = int(solver.group('ITER'))
                if iters == max_iter:
                    found_solver = True
                if not found_solver and (iters % snapshot_interval != 0 or iters > max_iter):
                    logging.info('Ignore snapshot interval: {} snapshot: {} max_iter: {}'.format(
                        snapshot_interval, fname, max_iter))
                elif iters > iterations or found_solver:
                    snapshot = op.join(path, fname)
                    iterations = iters
            if model:
                iters = int(model.group('ITER'))
                if iters == max_iter:
                    found_model = True
                if not found_model and (iters % snapshot_interval != 0 or iters > max_iter):
                    logging.info('Ignore snapshot interval: {} snapshot: {} max_iter: {}'.format(
                        snapshot_interval, fname, max_iter))
                elif iters > model_iterations or found_model:
                    weights = op.join(path, fname)
                    model_iterations = iters

    if iterations < 0:
        iterations = 0
    return snapshot, iterations, weights


def iterate_weights(prefixes, snapshot_intervals, max_iters, caffemodel=None,
                    weights_ext=".caffemodel", processed_ext=".report.class_ap.json"):
    """Iterate over all the models without evaluation result
    Find in descending order, in each of the paths in prefixes
    :param prefixes: paths to check for the solverstate and weights
    :type prefixes: list
    :param snapshot_intervals: list of valid snapshot intervals
    :type snapshot_intervals: list
    :param max_iters: lsit of maximum number of iterations that could be found
    :type max_iters: list
    :param caffemodel: single caffemodel path to return
    :param weights_ext: file extension of saved weights  (pass None to ignore weights)
    :param processed_ext: file extension of the already-processed model
    :return: index: index of the prefix
             iteration: total iterations already computed
             cur_iter: current iteration number
             weights: caffe weights fiel paht
             done: if all the prefixes have reached the max_iter
             more: if there are more model weights
    :rtype: (int, int, int, str, bool, bool)
    """
    assert len(prefixes) == len(snapshot_intervals) == len(max_iters)

    model_iter_pattern = re.compile(r'iter_(?P<ITER>\d+){}$'.format(re.escape(weights_ext)))
    if caffemodel:
        model = re.search(model_iter_pattern, caffemodel)
        if not model:
            logging.error("caffemodel {} does not have standard filename pattern".format(caffemodel))
            yield 0, 1, caffemodel, True, False
            return
        iters = int(model.group('ITER'))
        yield 0, 0, iters, caffemodel, True, False
        return

    iteration = 0
    more = False
    done = [False] * len(prefixes)

    for idx, (prefix, interval, max_iter) in enumerate(zip(prefixes, snapshot_intervals, max_iters)):
        model_iterations = 0
        weights = ''
        path = op.dirname(prefix)
        base = op.basename(prefix)

        done[idx] = False
        for fname in os.listdir(path):
            if not fname.startswith(base):
                continue
            model = re.search(model_iter_pattern, fname)
            if not model:
                continue
            iters = int(model.group('ITER'))
            if iters == max_iter:
                done[idx] = True
            elif iters % interval != 0 or iters > max_iter:
                logging.debug('Ignore snapshot interval: {} snapshot: {} max_iter: {}'.format(
                    interval, fname, max_iter
                ))
                continue
            new_weights = op.join(path, fname)
            if len(glob.glob(new_weights + ".*" + processed_ext)) == len(prefixes):
                # model is already evaluated against all of its test data
                iteration += interval
                continue
            if iters == max_iter:
                logging.info('Last model checkpoint found: {}'.format(fname))
            more = True
            if iters > model_iterations:
                model_iterations = iters
                weights = new_weights
        if weights:
            yield idx, iteration, model_iterations, weights, np.all(done), more

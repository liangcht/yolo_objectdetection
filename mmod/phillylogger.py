import sys
import json
import os
import os.path as op
from bisect import bisect_left
from contextlib import contextmanager
import logging
import copy
import numpy as np

from mmod.filelock import FileLock, FileLockException, FakeFileLock
from mmod.philly_utils import iterate_log_dir


def bisect_keys_add_value(keyvals, key, val):
    """Add value to list with sorted keys
    :param keyvals: list of key-values with sorted keys followed by values: [[key0, val0], ...]
    :param key: new key to append/update in the list
    :param val: value to add for the above key
    """
    keys = [kv[0] for kv in keyvals]
    idx = bisect_left(keys, key)
    if idx >= len(keys):
        keyvals.append([key, val])
        return
    if keys[idx] == key:
        # update value of a previous key
        keyvals[idx][1] = val
        return
    # USe slice to add sorted keyvalue pair
    keyvals[idx:idx] = [[key, val]]


class PhillyLogger(object):
    def __init__(self, log_dir, max_iters=None, command_names=None,
                 subcommand_names=None, use_lock=False, prev_log_parent=None,
                 is_master=True):
        """Log formatting for philly
        :param log_dir: log directory
        :param max_iters: list of max_iter for each command
        :type max_iters: list[int]
        :param command_names: list of command name for each command
        :type command_names: list[str]
        :param subcommand_names: list of sub-command names
        :type subcommand_names: list[list[str]]
        :param use_lock: If shoudl use file locks (needed when multiple processes may write to the same file)
        :param prev_log_parent: parent of the previous log directory
        :param is_master: if this is the master logger
        """
        if not max_iters:
            max_iters = []
        if not command_names:
            command_names = []
        if subcommand_names is None:
            subcommand_names = [[] for _ in command_names]
        assert isinstance(max_iters, list)
        assert isinstance(command_names, list)
        assert isinstance(subcommand_names, list)

        self._is_master = is_master
        self._redirected = False
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._use_lock = use_lock

        self._progress = 0.0
        self._last_errs = [0.0]
        self._last_command = ''  # last command name
        self._min_err = None
        self._max_err = 0.0

        # Caffe expects the leading slash
        self.log_dir = op.join(log_dir, '')
        self._max_iters = max_iters
        self._iters = [0] * len(max_iters)
        self._total_iters = float(sum(self._max_iters))
        self._command_index = None
        self._commands = [
            {
                'name': '{}{}'.format('{}_'.format(idx) if len(max_iters) > 1 else '', name),
                # The progress (in percent) that the command started
                'progress': round(sum(self._max_iters[:idx]) / self._total_iters * 100, 2),
                'minibatch': [],
                'finEpochs': [],
                'totEpochs': self._max_iters[idx],
            } for idx, name in enumerate(command_names)
        ]

        last_cmd = len(self._commands)
        # sub-command range for a given command
        self._sub_command_range = []
        # append sub-commands to the end
        for idx, sc_list in enumerate(subcommand_names):
            next_cmd = last_cmd + len(sc_list)
            self._sub_command_range.append(range(last_cmd, next_cmd))
            last_cmd = next_cmd
            if not len(sc_list):
                continue
            command = copy.deepcopy(self._commands[idx])
            for sub_idx, sc in enumerate(sc_list):
                command.update({
                    'name': command['name'] + "/{}/{}".format(sub_idx, sc)
                })
                self._commands.append(command)

        self._progress_file = os.path.join(self.log_dir, 'progress.json')
        self._logrank_file = os.path.join(self.log_dir, 'logrank.0.log')

        # Load previous progress, and initializa it only for master nodes
        if max_iters:
            self._load_progress(prev_log_parent=prev_log_parent)
            # Write the progress file just so that the logrank will be returned by philly
            self._write_progress(self._last_progress())

    @contextmanager
    def not_redirected(self):
        """Context to lift redirection temporarily
        """
        if not self._redirected:
            return
        sys.stdout, sys.stderr = self._stdout, self._stderr
        try:
            for handler in logging.root.handlers:
                if hasattr(handler, 'stream') and handler.stream == self:
                    handler.stream = self._stderr
            self._redirected = False
            yield
        finally:
            sys.stdout, sys.stderr = self, self
            for handler in logging.root.handlers:
                if hasattr(handler, 'stream') and (handler.stream == self._stderr or handler.stream == self._stdout):
                    handler.stream = self
            self._redirected = True

    @contextmanager
    def redirected(self):
        """Context to redirect the stdout/stderr
        :return: An instance of the logger that redirects stdout/stderr
        :rtype: PhillyLogger
        """
        if self._redirected:
            return
        try:
            # Fix up root logging handlers
            for handler in logging.root.handlers:
                if hasattr(handler, 'stream') and (handler.stream == self._stderr or handler.stream == self._stdout):
                    handler.stream = self

            sys.stdout, sys.stderr = self, self
            self._redirected = True
            yield self
        finally:
            sys.stdout, sys.stderr = self._stdout, self._stderr
            for handler in logging.root.handlers:
                if hasattr(handler, 'stream') and handler.stream:
                    handler.stream = self._stderr
            self._redirected = False

    def flush(self):
        """Flush the streams
        """
        self._stdout.flush()
        self._stderr.flush()

    def write(self, buf):
        """Write the redirected buffer
        """
        with self.not_redirected():
            try:
                with FileLock(self._logrank_file) if self._use_lock else FakeFileLock(self._logrank_file):
                    with open(self._logrank_file, 'a') as f:
                        for line in buf.rstrip().splitlines():
                            f.write(line + '\n')
            except FileLockException:
                logging.debug('Could not lock {}'.format(self._logrank_file))
            except Exception as e:
                logging.debug('File: {} error: {}'.format(self._logrank_file, e))

            # Also Tee to the actual stdout/stderr
            for line in buf.rstrip().splitlines():
                print(line)  # must be print
                sys.stdout.flush()

    def set_max_iter(self, new_max_iter):
        """max_ter may have changed, update accordingly
        :param new_max_iter: maximum number of iterations in this run
        """
        if not self._is_master:
            return
        if not new_max_iter:
            logging.warn("Ignore invalid new max_iter")
            return

        assert self._command_index is not None, "No command specified yet"

        max_iter = self._max_iters[self._command_index]
        if new_max_iter == max_iter:
            return

        logging.info("max_iter changed from {} to {}".format(max_iter, new_max_iter))
        self._max_iters[self._command_index] = new_max_iter
        self._total_iters = float(sum([(mi or 0) for mi in self._max_iters]))
        command = self._commands[self._command_index]
        command['totEpochs'] = self._max_iters[self._command_index]

    def _load_progress(self, progress_file=None, prev_log_parent=None):
        """Read last progress
        :param progress_file: the progress file to load from
        :type progress_file: str
        :param prev_log_parent: previous log directory
        """
        if not self._is_master:
            return
        if progress_file is None:
            progress_file = self._progress_file
            if op.isfile(progress_file):
                self._load_progress(progress_file)
                return
            for log_dir in iterate_log_dir():
                progress_file = op.join(log_dir, 'progress.json')
                if op.isfile(progress_file):
                    self._load_progress(progress_file)
                    return
            if prev_log_parent:
                for log_dir in iterate_log_dir(prev_log_parent):
                    progress_file = op.join(log_dir, 'progress.json')
                    if op.isfile(progress_file):
                        self._load_progress(progress_file)
                        return
            return

        logging.info('Loading progress from {}'.format(progress_file))
        try:
            with open(progress_file, 'r') as jsonFile:
                progress = json.load(jsonFile)
        except Exception as e:
            logging.error('Ignore loading: {} error: {}'.format(progress_file, e))
            return

        if 'commands' not in progress:
            logging.warn('Ignore loading: {}'.format(progress_file))
            return

        old_commands = progress['commands']
        if len(old_commands) != len(self._commands):
            logging.info("Loading {} old commands into {} new commands".format(len(old_commands), len(self._commands)))
        if len(old_commands) < len(self._commands):
            self._commands[:len(old_commands)] = old_commands
        else:
            self._commands = old_commands[:len(self._commands)]
        self._min_err = progress['gFMinErr']
        self._max_err = progress['gFMaxErr']

    def _last_progress(self):
        """Get the current progress
        :rtype: dict
        """
        progress = {
            'curCommand': self._last_command,
            'lastErr': self._last_errs[0],
            'gFMinErr': round(self._min_err or 0, 2),
            'gFMaxErr': round(self._max_err or 0, 2),
            'logfilename': 'logrank.0.log',
            'gMMinErr': round(self._min_err or 0, 2),
            'gMMaxErr': round(self._max_err or 0, 2),
            'lastProgress': round(self._progress, 2),
            'progress': round(self._progress or 0, 2),
            'totEpochs': sum(self._max_iters),
            'commands': self._commands
        }

        return progress

    def _update_progress(self, cur_iter):
        """Progress of the commands
        :param cur_iter: current iteration number
        :return: A dictionary of the progress
        :rtype: dict
        """
        assert self._command_index is not None, "No command specified yet"

        command = self._commands[self._command_index]
        subcmd_rng = self._sub_command_range[self._command_index]

        assert len(self._last_errs) == 1 + len(subcmd_rng), "Invalid sub-command range"
        self._last_command = command['name']

        x = cur_iter + sum(self._max_iters[:self._command_index])
        for idx, err in enumerate(self._last_errs):
            if idx:
                command = self._commands[subcmd_rng[idx - 1]]
            err = round(err, 2)
            minibatch = command['minibatch']
            assert isinstance(minibatch, list)
            bisect_keys_add_value(minibatch, x, err)

            finepochs = command['finEpochs']
            assert isinstance(finepochs, list)
            bisect_keys_add_value(finepochs, x, err)

            command.update({
                'finEpochs': finepochs,
                'minibatch': minibatch
            })

        min_last = min(self._last_errs)
        max_last = max(self._last_errs)
        if cur_iter == 0 or self._min_err is None or min_last < self._min_err:
            self._min_err = min_last
        if cur_iter == 0 or self._max_err is None or max_last > self._max_err:
            self._max_err = max_last

        return self._last_progress()

    def _write_progress(self, progress):
        """Write the progress to the progress file
        :param progress: the progress dictionary
        :type progress: dict
        """
        if not self._is_master:
            return
        # We do not lock for this file, it is assumed that only a single process writes to progress
        try:
            with open(self._progress_file, 'w') as jsonFile:
                json.dump(progress, jsonFile)
        except Exception as e:
            logging.error('File: {} error: {}'.format(self._progress_file, e))

    def set_iterations(self, iterations, losses=None, cur_iter=None, idx=None):
        """set number of iterations processed so far in this command
        :param iterations: total iterations processed so far in this command
        :param losses: train loss(es) so far in this command
        :type losses: list[float] | float
        :param cur_iter: current iteration number
        :param idx: command index
        """
        if not self._is_master:
            return
        if idx is not None:
            self.new_command(idx)
        assert self._command_index is not None, "No command specified yet"
        if cur_iter is None:
            cur_iter = iterations
        self._iters[self._command_index] = iterations
        self._progress = sum(self._iters) / self._total_iters * 100
        print("PROGRESS: {}%".format(round(self._progress, 4)))  # must be print
        if losses is None:
            return
        if not isinstance(losses, list):
            losses = [losses]
        if not np.all(np.isfinite(losses)):
            logging.info("Ignore all infinite losses at iteration: {}".format(iterations))
            return
        print("EVALERR: {}%".format(max(losses)))  # must be print
        self._last_errs = losses

        self._write_progress(self._update_progress(cur_iter))

    def new_command(self, idx):
        """Start a new command
        :param idx: current command index
        :type idx: int
        """
        if not self._is_master:
            return
        assert idx < len(self._sub_command_range), "Command index {} must be < {}".format(
            idx, len(self._sub_command_range))
        self._command_index = idx

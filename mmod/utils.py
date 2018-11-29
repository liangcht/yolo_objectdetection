import sys
import os
import shutil
import os.path as op
from contextlib import contextmanager
import subprocess
from threading import Timer
import numpy as np
import socket
import logging
import re

if sys.version_info >= (3, 0):
    # noinspection PyUnresolvedReferences
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


def ompi_rank():
    """Find OMPI world rank without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_RANK') or 0)


def ompi_size():
    """Find OMPI world size without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_SIZE') or 1)


def ompi_local_rank():
    """Find OMPI local rank without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK') or 0)


def ompi_local_size():
    """Find OMPI local size without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE') or 1)


def _get_mpi_name():
    """Cache machine MPI name without calling mpi functions
    :rtype: str
    """
    return "{}({}/{})".format(
        socket.gethostname(),
        ompi_rank(),
        ompi_size()
    )


_MPI_NAME = _get_mpi_name()


def get_mpi_name():
    """Find machine MPI name without calling mpi functions
    :rtype: str
    """
    return _MPI_NAME


def init_logging():
    """Initialize the logging
    :return:
    """
    np.seterr(all='raise')
    logging.basicConfig(
        level=logging.INFO,
        format=(
            '{}:%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s '
            '%(funcName)10s(): %(message)s').format(
                get_mpi_name()
            ),
        datefmt='%m-%d %H:%M:%S',
    )


@contextmanager
def cwd(path):
    """Change directory to the given path and back
    """
    if path == '.':
        yield
        return
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


def tail_path(path):
    """Return the tail of a path
    :type path: str
    :rtype: str
    """
    return op.join(op.basename(op.dirname(path)), op.basename(path))


def is_number(s, type_cast=float):
    try:
        type_cast(s)
        return True
    except (ValueError, TypeError):
        return False


@contextmanager
def run_and_terminate_process(*args, **kwargs):
    """Run a process and terminate it at the end
    """
    p = None
    try:
        p = subprocess.Popen(*args, **kwargs)
        yield p
    finally:
        if not p:
            return
        try:
            p.terminate()  # send sigterm
        except OSError:
            pass
        try:
            p.kill()       # send sigkill
        except OSError:
            pass


def kill_after(process, timeout):
    """Kill a process after given number of seconds
    :param process: process object
    :param timeout: timeout in seconds
    """
    def _force_kill():
        try:
            process.kill()
        except OSError:
            pass

    # break after few seconds
    Timer(timeout, _force_kill).start()


def _try_get_gpus():
    try:
        import torch
        return range(torch.cuda.device_count())
    except ImportError:
        return []


def get_gpus_nocache():
    """List of NVIDIA GPUs
    """
    cmds = 'nvidia-smi --query-gpu=name --format=csv,noheader'.split(' ')
    try:
        with run_and_terminate_process(cmds,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       bufsize=1) as process:
            return [line.strip() for line in iter(process.stdout.readline, "")]
    except RuntimeError:
        return


_GPUS = get_gpus_nocache()


def get_gpus():
    """List of NVIDIA GPUs
    """
    global _GPUS
    if _GPUS is None:
        logging.error("nvidia-smi was not found")
        _GPUS = _try_get_gpus()
    return _GPUS


def gpu_indices(divisible=True):
    """Get the GPU device indices for this process/rank
    :param divisible: if GPU count of all ranks must be the same
    :rtype: list[int]
    """
    local_size = ompi_local_size()
    local_rank = ompi_local_rank()
    assert 0 <= local_rank < local_size, "Invalid local_rank: {} local_size: {}".format(local_rank, local_size)
    gpu_count = len(get_gpus())
    assert gpu_count >= local_size > 0, "GPU count: {} must be >= LOCAL_SIZE: {} > 0".format(gpu_count, local_size)
    if divisible:
        ngpu = gpu_count / local_size
        gpus = np.arange(local_rank * ngpu, (local_rank + 1) * ngpu)
        if gpu_count % local_size != 0:
            logging.warning("gpu_count: {} not divisible by local_size: {}; some GPUs may be unused".format(
                gpu_count, local_size
            ))
    else:
        gpus = np.array_split(xrange(gpu_count), local_size)[local_rank]
    return gpus


def total_gpu_count():
    """Total number of GPUs we parallelize over
    :return:
    :rtype: int
    """
    return len(get_gpus()) * ompi_size()


def tsv_split(fp, sep=None, bufsize=1024):
    """Read a file and split the line (this is more efficient than reading entire line first)
    :param fp: a filelike object
    :param sep: seperator(s) (regexp pattern)
    :param bufsize: the chunk size to read the file (should not be small fo performance reasons)
    """
    if sep is None:
        sep = '\n|\t'
    prev = ''
    while True:
        s = fp.read(bufsize)
        if not s:
            break
        split = re.split(sep, s)
        if len(split) > 1:
            yield prev + split[0]
            prev = split[-1]
            for x in split[1:-1]:
                yield x
        else:
            prev += s
    if prev:
        yield prev


def tsv_read(fp, size, seek_offset=None, sep=None):
    """Read few elements from file
    :param fp: file-like object
    :param size: number of elements to read
    :param seek_offset: seek offset
    :param sep: seperator(s) (regexp pattern)
    """
    if seek_offset is not None:
        fp.seek(seek_offset)
    cols = []
    for col in tsv_split(fp, sep=sep):
        cols.append(col)
        if len(cols) == size:
            break
    return cols


@contextmanager
def open_file(fp):
    """Yield the fp
    """
    yield fp


def tsv_multi_column(f, bufsize=1024):
    """Find if a tsv has more than one columns (efficient check)
    :param f: file-like object (or path to file)
    :param bufsize: the chunk size to read the file (should not be small fo performance reasons)
    :rtype: bool
    """
    with open(f, 'r') if isinstance(f, basestring) else open_file(f) as fp:
        while True:
            s = fp.read(bufsize)
            if not s:
                return False
            tab_idx = s.find('\t')
            cr_idx = s.find('\n')
            if tab_idx < 0 and cr_idx < 0:
                continue
            if cr_idx < 0:
                return True
            if tab_idx < 0:
                return False
            if cr_idx < tab_idx:
                return False
            break
    return True


def splitex_ver(path):
    """Split the path to base name and versioned extension
    :param path: path to the file name
    :type path: str
    :rtype: (str, str)
    """
    base, ext = op.splitext(path)
    base2, ext2 = op.splitext(base)
    if len(ext2) > 2 and ext2[1].lower() == 'v' and is_number(ext2[2:], type_cast=int):
        return base2, ext2 + ext
    return base, ext


def splitfilename(path, splitname, is_composite=False, keep_ext=True):
    """Get a versioned file name and return the file name for given split name, with the same version
    :param path: path to potentially versioned training file
    :type path: str
    :param splitname: label, labelmap, inverted.label
    :type splitname: str
    :param is_composite: if this is a composite tsv (weird name)
    :param keep_ext: if should preserve the extension
    :rtype: str
    """
    base, ext = splitex_ver(path)
    if not splitname.startswith("."):
        splitname = "." + splitname
    if is_composite and base.endswith('X'):
        base = base[:-1]
    if not keep_ext:
        ext = ''
    return base + splitname + ext


def search_in_sorted(a, v):
    """Search v in sorted a and return the index
    :type a: xrange | numpy.ndarray | list
    :type v: int
    :rtype: int
    """
    if isinstance(a, xrange):
        a_min = a[0]
        a_max = a[-1]
        if v < a_min or v > a_max:
            return
        return v - a_min
    idx = np.searchsorted(a, v)
    if idx >= len(a):
        return
    if a[idx] != v:
        return
    return idx


def is_in_sorted(a, v):
    """Return if v is in sorted a
    :type a: xrange | numpy.ndarray | list
    :type v: int
    :rtype: bool
    """
    if isinstance(a, xrange):
        return v in a
    idx = np.searchsorted(a, v)
    if idx >= len(a):
        return False
    if a[idx] != v:
        return False
    return True


def search_both_sorted(a, vs):
    """Search a for values in vs and yield the indices for both
    Both a and v must be sorted. If only a is sorted we can just use np.searchsorted(a, vs)
    :param a: array to search into
    :type a: xrange | numpy.ndarray | list
    :param vs: array to search for
    :type vs: list[int] | numpy.ndarray
    :returns index and value
    :rtype: int, int
    """
    # go from both sides and narrow down the search
    if isinstance(a, xrange):
        a_min = a[0]
        a_max = a[-1]
        if vs[-1] < a_min:
            # right-most value is to the left of target
            return
        for left_v_idx, v in enumerate(vs):
            if v > a_max:
                return
            if v < a_min:
                continue
            yield v - a_min, left_v_idx
        return

    left = 0
    right = len(a)
    left_v_idx = 0
    right_v_idx = len(vs)
    while left_v_idx < right_v_idx:
        ln = vs[left_v_idx]
        a_idx = np.searchsorted(a[left: right], ln)
        width = right - left
        if a_idx >= width:
            # left-most value is to the right of a
            return
        if a_idx >= 0 and a[a_idx] == ln:
            a_idx += left
            left = a_idx
            yield a_idx, left_v_idx
        left_v_idx += 1
        if left_v_idx >= right_v_idx:
            break

        ln = vs[right_v_idx - 1]
        a_idx = np.searchsorted(a[left: right], ln)
        if a_idx < 0:
            # right-most value is to the left of target
            return
        width = right - left
        if width > a_idx:
            a_idx += left
            if a[a_idx] == ln:
                right = a_idx + 1
                yield a_idx, right_v_idx
            elif a_idx == 0:
                return
        right_v_idx -= 1


def window_stack(a, width):
    """Sliding window over first dimension of a
    :type a: numpy.ndarray or list
    :type width: int
    :rtype: numpy.ndarray
    """
    n = len(a)
    assert width <= n
    return np.vstack(a[i:i+width] for i in range(0, n - width + 1))


class FileCache(object):
    def __init__(self):
        self._is_closed = False
        self._open_files = {}

    def __repr__(self):
        return 'FileCache({})'.format(
            'close' if self._is_closed else 'open: {}'.format(len(self._open_files))
        )

    def open(self, path, mode="r", create_parents=False):
        """Open a file in the cache
        """
        fp = self._open_files.get(path)
        if fp is None:
            if create_parents and mode.startswith("w"):
                makedirs(op.dirname(path), exist_ok=True)
            fp = self._open_files[path] = open(path, mode)
        return fp

    @contextmanager
    def open_file(self, *args, **kwargs):
        yield self.open(*args, **kwargs)

    def close(self):
        assert not self._is_closed, "File cache is already closed"
        for fp in self._open_files.values():
            fp.close()
        self._open_files = {}
        self._is_closed = True

    def __getstate__(self):
        return {
            "_open_files": {},
            "_is_closed": True
        }


@contextmanager
def file_cache():
    """Open a file for reading and cache its file object
    This is for fielsystems like HDFS
    :rtype: FileCache
    """

    cache = None
    try:
        cache = FileCache()
        yield cache
    finally:
        if cache:
            cache.close()


@contextmanager
def open_with_lineidx(path, with_temp=False):
    """Write to a file and update its line index
    :param path: Path to tsv file
    :param with_temp: if should create temporary file before moving it
    :rtype: _WithLineIdx
    """

    tsv_file = path
    if with_temp:
        # TODO: fix for Windows
        tsv_file = "/tmp/to_move/" + path
        try:
            makedirs(os.path.dirname(tsv_file), exist_ok=True)
        except OSError as e:
            logging.error("Could not create temporary path: {} err: {}".format(tsv_file, e))
            with_temp = False
            tsv_file = path
    lineidx_file = op.splitext(tsv_file)[0] + ".lineidx"
    with open(tsv_file, 'wb') as fp, open(lineidx_file, 'wb') as fpidx:
        class _WithLineIdx(object):
            def __init__(self):
                self._offset = 0

            def write(self, buf):
                """Write the buffer to file
                """
                fp.write(buf)
                fpidx.write("{}\n".format(self._offset))
                self._offset += len(buf)

            def tell(self):
                return self._offset

        yield _WithLineIdx()

    if with_temp:
        shutil.move(tsv_file, path)
        lineidx_path = op.splitext(path)[0] + ".lineidx"
        shutil.move(lineidx_file, lineidx_path)


def range_split(rng, sections):
    """Split a contigious range to given number
    The last split may have a longer length
    :param rng: range or array
    :type rng: xrange
    :param sections: number of sections to split rng to
    :type sections: int
    :rtype: list[xrange]
    """
    _min = int(rng[0])
    _max = int(rng[-1])
    assert len(rng) == _max - _min + 1, "{} is not contiguous".format(rng)
    sections = int(sections)
    assert _max - _min > sections, "too many sections in {} to divide {}".format(sections, rng)
    step = int(len(rng) / sections)

    rngs = [xrange(a, a + step) for a in xrange(_min, _max, step)]
    last = rngs[-1]
    if last[-1] != _max + 1:
        rngs[-1] = xrange(last[0], _max + 1)

    return rngs

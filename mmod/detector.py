import logging
import threading
import multiprocessing as mp
import json
from loky import get_reusable_executor, cpu_count
try:
    import caffe
except ImportError:
    caffe = None

from mmod.experiment import Experiment
from mmod.utils import init_logging, open_with_lineidx
from mmod.detection import im_detect, result2bblist


def mp_logger():
    console_handler = logging.StreamHandler()
    log_formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d [{}]  "
        "%(filename)s:%(lineno)s %(funcName)10s(): %(message)s".format(
            mp.current_process().name
        ))
    console_handler.setFormatter(log_formatter)
    lg = mp.util.get_logger()
    lg.addHandler(console_handler)
    lg.setLevel(logging.INFO)
    return lg


_logger = None
_init_done = False
_gpu = None
_exp = None  # type: Experiment
_net = None
_last_gpu = -1
_gpu_lock = threading.RLock()


def read_image(imdb, input_range, out_queue):
    """Save prediction results to tsv file
    :type imdb: mmod.imdb.ImageDatabase
    :type input_range: xrange
    :type out_queue: loky.backend.queues.Queue
    """
    global _logger
    logger = _logger or logging
    try:
        # noinspection PyBroadException
        try:
            with imdb.open():
                for idx in input_range:
                    key = imdb.normkey(idx)
                    im = imdb.image(key)
                    out_queue.put((key, im))
        except Exception as e:
            logger.info("Exception {}".format(e))
            raise
    finally:
        out_queue.put(None)


def write_predict(outtsv_file, in_queue):
    """Save prediction results to tsv file
    :type outtsv_file: str
    :type in_queue: loky.backend.queues.Queue
    """
    global _logger
    logger = _logger or logging
    # noinspection PyBroadException
    try:
        with open_with_lineidx(outtsv_file, with_temp=True) as fp, \
                open_with_lineidx(outtsv_file + ".keys", with_temp=True) as kfp:
            while True:
                result = in_queue.get()
                if not result:
                    break
                uid, image_key, result = result
                tell = fp.tell()
                fp.write("{}\t{}\n".format(
                    image_key,
                    json.dumps(result, separators=(',', ':'), sort_keys=True),
                ))
                kfp.write("{}\t{}\n".format(
                    uid, tell
                ))
    except Exception as e:
        logger.info("Exception {}".format(e))
        raise


def detinit(exp, num_gpu=0, gpu=None):
    """Cache initialization of caffe network for an experiment on a given GPU
    :param exp: The experiment to initialize with
    :type exp: Experiment
    :param num_gpu: Number of GPUs
    :param gpu: GPU device to run detection on
    :return: intialized network and cmap list
    :rtype: (object, Experiment)
    """
    global _init_done, _gpu, _exp, _net, _logger
    logger = _logger or logging
    if num_gpu and gpu is None:
        global _last_gpu
        with _gpu_lock:
            _last_gpu = (_last_gpu + 1) % num_gpu
            gpu = _last_gpu
    if exp is not None and exp != _exp:
        # if something has changed, re-initialize
        if _exp is not None:
            logger.info("exp {}->{} gpu {}->{}".format(
                _exp, exp, _gpu, gpu
            ))
        _init_done = False
    if _init_done:
        assert _exp and _net, "Experiment is not set"
        return _net, _exp
    logger.info("Initializing worker with {}".format(exp))
    # open the db to keep the file object open
    if not exp.imdb.is_open():
        exp.imdb.open_db()
    if num_gpu:
        assert num_gpu > gpu >= 0
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    _net = caffe.Net(str(exp.caffenet), str(exp.caffemodel), caffe.TEST)
    _init_done = True
    _gpu = gpu
    _exp = exp
    return _net, _exp


def detect(exp, num_gpu, gpu, to_json, key, im=None, thresh=None, class_thresh=None, obj_thresh=None, **kwargs):
    """Detect the key in the experiment
    :type exp: Experiment
    :param num_gpu: Number of GPUs
    :param gpu: GPU device to run detection on
    :param to_json: if should covert the output to json
    :param key: the key to detect in the experiment
    :param im: image
    :param thresh: global class threshold
    :param class_thresh: per-class threshold
    :param obj_thresh: objectness threshold
    """
    net, exp = detinit(exp, num_gpu, gpu)
    if im is None:
        im = exp.imdb.image(key)
    scores, boxes = im_detect(net, im, **kwargs)
    result = result2bblist(im, scores, boxes, exp.cmap,
                           thresh=thresh, obj_thresh=obj_thresh, class_thresh=class_thresh)
    if to_json:
        result = json.dumps(result, separators=(',', ':'), sort_keys=True)
    uid = exp.imdb.uid(key)
    image_key = exp.imdb.image_key(key)
    return uid, image_key, result


def initializer():
    init_logging()
    global _logger
    if not _logger:
        _logger = mp_logger()
    _logger.info("init worker once")


class Detector(object):

    def __init__(self, exp, num_gpu=0, gpu=None, max_workers=None):
        self.exp = exp  # type: Experiment
        self.num_gpu = num_gpu
        self.gpu = gpu
        if gpu:
            assert 0 <= gpu < num_gpu
        self.max_workers = max_workers or num_gpu or cpu_count()
        self.executor = None  # type: loky.process_executor.ProcessPoolExecutor

    def __repr__(self):
        return "Detector(exp: {}{})".format(
            self.exp,
            ", num_gpu: {}".format(self.num_gpu) if self.num_gpu else ""
        )

    def detect_async(self, key, to_json=False, im=None, **kwargs):
        """Get the async detection result for a single key
        :param key: the key in the experiment
        :param to_json: if should covert the output to json
        :param im: image to detect (if not given, key will be used to retrieve the image)
        :return: detection result future, or None if not ready yet
        """
        assert caffe, "caffe is not available"
        assert key is not None
        if self.executor is None:
            # if it is GPU we set no timeout because the process is bound to GPU
            self.executor = get_reusable_executor(max_workers=self.max_workers,
                                                  timeout=None if self.num_gpu else 100*60,
                                                  initializer=initializer)
        result = self.executor.submit(detect, self.exp, self.num_gpu, self.gpu, to_json, key, im=im, **kwargs)
        return result

    def detect(self, key, to_json=False, **kwargs):
        """Get the detection result for a single key
        :param key: the key in the experiment
        :param to_json: if should covert the output to json
        :return: detection result, or None if not ready yet
        """
        result = self.detect_async(key, to_json=to_json, **kwargs)
        _, _, result = result.result()
        return result

    def shutdown(self):
        if not self.executor:
            return
        # wait for all the futures and shitdown
        self.executor.shutdown(wait=True)
        self.executor = None

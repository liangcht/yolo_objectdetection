import sys
import os
import os.path as op
import json
import numpy as np
import requests

try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = op.abspath(this_file)

if __name__ == '__main__':
    # When run as script, modify path assuming absolute import
    sys.path.append(op.join(op.dirname(this_file), '..'))

#sys.path.insert(0, "D:/development/Caffe_iris/python/")
#sys.path.insert(0, "/home/ehazar/repos/quickdetection/src/CCSCaffe/python/")

from mmod.utils import makedirs, cwd, init_logging, open_with_lineidx, \
    file_cache, splitex_ver, splitfilename, FileCache, splitfilename, gpu_indices
from mmod.runcaffe import moved_solvers
from mmod.philly_utils import abspath, vc_hdfs_root, get_log_parent, get_arg, expand_vc_user
from mmod.checkpoint import iterate_weights
from mmod.io_utils import write_to_file, read_to_buffer
from mmod.deteval import deteval
from mmod.detector import Detector, write_predict, read_image, detinit
from mmod.imdb import ImageDatabase
from mmod.experiment import Experiment
from mmod.detection import im_detect, result2bblist
from mmod.visual import visualize
from mmod.api_utils import convert_api
from mmod.runeval import run_eval
from mmod.im_utils import img_from_base64, im_rescale

init_logging()

# %cd /tmp/work
#% cd z: /

caffenet = 'data/mit1k_test_Tax1300V11_3/test.prototxt'
caffemodel = 'data/mit1k_test_Tax1300V11_3/snapshot/model_iter_139962.caffemodel'
assert op.isfile(caffemodel) and op.isfile(caffenet)

path = "data/Tagger/groundtruth/mit1k.images.tsv"
assert op.isfile(path)
imdb = ImageDatabase(path)

cmapfile = 'data/Tax1300V11_3/labelmap.txt'
assert op.isfile(cmapfile)

test_data = "imdb1k_parity"
# test_data='Tax1300V11_1_with_bb'
# intsv_file = abspath(op.join('data', test_data, 'testX.tsv'), roots=['.'])
name = op.basename(caffemodel) + "." + test_data
# imdb = ImageDatabase(intsv_file, name=test_data)
exp = Experiment(imdb, caffenet, caffemodel, name=name, cmapfile=cmapfile)
outtsv_file = exp.predict_path
new_label_file = "data/mit1k_test_Tax1300V11_3/test0.tsv"

# truth = exp.imdb.all_truths()
# from mmod.deteval import load_dets
# detresults = load_dets(outtsv_file)
# ovthresh = ov_th = 0.0

gpus = list(gpu_indices())
num_gpu = len(gpus)
# create one detector for each GPU
detectors = [
    Detector(exp, num_gpu, gpu) for gpu in gpus
]
# detectors = [
#    Detector(exp, max_workers=1)
# ]
detector = detectors[0]

# input_range = range(len(imdb))
np.random.seed(7889)
input_range = np.random.choice(len(imdb) - 1, replace=False, size=(100,))

thresh = 0.1
obj_thresh = 0.2


def _conv_label(lb):
    if lb in exp.cmap:
        return lb
    lb = lb.replace('_', ' ')
    assert lb in exp.cmap
    return lb


class_thresh_file = 'data/mit1k_test_Tax1300V11_3/class_thresh.txt'
assert op.isfile(class_thresh_file)
with open(class_thresh_file) as f:
    class_thresh = {_conv_label(l[0]): np.maximum(float(l[1].rstrip()), thresh) for l in
                    [line.split('\t') for line in f.readlines()]}

all_det = {}
processed = 0
results = []
total_count = len(input_range)
with open_with_lineidx(outtsv_file) as fp:
    with imdb.open():
        for idx in input_range:
            gpu_idx = idx % len(detectors)
            detector = detectors[gpu_idx]
            key = exp.imdb.normkey(idx)
            uid = exp.imdb.uid(key)
            results.append(detector.detect_async(key, maintain_ratio=True))
            while len(results) > 2 * len(detectors):
                result = results.pop().result()
                uid, result = result
                result = [crect for crect in result if
                          crect['conf'] >= class_thresh[crect['class']] and crect['obj'] >= obj_thresh]

                fp.write("{}\t{}\n".format(
                    uid,
                    json.dumps(result, separators=(',', ':'), sort_keys=True),
                ))
                all_det[uid] = result
                processed += 1
                print("{} of {}".format(processed, total_count))

        while results:
            result = results.pop().result()
            uid, result = result
            result = [crect for crect in result if
                      crect['conf'] >= class_thresh[crect['class']] and crect['obj'] >= obj_thresh]

            fp.write("{}\t{}\n".format(
                uid,
                json.dumps(result, separators=(',', ':'), sort_keys=True),
            ))
            all_det[uid] = result
            processed += 1
            print("{} of {}".format(processed, total_count))

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

path = "data/mit1k_test_Tax1300V11_3/testX.tsv"
assert op.isfile(path)
imdb = ImageDatabase(path)
name = op.basename(caffemodel) + ".cvbrowser"
exp = Experiment(imdb, caffenet, caffemodel, name=name, cmapfile=cmapfile)
outtsv_file = exp.predict_path

session = requests.Session()
with open(op.expanduser("~/auth/cvbrowser.txt")) as fp:
    session.auth = ("Basic", fp.readline().strip())
session.headers.update({'Content-Type': 'application/octet-stream'})

processed = 0
total_count = len(imdb)
with open_with_lineidx(outtsv_file, "w") as fp:
    for key in imdb:
        uid = imdb.uid(key)
        data = imdb.raw_image(uid)
        res = session.post("http://tax1300v11x3e.westus.azurecontainer.io/api/detect", data=data)
        assert res.status_code == 200, "failed {}".format(uid)
        result = convert_api(json.loads(res.content)['objects'])
        result = [crect for crect in result if crect['conf'] >= class_thresh[crect['class']]]
        fp.write("{}\t{}\n".format(
            uid,
            json.dumps(result, separators=(',', ':'), sort_keys=True),
        ))
        processed += 1
        print("{} of {}".format(processed, total_count))

run_eval(imdb.all_truths(), outtsv_file)

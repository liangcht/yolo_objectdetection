import json
import base64
import os.path as op
from mmod.utils import open_with_lineidx, cwd
from mmod.im_utils import recursive_files_list

annotations_path = op.expanduser('~/Downloads/annotations.txt')
# parse annotations
annotations = {}
with open(annotations_path) as fp:
    for line in fp:
        uid, x, y, w, h, label = line.split(",")
        # Check if it is xywh
        x, y, w, h = [int(c) for c in [x, y, w, h]]
        annotations[uid] = json.dumps([{'rect': [x, y, x + w, y + h], 'class': label.strip()}])

# annotations file references relative paths, the paths must be relative to the root
root = op.expanduser('~/Pictures')

outtsv_path = op.expanduser('~/Desktop/boz.tsv')

with open_with_lineidx(outtsv_path) as fp:
    with cwd(root):
        for fname in recursive_files_list('.'):
            fname = fname.replace('\\', '/').replace("./", "")
            with open(fname, 'r') as ifp:
                b64 = base64.b64encode(ifp.read())
            label = annotations.get(fname)
            if not label:
                print("No label for {}".format(fname))
                label = json.dumps([])
            fp.write("{}\t{}\t{}\n".format(fname, label, b64))

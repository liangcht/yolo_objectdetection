import argparse
import torch
import sys
import os.path as op

try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = op.abspath(this_file)

if __name__ == '__main__':
    # When run as script, modify path assuming absolute import
    sys.path.append(op.join(op.dirname(this_file), '..'))

from mtorch.caffenet import CaffeNet
from mtorch.region_target_loss import NUM_IMAGES_WARM


def convert(protofile, caffemodel, snapshot_pt):
    """ converts caffemodel to .pt snapshot
    :param protofile: file with network structure
    :param caffemodel: weights of your network
    :param snapshot_pt: where to save
    Example:
        ['--net', '/PROTOTXT_DIR/train_yolo_withSoftMaxTreeLoss.prototxt',
        '--caffemodel', '/CAFFEMODEL_DIR/model_iter_10022.caffemodel',
        '--snapshot_pt', '/PT_DIR/snapshot/model_iter_10022.pt']
    """
    model = CaffeNet(protofile)
    model.load_weights(caffemodel)
    model = model.cuda()
    try:
        seen_images_ini = model.region_target.seen_images.data.cpu().detach()
    except AttributeError:
        seen_images_ini = NUM_IMAGES_WARM

    state_dict = model.state_dict()
    state = {
        'state_dict': state_dict,
        'seen_images': seen_images_ini
    }
    torch.save(state, snapshot_pt)


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch snapshots to caffemodel.')
    parser.add_argument('--net',
                        help='Caffenet prototxt file for caffe training', required=True)
    parser.add_argument('--caffemodel',
                        help='caffemodel file with saved weights', required=True)
    parser.add_argument('--snapshot_pt',
                        help='path to save .pt snapshot', required=False)
    args = parser.parse_args()
    args = vars(args)

    if not args["snapshot_pt"]:
        args["snapshot_pt"] = args["caffemodel"][:-len(".caffemodel")] + ".pt"

    convert(args["net"], args["caffemodel"], args["snapshot_pt"])


if __name__ == '__main__':
    main()

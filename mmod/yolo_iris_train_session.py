import argparse
import os
import warnings
import torch
import sys
import traceback
import shutil
from mmod.simple_parser import load_labelmap_list
from mtorch.yolo_v2 import yolo_3extraconv
from mtorch.darknet import darknet_layers
from mtorch.yolo_v2_loss import YoloLossForPlainStructure
from mtorch.augmentation import DefaultDarknetAugmentation
from mtorch.multifixed_scheduler import MultiFixedScheduler
from mtorch.dataloaders import create_imdb_dataset
from mtorch.caffesgd import CaffeSGD
from mtorch.lr_scheduler import LinearDecreasingLR
import pdb

pretrain_model = '/home/tobai/ODExperiments/yoloSample/yolomodel/Logo_YoloV2_CaffeFeaturizer.pt'
total_epoch = 300
log_pth = './output_irisInit/'
# TODO: solver param
# steps = [100, 5000, 9000]
# lrs = [0.00001, 0.00001, 0.0001]
datafile = '/home/tobai/ODExperiments/dataset/benchmark_dataset/Ping-Logo-55/Ping-Logo-55.train_images.txt'
cmapfile = '/home/tobai/ODExperiments/dataset/benchmark_dataset/Ping-Logo-55/Ping-Logo_labels.txt'
label_map = cmapfile

def to_python_float(t):
    if isinstance(t, (float, int)):
        return t
    if hasattr(t, 'item'):
        return t.item()
    return t[0]


def train(model, num_class, device):
    # switch to train mode
    model.train()
    # model.freeze('dark6')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    criterion = YoloLossForPlainStructure(num_classes=num_class)
    criterion = criterion.cuda()

    # load training data
    augmenter = DefaultDarknetAugmentation()
    augmented_dataset = create_imdb_dataset(datafile,
                                            cmapfile, augmenter())
    data_loader = torch.utils.data.DataLoader(augmented_dataset, shuffle=True, batch_size=16) 
    scheduler = LinearDecreasingLR(optimizer, total_iter=len(data_loader)*total_epoch)

    for epoch in range(total_epoch):
        for inputs, labels in data_loader:
            scheduler.step()
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs.float().to(device), labels.float().to(device))
            loss.backward()
            optimizer.step()

        # pdb.set_trace()
        reduced_loss = to_python_float(loss.data)
        print("epoch {} loss {}".format(epoch, reduced_loss))
        print(scheduler.get_lr())
        state = {
            'epochs': epoch,
            'state_dict': model.state_dict(),
            'seen_images': criterion.seen_images,
            'region_target.biases': criterion.criterion.region_target.biases,
            'region_target.seen_images': criterion.criterion.seen_images
        }
        if optimizer:
            state.update({
                'optimizer': optimizer.state_dict(),
            })
        snapshot_pt = log_pth + "_epoch_{}".format(epoch + 1) + '.pt'
        if epoch % 20 == 0:
            print("Snapshotting to: {}".format(snapshot_pt))
            torch.save(state, snapshot_pt)


def main(args, log_pth):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cmap = load_labelmap_list(label_map)
    model = yolo_3extraconv(darknet_layers(),
        weights_file=pretrain_model,
        caffe_format_weights=True,
        ignore_mismatch = True,
        num_classes=len(cmap),
        map_location=lambda storage, loc: storage.cuda(0)).cuda()
    if model.pretrained_info:
        print("Pretrained model was loaded: " + model.pretrained_info)
    # TODO: set distributed

    # TODO: add solver_params
    train(model, len(cmap), device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--branch_name')
    # parser.add_argument('--task_config')
    # parser.add_argument('--experiment_id')
    # parser.add_argument('--random_seed', type=int, default=0)
    # parser.add_argument('--train_data')
    # parser.add_argument('--test_data')
    args = parser.parse_args()

    if os.path.exists(log_pth):
        shutil.rmtree(log_pth)
    else:
        os.makedirs(log_pth)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            main(args, log_pth)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback)
        print("error: {}".format(e.args[0]))
        raise SystemExit(exc_traceback)
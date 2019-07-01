import argparse
import os
import warnings
import torch
import sys
import traceback
import shutil
from torch.optim.lr_scheduler import StepLR
from iristorch.models.yolo_v2 import Yolo
from iristorch.layers.yolo_loss import YoloLoss
from iristorch.transforms.transforms import SSDTransform, IrisODTransform
from mtorch.augmentation import TestAugmentation
from mtorch.azureBlobODDataset import AzureBlobODDataset
import pdb
import json
import numpy as np
from mtorch.yolo_predict import PlainPredictorClassSpecificNMS
from mmod.meters import AverageMeter
from irisexperiment import ObjectDetectionEvaluator
from mmod.detection import result2bbIRIS
import time

# pretrain_model = '/home/tobai/ODExperiments/yoloSample/yolomodel/Logo_YoloV2_CaffeFeaturizer.pt'
pretrain_model = '/app/pretrain_model/Logo_YoloV2_CaffeFeaturizer.pt'

total_epoch = 300
log_pth = './output_irisInit/'
# TODO: solver param
# steps = [100, 5000, 9000]
# lrs = [0.00001, 0.00001, 0.0001]
trainingManifestFile = '/app/Ping-Logo/PingLogo_ltwh_trainingManifest.json'

def to_python_float(t):
    if isinstance(t, (float, int)):
        return t
    if hasattr(t, 'item'):
        return t.item()
    return t[0]

def eval(model, num_classes, test_loader):
    print("Creating test data loader")
    model.eval()
    yolo_predictor = PlainPredictorClassSpecificNMS(num_classes=num_classes).cuda()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    tic = time.time()
    results = list()
    gts = list()
    end = time.time()
    for i, inputs in enumerate(test_loader):
        data_time.update(time.time() - end)

        data, image_keys, hs, ws, gt_batch = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        gts += gt_batch
        # compute output
        for im, image_key, h, w in zip(data, image_keys, hs, ws):
            im = im.unsqueeze_(0)
            im = im.float().cuda()
            with torch.no_grad():
                features = model(im)
            prob, bbox = yolo_predictor(features, torch.Tensor((h, w)))

            bbox = bbox.cpu().numpy()
            prob = prob.cpu().numpy()
            assert bbox.shape[-1] == 4
            bbox = bbox.reshape(-1, 4)
            prob = prob.reshape(-1, prob.shape[-1])
            result = result2bbIRIS((h, w), prob, bbox, None,
                                   thresh=None, obj_thresh=None)
            # skip the background class
            for pre_idx, pre_box in enumerate(result):
                if pre_box[0] == 0:
                    del result[pre_idx]
            results.append(result)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    evaluator = ObjectDetectionEvaluator('a','a','a')
    evaluator.add_predictions(results, gts)
    eval_result =evaluator.get_report()
    print(eval_result)
    model.train()

def train(model, num_class, device):
    # switch to train mode
    model.train()
    # model.freeze('dark6')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    criterion = YoloLoss(num_classes=num_class)
    criterion = criterion.cuda()

    # load training data
    augmented_dataset = None
    with open(trainingManifestFile) as json_data:
        training_manifest = json.load(json_data)
        account_name = training_manifest["account_name"]
        container_name = training_manifest["container_name"]
        dataset_name = training_manifest["name"]
        sas_token = training_manifest["sas_token"]
        image_list = training_manifest["images"]['train']
        eval_image_list = training_manifest["images"]['val']
        augmented_dataset = AzureBlobODDataset(account_name, container_name, dataset_name, sas_token, image_list, SSDTransform(416))
        test_dataset = AzureBlobODDataset(account_name, container_name, dataset_name, sas_token, eval_image_list, TestAugmentation(), predict_phase=True)
    
    data_loader = torch.utils.data.DataLoader(augmented_dataset, shuffle=True, batch_size=16) 
    test_data_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=16) 
    scheduler = StepLR(optimizer, step_size=total_epoch//4, gamma=0.1)

    for epoch in range(total_epoch):
        for inputs, labels in data_loader:
            scheduler.step()
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs.float().to(device), labels.float().to(device))
            loss.backward()
            print(loss.data)
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
        if epoch % 1 == 0:
            print("Snapshotting to: {}".format(snapshot_pt))
            torch.save(state, snapshot_pt)
            eval(model, num_class, test_data_loader)


def main(args, log_pth):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_class = 0
    with open(trainingManifestFile) as json_data:
        training_manifest = json.load(json_data)
        num_class = len(training_manifest["tags"])

    model = Yolo(num_classes = num_class)
    
    pretrained_dict = torch.load(pretrain_model)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
    model.load_state_dict(pretrained_dict, strict=False)
    # TODO: set distributed

    # TODO: add solver_params
    model.to(device)
    train(model, num_class, device)


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

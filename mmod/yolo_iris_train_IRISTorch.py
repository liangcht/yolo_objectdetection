import argparse
import os
import warnings
import torch
from torch.utils.data import SequentialSampler
import sys
import traceback
import shutil
from mmod.simple_parser import load_labelmap_list
from iristorch.models.yolo_v2 import Yolo
from iristorch.layers.yolo_loss import YoloLoss
from iristorch.transforms.transforms import YoloV2InferenceTransform, YoloV2TrainingTransform
from iristorch.evaluators.evaluators import ObjectDetectionEvaluator
from iristorch.layers.yolo_predictor import PlainPredictorClassSpecificNMS
from torch.optim.lr_scheduler import StepLR
from mtorch.azureBlobODDataset import AzureBlobODDataset
from mtorch.IRIS_azureBlobODDataset import IRISAzureBlobODDataset
import json

import numpy as np
from mmod.detection import result2bbIRIS
import time

import math

#pretrain_model = '/app/pretrain_model/Logo_YoloV2_CaffeFeaturizer.pt'
pretrain_model = '/app/pretrain_model/YoloV2_CaffeFeaturizer_V2.pt'

total_epoch = 300
log_pth = './output_irisInit/'
# TODO: solver param
# steps = [100, 5000, 9000]
# lrs = [0.00001, 0.00001, 0.0001]
#datafile = '/app/Ping-Logo/Ping-Logo-55.train_images.txt'
#cmapfile = '/app/Ping-Logo/Ping-Logo_labels.txt'
#datafile = '/app/animal661/Animal.train_images.txt'
#testfile = '/app/animal661/Animal.test_images.txt'
#cmapfile = '/app/animal661/Animal-661_labels.txt'
#trainingManifestFile = '/app/Ping-Logo/PingLogo_trainingManifest.json'
#trainingManifestFile = '/app/animal661/Animal661_trainingManifest.json'
trainingManifestFile = '/app/fridge_trainingManifest.json'
#label_map = cmapfile

steps = [100, 5000, 9000, 10000000]
lrs = [0.001, 0.0006, 0.0003, 0.0001]
init_lr = lrs[3]

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
    results = list()
    gts = list()
    for inputs, targets in test_loader:
        gts += targets
        for im in inputs:
            im = im.unsqueeze_(0)
            im = im.float().cuda()
            with torch.no_grad():
                features = model(im)
            result = yolo_predictor(features)
            results.append(result)

    evaluator = ObjectDetectionEvaluator()
    evaluator.add_predictions(results, gts)
    eval_result =evaluator.get_report()
    print(eval_result)
    model.train()

def train(model, num_class, device):
    # switch to train mode
    model.train()
    # model.freeze('dark6')
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)
    criterion = YoloLoss(num_classes=num_class)
    criterion = criterion.cuda()


    # load training data
    augmented_dataset = None
    ''' Original json
    with open(trainingManifestFile) as json_data:
        training_manifest = json.load(json_data)
        account_name = training_manifest["account_name"]
        container_name = training_manifest["container_name"]
        dataset_name = training_manifest["name"]
        sas_token = training_manifest["sas_token"]
        image_list = training_manifest["images"]['train']
        test_image_list = training_manifest["images"]['val']
        augmented_dataset = AzureBlobODDataset(account_name, container_name, dataset_name, sas_token, image_list, YoloV2TrainingTransform(416))
        test_dataset = AzureBlobODDataset(account_name, container_name, dataset_name, sas_token, test_image_list, YoloV2InferenceTransform(416))
    '''

    with open(trainingManifestFile, encoding='utf-8-sig') as json_data:
        training_manifest = json.load(json_data)
        account_name = "irisliang"
        container_name = "aml-e1b16b23d7d041569d9c76db1b968d9e"
        dataset_name = "images"
        sas_token = "?sv=2017-04-17&ss=bfqt&srt=sco&sp=rl&st=2019-07-16T10%3A28%3A00Z&se=2020-07-17T10%3A28%3A00Z&sig=lFxhQ3zXdALYTc0MGmzRiAgiBURsvj%2Fej%2FuUrbV37oc%3D"
        image_list = training_manifest["DataSetManifestInfo"]['Images']
        test_image_list = training_manifest["ValidationDataSetManifestInfo"]['Images']
        augmented_dataset = IRISAzureBlobODDataset(account_name, container_name, dataset_name, sas_token, image_list, YoloV2TrainingTransform(416))
        test_dataset = IRISAzureBlobODDataset(account_name, container_name, dataset_name, sas_token, test_image_list, YoloV2InferenceTransform(416))

    # load training data
    # augmenter = DefaultDarknetAugmentation()
    # augmented_dataset = create_imdb_dataset(datafile, cmapfile, augmenter())
    

    # calculate config base on the dataset
    nSample = len(augmented_dataset)
    #total_epoch = max(5, math.ceil(50000 /(nSample+ 300)))
    if nSample < 1024:
        batch_size = 16
    elif nSample < 2048:
        batch_size = 32
    else:
        batch_size = 64
    steps = [i_step * total_epoch * nSample / batch_size for i_step in [80, 400, 600, 10000]]

    data_loader = torch.utils.data.DataLoader(augmented_dataset, shuffle=True, batch_size=batch_size)

    sampler = SequentialSampler(test_dataset)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, sampler=sampler, batch_size=32, num_workers=4, collate_fn=_list_collate)

    scheduler = StepLR(optimizer, step_size=total_epoch * len(data_loader) // 4, gamma=0.1)

    for epoch in range(total_epoch):
        start = time.time()
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
        print("epoch {} loss {} elapse {} seconds".format(epoch, reduced_loss, time.time() - start))
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
        if epoch % 10 == 0:
            print("Snapshotting to: {}".format(snapshot_pt))
            torch.save(state, snapshot_pt)
            eval(model, num_class, test_data_loader)

def _list_collate(batch):
    """ Function that collates lists or tuples together into one list (of lists/tuples).
    Use this as the collate function in a Dataloader,
    if you want to have a list of items as an output, as opposed to tensors
    """
    items = list(zip(*batch))
    return items

def main(args, log_pth):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with open(trainingManifestFile, encoding='utf-8-sig') as json_data:
        training_manifest = json.load(json_data)
        cmap = training_manifest["DataSetManifestInfo"]["Tags"] #training_manifest["tags"]
    model = Yolo(num_classes = len(cmap))

    if args.eval_only:
        model_dict = torch.load(args.model_file)
        model.load_state_dict(model_dict["state_dict"], strict=True)
        
        with open(trainingManifestFile) as json_data:
            training_manifest = json.load(json_data)
            account_name = training_manifest["account_name"]
            container_name = training_manifest["container_name"]
            dataset_name = training_manifest["name"]
            sas_token = training_manifest["sas_token"]

            test_image_list = training_manifest["images"]['val']
            test_dataset = AzureBlobODDataset(account_name, container_name, dataset_name, sas_token, test_image_list, YoloV2InferenceTransform(416))
        sampler = SequentialSampler(test_dataset)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, sampler=sampler, batch_size=32, num_workers=4, collate_fn=_list_collate)
        
        model.to(device)
        eval(model, len(cmap), test_data_loader)
    else:
        pretrained_dict = torch.load(pretrain_model)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                        (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        model.load_state_dict(pretrained_dict, strict=False)
        # TODO: set distributed

        from mtorch.custom_layers_ops import freeze_modules_for_training
        freeze_modules_for_training(model, 'dark5e')

        # TODO: add solver_params
        model.to(device)
        train(model, len(cmap), device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--branch_name')
    # parser.add_argument('--task_config')
    # parser.add_argument('--experiment_id')
    # parser.add_argument('--random_seed', type=int, default=0)
    # parser.add_argument('--train_data')
    # parser.add_argument('--test_data')
    parser.add_argument('--eval_only', type=bool)
    parser.add_argument('--model_file')
    args = parser.parse_args()

    if os.path.exists(log_pth):
        shutil.rmtree(log_pth)
    os.makedirs(log_pth)
    print(args)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            main(args, log_pth)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback)
        print("error: {}".format(e.args[0]))
        raise SystemExit(exc_traceback)

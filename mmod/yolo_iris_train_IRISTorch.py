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
from iristorch.schedulers import MultiFixedScheduler
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

def calculate_anchor(image_list):
    k = 5
    img_size = 416
    wh_list = []
    for image in image_list:
        for region in image["Regions"]:
            bbox = region["BoundingBox"]
            wh_list.append((bbox['Width'], bbox['Height']))

    bellevue_objects = np.asarray(wh_list)
    centroids = bellevue_objects[np.random.choice(np.arange(len(bellevue_objects)), k, replace=False)]
    anchors = kmeans_iou(k, centroids, bellevue_objects, feature_size=img_size / 32)
    anchor_list = []
    for a in anchors:
        anchor_list.append(a[0])
        anchor_list.append(a[1])
    return anchor_list

def _keep_max_num_bboxes(bboxes):
        """Discards boxes beyond num_bboxes"""
        num_bboxes = 30
        cur_num = bboxes.shape[0]
        diff_to_max = num_bboxes - cur_num
        if diff_to_max > 0:
            bboxes = np.lib.pad(bboxes, ((0, diff_to_max), (0, 0)),
                                "constant", constant_values=(0.0,))
        elif diff_to_max < 0:
            bboxes = bboxes[:num_bboxes, :]
        return bboxes

def train(model, num_class, device):
    # switch to train mode
    model.train()
    # model.freeze('dark6')
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)

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
        #anchors = calculate_anchor(image_list)

    criterion = YoloLoss(num_classes=num_class, seen_images=0)
    criterion = criterion.cuda()

    # load training data
    # augmenter = DefaultDarknetAugmentation()
    # augmented_dataset = create_imdb_dataset(datafile, cmapfile, augmenter())
    
    
    # calculate config base on the dataset
    num_samples =  len(augmented_dataset)
    num_epochs = max(5, 50000 // (num_samples + 300)) * 5

    if num_samples < 1024:
        batch_size = 16
    elif num_samples < 2048:
        batch_size = 32
    else:
        batch_size = 64

    total_iters = num_epochs * num_samples // batch_size
    learning_rates = [0.0001, 0.001, 0.0006, 0.0003, 0.0001]
    original_stage_iters = [50, 120, 400, 600, 10000]
    stage_iters = [(lambda x : x * total_iters // 1000)(x) for x in original_stage_iters]
        
    
    #steps = [i_step * total_epoch * nSample / batch_size for i_step in [80, 400, 600, 10000]]

    data_loader = torch.utils.data.DataLoader(augmented_dataset, shuffle=True, batch_size=batch_size, collate_fn=_list_collate)

    sampler = SequentialSampler(test_dataset)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, sampler=sampler, batch_size=batch_size, num_workers=4, collate_fn=_list_collate)

    #scheduler = StepLR(optimizer, step_size=total_epoch * len(data_loader) // 4, gamma=0.1)
    scheduler = MultiFixedScheduler(optimizer, stage_iters, learning_rates)

    for epoch in range(total_epoch):
        start = time.time()
        for inputs, labels in data_loader:
            import pdb
            pdb.set_trace()
            yolo_target = np.zeros(shape=(len(labels), 5), dtype=float)
            for i, t in enumerate(labels):
                yolo_target[i] = np.asarray([(t[1] + t[3]) / 2.0, (t[2] + t[4]) / 2.0, t[3] - t[1], t[4] - t[2], t[0]])
            yolo_target = _keep_max_num_bboxes(yolo_target).flatten()

            scheduler.step()
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs.float().to(device), yolo_target.float().to(device))
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


def area(x):
    if len(x.shape) == 1:
        return x[0] * x[1]
    else:
        return x[:, 0] * x[:, 1]

def kmeans_iou(k, centroids, points, iter_count=0, iteration_cutoff=25, feature_size=13):

    best_clusters = []
    best_avg_iou = 0
    best_avg_iou_iteration = 0

    npoi = points.shape[0]
    area_p = area(points)  # (npoi, 2) -> (npoi,)

    while True:
        cen2 = centroids.repeat(npoi, axis=0).reshape(k, npoi, 2)
        cdiff = points - cen2
        cidx = np.where(cdiff < 0)
        cen2[cidx] = points[cidx[1], cidx[2]]

        wh = cen2.prod(axis=2).T  # (k, npoi, 2) -> (npoi, k)
        dist = 1. - (wh / (area_p[:, np.newaxis] + area(centroids) - wh))  # -> (npoi, k)
        belongs_to_cluster = np.argmin(dist, axis=1)  # (npoi, k) -> (npoi,)
        clusters_niou = np.min(dist, axis=1)  # (npoi, k) -> (npoi,)
        clusters = [points[belongs_to_cluster == i] for i in range(k)]
        avg_iou = np.mean(1. - clusters_niou)
        if avg_iou > best_avg_iou:
            best_avg_iou = avg_iou
            best_clusters = clusters
            best_avg_iou_iteration = iter_count

        print("\nIteration {}".format(iter_count))
        print("Average iou to closest centroid = {}".format(avg_iou))
        print("Sum of all distances (cost) = {}".format(np.sum(clusters_niou)))

        new_centroids = np.array([np.mean(c, axis=0) for c in clusters])
        isect = np.prod(np.min(np.asarray([centroids, new_centroids]), axis=0), axis=1)
        aa1 = np.prod(centroids, axis=1)
        aa2 = np.prod(new_centroids, axis=1)
        shifts = 1 - isect / (aa1 + aa2 - isect)

        # for i, s in enumerate(shifts):
        #     print("{}: Cluster size: {}, Centroid distance shift: {}".format(i, len(clusters[i]), s))

        if sum(shifts) == 0 or iter_count >= best_avg_iou_iteration + iteration_cutoff:
            break

        centroids = new_centroids
        iter_count += 1

    # Get anchor boxes from best clusters
    anchors = np.asarray([np.mean(cluster, axis=0) for cluster in best_clusters])
    anchors = anchors[anchors[:, 0].argsort()]
    print("k-means clustering pascal anchor points (original coordinates) \
    \nFound at iteration {} with best average IoU: {} \
    \n{}".format(best_avg_iou_iteration, best_avg_iou, anchors*feature_size))

    return anchors

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

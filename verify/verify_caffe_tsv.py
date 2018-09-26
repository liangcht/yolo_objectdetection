from __future__ import with_statement
import sys
import os
import os.path as op
from collections import OrderedDict
import time
import numpy as np
#sys.path.append('/work/objectdetection/')
#import cv2
import torch
from torchvision import transforms
#import caffe
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mmod.utils import init_logging
from mmod.imdb import ImageDatabase
from mmod import deteval
from mtorch import Transforms
from PIL import ImageChops
import csv
import math, operator
import logging
import colorsys


EXT = ".csv"
OFFSET = "offset_"
CONSTANT = "const params_"
DISTORTION = "distortion_params_"
SPATIAL = "spatial_params_"

IMAGE = "image"
LABEL = "bbox"

BBOX_DIM = 4
COMPARISON_FOLDER = "data/mazontak/ComparisonCaffePython/test"
CAFFE_RES = "Caffe"
PYTHON_RES = "Python"#_Skimage"
DIFF_RES = "Abs_Diff"#_No_Antialising"

PYTHON_RES2 = "Python_Torchvision"
DIFF_RES2 = "Abs_Diff_Antialising"

PROTO = "/work/mnt/qd_output/Tax1300V10_5_darknet19_coco2017_bb_only/train_data_layer_only.prototxt"
CAFFEMODEL = "/work/mnt/qd_output/Tax1300V10_5_darknet19_coco2017_bb_only/snapshot/model_iter_134000.caffemodel"


#Statistics KEYS
MIN_DIST = "min_dist"
MAX_DIST = "max_dist"
MEAN_DIST = "mean_dist"
MIN_IOU = "min_iou"
MAX_IOU = "max_iou"
MEAN_IOU = "mean_iou"
RMSE = "rmse_"
KEYS_CONST = ['boxes', 'jitter', 'hue', 'saturation ', 'exposure', 'mean_r', 'mean_g',
            'mean_b', 'pixel_value_scale', 'max_samples', 'random_scale_min', 'random_scale_max', 
		    'output_ssd_label' , 'fix_offset' , 'rotate_max'] 
KEYS_SPATIAL = ['original_width', 'original_height', 'width', 'height', 'new_width', 'new_height',
	        'sized_width', 'sized_height', 'radians', 'dx', 'dy', 'flip']
KEYS_DISTORT = ['rand_hue', 'rand_sat', 'rand_exp']          
KEYS_STATS = [RMSE + "sk", RMSE + "tv", MEAN_IOU, MEAN_DIST, MIN_IOU, MAX_IOU, MIN_DIST, MAX_DIST]


def HSVToRGB(h, s, v):
    return colorsys.hsv_to_rgb(h, s, v)
 
def getDistinctColors(n):
    huePartition = 1.0 / (n + 1)
    return [HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n)]  
    

def read_const_params(offset):  
    with open(CONSTANT + str(offset*5 + 1) + EXT, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        print("read", CONSTANT)
        keys = KEYS_CONST
        for i,values in enumerate(data):
            res = dict(zip(keys, values))
    #os.remove(CONSTANT)
    return res

def read_spatial_params(offset):  
    with open(SPATIAL + str(offset*5 + 1) + EXT, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        keys = KEYS_SPATIAL
        for i,values in enumerate(data):
            res = dict(zip(keys, values))
              #os.remove(SPATIAL)
    return res

def read_distort_params(offset):  
    with open(DISTORTION + str(offset*5 + 1) + EXT, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        keys = KEYS_DISTORT
        for i,values in enumerate(data):
            res = dict(zip(keys, values))
    #os.remove(DISTORTION)
    return res

def read_index(offset):
    with open(OFFSET + str(offset*5 + 1) + EXT, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        keys = ['index']
        for i,values in enumerate(data):
            res = dict(zip(keys, values))
    #os.remove(OFFSET)
    return res


def get_bounding_boxes(truth):
    length = len(truth)
    bboxs = np.zeros(shape=(length , BBOX_DIM), dtype="float")
    for i, bbox in enumerate(truth):
        bboxs[i,:] = [float(val) for val in bbox['rect']]
    return bboxs

point_table = ([0] + ([255] * 255))

def black_or_b(im1, im2):
    diff = ImageChops.difference(im1.convert('L'), im2.convert('L'))
    #diff = diff.convert('L')
 #   diff = diff.point(point_table)
    new = diff# diff.convert('RGB')
   # new.paste(im1, mask=diff)
    return new


def rmsdiff(im1, im2):
    "Calculate the root-mean-square difference between two images"

    pixels = list(ImageChops.difference(im1, im2).convert('L').getdata())
    # calculate rms
    print(pixels[0:3])
    return math.sqrt(reduce(operator.add, map(lambda p: p*p, pixels))/ float(len(pixels)))

def visualize_result(result, title="", ax=None, crop_box=None, cols=getDistinctColors(30)):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(result[Transforms.IMAGE])
    #title += "\n Image Size: " + str(result[Transforms.IMAGE].size)
    for i, box in enumerate(result[Transforms.LABEL]):
        xmin, ymin, xmax, ymax = [int(x) for x in box[0:4]]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin - 1,
                             ymax - ymin - 1, fill=False,
                             edgecolor= cols[i] ,
                             linewidth=2.5)
        ax.add_patch(rect)
       # title += "\n BBox" + str(i) + ": " + str((xmin, ymin, xmax, ymax))

    if crop_box is not None:
        rect = plt.Rectangle(crop_box[0:2], crop_box[2] - 1,
                             crop_box[3] - 1, fill=False,
                             edgecolor='g',
                             linewidth=2.5)
        ax.add_patch(rect)

    ax.set_title(title)

def compare_bbox_results(bbox1, bbox2):
    if bbox1.shape[0] != bbox2.shape[0]:
        logging.info("Different number of boxes, 1 - {}, 2 - {}".format(bbox1.shape[0], bbox2.shape[0]))  
    min_dists = []
    box_index = []
    ious = []
    for box in bbox2:
        dists = np.sum(abs(bbox1 - box), axis=1)
        min_dists.append(np.min(dists, axis=0))
        min_ind = np.argmin(dists, axis=0)
        box_index.append(min_ind)
        ious.append(deteval.iou(bbox1[min_ind], box))
    
    return min_dists, box_index, ious


def print_comparison_reslts(bbox1, bbox2, min_dists, box_index, ious):
    for bb2, min_dist, ind, iou in zip(bbox2, min_dists, box_index, ious):
        logging.info("Box1:{}, Box2:{}, L1:{:.2}, iou:{:.2}".format(
            [round(x,2) for x in bbox1[ind,:]], [round(y,2) for y in bb2] , min_dist, round(iou,2)))
    
def calc_stats(min_dists, ious):
    stats = {}
    stats[MIN_DIST] = round(min(min_dists),2)  
    stats[MAX_DIST] = round(max(min_dists),2)  
    stats[MIN_IOU] = round(min(ious),2)  
    stats[MAX_IOU] = round(max(ious),2)  
    stats[MEAN_IOU] = round(sum(ious)/len(ious),2)  
    stats[MEAN_DIST] = round(sum(min_dists)/len(ious),2)
    return stats

def to_bounding_boxes(labels, num_boxes, width, height):
    truth = labels.cpu().reshape((num_boxes, -1))
    boxes = np.zeros_like(truth)
    boxes[:,0] = truth[:,0] - truth[:,2] / 2
    boxes[:,1] = truth[:,1] - truth[:,3] / 2
    boxes[:,2] = truth[:,0] + truth[:,2] / 2 
    boxes[:,3] = truth[:,1] + truth[:,3] / 2
    boxes[:,(0,2)] *= width
    boxes[:,(1,3)] *= height
    nonzero_row_indices =[i for i in range(boxes.shape[0]) if not np.allclose(boxes[i,:],0.0)]
    boxes = boxes[nonzero_row_indices,:] 
    return boxes



#os.environ['GLOG_minloglevel'] = '2'

def compare(offset_, log_name):
    comparison_results = {}
    protofile = PROTO
    caffemodel = CAFFEMODEL
    logging.basicConfig(filename=log_name, level=logging.INFO)

    try:
        assert op.isfile(protofile) 
    except:
        logging.info(protofile + "is not a file")

    try:
        assert op.isfile(caffemodel)
    except:
        logging.info(caffemodel + "is not a file")

        
    from mtorch.caffenet import CaffeNet

    init_logging()
    
    model = CaffeNet(protofile, keep_diffs=True, verbose=True)
    inputs = model.inputs
    model.load_weights(caffemodel)
    model = model.cuda()
    inputs = inputs.cuda()

    means = torch.tensor([float(i) for i in model.net_info['layers'][0]['transform_param']['mean_value']])
    means = means.resize_((3,1,1)).numpy()
    vdata, labels = inputs()

    try:
        const_params = read_const_params(offset_)
    except:
        logging.info("Could not read const params for offset " + str(offset_))
        return
    try:
        spatial_params = read_spatial_params(offset_)
    except:
        logging.info("Could not read spatial params for offset " + str(offset_))
        return
    try:
        distort_params = read_distort_params(offset_)
    except:
        logging.info("Could not read distortion params for offset " + str(offset_))
        return
    try:
        res = read_index(offset_)
    except:
        logging.info("Could not read offset params for offset " + str(offset_))
        return

    img = vdata.cpu().squeeze().numpy()
    bboxes_orig = to_bounding_boxes(labels, int(float(const_params["boxes"])), 
        float(spatial_params["width"]), float(spatial_params["height"]))

   
    db_path =  model.net_info['layers'][0]['tsv_data_param']['source']
    db = ImageDatabase(db_path)
    img_index = int(res["index"]) - 1 + offset_

    key = db.normkey(img_index)
    im = db.image(key)

    toPIL = transforms.ToPILImage()

    #plt.figure()
   
    #print(type(res))
    im_PIL = toPIL(im[:,:, (2,1,0)])
    
    #print(type(im))
    #print(im.mode)
    #print(im.size)
    bboxes = get_bounding_boxes(db.truth_list(key))
    sample = {IMAGE: im_PIL, LABEL: bboxes}
    crop300 = Transforms.Crop((0, 0, 300, 300), allow_outside_bb_center=False)

    sample = crop300(sample)
    random_distorter = Transforms.DistortColor(float(distort_params["rand_hue"]),
    float(distort_params["rand_sat"]), float(distort_params["rand_exp"]))
    resizer_skimage_no_antialising = Transforms.Resize(output_size=(float(spatial_params["new_width"]),float(spatial_params["new_height"])))
    resizer_torch_with_antialising = Transforms.Resize(output_size=(float(spatial_params["new_width"]),float(spatial_params["new_height"])),
    library=Transforms.TORCHVISION)

    canvas_adapter = Transforms.CanvasAdapter(size=(int(spatial_params["sized_width"]),int(spatial_params["sized_height"])),
    #defaul_pixel_val=tuple([int(float(i)) for i in model.net_info['layers'][0]['transform_param']['mean_value']]),
    defaul_pixel_val=(int(float(const_params["pixel_value_scale"])/2), 
    int(float(const_params["pixel_value_scale"])/2), int(float(const_params["pixel_value_scale"])/2)),
    dx=float(spatial_params["dx"]),dy=float(spatial_params["dy"]))

    horizontal_flipper = Transforms.RandomHorizontalFlip(int(spatial_params["flip"]))
    
    resized_no_antialising = resizer_skimage_no_antialising(sample)
    resized_antialising = resizer_torch_with_antialising(sample)

    resized = resized_no_antialising
    on_canvas = canvas_adapter(resized)
    flipped = horizontal_flipper(on_canvas)
    distorted = random_distorter(flipped)


    fig, (ax0, ax1) = plt.subplots(1, 2)
    caffe_res = (img + means).swapaxes(0,2).swapaxes(0,1)[:,:,(2,1,0)].astype(np.uint8)
    #ax1.imshow(caffe_res)
    title = "Flip {}, dx {:.2}, dy {:.2}, \n  hue {:.2}, sat {:.2}, exp {:.2} ".format(
    #, scale in x {:.1}, scale in _y {:.1}, \n  \
   
        spatial_params["flip"], spatial_params["dx"], spatial_params["dy"],
       # float(spatial_params["new_width"])/float(spatial_params["original_width"]), 
       # float(spatial_params["new_height"])/float(spatial_params["original_height"]),
        float(distort_params["rand_hue"]),float(distort_params["rand_sat"]),float(distort_params["rand_exp"])) 
   
    try:
        dists, inds, ious = compare_bbox_results(bboxes_orig[:,:4], distorted[Transforms.LABEL])
    except ValueError as err:
        logging.info("Number of boxes is not equal {}, for offset {}, exits...".format(err,offset_))
        return
    except:
        loggind.info("Unknown probelm, exits..")
        return
    
    caffe_res = toPIL(caffe_res)
    cols = getDistinctColors(max(bboxes_orig.shape[0], distorted[Transforms.LABEL].shape[0]))

    visualize_result(horizontal_flipper(sample), "Original Image", ax0, cols=cols)

    visualize_result({IMAGE: caffe_res, LABEL: bboxes_orig[inds,:]}, title, ax1, cols=cols)
    
    logging.info("Saving Caffe result for " + str(img_index))

    fig.savefig(op.join(COMPARISON_FOLDER, "Im_"  + str(img_index) + "_" + CAFFE_RES + ".png")) # save the figure to file
    plt.close(fig) # close the figure


    rmse = rmsdiff(distorted[Transforms.IMAGE], caffe_res)

    logging.info("Saving Python result for " + str(img_index))

    fig, (ax0, ax1)  =plt.subplots(1, 2)
    visualize_result(distorted, "{} RMSE:{:.2} out of 255".format(PYTHON_RES, rmse), ax1,  cols=cols)
    visualize_result(horizontal_flipper(sample), "Original Image", ax0, cols=cols)

    fig.savefig(op.join(COMPARISON_FOLDER,"Im_"  + str(img_index) + "_" + PYTHON_RES + ".png")) # save the figure to file
    plt.close(fig) # close the figure
    
    fig, ax = plt.subplots(1,1)
    diff = black_or_b(distorted[IMAGE], caffe_res)
    cs = ax.imshow(diff, cmap="gray")
    fig.colorbar(cs)
    ax.set_title("{}: RMSE {:.2}".format(PYTHON_RES, rmse))
    fig.savefig(op.join(COMPARISON_FOLDER,"Im_"  + str(img_index) + "_" +  DIFF_RES + ".png")) # save the figure to file
    plt.close(fig) # close the figure

    logging.info("Comparison of Bounding boxes for " + PYTHON_RES)
    print_comparison_reslts(bboxes_orig[:,:4], distorted[Transforms.LABEL], dists, inds, ious)
    stats = calc_stats(dists, ious)
    stats[RMSE + "sk"] = round(rmse, 2)
    stats["img_index"] = img_index 
    stats["scale_x"] = round(float(spatial_params["new_width"])/float(spatial_params["original_width"]),1) 
    stats["scale_y"] = round(float(spatial_params["new_height"])/float(spatial_params["original_height"]),1)


    #with antialising
    resized = resized_antialising
    on_canvas = canvas_adapter(resized)
    flipped = horizontal_flipper(on_canvas)
    distorted = random_distorter(flipped)
    rmse = round(rmsdiff(distorted[Transforms.IMAGE], caffe_res),2)

    logging.info("Saving Python result for " + str(img_index))

    fig, (ax0, ax1) =plt.subplots(1,1)
    visualize_result(distorted, "{} RMSE:{:.2} out of 255".format(PYTHON_RES2, rmse), ax1,cols=cols)
    visualize_result(horizontal_flipper(sample), "Original Image", ax0, cols=cols)
    fig.savefig(op.join(COMPARISON_FOLDER, "Im_"  + str(img_index) + "_" + PYTHON_RES2 + ".png")) # save the figure to file
    plt.close(fig) # close the figure
    
    fig, ax = plt.subplots(1,1)
    diff = black_or_b(distorted[IMAGE], caffe_res)
    cs = ax.imshow(diff, cmap="gray")
    fig.colorbar(cs)
    ax.set_title("{}: RMSE {:.2}".format(PYTHON_RES2,rmse))
    fig.savefig(op.join(COMPARISON_FOLDER, "Im_"  + str(img_index) + "_" + DIFF_RES2 + ".png")) # save the figure to file
    plt.close(fig) # close the figure
    stats[RMSE + "tv"] = round(rmse,2)

    return const_params, spatial_params, distort_params, stats


if __name__ == '__main__':
    main(0)

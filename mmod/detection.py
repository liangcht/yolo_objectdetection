import cv2
import json
try:
    import caffe
except ImportError:
    caffe = None
import numpy as np
from mmod.im_utils import im_rescale


def im_detect(caffe_net, im, pixel_mean=None, target_size=416, maintain_ratio=True, demean_after_pad=True):
    if pixel_mean is None:
        pixel_mean = [104.0, 117.0, 123.0]

    if maintain_ratio:
        h, w = im.shape[0:2]
        alpha = target_size / np.sqrt(h * w)
        height2 = int(np.round(alpha * h))
        width2 = int(np.round(alpha * w))
        if h > w:
            network_input_height = (height2 + 31) / 32 * 32
            network_input_width = ((network_input_height * w + h - 1) / h
                                   + 31) / 32 * 32
        else:
            network_input_width = (width2 + 31) / 32 * 32
            network_input_height = ((network_input_width * h + w - 1) / w +
                                    31) / 32 * 32
        target_size = max(network_input_width,
                          network_input_height)
    else:
        network_input_width = target_size
        network_input_height = target_size

    # first scale and then subtract the mean
    im_resized = im_rescale(im, target_size)
    if not demean_after_pad:
        im_resized = im_resized.astype(np.float32, copy=True)
        im_resized -= pixel_mean

    new_h, new_w = im_resized.shape[0:2]
    left = (network_input_width - new_w) / 2
    right = network_input_width - new_w - left
    top = (network_input_height - new_h) / 2
    bottom = network_input_height - new_h - top
    im_squared = cv2.copyMakeBorder(im_resized, top=top, bottom=bottom, left=left, right=right,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    blob = im_squared.astype(np.float32, copy=True)
    if demean_after_pad:
        blob -= np.float32(pixel_mean)
    # change blob dim order from h.w.c to c.h.w
    channel_swap = (2, 0, 1)
    blob = blob.transpose(channel_swap)
    if pixel_mean[0] == 0:
        blob /= 255.

    caffe_net.blobs['data'].reshape(1, *blob.shape)
    caffe_net.blobs['data'].data[...] = blob.reshape(1, *blob.shape)
    caffe_net.blobs['im_info'].reshape(1, 2)
    caffe_net.blobs['im_info'].data[...] = (im.shape[0:2],)

    caffe_net.forward()

    bbox = caffe_net.blobs['bbox'].data[0]
    prob = caffe_net.blobs['prob'].data[0]

    prob = prob.reshape(-1, prob.shape[-1])
    assert bbox.shape[-1] == 4
    bbox = bbox.reshape(-1, 4)

    return prob, bbox


def result2bblist(im, probs, boxes, class_map, thresh=None, obj_thresh=None, class_thresh=None):
    if thresh is None:
        thresh = 0
    if obj_thresh is None:
        obj_thresh = 0
    class_num = probs.shape[1] - 1  # the last one is obj_score * max_prob

    det_results = []
    for i, box in enumerate(boxes):
        if probs[i, -1] <= obj_thresh:
            continue
        if probs[i, 0:-1].max() <= thresh:
            continue
        for j in range(class_num):
            if probs[i, j] <= thresh:
                continue
            label = class_map[j]
            if class_thresh and probs[i, j] < class_thresh[label]:
                continue

            x, y, w, h = box

            im_h, im_w = im.shape[0:2]
            left = (x - w / 2.)
            right = (x + w / 2.)
            top = (y - h / 2.)
            bot = (y + h / 2.)

            left = int(max(left, 0))
            right = int(min(right, im_w - 1))
            top = int(max(top, 0))
            bot = int(min(bot, im_h - 1))

            crect = dict()
            crect['rect'] = map(float, [left, top, right, bot])
            assert j < len(class_map), "Invalid labelmap or network: predicted class index: {} >= count: {}".format(
                j, len(class_map)
            )
            crect['class'] = label
            crect['conf'] = max(round(probs[i, j], 4), 0.00001)
            crect['obj'] = max(round(probs[i, -1], 4), 0.00001)
            det_results.append(crect)

    return det_results


def result2json(*args, **kwargs):
    return json.dumps(result2bblist(*args, **kwargs), separators=(',', ':'))

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Sergey Karayev
# --------------------------------------------------------

cimport cython
import numpy as np
cimport numpy as np

#np.float32_t = np.float32_t
#ctypedef np.float32_t np.float32_t

cdef extern from "math.h":
    double abs(double m)
    double log(double x)


def bbox_overlaps(np.ndarray[np.float32_t, ndim=2] boxes,
        np.ndarray[np.float32_t, ndim=2] query_boxes):
    return bbox_overlaps_c(boxes, query_boxes)

cdef np.ndarray[np.float32_t, ndim=2] bbox_overlaps_c(
        np.ndarray[np.float32_t, ndim=2] boxes,
        np.ndarray[np.float32_t, ndim=2] query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] overlaps = np.zeros((N, K), dtype=np.float32)
    cdef np.float32_t iw, ih, box_area
    cdef np.float32_t ua
    cdef unsigned int k, n
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0]) *
            (query_boxes[k, 3] - query_boxes[k, 1])
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0])
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1])
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0]) *
                        (boxes[n, 3] - boxes[n, 1]) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bbox_intersections(
        np.ndarray[np.float32_t, ndim=2] boxes,
        np.ndarray[np.float32_t, ndim=2] query_boxes):
    return bbox_intersections_c(boxes, query_boxes)


cdef np.ndarray[np.float32_t, ndim=2] bbox_intersections_c(
        np.ndarray[np.float32_t, ndim=2] boxes,
        np.ndarray[np.float32_t, ndim=2] query_boxes):
    """
    For each query box compute the intersection ratio covered by boxes
    ----------
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of intersec between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] intersec = np.zeros((N, K), dtype=np.float32)
    cdef np.float32_t iw, ih, box_area
    cdef np.float32_t ua
    cdef unsigned int k, n
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0]) *
            (query_boxes[k, 3] - query_boxes[k, 1])
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0])
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1])
                )
                if ih > 0:
                    intersec[n, k] = iw * ih / box_area
    return intersec

def bbox_ious_diag(
        np.ndarray[np.float32_t, ndim=2] boxes,
        np.ndarray[np.float32_t, ndim=2] query_boxes):
    return bbox_ious_diag_c(boxes, query_boxes)

cdef np.ndarray[np.float32_t, ndim=2] bbox_ious_diag_c(
        np.ndarray[np.float32_t, ndim=2] boxes,
        np.ndarray[np.float32_t, ndim=2] query_boxes):
    """
    For each query box compute the IOU covered by corresponding boxes
    ----------
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (N, 4) ndarray of float
    Returns
    -------
    overlaps: (N) ndarray of intersec between boxes and query_boxes
    """
    assert boxes.shape[0] == query_boxes.shape[0]
    cdef unsigned int N = boxes.shape[0]
    cdef np.ndarray[np.float32_t, ndim=1] intersec = np.zeros((N, ), dtype=np.float32)
    cdef np.float32_t iw, ih, qbox_area, box_area, inter_area
    cdef unsigned int k, n

    for n in range(N):
        qbox_area = (
            (query_boxes[n, 2] - query_boxes[n, 0]) *
            (query_boxes[n, 3] - query_boxes[n, 1])
        )
        iw = (
            min(boxes[n, 2], query_boxes[n, 2]) -
            max(boxes[n, 0], query_boxes[n, 0])
        )
        if iw > 0:
            ih = (
                min(boxes[n, 3], query_boxes[n, 3]) -
                max(boxes[n, 1], query_boxes[n, 1])
            )
            if ih > 0:
                box_area = (
                    (boxes[n, 2] - boxes[n, 0]) *
                    (boxes[n, 3] - boxes[n, 1])
                )
                inter_area = iw * ih
                intersec[n] = inter_area / (qbox_area + box_area - inter_area)
    return intersec

def bbox_ious(
        np.ndarray[np.float32_t, ndim=2] boxes,
        np.ndarray[np.float32_t, ndim=2] query_boxes):
    return bbox_ious_c(boxes, query_boxes)


cdef np.ndarray[np.float32_t, ndim=2] bbox_ious_c(
        np.ndarray[np.float32_t, ndim=2] boxes,
        np.ndarray[np.float32_t, ndim=2] query_boxes):
    """
    For each query box compute the IOU covered by boxes
    ----------
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of intersec between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] intersec = np.zeros((N, K), dtype=np.float32)
    cdef np.float32_t iw, ih, qbox_area, box_area, inter_area
    cdef unsigned int k, n
    for k in range(K):
        qbox_area = (
            (query_boxes[k, 2] - query_boxes[k, 0]) *
            (query_boxes[k, 3] - query_boxes[k, 1])
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0])
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1])
                )
                if ih > 0:
                    box_area = (
                        (boxes[n, 2] - boxes[n, 0]) *
                        (boxes[n, 3] - boxes[n, 1])
                    )
                    inter_area = iw * ih
                    intersec[n, k] = inter_area / (qbox_area + box_area - inter_area)
    return intersec


def anchor_intersections(
        np.ndarray[np.float32_t, ndim=2] anchors,
        np.ndarray[np.float32_t, ndim=2] query_boxes):
    return anchor_intersections_c(anchors, query_boxes)


cdef np.ndarray[np.float32_t, ndim=2] anchor_intersections_c(
        np.ndarray[np.float32_t, ndim=2] anchors,
        np.ndarray[np.float32_t, ndim=2] query_boxes):
    """
    For each query box compute the intersection ratio covered by anchors
    ----------
    Parameters
    ----------
    boxes: (N, 2) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of intersec between boxes and query_boxes
    """
    cdef unsigned int N = anchors.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] intersec = np.zeros((N, K), dtype=np.float32)
    cdef np.float32_t iw, ih, anchor_area, inter_area
    cdef np.float32_t boxw, boxh
    cdef unsigned int k, n
    for n in range(N):
        anchor_area = anchors[n, 0] * anchors[n, 1]
        for k in range(K):
            boxw = (query_boxes[k, 2] - query_boxes[k, 0])
            boxh = (query_boxes[k, 3] - query_boxes[k, 1])
            iw = min(anchors[n, 0], boxw)
            ih = min(anchors[n, 1], boxh)
            inter_area = iw * ih
            intersec[n, k] = inter_area / (anchor_area + boxw * boxh - inter_area)

    return intersec


def bbox_intersections_self(
        np.ndarray[np.float32_t, ndim=2] boxes):
    return bbox_intersections_self_c(boxes)


cdef np.ndarray[np.float32_t, ndim=2] bbox_intersections_self_c(
        np.ndarray[np.float32_t, ndim=2] boxes):
    """
    For each query box compute the intersection ratio covered by boxes
    ----------
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    Returns
    -------
    overlaps: (N, N) ndarray of intersec between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] intersec = np.zeros((N, N), dtype=np.float32)
    cdef np.float32_t iw, ih, box_area
    cdef unsigned int k, n

    for k in range(N):
        box_area = (
            (boxes[k, 2] - boxes[k, 0] + 1) *
            (boxes[k, 3] - boxes[k, 1] + 1)
        )
        for n in range(k+1, N):
            iw = (
                min(boxes[n, 2], boxes[k, 2]) -
                max(boxes[n, 0], boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], boxes[k, 3]) -
                    max(boxes[n, 1], boxes[k, 1]) + 1
                )
                if ih > 0:
                    intersec[k, n] = iw * ih / box_area
    return intersec


def bbox_similarities(
        np.ndarray[np.float32_t, ndim=2] boxes,
        np.ndarray[np.float32_t, ndim=2] query_boxes):
    return bbox_similarities_c(boxes, query_boxes)

cdef np.ndarray[np.float32_t, ndim=2] bbox_similarities_c(
        np.ndarray[np.float32_t, ndim=2] boxes,
        np.ndarray[np.float32_t, ndim=2] query_boxes):
    """
    For each query box compute the intersection ratio covered by boxes
    ----------
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float (dets)
    Returns
    -------
    overlaps: (N, K) ndarray of similarity scores between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] sims = np.zeros((N, K), dtype=np.float32)
    cdef np.float32_t cx1, cy1, w1, h1
    cdef np.float32_t cx2, cy2, w2, h2

    cdef np.float32_t loc_dist, shape_dist

    cdef unsigned int k, n
    for n in range(N):
        cx1 = (boxes[n, 0] + boxes[n, 2]) * 0.5
        cy1 = (boxes[n, 1] + boxes[n, 3]) * 0.5
        w1 = boxes[n, 2] - boxes[n, 0] + 1
        h1 = boxes[n, 3] - boxes[n, 1] + 1

        for k in range(K):
            cx2 = (query_boxes[k, 0] + query_boxes[k, 2]) * 0.5
            cy2 = (query_boxes[k, 1] + query_boxes[k, 3]) * 0.5
            w2 = query_boxes[k, 2] - query_boxes[k, 0] + 1
            h2 = query_boxes[k, 3] - query_boxes[k, 1] + 1

            loc_dist = abs(cx1 - cx2) / (w1 + w2) + abs(cy1 - cy2) / (h1 + h2)
            shape_dist = abs(w2 * h2 / (w1 * h1) - 1.0)

            sims[n, k] = -log(loc_dist + 0.001) - shape_dist * shape_dist + 1

    return sims

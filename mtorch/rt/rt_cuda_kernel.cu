#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include "caffe_cuda.h"
#include "region_common.hpp"

namespace {

template <typename scalar_t>
__global__ void ExtractBoundingBox(int total, int num_anchor, int height, int width, 
        scalar_t* bbs_data, const scalar_t* blob_xy_data, const scalar_t* blob_wh_data,
        const scalar_t* biases) {
  CUDA_KERNEL_LOOP(index, total) {
      int b = index / (num_anchor * height * width);
      int left = index % (num_anchor * height * width);
      int n = left / (height * width);
      left = left % (height * width);
      int j = left / width;
      int i = left % width;
      scalar_t* curr_bbs_data = bbs_data + index * 4;
      int offset_double_bnji = b * (2 * num_anchor) * height * width + n * height * width + j * width + i;
      int offset_double_bnji_next = offset_double_bnji + num_anchor * height * width;
      *(curr_bbs_data + 0) = (*(blob_xy_data + offset_double_bnji) + i) / width;
      *(curr_bbs_data + 1) = (*(blob_xy_data + offset_double_bnji_next) + j) / height;
      double w = *(blob_wh_data + offset_double_bnji);
      double h = *(blob_wh_data + offset_double_bnji_next);
      *(curr_bbs_data + 2) = exp(w) * biases[2 * n] / width;
      *(curr_bbs_data + 3) = exp(h) * biases[2 * n + 1] / height;
  }
}

template <typename scalar_t>
__global__ void CalculateIOU(int total, scalar_t* iou_data, const scalar_t* bbs_data, const scalar_t* truth_data, int num_anchor, int height, int width, int max_gt,
                             scalar_t positive_thresh, const scalar_t* blob_obj_data, scalar_t* target_obj_noobj_data) {
  CUDA_KERNEL_LOOP(index, total) {
      int b = index / (num_anchor * height * width * max_gt);
      int left = index % (num_anchor * height * width * max_gt);
      int n = left / (height * width * max_gt);
      left = left % (height * width * max_gt);
      int j = left / (width * max_gt);
      left = left % (width * max_gt);
      int i = left / max_gt;
      int t = left % max_gt;
      scalar_t tx = *(truth_data + b * 5 * max_gt + t * 5 + 0);
      scalar_t ty = *(truth_data + b * 5 * max_gt + t * 5 + 1);
      scalar_t tw = *(truth_data + b * 5 * max_gt + t * 5 + 2);
      scalar_t th = *(truth_data + b * 5 * max_gt + t * 5 + 3);
      scalar_t curr_iou = 0;
      if (tx) {
          int curr_index = (b * num_anchor * height * width + n * height * width + j * width + i) * 4;
          scalar_t px = *(bbs_data + curr_index + 0);
          scalar_t py = *(bbs_data + curr_index + 1);
          scalar_t pw = *(bbs_data + curr_index + 2);
          scalar_t ph = *(bbs_data + curr_index + 3);
          curr_iou = TBoxIou(px, py, pw, ph,
                  tx, ty, tw, th);
          // if the iou is large enough, let's not penalize the objectiveness
          if (curr_iou > positive_thresh) {
              // multiple threads might write this address at the same time, but
              // at least one will succeeds. It is safe to do this.
              *(target_obj_noobj_data + index / max_gt) =
                  *(blob_obj_data + index / max_gt);
          }
      }
      *(iou_data + index) = curr_iou;
  }
}

template <typename scalar_t>
__global__ void GroundTruthTarget(int total, int max_gt,
        const scalar_t* truth_data, int num_anchor, int height, int width,
        const scalar_t* biases, int* gt_target_data) {
    CUDA_KERNEL_LOOP(index, total) {
        int b = index / max_gt;
        int t = index % max_gt;
        scalar_t tx = *(truth_data + b * max_gt * 5 + 5 * t + 0);
        scalar_t ty = *(truth_data + b * max_gt * 5 + 5 * t + 1);

        int target_i = -1;
        int target_j = -1;
        int target_n = -1;
        if (tx > 0 && ty > 0 && tx < 1 && ty < 1) {
            target_i = tx * width;
            target_j = ty * height;
            scalar_t tw = *(truth_data + b * max_gt * 5 + 5 * t + 2);
            scalar_t th = *(truth_data + b * max_gt * 5 + 5 * t + 3);

            scalar_t max_iou = -1;

            target_n = -1;
            for (int n = 0; n < num_anchor; n++) {
                scalar_t curr_iou = TBoxIou<scalar_t>(0, 0, tw, th, 0, 0, biases[2 * n] / width, biases[2 * n + 1] / height);
                if (curr_iou > max_iou) {
                    max_iou = curr_iou;
                    target_n = n;
                }
            }
        }

        *(gt_target_data + b * max_gt * 3 + t * 3 + 0) = target_i;
        *(gt_target_data + b * max_gt * 3 + t * 3 + 1) = target_j;
        *(gt_target_data + b * max_gt * 3 + t * 3 + 2) = target_n;
    }
}

template <typename scalar_t>
__global__ void RemoveDuplicateTarget(int total, int* gt_target_data, int max_gt) {
    CUDA_KERNEL_LOOP(index, total) {
        int b = index / (max_gt * max_gt);
        int left_index = index % (max_gt * max_gt);
        int left_t = left_index / max_gt;
        int right_t = left_index % max_gt;
        if (left_t == right_t) {
            continue;
        }

        int left_target_i = *(gt_target_data + b * max_gt * 3 + left_t * 3 + 0);
        int left_target_j = *(gt_target_data + b * max_gt * 3 + left_t * 3 + 1);
        int left_target_n = *(gt_target_data + b * max_gt * 3 + left_t * 3 + 2);
        if (left_target_i < 0) {
            continue;
        }

        int right_target_i = *(gt_target_data + b * max_gt * 3 + right_t * 3 + 0);
        int right_target_j = *(gt_target_data + b * max_gt * 3 + right_t * 3 + 1);
        int right_target_n = *(gt_target_data + b * max_gt * 3 + right_t * 3 + 2);
        if (right_target_i < 0) {
            continue;
        }
        if (left_target_i == right_target_i &&
                left_target_j == right_target_j &&
                left_target_n == right_target_n) {
            if (left_t < right_t) {
                *(gt_target_data + b * max_gt * 3 + left_t * 3 + 0) = -1;
                *(gt_target_data + b * max_gt * 3 + left_t * 3 + 1) = -1;
                *(gt_target_data + b * max_gt * 3 + left_t * 3 + 2) = -1;
            } else {
                *(gt_target_data + b * max_gt * 3 + right_t * 3 + 0) = -1;
                *(gt_target_data + b * max_gt * 3 + right_t * 3 + 1) = -1;
                *(gt_target_data + b * max_gt * 3 + right_t * 3 + 2) = -1;
            }
        }
    }
}

template <typename scalar_t>
__global__ void AlignGroudTruth(int total, const int* gt_target_data, int max_gt,
        const scalar_t* truth_data, scalar_t* target_xy_data, scalar_t* target_wh_data,
        scalar_t* target_xywh_weight_data, scalar_t coord_scale,
        int num_anchor, int height, int width, bool rescore, scalar_t* target_obj_obj_data,
        const scalar_t* iou_data, scalar_t* target_obj_noobj_data, scalar_t* target_class_data,
        const scalar_t* biases, const scalar_t* blob_obj_data) {
    CUDA_KERNEL_LOOP(index, total) {
        int b = index / max_gt;
        int t = index % max_gt;

        int target_i = *(gt_target_data + b * max_gt * 3 + t * 3 + 0);
        int target_j = *(gt_target_data + b * max_gt * 3 + t * 3 + 1);
        int target_n = *(gt_target_data + b * max_gt * 3 + t * 3 + 2);

        if (target_i < 0) {
            continue;
        }

        int offset_bt = b * max_gt * 5 + 5 * t;
        scalar_t tx = *(truth_data + offset_bt + 0);
        scalar_t ty = *(truth_data + offset_bt + 1);
        scalar_t tw = *(truth_data + offset_bt + 2);
        scalar_t th = *(truth_data + offset_bt + 3);

        if (tw <= 0 || th <= 0) {
            // we explicitly ignore this zero-length bounding boxes
            // note: this layer is not designed to support image-level labels
            continue;
        }

        int offset_bnji = b * num_anchor * height * width + target_n * height * width +
            target_j * width + target_i;

        int offset_double_bnji = offset_bnji + b * num_anchor * height * width;
        int offset_double_bnji_next = offset_double_bnji + num_anchor * width * height;

        *(target_xy_data + offset_double_bnji) = tx * width - target_i;
        *(target_xy_data + offset_double_bnji_next) = ty * height - target_j;
        *(target_wh_data + offset_double_bnji) = log(tw * width / biases[2 * target_n]);
        *(target_wh_data + offset_double_bnji_next) = log(th * height / biases[2 * target_n + 1]);
        *(target_xywh_weight_data + offset_double_bnji) = coord_scale * (2 - tw * th);
        *(target_xywh_weight_data + offset_double_bnji_next) = coord_scale * (2 - tw * th);

        if (!rescore) {
            *(target_obj_obj_data + offset_bnji) = 1;
        } else {
            *(target_obj_obj_data + offset_bnji) =  *(iou_data + offset_bnji * max_gt + t);
        }
        *(target_obj_noobj_data + offset_bnji) = *(blob_obj_data + offset_bnji);

        int cls = *(truth_data + offset_bt + 4);
        *(target_class_data + offset_bnji) = cls;
    }
}


} // namespace

std::vector<at::Tensor> rt_cuda_forward(
    at::Tensor xy, at::Tensor wh, at::Tensor obj, at::Tensor truth,
    at::Tensor biases,
    float coord_scale, float positive_thresh,
    bool warmup, bool rescore,
    int batches, int num_anchor, int height, int width, int max_gt
) {

  // if it is at the very begiining, let's align the output
  // by default, we set the target of xywh as itself, that mean 0 penalty
  auto target_xy = warmup ? at::full_like(xy, 0.5) : xy.clone();
  auto target_wh = warmup ? at::zeros_like(wh) : wh.clone();
  auto target_xywh_weight = warmup ? at::full_like(xy, 0.01) : at::zeros_like(xy);

  // for no-objectiveness, by default all of them be 0. we will zero-out the
  // position if it is 1) gt or 2) the predicted result is good enought
  auto target_obj_noobj = at::zeros_like(obj);
  // For this one, we will only pernalize the position which should be
  // responsible for the gt
  auto target_obj_obj = obj.clone();
  // by default, dont penalize the results
  auto target_class = at::full_like(obj, -1);

  // intermediate variables
  auto ious = at::zeros({batches, num_anchor, height, width, max_gt}, xy.type());
  auto bbs  = at::empty({batches, num_anchor, height, width, 4}, xy.type());
  auto gt_target = at::empty({batches, max_gt, 3, 1}, at::CUDA(at::kInt));

  int total = batches * num_anchor * height * width;
  AT_DISPATCH_FLOATING_TYPES(xy.type(), "ExtractBoundingBox", ([&] {
    ExtractBoundingBox<scalar_t><<<GET_BLOCKS(total), CUDA_NUM_THREADS>>>(
        total, num_anchor, height, width,
        bbs.data<scalar_t>(), xy.data<scalar_t>(), wh.data<scalar_t>(), biases.data<scalar_t>());
  }));

  total = batches * num_anchor * height * width * max_gt;
  AT_DISPATCH_FLOATING_TYPES(xy.type(), "CalculateIOU", ([&] {
    CalculateIOU<scalar_t><<<GET_BLOCKS(total), CUDA_NUM_THREADS>>>(
        total, ious.data<scalar_t>(),
        bbs.data<scalar_t>(), truth.data<scalar_t>(), num_anchor, height, width, max_gt,
        positive_thresh, obj.data<scalar_t>(), target_obj_noobj.data<scalar_t>());
  }));

  total = batches * max_gt;
  AT_DISPATCH_FLOATING_TYPES(xy.type(), "GroundTruthTarget", ([&] {
    GroundTruthTarget<scalar_t><<<GET_BLOCKS(total), CUDA_NUM_THREADS>>>(total, max_gt,
        truth.data<scalar_t>(), num_anchor, height, width,
        biases.data<scalar_t>(), gt_target.data<int>());
  }));

  total = max_gt * max_gt * batches;
  AT_DISPATCH_FLOATING_TYPES(xy.type(), "RemoveDuplicateTarget", ([&] {
    RemoveDuplicateTarget<scalar_t><<<GET_BLOCKS(total), CUDA_NUM_THREADS>>>(total, gt_target.data<int>(), max_gt);
  }));

  total = batches * max_gt;
  AT_DISPATCH_FLOATING_TYPES(xy.type(), "AlignGroudTruth", ([&] {
    AlignGroudTruth<scalar_t><<<GET_BLOCKS(total), CUDA_NUM_THREADS>>>(total, gt_target.data<int>(), max_gt,
        truth.data<scalar_t>(), target_xy.data<scalar_t>(), target_wh.data<scalar_t>(),
        target_xywh_weight.data<scalar_t>(), coord_scale,
        num_anchor, height, width, rescore, target_obj_obj.data<scalar_t>(),
        ious.data<scalar_t>(), target_obj_noobj.data<scalar_t>(), target_class.data<scalar_t>(), biases.data<scalar_t>(), obj.data<scalar_t>());
  }));

  return {target_xy, target_wh, target_xywh_weight, target_obj_obj, target_obj_noobj, target_class};
}

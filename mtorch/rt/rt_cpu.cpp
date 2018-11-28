#include <torch/torch.h>

#include <vector>
#include "mtorch_common.h"
#include "region_common.hpp"

// C++ interface
std::vector<at::Tensor> rt_forward(
    at::Tensor xy, at::Tensor wh, at::Tensor obj, at::Tensor truth,
    at::Tensor biases,
    float coord_scale, float positive_thresh,
    bool warmup, bool rescore) {
  CHECK_INPUT_CPU(xy);
  CHECK_INPUT_CPU(wh);
  CHECK_INPUT_CPU(obj);
  CHECK_INPUT_CPU(truth);
  CHECK_INPUT_CPU(biases);

  int num = xy.size(0);
  int num_anchor = xy.size(1) / 2;
  int height = xy.size(2);
  int width = xy.size(3);

  AT_ASSERTM(num_anchor > 0, "number of anchors must be positive");
  AT_ASSERTM(biases.numel() == 2 * num_anchor, "invalid number of biases");

  AT_ASSERTM(xy.dim() == 4, "invalid xy dim");
  AT_ASSERTM(wh.dim() == 4, "invalid wh dim");
  AT_ASSERTM(obj.dim() == 4, "invalid obj dim");

  AT_ASSERTM(wh.size(0) == num, "invalid wh.size(0)");
  AT_ASSERTM(wh.size(1) == 2 * num_anchor, "invalid wh.size(1)");
  AT_ASSERTM(wh.size(2) == height, "invalid wh.size(2)");
  AT_ASSERTM(wh.size(3) == width, "invalid wh.size(3)");
  AT_ASSERTM(obj.size(0) == num, "invalid obj.size(0)");
  AT_ASSERTM(obj.size(1) == num_anchor, "invalid obj.size(1)");
  AT_ASSERTM(obj.size(2) == height, "invalid obj.size(2)");
  AT_ASSERTM(obj.size(3) == width, "invalid obj.size(3)");

  int num_gt = truth.size(1) / 5;
  AT_ASSERTM(truth.size(0) == num, "invalid truth.size(0)");
  AT_ASSERTM(truth.size(1) == 5 * num_gt, "invalid truth.size(1)");
  AT_ASSERTM(truth.numel() == num * 5 * num_gt, "invalid truth size");

  int batches = num;
  int max_gt = num_gt;

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
  auto gt_target = at::empty({batches, max_gt, 3, 1}, at::kInt);

  AT_DISPATCH_FLOATING_TYPES(xy.type(), "ExtractBoundingBox", ([&] {
    auto bbs_data = bbs.data<scalar_t>();
    const auto blob_xy_data = xy.data<scalar_t>();
    const auto blob_wh_data = wh.data<scalar_t>();
    const auto biases_data = biases.data<scalar_t>();

    // Calculate the bbs
    for (int b = 0; b < batches; b++) {
      for (int n = 0; n < num_anchor; n++) {
        for (int j = 0; j < height; j++) {
          for (int i = 0; i < width; i++) {
            int offset_double_bnji = b * (2 * num_anchor) * height * width + n * height * width + j * width + i;
            int offset_double_bnji_next = offset_double_bnji + num_anchor * height * width;
            int index = b * num_anchor * height * width + n * height * width + j * width + i;
            scalar_t* curr_bbs_data = bbs_data + index * 4;
            *(curr_bbs_data + 0) = (*(blob_xy_data + offset_double_bnji) + i) / width;
            *(curr_bbs_data + 1) = (*(blob_xy_data + offset_double_bnji_next) + j) / height;
            double w = *(blob_wh_data + offset_double_bnji);
            double h = *(blob_wh_data + offset_double_bnji_next);
            *(curr_bbs_data + 2) = exp(w) * biases_data[2 * n] / width;
            *(curr_bbs_data + 3) = exp(h) * biases_data[2 * n + 1] / height;
          }
        }
      }
    }
  }));

  AT_DISPATCH_FLOATING_TYPES(xy.type(), "CalculateIOU", ([&] {
    const auto bbs_data = bbs.data<scalar_t>();
    const auto truth_data = truth.data<scalar_t>();
    const auto blob_obj_data = obj.data<scalar_t>();
    auto target_obj_noobj_data = target_obj_noobj.data<scalar_t>();
    auto iou_data = ious.data<scalar_t>();
    // calculate the IOU
    for (int b = 0; b < batches; b++) {
      for (int n = 0; n < num_anchor; n++) {
        for (int j = 0; j < height; j++) {
          for (int i = 0; i < width; i++) {
            int index = b * num_anchor * height * width + n * height * width + j * width + i;
            int curr_index = index * 4;
            scalar_t px = *(bbs_data + curr_index + 0);
            scalar_t py = *(bbs_data + curr_index + 1);
            scalar_t pw = *(bbs_data + curr_index + 2);
            scalar_t ph = *(bbs_data + curr_index + 3);
            for (int t = 0; t < max_gt; ++t) {
              scalar_t tx = *(truth_data + b * 5 * max_gt + t * 5 + 0);
              scalar_t ty = *(truth_data + b * 5 * max_gt + t * 5 + 1);
              scalar_t tw = *(truth_data + b * 5 * max_gt + t * 5 + 2);
              scalar_t th = *(truth_data + b * 5 * max_gt + t * 5 + 3);
              scalar_t curr_iou = 0;
              if (tx) {
                curr_iou = TBoxIou<scalar_t>(px, py, pw, ph,
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
        }
      }
    }
  }));

  AT_DISPATCH_FLOATING_TYPES(xy.type(), "GroundTruthTarget", ([&] {
    const auto truth_data = truth.data<scalar_t>();
    auto gt_target_data = gt_target.data<int>();
    const auto biases_data = biases.data<scalar_t>();
    for (int b = 0; b < batches; b++) {
      for (int t = 0; t < max_gt; ++t) {
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
                auto curr_iou = TBoxIou<scalar_t>(0, 0, tw, th, 0, 0, biases_data[2 * n] / width, biases_data[2 * n + 1] / height);
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
  }));

  // RemoveDuplicateTarget
  {
    auto gt_target_data = gt_target.data<int>();
    for (int b = 0; b < batches; b++) {
      for (int left_t = 0; left_t < max_gt; ++left_t) {
        for (int right_t = 0; right_t < max_gt; ++right_t) {
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
    }
  }

  AT_DISPATCH_FLOATING_TYPES(xy.type(), "AlignGroudTruth", ([&] {
    const auto gt_target_data = gt_target.data<int>();
    const auto truth_data = truth.data<scalar_t>();
    auto target_xy_data = target_xy.data<scalar_t>();
    auto target_wh_data = target_wh.data<scalar_t>();
    auto target_xywh_weight_data = target_xywh_weight.data<scalar_t>();
    auto target_class_data = target_class.data<scalar_t>();
    auto target_obj_noobj_data = target_obj_noobj.data<scalar_t>();
    auto target_obj_obj_data = target_obj_obj.data<scalar_t>();
    const auto biases_data = biases.data<scalar_t>();
    const auto blob_obj_data = obj.data<scalar_t>();
    const auto iou_data = ious.data<scalar_t>();
    for (int b = 0; b < batches; b++) {
      for (int t = 0; t < max_gt; ++t) {
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

        if (tw <= 0.00001 || th <= 0.00001) {
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
        *(target_wh_data + offset_double_bnji) = log(tw * width / biases_data[2 * target_n]);
        *(target_wh_data + offset_double_bnji_next) = log(th * height / biases_data[2 * target_n + 1]);
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
  }));

  return {target_xy, target_wh, target_xywh_weight, target_obj_obj, target_obj_noobj, target_class};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rt_forward, "RegionTarget forward (CPU)");
}

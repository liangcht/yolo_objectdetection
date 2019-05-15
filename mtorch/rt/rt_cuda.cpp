#include <torch/extension.h>

#include <vector>
#include "mtorch_common.h"

// CUDA forward declarations

std::vector<at::Tensor> rt_cuda_forward(
    at::Tensor xy, at::Tensor wh, at::Tensor obj, at::Tensor truth,
    at::Tensor biases,
    float coord_scale, float positive_thresh,
    bool warmup, bool rescore,
    int batches, int num_anchor, int height, int width, int max_gt
);

// C++ interface
std::vector<at::Tensor> rt_forward(
    at::Tensor xy, at::Tensor wh, at::Tensor obj, at::Tensor truth,
    at::Tensor biases,
    float coord_scale, float positive_thresh,
    bool warmup, bool rescore) {
  CHECK_INPUT(xy);
  CHECK_INPUT(wh);
  CHECK_INPUT(obj);
  CHECK_INPUT(truth);
  CHECK_INPUT(biases);

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

  return rt_cuda_forward(
      xy, wh, obj, truth,
      biases,
      coord_scale, positive_thresh,
      warmup, rescore,
      num, num_anchor, height, width, num_gt
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rt_forward, "RegionTarget forward (CUDA)");
}

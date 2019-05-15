#include <torch/extension.h>

#include <vector>
#include "mtorch_common.h"

// CUDA forward declarations

std::vector<at::Tensor> smtl_cuda_forward(
    at::Tensor prob, at::Tensor label,
    at::Tensor parent,
    int outer_num, int inner_num, int dim,
    bool has_ignore_label, int ignore_label, bool valid_normalization);

std::vector<at::Tensor> smtl_cuda_backward(
    at::Tensor prob, at::Tensor label,
    at::Tensor parent, at::Tensor group_offset, at::Tensor group_size, at::Tensor group,
    int outer_num, int inner_num, int dim,
    bool has_ignore_label, int ignore_label);

// C++ interface
std::vector<at::Tensor> smtl_forward(
    at::Tensor prob, at::Tensor label,
    at::Tensor parent,
    bool has_ignore_label, int ignore_label,
    int axis, bool valid_normalization) {
  CHECK_INPUT(prob);
  CHECK_INPUT(label);
  CHECK_INPUT(parent);

  int outer_num = 1;
  for (int i = 0; i < axis; ++i)
    outer_num *= prob.size(i);
  int inner_num = 1;
  for (int i = axis + 1; i < prob.dim(); ++i)
    inner_num *= prob.size(i);
  int dim = prob.numel() / outer_num;

  AT_ASSERTM(label.numel() == outer_num * inner_num, "number of labels must match number of predictions")
  AT_ASSERTM(parent.numel() == prob.size(axis), "Channel count must match tree node count")

  return smtl_cuda_forward(
      prob, label,
      parent,
      outer_num, inner_num, dim,
      has_ignore_label, ignore_label, valid_normalization);
}

std::vector<at::Tensor> smtl_backward(
    at::Tensor prob, at::Tensor label,
    at::Tensor parent, at::Tensor group_offset, at::Tensor group_size, at::Tensor group,
    bool has_ignore_label, int ignore_label,
    int axis) {
  CHECK_INPUT(prob);
  CHECK_INPUT(label);
  CHECK_INPUT(parent);
  CHECK_INPUT(group_offset);
  CHECK_INPUT(group_size);
  CHECK_INPUT(group);

  int outer_num = 1;
  for (int i = 0; i < axis; ++i)
    outer_num *= prob.size(i);
  int inner_num = 1;
  for (int i = axis + 1; i < prob.dim(); ++i)
    inner_num *= prob.size(i);
  int dim = prob.numel() / outer_num;

  return smtl_cuda_backward(
      prob, label,
      parent, group_offset, group_size, group,
      outer_num, inner_num, dim,
      has_ignore_label, ignore_label);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &smtl_forward, "SMTL forward (CUDA)");
  m.def("backward", &smtl_backward, "SMTL backward (CUDA)");
}

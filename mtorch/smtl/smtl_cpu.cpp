#include <torch/torch.h>

#include <vector>
#include <cfloat>
#include "mtorch_common.h"

// C++ interface
std::vector<at::Tensor> smtl_forward(
    at::Tensor prob, at::Tensor label,
    at::Tensor parent,
    bool has_ignore_label, int ignore_label,
    int axis, bool valid_normalization) {
  CHECK_INPUT_CPU(prob);
  CHECK_INPUT_CPU(label);
  CHECK_INPUT_CPU(parent);

  int outer_num = 1;
  for (int i = 0; i < axis; ++i)
    outer_num *= prob.size(i);
  int inner_num = 1;
  for (int i = axis + 1; i < prob.dim(); ++i)
    inner_num *= prob.size(i);
  int dim = prob.numel() / outer_num;
  int spatial_dim = inner_num;

  AT_ASSERTM(label.numel() == outer_num * inner_num, "number of labels must match number of predictions")
  AT_ASSERTM(parent.numel() == prob.size(axis), "Channel count must match tree node count")

  auto normalization = at::tensor(outer_num, prob.options());

  // Intermediate variables
  auto loss = at::empty_like(label);
  int nthreads = outer_num * inner_num;
  at::Tensor counts;
  AT_DISPATCH_FLOATING_TYPES(prob.type(), "smtl_forward", ([&] {
    auto loss_data = loss.data<scalar_t>();
    const auto label_data = label.data<scalar_t>();
    const auto prob_data = prob.data<scalar_t>();
    const auto parent_data = parent.data<int>();
    scalar_t* counts_data = nullptr;
    if (valid_normalization) {
      counts = at::empty_like(label);
      counts_data = counts.data<scalar_t>();
    }
    for (int index = 0; index < nthreads; ++index) {
      // index == n * inner_num + s
      const int n = index / inner_num;
      const int s = index % inner_num;

        if (counts_data)
            counts_data[index] = 0;
        loss_data[index] = 0;
        int label_value = static_cast<int>(label_data[index]);
        if (has_ignore_label && label_value == ignore_label)
            continue;

        while (label_value >= 0) {
            loss_data[index] -= log(std::max(prob_data[n * dim + label_value * spatial_dim + s], scalar_t(FLT_MIN)));
            if (counts_data)
                counts_data[index]++;
            label_value = parent_data[label_value];
        }

    }
    if (valid_normalization)
      normalization = counts.sum();
  }));

  return {loss.sum() / normalization, normalization};
}

std::vector<at::Tensor> smtl_backward(
    at::Tensor prob, at::Tensor label,
    at::Tensor parent, at::Tensor group_offset, at::Tensor group_size, at::Tensor group,
    bool has_ignore_label, int ignore_label,
    int axis) {
  CHECK_INPUT_CPU(prob);
  CHECK_INPUT_CPU(label);
  CHECK_INPUT_CPU(parent);
  CHECK_INPUT_CPU(group_offset);
  CHECK_INPUT_CPU(group_size);
  CHECK_INPUT_CPU(group);

  int outer_num = 1;
  for (int i = 0; i < axis; ++i)
    outer_num *= prob.size(i);
  int inner_num = 1;
  for (int i = axis + 1; i < prob.dim(); ++i)
    inner_num *= prob.size(i);
  int dim = prob.numel() / outer_num;
  int spatial_dim = inner_num;

  auto diff = at::zeros_like(prob);
  int nthreads = outer_num * inner_num;

  AT_DISPATCH_FLOATING_TYPES(prob.type(), "smtl_backward", ([&] {
    const auto parent_data = parent.data<int>();
    const auto group_data = group.data<int>();
    const auto group_offset_data = group_offset.data<int>();
    const auto group_size_data = group_size.data<int>();
    const auto label_data = label.data<scalar_t>();
    const auto prob_data = prob.data<scalar_t>();
    auto bottom_diff = diff.data<scalar_t>();
    for (int index = 0; index < nthreads; ++index) {
      // index == n * inner_num + s
      const int n = index / inner_num;
      const int s = index % inner_num;
      int label_value = static_cast<int>(label_data[index]);
      if (has_ignore_label && label_value == ignore_label)
        continue;
      while (label_value >= 0) {
        int g = group_data[label_value];
        int offset = group_offset_data[g];
        // TODO: Use dynamic parallelism for devices with 3.5 compute capability
        for (int c = 0; c < group_size_data[g]; ++c)
          bottom_diff[n * dim + (offset + c) * spatial_dim + s] = prob_data[n * dim + (offset + c) * spatial_dim + s];

        bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
        label_value = parent_data[label_value];
      }
    }
  }));

  return {diff};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &smtl_forward, "SMTL forward (CPU)");
  m.def("backward", &smtl_backward, "SMTL backward (CPU)");
}

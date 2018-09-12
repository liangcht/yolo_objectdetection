#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cfloat>

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)


namespace {

// CUDA: use 1024 threads per block
const int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


template <typename scalar_t>
__global__ void smtl_cuda_forward_kernel(
    const int nthreads,
    const int* __restrict__ parent_data, const scalar_t* __restrict__ prob_data, const scalar_t* __restrict__ label,
    const int dim, const int spatial_dim,
    const bool has_ignore_label, const int ignore_label,
    scalar_t* __restrict__ loss_data, scalar_t* __restrict__ counts) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        // index == n * spatial_dim + s
        const int n = index / spatial_dim;
        const int s = index % spatial_dim;
        if (counts)
            counts[index] = 0;
        loss_data[index] = 0;
        int label_value = static_cast<int>(label[index]);
        if (has_ignore_label && label_value == ignore_label)
            continue;

        while (label_value >= 0) {
            loss_data[index] -= log(max(prob_data[n * dim + label_value * spatial_dim + s], scalar_t(FLT_MIN)));
            if (counts)
                counts[index]++;
            label_value = parent_data[label_value];
        }
    }
}

template <typename scalar_t>
__global__ void smtl_cuda_backward_kernel(
    const int nthreads,
    const int* __restrict__ parent_data, const int* __restrict__ group_offset_data, const int* __restrict__ group_size_data, const int* __restrict__ group_data,
    const scalar_t* __restrict__ label, const scalar_t* __restrict__ prob_data, scalar_t* __restrict__ bottom_diff,
    const int dim, const int spatial_dim,
    const bool has_ignore_label, const int ignore_label) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        // index == n * spatial_dim + s
        const int n = index / spatial_dim;
        const int s = index % spatial_dim;
        int label_value = static_cast<int>(label[index]);
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
}

} // namespace

std::vector<at::Tensor> smtl_cuda_forward(
    at::Tensor prob, at::Tensor label,
    at::Tensor parent,
    int outer_num, int inner_num, int dim,
    bool has_ignore_label, int ignore_label) {

  int nthreads = outer_num * inner_num;

  // Intermediate variables
  auto loss = at::zeros_like(label);

  AT_DISPATCH_FLOATING_TYPES(prob.type(), "smtl_cuda_forward", ([&] {
    smtl_cuda_forward_kernel<scalar_t><<<GET_BLOCKS(nthreads), CUDA_NUM_THREADS>>>(
        nthreads,
        parent.data<int>(),
        prob.data<scalar_t>(),
        label.data<scalar_t>(),
        dim,
        inner_num,
        has_ignore_label, ignore_label,
        loss.data<scalar_t>(),
        nullptr);
  }));

  // TODO: implement VALID normalization
  // batch normalization
  return {loss.sum() / outer_num};
}

std::vector<at::Tensor> smtl_cuda_backward(
    at::Tensor prob, at::Tensor label,
    at::Tensor parent, at::Tensor group_offset, at::Tensor group_size, at::Tensor group,
    int outer_num, int inner_num, int dim,
    bool has_ignore_label, int ignore_label) {

  int nthreads = outer_num * inner_num;

  auto diff = at::zeros_like(prob);

  AT_DISPATCH_FLOATING_TYPES(prob.type(), "smtl_cuda_backward", ([&] {
    smtl_cuda_backward_kernel<scalar_t><<<GET_BLOCKS(nthreads), CUDA_NUM_THREADS>>>(
        nthreads,
        parent.data<int>(),
        group_offset.data<int>(),
        group_size.data<int>(),
        group.data<int>(),
        label.data<scalar_t>(),
        prob.data<scalar_t>(),
        diff.data<scalar_t>(),
        dim,
        inner_num,
        has_ignore_label, ignore_label);
  }));

  // TODO: implement VALID normalization
  // batch normalization
  return {diff / outer_num};
}

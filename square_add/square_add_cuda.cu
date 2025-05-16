#include <torch/extension.h>

__global__ void square_add_kernel(const float* a, const float* b, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = a[i] * a[i] + b[i] * b[i];
    }
}

at::Tensor square_add_cuda(const at::Tensor& a, const at::Tensor& b) {
    auto out = torch::zeros_like(a);
    int size = a.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    square_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
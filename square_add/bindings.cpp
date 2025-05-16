#include <torch/extension.h>

at::Tensor square_add_cpu(const at::Tensor& a, const at::Tensor& b);
at::Tensor square_add_cuda(const at::Tensor& a, const at::Tensor& b);

at::Tensor square_add(const at::Tensor& a, const at::Tensor& b) {
    if (a.device().is_cuda()) {
        return square_add_cuda(a, b);
    }
    return square_add_cpu(a, b);
}

TORCH_LIBRARY(square_add, m) {
    m.def("square_add", &square_add);
}
#include <torch/extension.h>

at::Tensor square_add_cpu(const at::Tensor& a, const at::Tensor& b) {
    return a.square() + b.square();
}
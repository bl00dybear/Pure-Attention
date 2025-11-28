#pragma once

// libs
#include <memory>

namespace core {
    // forward declaration
    class Tensor;

    std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& A, const std::shared_ptr<Tensor>& B);
    std::shared_ptr<Tensor> matadd(const std::shared_ptr<Tensor>& A, const std::shared_ptr<Tensor>& X);
};

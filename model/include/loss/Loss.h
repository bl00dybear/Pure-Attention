#pragma once
#include "core/Tensor.h"

class MSELoss {
public:
    // Returnează un scalar Tensor (loss value)
    // Setează începutul lanțului de backpropagation
    std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> predictions, std::shared_ptr<Tensor> targets);
};
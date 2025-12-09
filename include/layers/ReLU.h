#pragma once

// headers
#include <layers/Module.h>
#include <core/Functional.h>
#include <core/Tensor.h>

namespace layers {
    class ReLU : public Module {
    public:
        ReLU() = default;

        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor> &In) override;
    };
};

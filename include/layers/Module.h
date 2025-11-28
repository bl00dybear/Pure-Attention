#pragma once

#include <memory>
#include <vector>
#include "../core/Tensor.h"

namespace layers{
    using float32_t = float;

    class Module{
    public:
        virtual ~Module() = default;
        virtual std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) = 0;
        virtual std::vector<std::shared_ptr<core::Tensor>> parameters() = 0;
    };
};


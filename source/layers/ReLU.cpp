// headers
#include <layers/ReLU.h>

namespace layers {
    std::shared_ptr<core::Tensor> ReLU::forward(const std::shared_ptr<core::Tensor> &In) {
        return relu(In);
    };

    std::vector<std::shared_ptr<core::Tensor>> ReLU::parameters() {
        return {};
    }
};

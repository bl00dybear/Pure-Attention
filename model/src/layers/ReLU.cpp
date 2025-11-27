#include "layers/Activation.h"

std::shared_ptr<Tensor> ReLU::forward(std::shared_ptr<Tensor> input) {
    return input->relu();
}
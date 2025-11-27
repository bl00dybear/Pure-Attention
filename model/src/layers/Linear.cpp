#include "layers/Linear.h"

Linear::Linear(int in_features, int out_features) 
    : in_features(in_features), out_features(out_features) {
    
    // Weights: (In, Out)
    weights = Tensor::randn({in_features, out_features}, true);
    // Bias: (1, Out)
    bias = Tensor::zeros({1, out_features}, true);
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
    // Y = XW + B
    auto xw = input->matmul(weights);
    return xw->add(bias);
}

std::vector<std::shared_ptr<Tensor>> Linear::parameters() {
    return {weights, bias};
}
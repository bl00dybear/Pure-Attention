#pragma once
#include "Module.h"

class Linear : public Module {
private:
    std::shared_ptr<Tensor> weights;
    std::shared_ptr<Tensor> bias;
    int in_features;
    int out_features;

public:
    Linear(int in_features, int out_features);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
};
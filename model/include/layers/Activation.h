#pragma once
#include "Module.h"

class ReLU : public Module {
public:
    ReLU() = default;

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;

    // ReLU nu are parametri antrenabili
    std::vector<std::shared_ptr<Tensor>> parameters() override {
        return {};
    }
};
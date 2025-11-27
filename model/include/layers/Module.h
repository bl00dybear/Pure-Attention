#pragma once
#include "core/Tensor.h"
#include <vector>

class Module {
public:
    virtual ~Module() = default;

    // Metoda principală: primește input, returnează output
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) = 0;

    // Returnează lista tuturor parametrilor antrenabili (weights, biases)
    // Folosită de optimizer
    virtual std::vector<std::shared_ptr<Tensor>> parameters() = 0;
};
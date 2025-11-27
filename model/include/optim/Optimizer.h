#pragma once
#include "core/Tensor.h"
#include <vector>
#include <unordered_map>

class Adam {
private:
    std::vector<std::shared_ptr<Tensor>> parameters;
    float lr;     // Learning rate
    float beta1;
    float beta2;
    float epsilon;
    int t;        // Time step (epoch/iteration index)

    // Stocăm istoricul (m și v) pentru fiecare parametru.
    // Folosim pointerul raw ca cheie unică.
    struct AdamState {
        float* m_device; // Moving average of gradients
        float* v_device; // Moving average of squared gradients
    };
    
    std::unordered_map<Tensor*, AdamState> state;

public:
    Adam(std::vector<std::shared_ptr<Tensor>> params, float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999);
    ~Adam(); // Trebuie să eliberăm memoria pentru m și v

    void step();      // Aplică regula de update
    void zero_grad(); // Resetează gradienții
};
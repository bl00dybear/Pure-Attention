#include <iostream>
#include <vector>
#include "core/Tensor.h"
#include "layers/Linear.h"
#include "layers/Activation.h"
#include "loss/Loss.h"
#include "optim/Optimizer.h"

int main() {
    // Hyperparameters
    int batch_size = 64;
    int input_dim = 10;
    int hidden_dim = 32;
    int output_dim = 1;
    int epochs = 100;
    float learning_rate = 0.01f;

    std::cout << "Initializing Neural Network..." << std::endl;

    // 1. Create Data (Dummy data)
    auto input = Tensor::randn({batch_size, input_dim});
    auto target = Tensor::randn({batch_size, output_dim});

    // 2. Define Model Layers
    Linear fc1(input_dim, hidden_dim);
    ReLU relu;
    Linear fc2(hidden_dim, output_dim);

    // 3. Collect Parameters
    std::vector<std::shared_ptr<Tensor>> params;
    auto p1 = fc1.parameters();
    auto p2 = fc2.parameters();
    params.insert(params.end(), p1.begin(), p1.end());
    params.insert(params.end(), p2.begin(), p2.end());

    // 4. Optimizer & Loss
    Adam optimizer(params, learning_rate);
    MSELoss criterion;

    std::cout << "Start Training..." << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // A. Zero Gradients
        optimizer.zero_grad();

        // B. Forward Pass
        auto x1 = fc1.forward(input);
        auto x2 = relu.forward(x1);
        auto output = fc2.forward(x2);

        // C. Compute Loss
        auto loss = criterion(output, target);

        // D. Backward Pass
        loss->backward();

        // E. Optimizer Step
        optimizer.step();

        // Logging
        if (epoch % 10 == 0) {
            float loss_val;
            loss->to_cpu(&loss_val);
            std::cout << "Epoch [" << epoch << "/" << epochs << "] Loss: " << loss_val << std::endl;
        }
    }

    std::cout << "Training finished successfully!" << std::endl;
    return 0;
}
#include "loss/Loss.h"
#include "backend/Kernels.cuh"

std::shared_ptr<Tensor> MSELoss::operator()(std::shared_ptr<Tensor> predictions, std::shared_ptr<Tensor> targets) {
    // Output is scalar
    auto loss = std::make_shared<Tensor>(std::vector<int>{1}, true);
    
    LaunchMSELoss(predictions->data, targets->data, loss->data, predictions->size);

    loss->grad_fn = [loss, predictions, targets]() {
        LaunchMSELossBackward(predictions->data, targets->data, predictions->grad, predictions->size);
        // Targets nu au nevoie de gradient de obicei
        if (predictions->grad_fn) predictions->grad_fn();
    };

    return loss;
}
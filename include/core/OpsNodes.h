#pragma once

#include <core/Tensor.h>

namespace core {
    struct MatMulFunction : public Function {
        std::shared_ptr<Tensor> X_input;
        std::shared_ptr<Tensor> W_input;
        std::weak_ptr<Tensor> Y_output;

        MatMulFunction(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w, std::shared_ptr<Tensor> y);

        void apply_backward() override;
    };

    struct AddFunction : public Function {
        std::shared_ptr<Tensor> X_input;
        std::shared_ptr<Tensor> bias_input;
        std::weak_ptr<Tensor> Y_output;

        AddFunction(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> bias, std::shared_ptr<Tensor> y);

        void apply_backward() override;
    };

    struct ReLUFunction : public Function {
        std::shared_ptr<Tensor> Input;
        std::weak_ptr<Tensor> Output;

        ReLUFunction(std::shared_ptr<Tensor> in, std::shared_ptr<Tensor> out);

        void apply_backward() override ;
    };

    struct MSEFunction : public Function {
        std::shared_ptr<Tensor> predictions;
        std::shared_ptr<Tensor> targets;
        std::weak_ptr<Tensor> output_loss;

        // Constructor
        MSEFunction(const std::shared_ptr<Tensor>& preds,
                    const std::shared_ptr<Tensor>& targs,
                    const std::shared_ptr<Tensor>& out)
            : predictions(preds), targets(targs), output_loss(out) {}

        void apply_backward() override {
            // 1. Verificam daca output-ul (scalarul Loss) mai exista
            auto loss_ptr = output_loss.lock();
            if (!loss_ptr) return;

            // 2. Calculam dimensiunea totala N (pentru impartirea 2/N)
            // Presupunem ca shape-ul e [Batch, Features...]
            uint32_t N = 1;
            for(auto s : predictions->get_shape()) N *= s;

            // 3. Calculam gradientul doar pentru Predicții (Target-ul nu are gradient)
            if (predictions->requires_grad()) {

                // Matematica: Grad = (2/N) * (Pred - Target) * Grad_Loss_Initial
                // Grad_Loss_Initial este de obicei 1.0 (setat de .backward())

                launch_mse_backward(
                    predictions->get_data_ptr(),
                    targets->get_data_ptr(),
                    loss_ptr->get_gradient_ptr(), // Gradientul scalarului (1.0)
                    predictions->get_gradient_ptr(), // Unde scriem rezultatul
                    N,
                    CudaContext::getStream()
                );

                // 4. Propagam recursiv
                predictions->backward();
            }
        }
    };
};
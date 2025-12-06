// #pragma once
//
// #include <layers/Module.h>
// #include <core/Tensor.h>
//
// #include <memory>
// #include <vector>
//
// namespace layers {
//     class MultiheadAttention : public Module {
//     private:
//         uint32_t model_dimension;
//         uint32_t heads_number;
//     public:
//         MultiheadAttention(const uint32_t model_dimension, const uint32_t heads_number);
//
//         std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor> &input) override;
//         std::shared_ptr<core::Tensor> forward (
//             const std::shared_ptr<core::Tensor> &query,
//             const std::shared_ptr<core::Tensor> &key,
//             const std::shared_ptr<core::Tensor> &value,
//             const std::shared_ptr<core::Tensor> &mask);
//
//         std::vector<std::shared_ptr<core::Tensor>> parameters() override;
//     };
// }
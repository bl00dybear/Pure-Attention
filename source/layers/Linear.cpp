#include <layers/Linear.h>
#include <core/Tensor.h>
#include <core/Functional.h>


namespace layers{
    using namespace core;

    Linear::Linear(uint32_t in_channels, uint32_t out_channels){
        // x: batch x in_channels
        // W: in_channels x out_channles
        // b: 1 x out_channels
        // y = x @ W + b -> y: batch x out_channels    
        weight = std::make_shared<Tensor>(std::vector<uint32_t>{in_channels,out_channels});
        bias = std::make_shared<Tensor>(std::vector<uint32_t>{1,out_channels});

        // populare vram cu w N(0,1) si b cu 0

    }


    std::shared_ptr<Tensor> Linear::forward(const std::shared_ptr<Tensor>& input){
        return matadd(matmul(input, weight), bias);
    }

    std::vector<std::shared_ptr<Tensor>>  Linear::parameters(){
        return { weight, bias };
    }
}


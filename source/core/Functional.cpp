// headers
#include <core/Tensor.h>
#include <backend/Launchers.h>
#include <core/Context.h>

// libs
#include <memory>

namespace core {
    std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& A, const std::shared_ptr<Tensor>& B) {
        uint32_t M = A->get_shape()[0];
        uint32_t K = A->get_shape()[1];
        uint32_t N = B->get_shape()[1];

        auto C = std::make_shared<Tensor>(std::vector<uint32_t>{M, N});

        launch_matmul_tiled(
            A->get_data_ptr(), 
            B->get_data_ptr(), 
            C->get_data_ptr(), 
            M, N, K,
            CudaContext::getStream()
        );

        return C;
    }
    
    std::shared_ptr<Tensor> matadd(const std::shared_ptr<Tensor>& A, const std::shared_ptr<Tensor>& X) {
        uint32_t M = A->get_shape()[0];
        uint32_t N = A->get_shape()[1];

        auto B = std::make_shared<Tensor>(std::vector<uint32_t>{M, N});

        launch_matadd_tiled(
            A->get_data_ptr(), 
            X->get_data_ptr(), 
            B->get_data_ptr(), 
            M, N,
            CudaContext::getStream()
        );

        return B;
    }

    std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& In) {
        uint32_t N = In->get_shape()[0];

        auto Out = std::make_shared<Tensor>(std::vector<uint32_t>{N});

        launch_ReLU_tiled(
            In->get_data_ptr(), 
            Out->get_data_ptr(),
            N,
            CudaContext::getStream()
        );

        return Out;
    }

    void popzeros(const std::shared_ptr<Tensor>& A){
        int M = A->get_shape()[0];
        int N = A->get_shape()[1];

        launch_zero_population(A->get_data_ptr(), M, N, CudaContext::getStream());
    }

    void popnormal(const std::shared_ptr<Tensor>& A){
        int M = A->get_shape()[0];
        int N = A->get_shape()[1];

        launch_normal_population(A->get_data_ptr(), M, N, CudaContext::getStream());
    }
};
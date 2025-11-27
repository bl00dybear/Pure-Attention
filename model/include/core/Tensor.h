#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <string>
#include "Utils.h"

// Forward declaration
class Tensor;

// Definim un tip pentru funcția de backward (un lambda care nu returnează nimic)
using BackwardFn = std::function<void()>;

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    // Device Data (GPU pointers)
    float* data;
    float* grad;
    
    // Shape and Metadata
    std::vector<int> shape;
    int size;
    bool requires_grad;

    // Autograd Graph
    // Această funcție este populată DOAR dacă tensorul este rezultatul unei operații
    BackwardFn grad_fn; 

    // Constructor & Destructor
    Tensor(std::vector<int> shape, bool requires_grad = false);
    ~Tensor();

    // Initializers (Factories)
    static std::shared_ptr<Tensor> zeros(std::vector<int> shape, bool requires_grad = false);
    static std::shared_ptr<Tensor> randn(std::vector<int> shape, bool requires_grad = false); // Random normal initialization

    // Operations (Create the Graph)
    std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> relu();
    
    // Backward Pass initiation
    void backward();

    // Helper functions
    void zero_grad();           // Resetează gradienții la 0
    void to_cpu(float* dest);   // Copiază datele de pe GPU pe CPU pentru printare/verificare
    int  get_rows() const { return shape.empty() ? 0 : shape[0]; }
    int  get_cols() const { return shape.empty() ? 0 : (shape.size() > 1 ? shape[1] : 1); }
};
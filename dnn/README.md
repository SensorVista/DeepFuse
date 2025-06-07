# DeepFuse - Deep Learning Framework

DeepFuse is a high-performance deep learning framework built with CUDA, providing efficient implementations of common neural network operations and layers.

## Table of Contents
- [Core Components](#core-components)
- [Layers](#layers)
- [Loss Functions](#loss-functions)
- [Optimizers](#optimizers)
- [Usage Examples](#usage-examples)

## Core Components

### CUDA Context Management
The framework provides robust CUDA context management through the `Cuda` class:

```cpp
namespace dnn {
class Cuda {
    // Constructor with optional device selection
    Cuda(int device_id = 0);
    
    // Get current CUDA instance
    static Cuda& current();
    
    // Device information
    int id() const;
    static const std::vector<Device>& get_devices();
    
    // Stream management
    static void set_stream(cudaStream_t stream);
    static cudaStream_t stream();
    
    // Optional cuDNN support
    #ifdef ENABLE_CUDNN
    cudnnHandle_t cudnn() const;
    cublasHandle_t cublas() const;
    #endif
};
}
```

### Tensor Operations
The framework includes a comprehensive tensor implementation with CUDA acceleration:

```cpp
namespace dnn {
class Tensor {
    // Tensor creation and management
    Tensor(const Shape& shape);
    Tensor(const std::vector<float>& data, const Shape& shape);
    
    // Data access and manipulation
    float* data();
    const float* data() const;
    Shape shape() const;
    
    // CUDA operations
    void to_device();
    void to_host();
};
}
```

## Layers

### Convolutional Layer
```cpp
class ConvLayer : public Layer {
    ConvLayer(int in_channels, int out_channels, 
             int kernel_size, int stride = 1, 
             int padding = 0);
};
```

### Fully Connected Layer
```cpp
class FullyConnectedLayer : public Layer {
    FullyConnectedLayer(int in_features, int out_features);
};
```

### Activation Layers
```cpp
class ActivationLayer : public Layer {
    ActivationLayer(ActivationType type);
};
```

### Pooling Layer
```cpp
class PoolingLayer : public Layer {
    PoolingLayer(int kernel_size, int stride = 1, 
                int padding = 0, PoolingType type = PoolingType::MAX);
};
```

### Flatten Layer
```cpp
class FlattenLayer : public Layer {
    FlattenLayer();
};
```

## Loss Functions

### Cross Entropy Loss
```cpp
class CrossEntropyLoss : public Loss {
    CrossEntropyLoss();
    float forward(const Tensor& predictions, const Tensor& targets);
    Tensor backward(const Tensor& predictions, const Tensor& targets);
};
```

### Binary Cross Entropy Loss
```cpp
class BinaryCrossEntropyLoss : public Loss {
    BinaryCrossEntropyLoss();
    float forward(const Tensor& predictions, const Tensor& targets);
    Tensor backward(const Tensor& predictions, const Tensor& targets);
};
```

## Optimizers

### Stochastic Gradient Descent (SGD)
```cpp
class SGDOptimizer : public Optimizer {
    SGDOptimizer(float learning_rate = 0.01, 
                 float momentum = 0.0, 
                 float weight_decay = 0.0);
    
    void step();
    void zero_grad();
};
```

## Usage Examples

### Basic Network Creation
```cpp
#include <dnn/core/cuda.h>
#include <dnn/layers/conv_layer.h>
#include <dnn/layers/fully_connected_layer.h>

// Initialize CUDA context
dnn::Cuda cuda(0);  // Use GPU 0

// Create a simple CNN
auto conv1 = std::make_shared<ConvLayer>(3, 32, 3);  // 3 input channels, 32 output channels, 3x3 kernel
auto fc1 = std::make_shared<FullyConnectedLayer>(32 * 28 * 28, 10);  // Flattened conv output to 10 classes
```

### Training Loop
```cpp
#include <dnn/losses/cross_entropy.h>
#include <dnn/optimizers/sgd_optimizer.h>

// Setup loss and optimizer
CrossEntropyLoss loss;
SGDOptimizer optimizer(0.01);

// Training loop
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (const auto& batch : dataloader) {
        // Forward pass
        auto output = model.forward(batch.input);
        
        // Compute loss
        float loss_value = loss.forward(output, batch.target);
        
        // Backward pass
        auto grad = loss.backward(output, batch.target);
        model.backward(grad);
        
        // Update weights
        optimizer.step();
        optimizer.zero_grad();
    }
}
```

## Requirements
- CUDA Toolkit 11.0 or higher
- cuDNN 8.0 or higher (optional)
- C++17 or higher
- CMake 3.15 or higher

## Building
```bash
mkdir build && cd build
cmake ..
make
```

## License
[Add your license information here]

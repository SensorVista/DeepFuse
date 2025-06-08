# DeepFuse Applications

This directory contains example applications demonstrating the capabilities of the DeepFuse framework. Each application showcases different aspects of the framework's functionality.

## Available Applications

### Perceptron (`perceptron/`)
A simple binary classification example using a single-layer perceptron. Demonstrates:
- Basic neural network training
- Binary classification on synthetic XOR-like data
- Batch processing
- Model evaluation and accuracy tracking

### LeNet-5 (`lenet5/`)
Implementation of the classic LeNet-5 architecture for MNIST digit recognition. Features:
- Convolutional neural network architecture
- Multi-layer design with conv, pooling, and fully connected layers
- MNIST dataset integration
- Real-world image classification

### ResNet-20 (`resnet20/`)
A lightweight implementation of ResNet-20 architecture. Demonstrates:
- Deep residual network architecture
- Skip connections and residual blocks
- Advanced CNN training techniques
- Model complexity management

### Multi-GPU Host (`host-multi/`)
Advanced multi-GPU training application with host-side parallelization. Features:
- Multi-GPU training support
- Thread-safe model execution
- Dynamic GPU memory management
- Parallel data processing
- TBB (Threading Building Blocks) integration
- NVMe data loading capabilities

## Building and Running

Each application can be built independently:

```bash
# Build a specific application
cd apps/<app_name>
mkdir build && cd build
cmake ..
make

# Run the application
./<app_name>
```

## Requirements

- CUDA Toolkit 11.0 or higher
- C++17 or higher
- CMake 4.0 or higher
- NVMe support (optional, for host-multi example)

## Performance Considerations

- **Single-GPU Applications**: Optimized for single GPU usage with efficient memory management
- **Multi-GPU Application**: Implements dynamic load balancing and memory optimization across multiple GPUs
- **Data Loading**: Supports both in-memory and NVMe-based data loading for large datasets
- **Batch Processing**: Configurable batch sizes for optimal performance

## Usage Examples

### Basic Perceptron Training
```cpp
dnn::Perceptron<float> network(input_dim, hidden_dim, output_dim);
network.train_step(input_batch, target_batch);
```

### Multi-GPU Training
```cpp
// Configure multi-GPU training
TrainingTask task{
    device_id,
    num_samples,
    batch_size,
    num_epochs
};
tbb::parallel_for(tbb::blocked_range<size_t>(0, tasks.size()),
    [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
            tasks[i]();
        }
    }
);
```

## Notes

- All applications include error handling and proper resource management
- CUDNN acceleration is optional and can be enabled during build
- Applications demonstrate both basic and advanced usage of the DeepFuse framework
- Each application includes example data or data generation code

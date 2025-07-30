# DeepFuse: Scalable Transformers on Commodity GPUs

C++ 17 CUDA framework. Support next-gen Blackwell architecture and legacy CUDA devices. Supports e5m2 and e4m3 8-bit 
quantization. FP32, FP16 and BF16. Advanced support for device management, tensor ops,  cuBLAS and cuDNN.

This project demonstrates CUDA functionality and provides a advanced test environment for CUDA operations. 
Breaks the VRAM ceiling, reduces overhead, and enables billion-scale transformers on consumer GPUs.
Azure, MetaAI and DeepMind scaled with big budgets. We scale with efficiency.

## Documentation

### Core Framework ([dnn/](dnn/))
- Core CUDA components and tensor operations
- Neural network layers implementation:
  - Convolutional, Fully Connected, Flatten, Pooling (Max, Average, Global)
  - Batch Normalization, Layer Normalization
  - Activation: ReLU, Sigmoid, Tanh, ClippedReLU, ELU, GELU
  - Residual Block, Residual Add
  - Multi-Head Attention, Transformer Block, Softmax
- Loss functions:
  - Cross Entropy Loss
  - Binary Cross Entropy Loss
- Optimizers:
  - SGD (Stochastic Gradient Descent)
  - Adam
- NLP utilities:
  - BPE Tokenizer, Token Embedding, Positional Encoding, Vocabulary Loader
- API reference and usage examples

### Unit Tests ([tests/](tests/))
- Comprehensive test coverage
- CUDNN integration testing
- Performance benchmarks
- Testing best practices

### Example Applications ([apps/](apps/))
- Perceptron: Basic binary classification
- LeNet-5: MNIST digit recognition
- ResNet-20: Deep residual network
- ResNet-32: Deeper residual network for CIFAR-10
- ResNet-56: Benchmark deep residual network
- GPT-2: Transformer-based LLM
- Multi-GPU Host: Advanced multi-GPU training

## Features

### Layer-Serialized Architecture
- **Streaming Execution**: Custom implementation of layer-by-layer computation
- **Memory Management**: Advanced techniques for parameter streaming and caching
- **Attention Optimization**: Novel approaches to sliding-window attention
- **Gradient Handling**: Efficient gradient accumulation and recomputation

### CUDA Optimizations
- **Kernel Fusion**: Implementation of fused transformer operations and other kernels
- **Memory Streaming**: Optimized host-device transfer patterns
- **Stream Management**: Advanced scheduling for layer execution
- **Asynchronous Operations**: Non-blocking memory transfers and computations

### Transformer Features
- **QKV Attention**: Optimized implementation with custom CUDA kernels
- **Rotary Embeddings**: Efficient positional encoding
- **Layer Normalization**: In-place operations with numerical stability
- **Residual Connections**: Memory-efficient implementation

### Benefits
- **Disk-backed execution**  
  Weight tensors, optimizer state, and activations are managed off-GPU using pinned host memory and NVMe streaming, removing memory ceilings.

- **Stream layers, not models**  
  Execute arbitrarily deep networks by paging individual layers in and out of GPU memory, enabling transformer stacks that exceed device limits.

- **Train GPT-class models on consumer GPUs**  
  Pretrain and fine-tune billion-parameter transformers on devices like the RTX 4090 or older multi-GPU rigs, without needing massive VRAM.

- **Bypass framework overhead**  
  Fully custom CUDA/C++ implementation with zero reliance on PyTorch, TensorFlow, or external autograd frameworks.

- **Precision without compromise**  
  Supports Tensor Core acceleration, FP16/BF16 mixed precision, and warp-level primitives for fused, high-throughput kernels.

- **Memory-aware scheduling**  
  Optimized for cache hierarchy, shared memory tiling, and async overlap of I/O and compute using cooperative groups and CUDA streams.

- **Token-efficient transformer scaling**  
  Designed for long sequence processing using rotary embeddings, sliding window attention, and Flash-style optimizations.

- **Research-grade control**  
  Every byte, warp, and instruction is under explicit control—ideal for systems-level AI research, AGI experimentation, and real-time inference.

- **Granular checkpointing**
  Supports checkpoints for the entire model, individual layers, weights, activations, and input batches. Enables precise recovery and flexible, layer-wise debugging.  

## Example Project Structure

```
DeepFuse/
├── dnn/           # Core framework implementation
│   ├── core/      # CUDA context and tensor operations
│   ├── layers/    # Neural network layers
│   ├── losses/    # Loss functions
│   ├── models/    # Model architectures
│   └── optimizers/# Optimization algorithms
├── tests/         # Unit tests and benchmarks
│   ├── core/      # Core functionality tests
│   ├── layers/    # Layer implementation tests
│   └── models/    # Model architecture tests
└── apps/          # Example applications
    ├── perceptron/# Basic neural network example
    ├── gpt2/      # LLM transformer example
    ├── lenet5/    # MNIST classification
    ├── resnet20/  # Deep residual network
    ├── resnet32/  # Deeper residual network
    ├── resnet56/  # Benchmark deep residual network
    └── host-multi/# Multi-GPU training
```
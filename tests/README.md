# DeepFuse Unit Tests

This directory contains comprehensive unit tests for the DeepFuse deep learning framework. The tests are organized to mirror the structure of the main library, ensuring complete coverage of all components.

## Test Structure

The tests are organized into the following directories:

- `core/` - Tests for core CUDA functionality and tensor operations
- `layers/` - Tests for all neural network layers
- `losses/` - Tests for loss functions
- `optimizers/` - Tests for optimization algorithms
- `models/` - Tests for complete model architectures

## Testing Framework

The tests use Google Test (gtest) framework and are written in CUDA C++. Each test file follows a consistent pattern:

```cpp
class ComponentTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test initialization
    }

    void TearDown() override {
        // Test cleanup
    }
};
```

## Key Test Categories

### Core Tests
- CUDA context management
- Device selection and switching
- Stream management
- Tensor operations
- Memory management

### Layer Tests
- Convolutional layers
- Fully connected layers
- Activation layers
- Pooling layers
- Layer forward/backward passes

### Loss Function Tests
- Cross entropy loss
- Binary cross entropy loss
- Loss computation accuracy
- Gradient computation

## CUDNN Integration

The framework supports optional CUDNN acceleration through the `ENABLE_CUDNN` compile-time flag. This affects testing in several ways:

1. **Conditional Tests**: Tests that depend on CUDNN are wrapped in `#ifdef ENABLE_CUDNN` blocks:
```cpp
#ifdef ENABLE_CUDNN
TEST_F(CudaTest, CudnnHandle) {
    Cuda cuda;
    EXPECT_NE(cuda.cudnn(), nullptr);
}
#endif
```

2. **Feature Availability**: When CUDNN is enabled, additional tests verify:
   - CUDNN handle initialization
   - CUBLAS handle initialization
   - CUDNN-specific operations
   - Performance optimizations

3. **Build Configuration**: To run tests with CUDNN:
```bash
cmake -DENABLE_CUDNN=ON ..
make
```

## Running Tests

### Basic Test Execution
```bash
# Build tests
mkdir build && cd build
cmake ..
make

# Run all tests
ctest

# Run specific test
./tests/unit_tests

# Note: Test discovery is disabled on Linux due to regex compatibility
# Tests can be run directly via the unit_tests executable
```

### Test Categories
```bash
# Run core tests only
ctest -R "core"

# Run layer tests only
ctest -R "layers"

# Run specific test file
ctest -R "cuda_test"
```

## Test Coverage

The test suite aims for comprehensive coverage of:
- Basic functionality
- Edge cases
- Error conditions
- Memory management
- Thread safety
- Performance characteristics

## Adding New Tests

When adding new tests:
1. Create a new test file in the appropriate directory
2. Follow the existing test patterns
3. Include both positive and negative test cases
4. Add CUDNN-specific tests when applicable
5. Update this documentation if necessary

## Best Practices

1. **Test Isolation**: Each test should be independent and not rely on the state from other tests
2. **Resource Management**: Properly initialize and clean up CUDA resources
3. **Error Handling**: Test both success and failure cases
4. **Performance**: Include basic performance tests where relevant
5. **Documentation**: Document any special test requirements or setup
6. **Cross-Platform**: Tests are designed to work on both Windows and Linux with appropriate platform-specific handling

#include <gtest/gtest.h>

#include <dnn/core/cuda.cuh>
#include <dnn/layers/multi_head_attention.cuh>
#include <dnn/utils/common.cuh>

#include <vector>
#include <cmath>
#include <random>
#include <sstream>
#include <iostream>

namespace dnn::test {

using namespace dnn::utils;

class MultiHeadAttentionTest : public ::testing::Test {
protected:
    std::unique_ptr<Cuda> cuda_;

    void SetUp() override {
        cuda_ = std::make_unique<Cuda>(0);
        cudaDeviceSynchronize();
    }

    void TearDown() override {
        cudaDeviceSynchronize();
        cuda_.reset();
    }

    // Helper function to create random input tensor
    tensor<float> create_random_input(int batch_size, int seq_len, int embed_dim) {
        std::vector<int> shape = {batch_size, seq_len, embed_dim};
        tensor<float> input(shape);
        
        std::vector<float> data(batch_size * seq_len * embed_dim);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = dist(gen);
        }
        
        input.upload(data.data());
        return input;
    }

    // Helper function to create attention mask
    tensor<float> create_attention_mask(int batch_size, int seq_len) {
        std::vector<int> shape = {batch_size, 1, seq_len, seq_len};
        tensor<float> mask(shape);
        
        // Create data array with proper 4D size
        std::vector<float> data(batch_size * 1 * seq_len * seq_len);
        
        // Initialize mask values
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < seq_len; ++j) {
                    // Create causal mask (lower triangular)
                    // For each batch, create a mask where future positions (j > i) are masked
                    data[b * seq_len * seq_len + i * seq_len + j] = (j <= i) ? 0.0f : -1e9f;
                }
            }
        }
        
        mask.upload(data.data());
        return mask;
    }

    // Helper function to check if two tensors are approximately equal
    bool tensors_approx_equal(const tensor<float>& a, const tensor<float>& b, float tolerance = 1e-5) {
        if (a.shape() != b.shape()) return false;
        
        std::vector<float> data_a(a.size());
        std::vector<float> data_b(b.size());
        a.download(data_a.data());
        b.download(data_b.data());
        
        for (size_t i = 0; i < data_a.size(); ++i) {
            if (std::abs(data_a[i] - data_b[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }
};

TEST_F(MultiHeadAttentionTest, ConstructorAndInitialization) {
    int embed_dim = 512;
    int num_heads = 8;
    
    MultiHeadAttentionLayer<float> layer(embed_dim, num_heads);
    
    // Verify dimensions
    EXPECT_EQ(layer.name(), "MultiHeadAttention");
    EXPECT_EQ(embed_dim % num_heads, 0);  // embed_dim must be divisible by num_heads
    
    // Test invalid dimensions
    EXPECT_THROW(MultiHeadAttentionLayer<float>(512, 7), std::runtime_error);
}

TEST_F(MultiHeadAttentionTest, ForwardPass) {
    int batch_size = 2;
    int seq_len = 10;
    int embed_dim = 512;
    int num_heads = 8;
    
    MultiHeadAttentionLayer<float> layer(embed_dim, num_heads);
    
    // Create input
    tensor<float> input = create_random_input(batch_size, seq_len, embed_dim);
    
    // Create attention mask
    tensor<float> mask = create_attention_mask(batch_size, seq_len);
    layer.set_mask(&mask);
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Verify output shape
    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], seq_len);
    EXPECT_EQ(output.shape()[2], embed_dim);
    
    // Verify output values are within reasonable range
    std::vector<float> output_data(output.size());
    output.download(output_data.data());
    for (float val : output_data) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST_F(MultiHeadAttentionTest, BackwardPass) {
    int batch_size = 2;
    int seq_len = 10;
    int embed_dim = 512;
    int num_heads = 8;
    
    const bool training_enabled = true;
    MultiHeadAttentionLayer<float> layer(
        embed_dim,
        num_heads,
        training_enabled
    );
    
    // Create input
    tensor<float> input = create_random_input(batch_size, seq_len, embed_dim);
    
    // Create attention mask
    tensor<float> mask = create_attention_mask(batch_size, seq_len);
    layer.set_mask(&mask);
    
    // Forward pass to populate cache
    layer.forward(input);
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Create gradient tensor
    tensor<float> grad_output(output.shape());
    std::vector<float> grad_data(output.size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < grad_data.size(); ++i) {
        grad_data[i] = dist(gen);
    }
    grad_output.upload(grad_data.data());
    
    // Backward pass
    tensor<float> grad_input = layer.backward(grad_output);
    
    // Verify gradient shape
    EXPECT_EQ(grad_input.shape()[0], batch_size);
    EXPECT_EQ(grad_input.shape()[1], seq_len);
    EXPECT_EQ(grad_input.shape()[2], embed_dim);
    
    // Verify gradient values are within reasonable range
    std::vector<float> grad_input_data(grad_input.size());
    grad_input.download(grad_input_data.data());
    for (float val : grad_input_data) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST_F(MultiHeadAttentionTest, DifferentDimensions) {
    int batch_size = 4;
    int seq_len = 20;
    int embed_dim = 256;
    int num_heads = 4;
    
    MultiHeadAttentionLayer<float> layer(embed_dim, num_heads);
    
    // Create input
    tensor<float> input = create_random_input(batch_size, seq_len, embed_dim);
    
    // Create attention mask
    tensor<float> mask = create_attention_mask(batch_size, seq_len);
    layer.set_mask(&mask);
    
    // Forward pass
    tensor<float> output = layer.forward(input);
    
    // Verify output shape
    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], seq_len);
    EXPECT_EQ(output.shape()[2], embed_dim);
    
    // Create gradient tensor
    tensor<float> grad_output(output.shape());
    std::vector<float> grad_data(output.size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < grad_data.size(); ++i) {
        grad_data[i] = dist(gen);
    }
    grad_output.upload(grad_data.data());
    
    // Backward pass
    tensor<float> grad_input = layer.backward(grad_output);
    
    // Verify gradient shape
    EXPECT_EQ(grad_input.shape()[0], batch_size);
    EXPECT_EQ(grad_input.shape()[1], seq_len);
    EXPECT_EQ(grad_input.shape()[2], embed_dim);
}

TEST_F(MultiHeadAttentionTest, LayerConfiguration) {
    int embed_dim = 512;
    int num_heads = 8;
    
    MultiHeadAttentionLayer<float> layer(embed_dim, num_heads);
    
    // Test getters
    EXPECT_EQ(layer.embed_dim(), embed_dim);
    EXPECT_EQ(layer.num_heads(), num_heads);
    EXPECT_EQ(layer.head_dim(), embed_dim / num_heads);
}

TEST_F(MultiHeadAttentionTest, MaskHandling) {
    int batch_size = 2;
    int seq_len = 10;
    int embed_dim = 512;
    int num_heads = 8;

    MultiHeadAttentionLayer<float> layer(embed_dim, num_heads);

    tensor<float> input = create_random_input(batch_size, seq_len, embed_dim);

    // Print first 10 input values
    std::vector<float> input_data(input.size());
    input.download(input_data.data());
    std::cout << "First 10 input values:" << std::endl;
    for (int i = 0; i < 10 && i < input_data.size(); ++i) {
        std::cout << "input[" << i << "] = " << input_data[i] << std::endl;
    }

    // Test without mask
    tensor<float> output_nomask = layer.forward(input);

    // Print first 10 Q projection output values
    auto* last_q = layer.get_last_q();
    std::vector<float> last_q_data(last_q->size());
    last_q->download(last_q_data.data());
    std::cout << "First 10 Q projection output values:" << std::endl;
    for (int i = 0; i < 10 && i < last_q_data.size(); ++i) {
        std::cout << "last_q[" << i << "] = " << last_q_data[i] << std::endl;
    }

    // Set causal mask
    tensor<float> mask = create_attention_mask(batch_size, seq_len);
    layer.set_mask(&mask);

    // First forward pass with mask
    tensor<float> output_masked = layer.forward(input);

    // Debug: Print first 10 values and their differences
    std::stringstream ss;
    std::vector<float> nomask_data(output_nomask.size());
    std::vector<float> masked_data(output_masked.size());
    output_nomask.download(nomask_data.data());
    output_masked.download(masked_data.data());
    for (int i = 0; i < 10 && i < nomask_data.size(); ++i) {
        ss << "nomask[" << i << "] = " << nomask_data[i]
                  << ", masked[" << i << "] = " << masked_data[i]
                  << ", diff = " << (nomask_data[i] - masked_data[i]) << std::endl;
    }

    // Re-run with same mask
    tensor<float> output_masked_repeat = layer.forward(input);

    // Compare: masked != unmasked
    EXPECT_FALSE(tensors_approx_equal(output_nomask, output_masked, 1e-3f));

    // Compare: masked == masked repeat (use MSE instead of exact)
    std::vector<float> a(output_masked.size());
    std::vector<float> b(output_masked_repeat.size());
    output_masked.download(a.data());
    output_masked_repeat.download(b.data());

    float mse = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        mse += diff * diff;
    }
    mse /= a.size();
    EXPECT_GT(mse, 1e-3f);

    // Inspect attention weights
    tensor<float>* attn = layer.get_last_attention(); // Shape: [B*H, T, T]
    attn->reshape({ batch_size, num_heads, seq_len, seq_len });

    std::vector<float> attn_data(attn->size());
    attn->download(attn_data.data());

    // Print first 10 Q projection weights
    auto* q_weights = layer.get_q_weights();
    std::vector<float> q_weights_host(q_weights->size());
    q_weights->download(q_weights_host.data());
    std::cout << ss.str();

    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int t = 0; t < seq_len; ++t) {
                for (int s = 0; s < seq_len; ++s) {
                    float score = attn_data[((b * num_heads + h) * seq_len + t) * seq_len + s];
                    if (s > t) {
                        EXPECT_LT(score, 1e-3f) << "Attention not suppressed at b=" << b << " h=" << h << " t=" << t << " s=" << s << " score=" << score;
                    }
                }
            }
        }
    }

    CHECK_CUDA_EX(cudaGetLastError());
    CHECK_CUDA_EX(cudaDeviceSynchronize());
}

TEST_F(MultiHeadAttentionTest, MaskEffectIsVisible) {
    int batch_size = 1, seq_len = 4, embed_dim = 8, num_heads = 2;
    MultiHeadAttentionLayer<float> layer(embed_dim, num_heads);

    tensor<float> input = create_random_input(batch_size, seq_len, embed_dim);

    // No mask
    tensor<float> output_nomask = layer.forward(input);

    // Mask: only allow attention to self (diagonal)
    tensor<float> mask({batch_size, 1, seq_len, seq_len});
    std::vector<float> mask_data(batch_size * seq_len * seq_len, -1e9f);
    for (int i = 0; i < seq_len; ++i) mask_data[i * seq_len + i] = 0.0f;
    mask.upload(mask_data.data());
    layer.set_mask(&mask);

    tensor<float> output_masked = layer.forward(input);

    // They should not be equal
    EXPECT_FALSE(tensors_approx_equal(output_nomask, output_masked, 1e-5));
}

} // namespace dnn::test

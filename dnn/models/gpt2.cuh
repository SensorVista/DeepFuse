#pragma once

#include "natural_language_model.cuh"
#include "../tokens/token_embedding.cuh"
#include "../tokens/tokenizer.cuh"
#include "../tokens/positional_encoding.cuh"
#include "../layers/transformer_block.cuh"
#include "../layers/fully_connected_layer.cuh"
#include "../losses/cross_entropy.cuh"
#include "../optimizers/adam_optimizer.cuh"

#include <memory>
#include <random>

namespace dnn {

template<typename T>
class Gpt2 : public NaturalLanguageModel<T> {
public:
    Gpt2(std::shared_ptr<Tokenizer> tokenizer,
         int vocab_size = 50257,
         int max_seq_len = 1024,
         int num_layers = 12,
         int num_heads = 12,
         int hidden_dim = 768,
         int intermediate_dim = 3072,
         T learning_rate = static_cast<T>(1e-4),
         T beta1 = static_cast<T>(0.9),
         T beta2 = static_cast<T>(0.98),
         T epsilon = static_cast<T>(1e-8))
        : NaturalLanguageModel<T>(tokenizer,
                                  std::make_unique<CrossEntropyLoss<T>>(),
                                  std::make_unique<AdamOptimizer<T>>(learning_rate, beta1, beta2, epsilon)),
          vocab_size_(vocab_size),
          max_seq_len_(max_seq_len),
          hidden_dim_(hidden_dim)
    {
        // Initialize token embeddings with proper scaling
        auto token_embedding = std::make_unique<TokenEmbedding<T>>(vocab_size, hidden_dim, max_seq_len);
        initialize_weights(token_embedding->weights(), 0.02f);
        this->add_layer(std::move(token_embedding));

        // Add positional encoding
        this->add_layer(std::make_unique<PositionalEncoding<T>>(hidden_dim, max_seq_len));

        // Add transformer blocks with proper initialization
        for (int i = 0; i < num_layers; ++i) {
            auto block = std::make_unique<TransformerBlock<T>>(hidden_dim, intermediate_dim, num_heads, true);
            initialize_transformer_block(block.get());
            this->add_layer(std::move(block));
        }

        // Add final layer with proper initialization
        auto final_layer = std::make_unique<FullyConnectedLayer<T>>(hidden_dim, vocab_size);
        initialize_weights(final_layer->weights(), 0.02f);
        this->add_layer(std::move(final_layer));
    }

    std::string name() const override {
        return "Gpt2";
    }

private:
    void initialize_weights(tensor<T>& weights, T scale) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> dist(0.0f, scale);
        
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = dist(gen);
        }
    }

    void initialize_transformer_block(TransformerBlock<T>* block) {
        // Initialize attention weights
        initialize_weights(block->query_weights(), 0.02f);
        initialize_weights(block->key_weights(), 0.02f);
        initialize_weights(block->value_weights(), 0.02f);
        initialize_weights(block->output_weights(), 0.02f);

        // Initialize MLP weights
        initialize_weights(block->mlp_fc1_weights(), 0.02f);
        initialize_weights(block->mlp_fc2_weights(), 0.02f);
    }

    int vocab_size_;
    int max_seq_len_;
    int hidden_dim_;
};

} // namespace dnn

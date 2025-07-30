#pragma once

#include <dnn/models/natural_language_model.cuh>
#include <dnn/tokens/token_embedding.cuh>
#include <dnn/tokens/bpe_tokenizer.cuh>
#include <dnn/tokens/positional_encoding.cuh>
#include <dnn/layers/transformer_block.cuh>
#include <dnn/layers/fully_connected_layer.cuh>
#include <dnn/losses/cross_entropy.cuh>
#include <dnn/optimizers/adam_optimizer.cuh>

#include <memory>
#include <random>
#include <fstream>
#include <stdexcept>

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
         T epsilon = static_cast<T>(1e-8),
         bool training_enabled = false);

    std::string name() const override;

    void train_step(const tensor<T>& input, const tensor<T>& target) override;
    void train_step(const std::vector<int>& input_token_ids, const std::vector<int>& target_token_ids);

    tensor<T> forward(const tensor<T>& input) override;
    tensor<T> forward(const std::vector<int>& input_token_ids);

    void save(const std::string& path) const override;
    static std::unique_ptr<Gpt2<T>> load(const std::string& path, bool training_enabled);

    std::vector<BaseLayer*> layers() override;

private:
    void initialize_weights(tensor<T>* weights, T scale);

    int vocab_size_;
    int max_seq_len_;
    int hidden_dim_;
    std::unique_ptr<TokenEmbedding<T>> token_embedding_;
    std::unique_ptr<PositionalEncoding<T>> positional_encoding_;
    std::vector<std::unique_ptr<TransformerBlock<T>>> transformer_blocks_;
    std::unique_ptr<FullyConnectedLayer<T>> final_fc_;
};

} // namespace dnn

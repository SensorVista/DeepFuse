#include "gpt2.cuh"

#include <random>
#include <stdexcept>
#include <fstream>
#include <memory>
#include <vector>
#include <string>

namespace dnn {

template<typename T>
Gpt2<T>::Gpt2(std::shared_ptr<Tokenizer> tokenizer,
         int vocab_size,
         int max_seq_len,
         int num_layers,
         int num_heads,
         int hidden_dim,
         int intermediate_dim,
         T learning_rate,
         T beta1,
         T beta2,
         T epsilon,
         bool training_enabled)
    : NaturalLanguageModel<T>(tokenizer,
                              max_seq_len,
                              training_enabled),
      vocab_size_(vocab_size),
      max_seq_len_(max_seq_len),
      hidden_dim_(hidden_dim)
{
    token_embedding_ = std::make_unique<TokenEmbedding<T>>(vocab_size, hidden_dim, max_seq_len, training_enabled);
    initialize_weights(token_embedding_->weights(), 0.02f);
    positional_encoding_ = std::make_unique<PositionalEncoding<T>>(hidden_dim, max_seq_len, training_enabled);
    for (int i = 0; i < num_layers; ++i) {
        auto block = std::make_unique<TransformerBlock<T>>(hidden_dim, num_heads, intermediate_dim, training_enabled);
        block->initialize_all_weights();
        transformer_blocks_.push_back(std::move(block));
    }
    final_fc_ = std::make_unique<FullyConnectedLayer<T>>(hidden_dim, vocab_size, training_enabled);
    initialize_weights(final_fc_->weights(), 0.02f);

    this->set_loss(std::make_unique<CrossEntropyLoss<T>>());
    this->set_optimizer(std::make_unique<AdamOptimizer<T>>(learning_rate, beta1, beta2, epsilon));
}

template<typename T>
std::string Gpt2<T>::name() const {
    return "Gpt2";
}

template<typename T>
void Gpt2<T>::train_step(const tensor<T>& input, const tensor<T>& target) {
    throw std::runtime_error("Use token-based train_step for NLP models.");
}

template<typename T>
void Gpt2<T>::train_step(const std::vector<int>& input_token_ids, const std::vector<int>& target_token_ids) {
    int B = 1;
    int T = static_cast<int>(input_token_ids.size());
    dnn::tensor<int> input_tensor({B, T});
    input_tensor.upload(input_token_ids.data());
    dnn::tensor<int> target_tensor({B, T});
    target_tensor.upload(target_token_ids.data());

    auto* embedding_layer = dynamic_cast<dnn::LayerAsymmetric<float, int>*>(token_embedding_.get());
    if (!embedding_layer) throw std::runtime_error("First layer is not a LayerAsymmetric<float, int>");
    dnn::tensor<float> x = embedding_layer->forward(input_tensor);

    auto* pos_layer = dynamic_cast<dnn::Layer<float>*>(positional_encoding_.get());
    if (!pos_layer) throw std::runtime_error("Second layer is not a Layer<float>");
    x = pos_layer->forward(x);

    for (size_t i = 0; i < transformer_blocks_.size(); ++i) {
        auto* typed_layer = dynamic_cast<dnn::Layer<float>*>(transformer_blocks_[i].get());
        if (!typed_layer) throw std::runtime_error("Layer type mismatch in Gpt2 forward");
        x = typed_layer->forward(x);
    }

    x.reshape({B * T, hidden_dim_});
    auto* final_layer = dynamic_cast<dnn::FullyConnectedLayer<float>*>(final_fc_.get());
    if (!final_layer) throw std::runtime_error("Final layer is not FullyConnectedLayer<float>");
    dnn::tensor<float> logits = final_layer->forward(x);

    target_tensor.reshape({B * T});
    auto* ce_loss = dynamic_cast<dnn::CrossEntropyLoss<float>*>(this->loss_.get());
    if (!ce_loss) throw std::runtime_error("Loss is not CrossEntropyLoss<float>");
    this->current_loss_ = ce_loss->compute(logits, target_tensor);
    dnn::tensor<float> grad = ce_loss->compute_gradient(logits, target_tensor);

    grad = final_layer->backward(grad);
    grad.reshape({B, T, hidden_dim_});
    for (int i = static_cast<int>(transformer_blocks_.size()) - 1; i >= 0; --i) {
        auto* typed_layer = dynamic_cast<dnn::Layer<float>*>(transformer_blocks_[i].get());
        grad = typed_layer->backward(grad);
    }
    grad = pos_layer->backward(grad);
    grad = embedding_layer->backward(grad);

    std::vector<dnn::tensor<float>*> all_grads;
    {
        auto grads = token_embedding_->gradients();
        all_grads.insert(all_grads.end(), grads.begin(), grads.end());
    }
    {
        auto grads = positional_encoding_->gradients();
        all_grads.insert(all_grads.end(), grads.begin(), grads.end());
    }
    for (const auto& block : transformer_blocks_) {
        auto grads = block->gradients();
        all_grads.insert(all_grads.end(), grads.begin(), grads.end());
    }
    {
        auto grads = final_fc_->gradients();
        all_grads.insert(all_grads.end(), grads.begin(), grads.end());
    }
    dnn::utils::clip_grad_norm(all_grads, 1.0f);
    this->optimizer_->step();
}

template<typename T>
tensor<T> Gpt2<T>::forward(const tensor<T>& input) {
    throw std::runtime_error("Use token-based forward for NLP models.");
}

template<typename T>
tensor<T> Gpt2<T>::forward(const std::vector<int>& input_token_ids) {
    tensor<int> input_tensor({1, static_cast<int>(input_token_ids.size())});
    input_tensor.upload(input_token_ids.data());
    auto* embedding_layer = dynamic_cast<LayerAsymmetric<T, int>*>(token_embedding_.get());
    if (!embedding_layer) throw std::runtime_error("First layer is not a LayerAsymmetric<T, int>");
    tensor<T> x = embedding_layer->forward(input_tensor);
    auto* pos_layer = dynamic_cast<Layer<T>*>(positional_encoding_.get());
    if (!pos_layer) throw std::runtime_error("Second layer is not a Layer<T>");
    x = pos_layer->forward(x);
    for (size_t i = 0; i < transformer_blocks_.size(); ++i) {
        auto* typed_layer = dynamic_cast<Layer<T>*>(transformer_blocks_[i].get());
        if (!typed_layer) throw std::runtime_error("Layer type mismatch in Gpt2 forward");
        x = typed_layer->forward(x);
    }
    return x;
}

template<typename T>
void Gpt2<T>::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open file for saving Gpt2");
    out.write(reinterpret_cast<const char*>(&vocab_size_), sizeof(vocab_size_));
    out.write(reinterpret_cast<const char*>(&max_seq_len_), sizeof(max_seq_len_));
    int num_layers = static_cast<int>(transformer_blocks_.size());
    out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    out.write(reinterpret_cast<const char*>(&hidden_dim_), sizeof(hidden_dim_));
    this->tokenizer_->save(out);
    out.write(reinterpret_cast<const char*>(&token_embedding_->name()[0]), token_embedding_->name().size());
    token_embedding_->save(out);
    out.write(reinterpret_cast<const char*>(&positional_encoding_->name()[0]), positional_encoding_->name().size());
    positional_encoding_->save(out);
    for (const auto& block : transformer_blocks_) {
        out.write(reinterpret_cast<const char*>(&block->name()[0]), block->name().size());
        block->save(out);
    }
    out.write(reinterpret_cast<const char*>(&final_fc_->name()[0]), final_fc_->name().size());
    final_fc_->save(out);
    out.close();
}

template<typename T>
std::unique_ptr<Gpt2<T>> Gpt2<T>::load(const std::string& path, bool training_enabled) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open file for loading Gpt2");
    int vocab_size, max_seq_len, num_layers, hidden_dim;
    in.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
    in.read(reinterpret_cast<char*>(&max_seq_len), sizeof(max_seq_len));
    in.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    in.read(reinterpret_cast<char*>(&hidden_dim), sizeof(hidden_dim));
    auto tokenizer = std::make_shared<BpeTokenizer>(std::make_shared<VocabLoader>());
    tokenizer->load(in);
    int num_transformer_blocks = num_layers - 2;
    auto model = std::make_unique<Gpt2<T>>(tokenizer, vocab_size, max_seq_len, num_transformer_blocks, 12, hidden_dim, 3072, static_cast<T>(1e-4), static_cast<T>(0.9), static_cast<T>(0.98), static_cast<T>(1e-8), training_enabled);
    model->token_embedding_->load(in);
    model->positional_encoding_->load(in);
    for (int i = 0; i < num_layers; ++i) {
        int type_len;
        in.read(reinterpret_cast<char*>(&type_len), sizeof(type_len));
        std::string type(type_len, '\0');
        in.read(&type[0], type_len);
        model->transformer_blocks_[i]->load(in);
    }
    model->final_fc_->load(in);
    in.close();
    return model;
}

template<typename T>
std::vector<BaseLayer*> Gpt2<T>::layers() {
    std::vector<BaseLayer*> result;
    result.push_back(token_embedding_.get());
    result.push_back(positional_encoding_.get());
    for (auto& block : transformer_blocks_) result.push_back(block.get());
    result.push_back(final_fc_.get());
    return result;
}

template<typename T>
void Gpt2<T>::initialize_weights(tensor<T>* weights, T scale) {
    std::vector<T> host_data(weights->size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dist(0.0f, scale);
    for (size_t i = 0; i < weights->size(); ++i) {
        host_data[i] = dist(gen);
    }
    weights->upload(host_data.data());
}

// Explicit template instantiations

template class Gpt2<float>;

} // namespace dnn

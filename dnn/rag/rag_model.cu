#include "rag_model.cuh"
#include "document_store.cuh"
#include "../utils/common.cuh"
#include "../tokens/tokenizer.cuh"
#include "../core/tensor.cuh"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <limits>
#include <iostream>

namespace dnn {

template<typename T>
RAGModel<T>::RAGModel(std::shared_ptr<dnn::TrainingModel<T>> generator_model,
                    std::shared_ptr<dnn::TrainingModel<T>> embedding_model,
                    std::shared_ptr<DocumentStore> document_store,
                    std::shared_ptr<dnn::Tokenizer> tokenizer,
                    int max_context_length, int retrieval_top_k, bool training_enabled)
    : TrainingModel<T>(training_enabled),
      generator_model_(generator_model),
      embedding_model_(embedding_model),
      document_store_(document_store),
      tokenizer_(tokenizer),
      max_context_length_(max_context_length),
      retrieval_top_k_(retrieval_top_k) {
    
    if (!generator_model_) {
        throw std::invalid_argument("Generator model cannot be null");
    }
    if (!embedding_model_) {
        throw std::invalid_argument("Embedding model cannot be null");
    }
    if (!document_store_) {
        throw std::invalid_argument("Document store cannot be null");
    }
    if (!tokenizer_) {
        throw std::invalid_argument("Tokenizer cannot be null");
    }
    if (max_context_length <= 0) {
        throw std::invalid_argument("Max context length must be positive");
    }
    if (retrieval_top_k <= 0) {
        throw std::invalid_argument("Retrieval top-k must be positive");
    }
    
    // Create retriever layer
    retriever_layer_ = std::make_unique<RetrieverLayer<T>>(
        document_store_, embedding_model_, retrieval_top_k_, 0.0f, training_enabled
    );
    
    // Set default context template
    format_context_template_ = "Context: {}\n\nQuestion: {}\n\nAnswer:";
    
    // Note: The RAG model's optimizer should be set up externally
    // to work with the generator model's parameters
}

template<typename T>
void RAGModel<T>::train_step(const tensor<T>& input, const tensor<T>& target) {
    // Joint training for RAG: train both retriever and generator

    // Step 1: Forward pass through retriever to get retrieved documents
    tensor<T> retrieved = retriever_layer_->forward(input);

    // Step 2: Train the generator model with retrieved context
    generator_model_->train_step(retrieved, target);
    T generator_loss = generator_model_->loss();

    // Step 3: Train the retriever (embedding model) to improve retrieval
    // For joint training, we can train the retriever to better represent the input
    // This is a simplified approach - in practice, you'd use more sophisticated methods
    if (embedding_model_ && retriever_optimizer_) {
        // Train embedding model on the input to improve representations
        embedding_model_->train_step(input, input);  // Auto-encoder style training
        T retriever_loss = embedding_model_->loss();

        // Combine losses (weighted combination)
        this->current_loss_ = generator_loss * static_cast<T>(0.7f) + retriever_loss * static_cast<T>(0.3f);
    } else {
        // Fallback to generator-only training
        this->current_loss_ = generator_loss;
    }
}

template<typename T>
void RAGModel<T>::set_optimizer(std::unique_ptr<dnn::Optimizer<T>> optimizer) {
    // For now, just set the generator optimizer (maintain backward compatibility)
    // In a full implementation, you'd create separate optimizers for retriever and generator
    if (generator_model_) {
        generator_model_->set_optimizer(std::move(optimizer));
    }
}

template<typename T>
void RAGModel<T>::set_retriever_optimizer(std::unique_ptr<dnn::Optimizer<T>> optimizer) {
    retriever_optimizer_ = std::move(optimizer);
    // Note: In a full implementation, you'd set this on the embedding model
}

template<typename T>
void RAGModel<T>::set_generator_optimizer(std::unique_ptr<dnn::Optimizer<T>> optimizer) {
    if (generator_model_) {
        generator_model_->set_optimizer(std::move(optimizer));
    }
}

template<typename T>
void RAGModel<T>::train_step(const std::string& query, const std::string& target_answer) {
    // Convert strings to tensors
    std::vector<int> query_tokens = tokenizer_->encode(query, true);
    std::vector<int> target_tokens = tokenizer_->encode(target_answer, true);
    
    // Create input tensor
    tensor<int> input_tokens({1, static_cast<int>(query_tokens.size())});
    input_tokens.upload(query_tokens.data());
    
    // Create target tensor
    tensor<int> target_tokens_tensor({1, static_cast<int>(target_tokens.size())});
    target_tokens_tensor.upload(target_tokens.data());
    
    // Convert to model's data type
    tensor<T> input_float = dnn::utils::to_one_hot<T>(input_tokens, tokenizer_->vocab_size());
    tensor<T> target_float = dnn::utils::to_one_hot<T>(target_tokens_tensor, tokenizer_->vocab_size());
    
    // Train step
    train_step(input_float, target_float);
}

template<typename T>
tensor<T> RAGModel<T>::forward(const tensor<T>& input) {
    // Forward pass through retriever
    tensor<T> retrieved = retriever_layer_->forward(input);
    
    // Forward pass through generator
    return generator_model_->forward(retrieved);
}

template<typename T>
std::string RAGModel<T>::generate(const std::string& query, int max_length) {
    if (query.empty()) {
        return "";
    }

    // RAG Step 1: Retrieve relevant documents from knowledge base
    std::vector<std::pair<Document*, float>> retrieved = retriever_layer_->retrieve_for_query(query);

    // RAG Step 2: Build context from retrieved documents
    std::string context = build_context(query, retrieved);

    // RAG Step 3: Prepare input combining query + retrieved context
    tensor<T> current_input = prepare_rag_input(query, context);

    // RAG Step 4: Generate response autoregressively using retrieved context
    std::vector<int> generated_tokens;
    generated_tokens.reserve(max_length);

    const int eos_token_id = tokenizer_->get_eos_token_id();
    const int vocab_size = tokenizer_->vocab_size();

    // Track token frequencies for repetition penalty
    std::unordered_map<int, int> token_counts;

    for (int step = 0; step < max_length; ++step) {
        // Forward pass
        tensor<T> logits = forward(current_input);

        // Apply repetition penalty
        apply_repetition_penalty(logits, generated_tokens, token_counts, 1.1f);

        // Get next token using temperature + nucleus sampling
        int next_token = sample_next_token_nucleus(logits, vocab_size, 1.0f, 0.9f);

        // Check for EOS token
        if (next_token == eos_token_id) {
            break;
        }

        // Add token to generated sequence
        generated_tokens.push_back(next_token);

        // Update token counts for repetition penalty
        token_counts[next_token]++;

        // Prepare next input by appending the generated token
        current_input = append_token_to_input(std::move(current_input), next_token);
    }

    return tokenizer_->decode(generated_tokens, true);
}

template<typename T>
void RAGModel<T>::add_knowledge_base(const std::vector<std::pair<std::string, std::string>>& documents) {
    if (documents.empty()) {
        return;
    }
    
    document_store_->add_documents(documents);
    update_embeddings();
}

template<typename T>
void RAGModel<T>::update_embeddings() {
    if (!document_store_ || !embedding_model_) {
        throw std::runtime_error("Document store or embedding model not available");
    }

    if constexpr (std::is_same_v<T, float>) {
        update_embeddings_float(document_store_.get(), embedding_model_);
    } else if constexpr (std::is_same_v<T, __half>) {
        update_embeddings_half(document_store_.get(), embedding_model_);
    } else {
        throw std::runtime_error("Unsupported tensor type for embedding updates");
    }
}

template<typename T>
void RAGModel<T>::update_document_embeddings() {
    if (!document_store_ || !embedding_model_) {
        throw std::runtime_error("Document store or embedding model not available");
    }

    // Wire the embedding model to the document store
    document_store_->set_embedding_model(embedding_model_);

    // Update embeddings using the document store's internal method
    document_store_->update_embeddings_internal();
}

template<typename T>
std::string RAGModel<T>::build_context(const std::string& query, 
                                     const std::vector<std::pair<Document*, float>>& retrieved) {
    if (retrieved.empty()) {
        return format_context_template_;
    }
    
    // Extract content from retrieved documents
    std::vector<std::string> retrieved_content = extract_retrieved_content(retrieved);
    
    // Format context
    return dnn::utils::TextProcessor::format_rag_context(
        query, retrieved_content, format_context_template_
    );
}

template<typename T>
void RAGModel<T>::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for saving: " + path);
    }
    
    // Save configuration
    out.write(reinterpret_cast<const char*>(&max_context_length_), sizeof(max_context_length_));
    out.write(reinterpret_cast<const char*>(&retrieval_top_k_), sizeof(retrieval_top_k_));
    
    // Save context template
    size_t template_len = format_context_template_.length();
    out.write(reinterpret_cast<const char*>(&template_len), sizeof(template_len));
    out.write(format_context_template_.c_str(), template_len);
    
    // Save retriever layer
    if (retriever_layer_) {
        retriever_layer_->save(out);
    }
    
    // Note: Generator and embedding models should be saved separately
    // as they are shared pointers and might be used elsewhere
}

template<typename T>
std::unique_ptr<RAGModel<T>> RAGModel<T>::load(const std::string& path, bool training_enabled) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for loading: " + path);
    }
    
    // Load configuration
    int max_context_length, retrieval_top_k;
    in.read(reinterpret_cast<char*>(&max_context_length), sizeof(max_context_length));
    in.read(reinterpret_cast<char*>(&retrieval_top_k), sizeof(retrieval_top_k));
    
    // Load context template
    size_t template_len;
    in.read(reinterpret_cast<char*>(&template_len), sizeof(template_len));
    std::string format_context_template(template_len, '\0');
    in.read(&format_context_template[0], template_len);
    
    // Note: This is a simplified load method. In practice, you would need to provide
    // the actual models, document store, and tokenizer externally
    throw std::runtime_error("RAGModel::load() requires external model setup - use constructor instead");
}

template<typename T>
std::vector<dnn::BaseLayer*> RAGModel<T>::layers() {
    std::vector<dnn::BaseLayer*> all_layers;
    
    // Add retriever layer
    if (retriever_layer_) {
        all_layers.push_back(retriever_layer_.get());
    }
    
    // Add generator model layers
    if (generator_model_) {
        auto gen_layers = generator_model_->layers();
        all_layers.insert(all_layers.end(), gen_layers.begin(), gen_layers.end());
    }
    
    return all_layers;
}

template<typename T>
int RAGModel<T>::sample_next_token_argmax(const tensor<T>& logits, int vocab_size) {
    // Download logits data
    std::vector<T> logits_data(logits.size());
    logits.download(logits_data.data());

    // Find argmax (highest probability token)
    int best_token = 0;
    float best_score = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < vocab_size && i < static_cast<int>(logits_data.size()); ++i) {
        float score = static_cast<float>(logits_data[i]);
        if (score > best_score) {
            best_score = score;
            best_token = i;
        }
    }

    return best_token;
}

template<typename T>
int RAGModel<T>::sample_next_token_temperature(const tensor<T>& logits, int vocab_size, float temperature) {
    // Download logits data
    std::vector<T> logits_data(logits.size());
    logits.download(logits_data.data());

    // Apply temperature scaling
    std::vector<float> scaled_logits;
    scaled_logits.reserve(vocab_size);

    for (int i = 0; i < vocab_size && i < static_cast<int>(logits_data.size()); ++i) {
        float logit = static_cast<float>(logits_data[i]);
        scaled_logits.push_back(logit / temperature);
    }

    // Compute softmax probabilities
    std::vector<float> probabilities = softmax(scaled_logits);

    // Sample from the distribution
    return sample_from_distribution(probabilities);
}

template<typename T>
int RAGModel<T>::sample_next_token_topk(const tensor<T>& logits, int vocab_size, float temperature, int top_k) {
    // Download logits data
    std::vector<T> logits_data(logits.size());
    logits.download(logits_data.data());

    // Create token-probability pairs
    std::vector<std::pair<float, int>> token_probs;
    token_probs.reserve(vocab_size);

    for (int i = 0; i < vocab_size && i < static_cast<int>(logits_data.size()); ++i) {
        float logit = static_cast<float>(logits_data[i]);
        token_probs.emplace_back(logit / temperature, i);
    }

    // Sort by logit value (descending)
    std::partial_sort(token_probs.begin(),
                     token_probs.begin() + std::min(top_k, static_cast<int>(token_probs.size())),
                     token_probs.end(),
                     std::greater<std::pair<float, int>>());

    // Keep only top-k tokens
    if (token_probs.size() > static_cast<size_t>(top_k)) {
        token_probs.resize(top_k);
    }

    // Extract logits for softmax
    std::vector<float> top_logits;
    top_logits.reserve(token_probs.size());
    for (const auto& pair : token_probs) {
        top_logits.push_back(pair.first);
    }

    // Compute softmax probabilities
    std::vector<float> probabilities = softmax(top_logits);

    // Sample from top-k distribution
    int selected_idx = sample_from_distribution(probabilities);
    return token_probs[selected_idx].second;
}

template<typename T>
void RAGModel<T>::apply_repetition_penalty(tensor<T>& logits, const std::vector<int>& generated_tokens,
                                         std::unordered_map<int, int>& token_counts, float penalty) {
    if (generated_tokens.empty() || penalty <= 0.0f) {
        return;
    }

    // Update token counts for recent tokens (last 10 tokens)
    size_t start_idx = generated_tokens.size() > 10 ? generated_tokens.size() - 10 : 0;
    for (size_t i = start_idx; i < generated_tokens.size(); ++i) {
        token_counts[generated_tokens[i]]++;
    }

    // Download logits for modification
    std::vector<T> logits_data(logits.size());
    logits.download(logits_data.data());

    // Apply penalty to recently used tokens
    for (const auto& pair : token_counts) {
        int token_id = pair.first;
        int count = pair.second;

        if (token_id >= 0 && token_id < static_cast<int>(logits_data.size())) {
            // Apply penalty: divide by (count * penalty)
            float current_logit = static_cast<float>(logits_data[token_id]);
            if (current_logit < 0) {
                logits_data[token_id] = static_cast<T>(current_logit * count * penalty);
            } else {
                logits_data[token_id] = static_cast<T>(current_logit / (count * penalty));
            }
        }
    }

    // Upload modified logits
    logits.upload(logits_data.data());
}

template<typename T>
int RAGModel<T>::sample_next_token_nucleus(const tensor<T>& logits, int vocab_size, float temperature, float top_p) {
    // Download logits data
    std::vector<T> logits_data(logits.size());
    logits.download(logits_data.data());

    // Apply temperature scaling
    std::vector<std::pair<float, int>> token_probs;
    token_probs.reserve(vocab_size);

    for (int i = 0; i < vocab_size && i < static_cast<int>(logits_data.size()); ++i) {
        float logit = static_cast<float>(logits_data[i]);
        token_probs.emplace_back(logit / temperature, i);
    }

    // Sort by logit value (descending)
    std::sort(token_probs.begin(), token_probs.end(),
              std::greater<std::pair<float, int>>());

    // Compute softmax for the sorted tokens
    std::vector<float> sorted_logits;
    sorted_logits.reserve(token_probs.size());
    for (const auto& pair : token_probs) {
        sorted_logits.push_back(pair.first);
    }

    std::vector<float> probabilities = softmax(sorted_logits);

    // Find nucleus (cumulative probability >= top_p)
    float cumulative_prob = 0.0f;
    size_t nucleus_size = 0;

    for (size_t i = 0; i < probabilities.size(); ++i) {
        cumulative_prob += probabilities[i];
        nucleus_size = i + 1;
        if (cumulative_prob >= top_p) {
            break;
        }
    }

    // Keep only nucleus tokens
    if (nucleus_size < token_probs.size()) {
        token_probs.resize(nucleus_size);
        probabilities.resize(nucleus_size);

        // Renormalize probabilities
        float prob_sum = 0.0f;
        for (float prob : probabilities) {
            prob_sum += prob;
        }
        for (float& prob : probabilities) {
            prob /= prob_sum;
        }
    }

    // Sample from nucleus
    int selected_idx = sample_from_distribution(probabilities);
    return token_probs[selected_idx].second;
}

template<typename T>
std::vector<float> RAGModel<T>::softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());

    // Find max logit for numerical stability
    float max_logit = *std::max_element(logits.begin(), logits.end());

    // Compute exp and sum
    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }

    // Normalize
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum_exp;
    }

    return probs;
}

template<typename T>
int RAGModel<T>::sample_from_distribution(const std::vector<float>& probabilities) {
    // Generate random number between 0 and 1
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float rand_val = dist(gen);

    // Sample from cumulative distribution
    float cumulative = 0.0f;
    for (size_t i = 0; i < probabilities.size(); ++i) {
        cumulative += probabilities[i];
        if (rand_val <= cumulative) {
            return static_cast<int>(i);
        }
    }

    // Fallback (should not happen with proper normalization)
    return 0;
}

template<typename T>
tensor<T> RAGModel<T>::append_token_to_input(tensor<T> current_input, int new_token) {
    // Get current input dimensions
    const auto& shape = current_input.shape();
    if (shape.size() != 2) {
        throw std::runtime_error("Input tensor must be 2D [batch_size, seq_len]");
    }

    int batch_size = shape[0];
    int seq_len = shape[1];

    // Create new input tensor with increased sequence length
    tensor<T> new_input({batch_size, seq_len + 1});

    // Copy existing data
    std::vector<T> current_data(current_input.size());
    std::vector<T> new_data(new_input.size());

    current_input.download(current_data.data());

    // Copy all existing tokens
    for (int i = 0; i < current_input.size(); ++i) {
        new_data[i] = current_data[i];
    }

    // Add new token at the end
    new_data[current_input.size()] = static_cast<T>(new_token);

    new_input.upload(new_data.data());
    return new_input;
}

template<typename T>
tensor<T> RAGModel<T>::prepare_rag_input(const std::string& query, const std::string& context) {
    // Combine context and query
    std::string full_input = context + " " + query;

    // Tokenize
    std::vector<int> tokens = tokenizer_->encode(full_input, true);

    // Truncate if too long
    if (static_cast<int>(tokens.size()) > max_context_length_) {
        tokens.resize(max_context_length_);
    }

    // Create input tensor with proper lifecycle management
    tensor<int> input_tokens({1, static_cast<int>(tokens.size())});
    input_tokens.upload(tokens.data());

    // Convert to model's data type and return (move semantics)
    return dnn::utils::to_one_hot<T>(std::move(input_tokens), tokenizer_->vocab_size());
}

template<typename T>
std::vector<std::string> RAGModel<T>::extract_retrieved_content(
    const std::vector<std::pair<Document*, float>>& retrieved) {
    
    std::vector<std::string> content;
    content.reserve(retrieved.size());
    
    for (const auto& [doc, score] : retrieved) {
        if (doc && !doc->content.empty()) {
            content.push_back(doc->content);
        }
    }
    
    return content;
}

// Helper functions to avoid template issues in header
void update_embeddings_float(DocumentStore* store, std::shared_ptr<dnn::TrainingModel<float>> embedding_model) {
    if (!embedding_model) {
        throw std::invalid_argument("Embedding model cannot be null");
    }
    
    store->update_embeddings_internal();
}

void update_embeddings_half(DocumentStore* store, std::shared_ptr<dnn::TrainingModel<__half>> embedding_model) {
    if (!embedding_model) {
        throw std::invalid_argument("Embedding model cannot be null");
    }

    store->update_embeddings_internal();
}

template<typename T>
void RAGModel<T>::train(const std::vector<std::pair<tensor<T>, tensor<T>>>& train_data,
                        const std::vector<std::pair<tensor<T>, tensor<T>>>& val_data,
                        int epochs, int batch_size, bool verbose) {
    if (train_data.empty()) {
        throw std::runtime_error("Training data cannot be empty");
    }

    // Initialize training state
    current_epoch_ = 0;
    best_val_loss_ = std::numeric_limits<T>::max();
    training_losses_.clear();
    validation_losses_.clear();

    std::cout << "Starting RAG training for " << epochs << " epochs..." << std::endl;
    std::cout << "Training samples: " << train_data.size() << std::endl;
    std::cout << "Validation samples: " << val_data.size() << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        current_epoch_ = epoch;

        // Training phase
        T epoch_train_loss = 0.0f;
        int num_train_batches = 0;

        // Create shuffled indices instead of copying tensors
        std::vector<size_t> indices(train_data.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(),
                    std::default_random_engine(std::random_device{}()));

        // Train in batches using shuffled indices
        for (size_t i = 0; i < indices.size(); i += batch_size) {
            size_t batch_end = std::min(i + batch_size, indices.size());
            T batch_loss = 0.0f;

            // Process batch
            for (size_t j = i; j < batch_end; ++j) {
                size_t idx = indices[j];
                const auto& [input, target] = train_data[idx];
                train_step(input, target);
                batch_loss += this->current_loss_;
            }

            batch_loss /= static_cast<T>(batch_end - i);
            epoch_train_loss += batch_loss;
            num_train_batches++;
        }

        epoch_train_loss /= static_cast<T>(num_train_batches);
        training_losses_.push_back(epoch_train_loss);

        // Validation phase
        T val_loss = validate(val_data);
        validation_losses_.push_back(val_loss);

        // Learning rate scheduling (simple decay)
        if (epoch > 0 && epoch % 10 == 0) {
            current_learning_rate_ *= 0.9f;
            set_learning_rate(current_learning_rate_);
        }

        // Logging
        if (verbose) {
            std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                      << " - Train Loss: " << static_cast<float>(epoch_train_loss)
                      << " - Val Loss: " << static_cast<float>(val_loss)
                      << " - LR: " << static_cast<float>(current_learning_rate_) << std::endl;
        }

        // Checkpoint saving (save best model)
        if (val_loss < best_val_loss_) {
            best_val_loss_ = val_loss;
            save_checkpoint("rag_model_best.bin", epoch, val_loss);
        }
    }

    std::cout << "Training completed! Best validation loss: " << static_cast<float>(best_val_loss_) << std::endl;
}

template<typename T>
void RAGModel<T>::train_with_text_data(const std::vector<std::pair<std::string, std::string>>& train_queries,
                                      const std::vector<std::pair<std::string, std::string>>& val_queries,
                                      int epochs, int batch_size, bool verbose) {
    // Convert text data to tensor format
    std::vector<std::pair<tensor<T>, tensor<T>>> train_tensors;
    std::vector<std::pair<tensor<T>, tensor<T>>> val_tensors;

    for (const auto& [query, answer] : train_queries) {
        std::vector<int> query_tokens = tokenizer_->encode(query, true);
        std::vector<int> answer_tokens = tokenizer_->encode(answer, true);

        tensor<int> query_tensor({1, static_cast<int>(query_tokens.size())});
        tensor<int> answer_tensor({1, static_cast<int>(answer_tokens.size())});

        query_tensor.upload(query_tokens.data());
        answer_tensor.upload(answer_tokens.data());

        tensor<T> query_float = dnn::utils::to_one_hot<T>(query_tensor, tokenizer_->vocab_size());
        tensor<T> answer_float = dnn::utils::to_one_hot<T>(answer_tensor, tokenizer_->vocab_size());

        train_tensors.emplace_back(std::move(query_float), std::move(answer_float));
    }

    for (const auto& [query, answer] : val_queries) {
        std::vector<int> query_tokens = tokenizer_->encode(query, true);
        std::vector<int> answer_tokens = tokenizer_->encode(answer, true);

        tensor<int> query_tensor({1, static_cast<int>(query_tokens.size())});
        tensor<int> answer_tensor({1, static_cast<int>(answer_tokens.size())});

        query_tensor.upload(query_tokens.data());
        answer_tensor.upload(answer_tokens.data());

        tensor<T> query_float = dnn::utils::to_one_hot<T>(query_tensor, tokenizer_->vocab_size());
        tensor<T> answer_float = dnn::utils::to_one_hot<T>(answer_tensor, tokenizer_->vocab_size());

        val_tensors.emplace_back(std::move(query_float), std::move(answer_float));
    }

    // Train with tensor data
    train(train_tensors, val_tensors, epochs, batch_size, verbose);
}

template<typename T>
T RAGModel<T>::validate(const std::vector<std::pair<tensor<T>, tensor<T>>>& val_data) {
    if (val_data.empty()) {
        return static_cast<T>(0.0f);
    }

    T total_loss = 0.0f;
    for (const auto& [input, target] : val_data) {
        // Forward pass (no gradients for validation)
        tensor<T> output = forward(input);

        // Compute loss (simplified - using MSE for now)
        // In a real implementation, you'd use proper loss functions
        T loss = 0.0f;
        if (output.size() == target.size()) {
            std::vector<T> output_data(output.size());
            std::vector<T> target_data(target.size());
            output.download(output_data.data());
            target.download(target_data.data());

            for (size_t i = 0; i < output_data.size(); ++i) {
                T diff = output_data[i] - target_data[i];
                loss += diff * diff;
            }
            loss /= static_cast<T>(output_data.size());
        }

        total_loss += loss;
    }

    return total_loss / static_cast<T>(val_data.size());
}

template<typename T>
void RAGModel<T>::save_checkpoint(const std::string& path, int epoch, T loss) {
    // Basic checkpoint saving - save model weights and training state
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Warning: Could not save checkpoint to " << path << std::endl;
        return;
    }

    // Save training state
    out.write(reinterpret_cast<const char*>(&epoch), sizeof(epoch));
    out.write(reinterpret_cast<const char*>(&loss), sizeof(loss));
    out.write(reinterpret_cast<const char*>(&current_learning_rate_), sizeof(current_learning_rate_));

    // Save model weights (simplified - in practice you'd save all model parameters)
    // Note: We can't call save on the generator model here because it expects a path,
    // not an ofstream. In a full implementation, you'd save each model separately.
    if (generator_model_) {
        std::string generator_path = path + "_generator.bin";
        generator_model_->save(generator_path);
    }

    if (embedding_model_) {
        std::string embedding_path = path + "_embedding.bin";
        embedding_model_->save(embedding_path);
    }

    std::cout << "Checkpoint saved to " << path << " (epoch: " << epoch << ", loss: " << static_cast<float>(loss) << ")" << std::endl;
}

template<typename T>
bool RAGModel<T>::load_checkpoint(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Warning: Could not load checkpoint from " << path << std::endl;
        return false;
    }

    // Load training state
    in.read(reinterpret_cast<char*>(&current_epoch_), sizeof(current_epoch_));
    in.read(reinterpret_cast<char*>(&best_val_loss_), sizeof(best_val_loss_));
    in.read(reinterpret_cast<char*>(&current_learning_rate_), sizeof(current_learning_rate_));

    // Load model weights (simplified - models don't have load methods in base class)
    // Note: In a full implementation, you'd need to implement load methods for each model
    // or handle model loading externally
    if (generator_model_) {
        std::string generator_path = path + "_generator.bin";
        std::ifstream gen_in(generator_path, std::ios::binary);
        if (gen_in) {
            std::cout << "Note: Generator model loading not implemented (no load method in base class)" << std::endl;
        }
    }

    if (embedding_model_) {
        std::string embedding_path = path + "_embedding.bin";
        std::ifstream emb_in(embedding_path, std::ios::binary);
        if (emb_in) {
            std::cout << "Note: Embedding model loading not implemented (no load method in base class)" << std::endl;
        }
    }

    std::cout << "Checkpoint loaded from " << path << std::endl;
    return true;
}

template<typename T>
void RAGModel<T>::set_learning_rate(T lr) {
    current_learning_rate_ = lr;
    // Update optimizer learning rates if available
    if (generator_optimizer_) {
        // Note: This would require optimizer to support LR updates
        // For now, we'll just store the value
    }
}

template<typename T>
T RAGModel<T>::get_learning_rate() const {
    return current_learning_rate_;
}

// Explicit template instantiations
template class RAGModel<float>;
template class RAGModel<__half>;

} // namespace dnn

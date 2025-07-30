#include "resnet32.cuh"
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <iostream>

namespace dnn {

template<typename T>
void ResNet32<T>::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open file for saving: " + path);
    // Save model metadata
    int num_blocks1 = static_cast<int>(res_stage1_.size());
    int num_blocks2 = static_cast<int>(res_stage2_.size());
    int num_blocks3 = static_cast<int>(res_stage3_.size());
    out.write(reinterpret_cast<const char*>(&num_blocks1), sizeof(num_blocks1));
    out.write(reinterpret_cast<const char*>(&num_blocks2), sizeof(num_blocks2));
    out.write(reinterpret_cast<const char*>(&num_blocks3), sizeof(num_blocks3));
    conv1_.save(out);
    bn1_.save(out);
    act1_.save(out);
    for (const auto& block : res_stage1_) block->save(out);
    for (const auto& block : res_stage2_) block->save(out);
    for (const auto& block : res_stage3_) block->save(out);
    global_pool_.save(out);
    fc_.save(out);
}

template<typename T>
std::unique_ptr<ResNet32<T>> ResNet32<T>::load(const std::string& path, bool training_enabled) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open file for loading: " + path);
    int num_blocks1, num_blocks2, num_blocks3;
    in.read(reinterpret_cast<char*>(&num_blocks1), sizeof(num_blocks1));
    in.read(reinterpret_cast<char*>(&num_blocks2), sizeof(num_blocks2));
    in.read(reinterpret_cast<char*>(&num_blocks3), sizeof(num_blocks3));
    auto model = std::make_unique<ResNet32<T>>(10, 0.1, 0.9, training_enabled);
    if (num_blocks1 != static_cast<int>(model->res_stage1_.size()) ||
        num_blocks2 != static_cast<int>(model->res_stage2_.size()) ||
        num_blocks3 != static_cast<int>(model->res_stage3_.size()))
        throw std::runtime_error("Block count mismatch in ResNet32::load");
    model->conv1_.load(in);
    model->bn1_.load(in);
    model->act1_.load(in);
    for (auto& block : model->res_stage1_) block->load(in);
    for (auto& block : model->res_stage2_) block->load(in);
    for (auto& block : model->res_stage3_) block->load(in);
    model->global_pool_.load(in);
    model->fc_.load(in);
    return model;
}

template class ResNet32<float>;
// template class ResNet32<__half>;

} // namespace dnn 
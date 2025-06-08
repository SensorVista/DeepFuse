#pragma once

#include <dnn/models/model.cuh>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace dnn {

class OnnxLoader {
public:

enum class DType : uint8_t {
    Float32 = 0,
    Half = 1,
    Int8 = 2,
    UInt8 = 3,
    BFloat16 = 4
};

struct Attribute {
    std::string key;
    float value; // Extend later with variant
};

struct SerializedTensor {
    std::string name;
    DType dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> raw_data;
};

struct SerializedLayer {
    std::string type;
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<Attribute> attributes;
};

struct SerializedModel {
    std::string name;
    std::vector<SerializedTensor> tensors;
    std::vector<SerializedLayer> layers;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::string author;
    std::string framework;
    std::string timestamp;
};

// Load and save ONNX v2 models with SerializedModel
static void load_onyx_v2(const std::string& path, SerializedModel& model);
static void save_onyx_v2(const SerializedModel& model, const std::string& path);

// Load and save ONNX v2 models with Model<T>
template<typename T>
static Model<T>* load_onyx_v2(const std::string& path);

template<typename T>
static void save_onyx_v2(const Model<T>* model, const std::string& path);

};

} // namespace dnn

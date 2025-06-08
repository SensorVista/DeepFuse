#include <gtest/gtest.h>

#include <dnn/utils/onnx_loader.cuh>
#include <dnn/core/tensor.cuh>
#include <dnn/models/training_model.cuh>

#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <memory>

namespace dnn {
namespace test {

class ONNXLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple ONNX test file
        create_test_onnx_file();
    }

    void TearDown() override {
        // Clean up test file
        std::filesystem::remove(onnx_filepath);
    }

    void write_string(std::ofstream& out, const std::string& str) {
        uint32_t len = static_cast<uint32_t>(str.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(str.data(), len);
    }

    void write_tensor(std::ofstream& out, const OnnxLoader::SerializedTensor& tensor) {
        write_string(out, tensor.name);
        uint8_t dtype = static_cast<uint8_t>(tensor.dtype);
        out.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
        uint32_t dim_count = static_cast<uint32_t>(tensor.shape.size());
        out.write(reinterpret_cast<const char*>(&dim_count), sizeof(dim_count));
        out.write(reinterpret_cast<const char*>(tensor.shape.data()), dim_count * sizeof(int64_t));
        uint32_t byte_size = static_cast<uint32_t>(tensor.raw_data.size());
        out.write(reinterpret_cast<const char*>(&byte_size), sizeof(byte_size));
        out.write(reinterpret_cast<const char*>(tensor.raw_data.data()), byte_size);
    }

    void write_layer(std::ofstream& out, const OnnxLoader::SerializedLayer& layer) {
        write_string(out, layer.type);
        write_string(out, layer.name);
        
        uint32_t input_count = static_cast<uint32_t>(layer.inputs.size());
        out.write(reinterpret_cast<const char*>(&input_count), sizeof(input_count));
        for (const auto& input : layer.inputs) {
            write_string(out, input);
        }
        
        uint32_t output_count = static_cast<uint32_t>(layer.outputs.size());
        out.write(reinterpret_cast<const char*>(&output_count), sizeof(output_count));
        for (const auto& output : layer.outputs) {
            write_string(out, output);
        }
        
        uint32_t attr_count = static_cast<uint32_t>(layer.attributes.size());
        out.write(reinterpret_cast<const char*>(&attr_count), sizeof(attr_count));
        for (const auto& attr : layer.attributes) {
            write_string(out, attr.key);
            out.write(reinterpret_cast<const char*>(&attr.value), sizeof(attr.value));
        }
    }

    void create_test_onnx_file() {
        std::ofstream file(onnx_filepath, std::ios::binary);
        
        // Write magic number ('TYNX')
        uint32_t magic = 0x584E5954;
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        
        // Write version
        uint32_t version = 2;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));

        // Write model metadata
        write_string(file, "test_model");
        write_string(file, "test_author");
        write_string(file, "test_framework");
        write_string(file, "test_timestamp");

        // Create test tensors
        std::vector<OnnxLoader::SerializedTensor> tensors;
        
        // Input tensor
        OnnxLoader::SerializedTensor input_tensor;
        input_tensor.name = "input";
        input_tensor.dtype = OnnxLoader::DType::Float32;
        input_tensor.shape = {1, 3, 32, 32};  // Batch, Channels, Height, Width
        input_tensor.raw_data.resize(4 * 3 * 32 * 32, 0);  // Initialize with zeros
        tensors.push_back(std::move(input_tensor));

        // Conv layer weights
        OnnxLoader::SerializedTensor conv_weight;
        conv_weight.name = "conv1_weight";
        conv_weight.dtype = OnnxLoader::DType::Float32;
        conv_weight.shape = {16, 3, 3, 3};  // Out channels, In channels, Kernel height, Kernel width
        conv_weight.raw_data.resize(4 * 16 * 3 * 3 * 3, 0);
        tensors.push_back(std::move(conv_weight));

        // Conv layer bias
        OnnxLoader::SerializedTensor conv_bias;
        conv_bias.name = "conv1_bias";
        conv_bias.dtype = OnnxLoader::DType::Float32;
        conv_bias.shape = {16};  // Out channels
        conv_bias.raw_data.resize(4 * 16, 0);
        tensors.push_back(std::move(conv_bias));

        // Write tensor count and tensors
        uint32_t tensor_count = static_cast<uint32_t>(tensors.size());
        file.write(reinterpret_cast<const char*>(&tensor_count), sizeof(tensor_count));
        for (const auto& tensor : tensors) {
            write_tensor(file, tensor);
        }

        // Create test layers
        std::vector<OnnxLoader::SerializedLayer> layers;
        
        // Conv layer
        OnnxLoader::SerializedLayer conv_layer;
        conv_layer.type = "Conv2D";
        conv_layer.name = "conv1";
        conv_layer.inputs = {"input", "conv1_weight", "conv1_bias"};
        conv_layer.outputs = {"conv1_output"};
        conv_layer.attributes.push_back({"stride", 1.0f});
        conv_layer.attributes.push_back({"padding", 1.0f});
        layers.push_back(std::move(conv_layer));

        // ReLU layer
        OnnxLoader::SerializedLayer relu_layer;
        relu_layer.type = "ReLU";
        relu_layer.name = "relu1";
        relu_layer.inputs = {"conv1_output"};
        relu_layer.outputs = {"relu1_output"};
        layers.push_back(std::move(relu_layer));

        // Write layer count and layers
        uint32_t layer_count = static_cast<uint32_t>(layers.size());
        file.write(reinterpret_cast<const char*>(&layer_count), sizeof(layer_count));
        for (const auto& layer : layers) {
            write_layer(file, layer);
        }

        // Write input/output lists
        uint32_t input_count = 1;
        file.write(reinterpret_cast<const char*>(&input_count), sizeof(input_count));
        write_string(file, "input");

        uint32_t output_count = 1;
        file.write(reinterpret_cast<const char*>(&output_count), sizeof(output_count));
        write_string(file, "relu1_output");
    }

    const std::string onnx_filepath = "test_model.onnx";
};

TEST_F(ONNXLoaderTest, LoadModelSuccess) {
    OnnxLoader loader;
    OnnxLoader::SerializedModel model;
    ASSERT_NO_THROW(loader.load_onyx_v2(onnx_filepath, model));
    
    // Basic verification that model loaded
    EXPECT_FALSE(model.name.empty());
    EXPECT_EQ(model.tensors.size(), 3);  // input, conv1_weight, conv1_bias
    EXPECT_EQ(model.layers.size(), 2);   // Conv2D and ReLU
}

TEST_F(ONNXLoaderTest, LoadInvalidFile) {
    OnnxLoader loader;
    OnnxLoader::SerializedModel model;
    EXPECT_THROW(
        loader.load_onyx_v2("nonexistent_file.onnx", model),
        std::runtime_error
    );
}

TEST_F(ONNXLoaderTest, ModelStructure) {
    OnnxLoader loader;
    OnnxLoader::SerializedModel model;
    loader.load_onyx_v2(onnx_filepath, model);
    
    // Verify model structure
    EXPECT_FALSE(model.name.empty());
    EXPECT_EQ(model.inputs.size(), 1);    // We created one input
    EXPECT_EQ(model.outputs.size(), 1);   // We created one output
    
    EXPECT_EQ(model.tensors.size(), 3);   // input, conv1_weight, conv1_bias
    EXPECT_EQ(model.layers.size(), 2);    // Conv2D and ReLU
}

TEST_F(ONNXLoaderTest, TrainingModelConversion) {
    OnnxLoader loader;
    auto model = loader.load_onyx_v2<float>(onnx_filepath);
    
    // Basic verification that model loaded
    EXPECT_NE(model, nullptr);
    EXPECT_EQ(model->layers().size(), 2);  // Conv2D and ReLU layers
}

TEST_F(ONNXLoaderTest, LoadModelWithData) {
    OnnxLoader loader;
    OnnxLoader::SerializedModel model;
    ASSERT_NO_THROW(loader.load_onyx_v2(onnx_filepath, model));
    
    // Verify model metadata
    EXPECT_EQ(model.name, "test_model");
    EXPECT_EQ(model.author, "test_author");
    EXPECT_EQ(model.framework, "test_framework");
    
    // Verify tensors
    EXPECT_EQ(model.tensors.size(), 3);
    EXPECT_EQ(model.tensors[0].name, "input");
    EXPECT_EQ(model.tensors[0].shape, std::vector<int64_t>({1, 3, 32, 32}));
    EXPECT_EQ(model.tensors[1].name, "conv1_weight");
    EXPECT_EQ(model.tensors[1].shape, std::vector<int64_t>({16, 3, 3, 3}));
    EXPECT_EQ(model.tensors[2].name, "conv1_bias");
    EXPECT_EQ(model.tensors[2].shape, std::vector<int64_t>({16}));
    
    // Verify layers
    EXPECT_EQ(model.layers.size(), 2);
    EXPECT_EQ(model.layers[0].type, "Conv2D");
    EXPECT_EQ(model.layers[0].name, "conv1");
    EXPECT_EQ(model.layers[0].inputs.size(), 3);
    EXPECT_EQ(model.layers[0].outputs.size(), 1);
    EXPECT_EQ(model.layers[1].type, "ReLU");
    EXPECT_EQ(model.layers[1].name, "relu1");
    
    // Verify inputs/outputs
    EXPECT_EQ(model.inputs.size(), 1);
    EXPECT_EQ(model.inputs[0], "input");
    EXPECT_EQ(model.outputs.size(), 1);
    EXPECT_EQ(model.outputs[0], "relu1_output");
}

TEST_F(ONNXLoaderTest, TrainingModelWithData) {
    OnnxLoader loader;
    auto model = loader.load_onyx_v2<float>(onnx_filepath);
    
    // Basic verification that model loaded
    EXPECT_NE(model, nullptr);
    EXPECT_EQ(model->layers().size(), 2);  // Conv + ReLU
    
    // Verify layer types
    const auto& layers = model->layers();
    EXPECT_EQ(layers[0]->name(), "Conv");
    EXPECT_EQ(layers[1]->name(), "Activation(ReLU)");
}

} // namespace test
} // namespace dnn
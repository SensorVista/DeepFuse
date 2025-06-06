#include "onnx_loader.cuh"
#include "dnn/models/training_model.cuh"
#include "dnn/layers/conv_layer.cuh"
#include "dnn/layers/activation_layer.cuh"
#include "dnn/layers/fully_connected_layer.cuh"
#include "dnn/layers/pooling_layer.cuh"
#include "dnn/layers/flatten_layer.cuh"

#include <fstream>  
#include <stdexcept>    
#include <unordered_map>

namespace dnn {

TrainingModel<float>* OnnxLoader::load_onyx_v2(const std::string& path) {
    SerializedModel smodel;
    load_onyx_v2(path, smodel);

    auto* model = new TrainingModel<float>();

    std::unordered_map<std::string, tensor<float>> tensor_map;
    for (const auto& t : smodel.tensors) {
        std::vector<int> shape(t.shape.begin(), t.shape.end());
        tensor<float> temp(shape);
        temp.upload(reinterpret_cast<const float*>(t.raw_data.data()));
        tensor_map.emplace(t.name, std::move(temp));
    }

    for (const auto& layer : smodel.layers) {
        if (layer.type == "Conv2D") {
            tensor<float> weight = std::move(tensor_map.at(layer.inputs[1]));
            tensor<float> bias   = std::move(tensor_map.at(layer.inputs[2]));

            int out_ch = weight.shape()[0];
            int in_ch  = weight.shape()[1];
            int k      = weight.shape()[2];

            int stride = 1, padding = 0;
            for (const auto& attr : layer.attributes) {
                if (attr.key == "stride")  stride  = static_cast<int>(attr.value);
                if (attr.key == "padding") padding = static_cast<int>(attr.value);
            }

            auto conv_layer = std::make_unique<ConvLayer<float>>(in_ch, out_ch, std::vector<int>{k, k}, stride, padding);
            if (auto* conv_ptr = dynamic_cast<LayerWeightBias<float>*>(conv_layer.get())) {
                conv_ptr->weights()->copy_from(weight);
                conv_ptr->bias()->copy_from(bias);
            }

            model->add_layer(std::move(conv_layer));
        } 
        else if (layer.type == "ReLU") {
            model->add_layer(std::make_unique<ActivationLayer<float>>(ActivationType::ReLU));
        } 
        else if (layer.type == "Gemm" || layer.type == "Linear") {
            tensor<float> weight = std::move(tensor_map.at(layer.inputs[1]));
            tensor<float> bias   = std::move(tensor_map.at(layer.inputs[2]));

            int in_f  = weight.shape()[1];
            int out_f = weight.shape()[0];

            auto fc_layer = std::make_unique<FullyConnectedLayer<float>>(in_f, out_f);
            if (auto* fc_ptr = dynamic_cast<LayerWeightBias<float>*>(fc_layer.get())) {
                fc_ptr->weights()->copy_from(weight);
                fc_ptr->bias()->copy_from(bias);
            }

            model->add_layer(std::move(fc_layer));
        } 
        else if (layer.type == "MaxPool" || layer.type == "AveragePool") {
            int kernel_size = 2, stride = 2;
            for (const auto& attr : layer.attributes) {
                if (attr.key == "kernel_size") kernel_size = static_cast<int>(attr.value);
                if (attr.key == "stride")      stride      = static_cast<int>(attr.value);
            }

            PoolingType type = (layer.type == "MaxPool") ? PoolingType::Max : PoolingType::Average;
            model->add_layer(std::make_unique<PoolingLayer<float>>(type, kernel_size, stride));
        } 
        else if (layer.type == "Flatten") {
            model->add_layer(std::make_unique<FlattenLayer<float>>());
        }
    }

    return model;
}

void OnnxLoader::save_onyx_v2(TrainingModel<float>* model, const std::string& path) {
    SerializedModel smodel;
    smodel.name = "exported_model";
    smodel.framework = "C++";
    smodel.author = "auto_export";
    smodel.timestamp = "now";

    int count = 0;
    for (const auto& layer : model->layers()) {
        for (auto* param : layer->parameters()) {
            SerializedTensor t;
            t.name = "param_" + std::to_string(count++);
            t.dtype = DType::Float32;
            t.shape = std::vector<int64_t>(param->shape().begin(), param->shape().end());
            t.raw_data.resize(param->size() * sizeof(float));
            param->download(reinterpret_cast<float*>(t.raw_data.data()));
            smodel.tensors.push_back(std::move(t));
        }
    }

    for (const auto& lptr : model->layers()) {
        SerializedLayer layer;
        layer.name = lptr->name();
        layer.type = lptr->name();
        smodel.layers.push_back(std::move(layer));
    }

    save_onyx_v2(smodel, path);
}

static void write_string(std::ostream& out, const std::string& str) {
    uint32_t len = static_cast<uint32_t>(str.size());
    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    out.write(str.data(), len);
}

static std::string read_string(std::istream& in) {
    uint32_t len;
    in.read(reinterpret_cast<char*>(&len), sizeof(len));
    std::string str(len, '\0');
    in.read(&str[0], len);
    return str;
}

void OnnxLoader::load_onyx_v2(const std::string& path, SerializedModel& model) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open file for reading");

    uint32_t magic, version;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (magic != 0x584E5954) throw std::runtime_error("Invalid ONYX format");

    model.name = read_string(in);
    model.author = read_string(in);
    model.framework = read_string(in);
    model.timestamp = read_string(in);

    uint32_t tensor_count;
    in.read(reinterpret_cast<char*>(&tensor_count), sizeof(tensor_count));
    model.tensors.resize(tensor_count);
    for (auto& t : model.tensors) {
        t.name = read_string(in);
        uint8_t dtype;
        in.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));
        t.dtype = static_cast<DType>(dtype);
        uint32_t dim_count;
        in.read(reinterpret_cast<char*>(&dim_count), sizeof(dim_count));
        t.shape.resize(dim_count);
        in.read(reinterpret_cast<char*>(t.shape.data()), dim_count * sizeof(int64_t));
        uint32_t byte_size;
        in.read(reinterpret_cast<char*>(&byte_size), sizeof(byte_size));
        t.raw_data.resize(byte_size);
        in.read(reinterpret_cast<char*>(t.raw_data.data()), byte_size);
    }

    uint32_t layer_count;
    in.read(reinterpret_cast<char*>(&layer_count), sizeof(layer_count));
    model.layers.resize(layer_count);
    for (auto& l : model.layers) {
        l.type = read_string(in);
        l.name = read_string(in);
        uint32_t input_count, output_count, attr_count;
        in.read(reinterpret_cast<char*>(&input_count), sizeof(input_count));
        l.inputs.resize(input_count);
        for (auto& s : l.inputs) s = read_string(in);
        in.read(reinterpret_cast<char*>(&output_count), sizeof(output_count));
        l.outputs.resize(output_count);
        for (auto& s : l.outputs) s = read_string(in);
        in.read(reinterpret_cast<char*>(&attr_count), sizeof(attr_count));
        l.attributes.resize(attr_count);
        for (auto& attr : l.attributes) {
            attr.key = read_string(in);
            in.read(reinterpret_cast<char*>(&attr.value), sizeof(attr.value));
        }
    }

    uint32_t input_tensor_count;
    in.read(reinterpret_cast<char*>(&input_tensor_count), sizeof(input_tensor_count));
    model.inputs.resize(input_tensor_count);
    for (auto& s : model.inputs) s = read_string(in);

    uint32_t output_tensor_count;
    in.read(reinterpret_cast<char*>(&output_tensor_count), sizeof(output_tensor_count));
    model.outputs.resize(output_tensor_count);
    for (auto& s : model.outputs) s = read_string(in);
}

void OnnxLoader::save_onyx_v2(const SerializedModel& model, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open file for writing");

    uint32_t magic = 0x584E5954; // 'TYNX'
    uint32_t version = 2;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));

    write_string(out, model.name);
    write_string(out, model.author);
    write_string(out, model.framework);
    write_string(out, model.timestamp);

    uint32_t tensor_count = static_cast<uint32_t>(model.tensors.size());
    out.write(reinterpret_cast<const char*>(&tensor_count), sizeof(tensor_count));
    for (const auto& t : model.tensors) {
        write_string(out, t.name);
        uint8_t dtype = static_cast<uint8_t>(t.dtype);
        out.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
        uint32_t dim_count = static_cast<uint32_t>(t.shape.size());
        out.write(reinterpret_cast<const char*>(&dim_count), sizeof(dim_count));
        out.write(reinterpret_cast<const char*>(t.shape.data()), dim_count * sizeof(int64_t));
        uint32_t byte_size = static_cast<uint32_t>(t.raw_data.size());
        out.write(reinterpret_cast<const char*>(&byte_size), sizeof(byte_size));
        out.write(reinterpret_cast<const char*>(t.raw_data.data()), byte_size);
    }

    uint32_t layer_count = static_cast<uint32_t>(model.layers.size());
    out.write(reinterpret_cast<const char*>(&layer_count), sizeof(layer_count));
    for (const auto& l : model.layers) {
        write_string(out, l.type);
        write_string(out, l.name);
        uint32_t input_count = static_cast<uint32_t>(l.inputs.size());
        out.write(reinterpret_cast<const char*>(&input_count), sizeof(input_count));
        for (const auto& s : l.inputs) write_string(out, s);
        uint32_t output_count = static_cast<uint32_t>(l.outputs.size());
        out.write(reinterpret_cast<const char*>(&output_count), sizeof(output_count));
        for (const auto& s : l.outputs) write_string(out, s);
        uint32_t attr_count = static_cast<uint32_t>(l.attributes.size());
        out.write(reinterpret_cast<const char*>(&attr_count), sizeof(attr_count));
        for (const auto& attr : l.attributes) {
            write_string(out, attr.key);
            out.write(reinterpret_cast<const char*>(&attr.value), sizeof(attr.value));
        }
    }

    uint32_t input_tensor_count = static_cast<uint32_t>(model.inputs.size());
    out.write(reinterpret_cast<const char*>(&input_tensor_count), sizeof(input_tensor_count));
    for (const auto& s : model.inputs) write_string(out, s);

    uint32_t output_tensor_count = static_cast<uint32_t>(model.outputs.size());
    out.write(reinterpret_cast<const char*>(&output_tensor_count), sizeof(output_tensor_count));
    for (const auto& s : model.outputs) write_string(out, s);
}

} // namespace dnn  

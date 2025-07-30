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
#include <memory>

static inline void write_string(std::ostream& out, const std::string& str) {
    uint32_t len = static_cast<uint32_t>(str.size());
    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    out.write(str.data(), len);
}

static inline std::string read_string(std::istream& in) {
    uint32_t len;
    in.read(reinterpret_cast<char*>(&len), sizeof(len));
    std::string str(len, '\0');
    in.read(&str[0], len);
    return str;
}

namespace dnn {

} // namespace dnn  

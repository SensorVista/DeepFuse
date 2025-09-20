# Define app name
set(TARGET_NAME dnn)

# Collect all core DNN CUDA and C++ sources
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

# Collect application headers
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh
)

# Organize sources for Visual Studio Solution Explorer
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${HEADERS})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${SOURCES})

# Real: SASS code for known GPUs (fastest)
# Virtual: PTX code for newer GPUs via JIT (slow fallback, but works)
set(CMAKE_CUDA_ARCHITECTURES "89-real;89-virtual")

# Define static library for core DNN framework
add_library(${TARGET_NAME} STATIC ${SOURCES})

# Make core headers available to dependents
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_SOURCE_DIR})

# Add nlohmann/json include directories for linter/IDE support (Linux only)
if(UNIX AND NOT APPLE)
    if(TARGET nlohmann_json::nlohmann_json)
        get_target_property(NLOHMANN_JSON_INCLUDE_DIRS nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
        if(NLOHMANN_JSON_INCLUDE_DIRS)
            target_include_directories(${TARGET_NAME} PUBLIC ${NLOHMANN_JSON_INCLUDE_DIRS})
        endif()
    endif()
endif()

# Attach headers for IDE only (no compile impact)
target_sources(${TARGET_NAME} PRIVATE ${HEADERS})

# Link CUDA runtime for device code support
target_link_libraries(${TARGET_NAME} PUBLIC 
    CUDA::cudart
)

# Link nlohmann/json (Linux only - Windows uses vcpkg)
if(UNIX AND NOT APPLE)
    target_link_libraries(${TARGET_NAME} PUBLIC nlohmann_json::nlohmann_json)
endif()

if(DEFINED USE_CUDNN AND USE_CUDNN)
    target_link_libraries(${TARGET_NAME} PUBLIC 
        "${CUDAToolkit_LIBRARY_DIR}/cudnn.lib"
        "${CUDAToolkit_LIBRARY_DIR}/cublas.lib"
    )
endif()

# Enable cuDNN
if(USE_CUDNN)
    target_compile_definitions(dnn PUBLIC ENABLE_CUDNN)
endif()

# Ensure C++17 is used for both CUDA and C++ sources
target_compile_features(${TARGET_NAME} PUBLIC cxx_std_17)
set_target_properties(${TARGET_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

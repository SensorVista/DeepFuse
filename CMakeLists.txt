cmake_minimum_required(VERSION 4.0)

# Configure CUDA host compiler before project() call
if(UNIX AND NOT APPLE)
    find_program(CLANG_14 clang-14)
    if(CLANG_14)
        set(CMAKE_CUDA_HOST_COMPILER ${CLANG_14})
        message(STATUS "Using clang-14 as CUDA host compiler: ${CLANG_14}")
    else()
        message(WARNING "clang-14 not found, NVCC will use default host compiler")
    endif()
endif()

project(DeepFuse 
    DESCRIPTION "Deep Learning Library for CUDA."
    LANGUAGES C CUDA CXX)

# Declare global option
set(USE_CUDNN OFF)

# Remove /RTC1 from CUDA debug flags to avoid conflict with /O2
if(MSVC)
    string(REPLACE "/RTC1" "" CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG}")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard - ensure NVCC uses the same C++ standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Force NVCC to use C++17 with clang-14
if(UNIX AND NOT APPLE)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")
endif()

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Enable cuDNN
if(DEFINED USE_CUDNN AND USE_CUDNN)
    add_library(cublas STATIC IMPORTED GLOBAL)
    set_target_properties(cublas PROPERTIES
        IMPORTED_LOCATION "${CUDAToolkit_LIBRARY_DIR}/cublas.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIR}"
    )
    
    add_library(cudnn STATIC IMPORTED GLOBAL)
    set_target_properties(cudnn PROPERTIES
        IMPORTED_LOCATION "${CUDAToolkit_LIBRARY_DIR}/cudnn.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIR}"
    )    
endif()

# Add CUDA include directories
include_directories(${CMAKE_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIR})

# Enable testing
enable_testing()
include(FetchContent)

# Try to find system GoogleTest first
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    # Fallback to FetchContent if system version not found
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY    https://github.com/google/googletest.git
      GIT_TAG           v1.14.0  # Updated version for GCC 13 compatibility
    )
    
    FetchContent_MakeAvailable(
      googletest
    )
endif()

# Find or fetch nlohmann/json
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    # Fallback to FetchContent if not found via find_package (Linux only)
    if(UNIX AND NOT APPLE)
        FetchContent_Declare(
          json
          GIT_REPOSITORY    https://github.com/nlohmann/json.git
          GIT_TAG           v3.11.3  # Latest stable version
        )
        
        FetchContent_MakeAvailable(
          json
        )
        message(STATUS "Using nlohmann/json from FetchContent")
    else()
        message(FATAL_ERROR "nlohmann/json not found. On Windows, please install via vcpkg: vcpkg install nlohmann-json:x64-windows")
    endif()
else()
    message(STATUS "Using system nlohmann/json")
endif()

# Add core DNN CUDA library
add_subdirectory(dnn)

# Add tests
add_subdirectory(tests)

# Add apps
add_subdirectory(apps/lenet5)
add_subdirectory(apps/perceptron)
add_subdirectory(apps/resnet20)
add_subdirectory(apps/gpt2)
add_subdirectory(apps/rag)
add_subdirectory(apps/host-multi)  # Now supports both Windows and Linux

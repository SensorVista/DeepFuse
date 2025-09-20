cmake_minimum_required(VERSION 4.0)

# Linux-specific CUDA host compiler setup (must be before project() command)
if(UNIX AND NOT APPLE)
    # Check if clang-14 is available and set it as CUDA host compiler
    find_program(CLANG14_EXECUTABLE clang-14)
    if(CLANG14_EXECUTABLE)
        set(CMAKE_CUDA_HOST_COMPILER ${CLANG14_EXECUTABLE})
        message(STATUS "Using clang-14 as CUDA host compiler: ${CLANG14_EXECUTABLE}")
    else()
        # Fallback: try to find a compatible GCC version
        find_program(GCC11_EXECUTABLE gcc-11)
        if(GCC11_EXECUTABLE)
            set(CMAKE_CUDA_HOST_COMPILER ${GCC11_EXECUTABLE})
            message(STATUS "Using gcc-11 as CUDA host compiler: ${GCC11_EXECUTABLE}")
        else()
            message(WARNING "No compatible CUDA host compiler found. Please install clang-14 or gcc-11")
        endif()
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

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)

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

# Fetch GoogleTest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           b796f7d44681514f58a683a3a71ff17c94edb0c1 # version 1.13.0
)

FetchContent_MakeAvailable(
  googletest
)

# Add core DNN CUDA library
add_subdirectory(dnn)

# Add tests
add_subdirectory(tests)

# Add apps
add_subdirectory(apps/lenet5)
add_subdirectory(apps/perceptron)
add_subdirectory(apps/resnet20)
add_subdirectory(apps/gpt2)
add_subdirectory(apps/host-multi)

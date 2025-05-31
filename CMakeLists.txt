cmake_minimum_required(VERSION 4.0)
project(DeepLearning 
    DESCRIPTION "Deep Learning Library for CUDA."
    LANGUAGES C CUDA CXX)

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

# # Add include directories
# include_directories(
#     ${CMAKE_CURRENT_SOURCE_DIR}
#     ${CUDAToolkit_INCLUDE_DIRS}
# )

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
# Define app name
set(TARGET_NAME unit_tests)

# Collect all unit test sources
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/**/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/**/*.cpp
)

# Ensure test sources compile as host C++
# foreach(src IN LISTS SOURCES)
#     set_source_files_properties(${src} PROPERTIES LANGUAGE CXX)
# endforeach()

# Organize test sources for Visual Studio Solution Explorer
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${SOURCES})

# Define unit test executable
add_executable(${TARGET_NAME} ${SOURCES})

# Include headers from core DNN library
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_SOURCE_DIR})

# Link CUDA runtime for device code support
# set_target_properties(${TARGET_NAME} PROPERTIES
#     LINKER_LANGUAGE CUDA
#     CUDA_RESOLVE_DEVICE_SYMBOLS ON
# )

# Link against core DNN library and GoogleTest
target_link_libraries(${TARGET_NAME} PRIVATE dnn GTest::gtest_main)

# Register tests with CTest
include(GoogleTest)
# gtest_discover_tests(${TARGET_NAME})  # Temporarily disabled due to regex error

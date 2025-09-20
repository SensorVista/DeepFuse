# Define app name
set(TARGET_NAME unit_tests)

# Collect all unit test sources
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/**/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/**/*.cpp
)

# Ensure test sources compile as CUDA (for .cu files) or CXX (for .cpp files)
foreach(src IN LISTS SOURCES)
    if(src MATCHES "\\.cu$")
        # CUDA files - keep as CUDA
        set_source_files_properties(${src} PROPERTIES LANGUAGE CUDA)
    elseif(src MATCHES "\\.cpp$")
        # C++ files - compile as C++
        set_source_files_properties(${src} PROPERTIES LANGUAGE CXX)
    endif()
endforeach()

# Organize test sources for Visual Studio Solution Explorer
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${SOURCES})

# Define unit test executable
add_executable(${TARGET_NAME} ${SOURCES})

# Include headers from core DNN library
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_SOURCE_DIR})

# Link CUDA runtime for device code support
set_target_properties(${TARGET_NAME} PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link against core DNN library and GoogleTest
target_link_libraries(${TARGET_NAME} PRIVATE dnn GTest::gtest_main)

# Register tests with CTest
include(GoogleTest)

# Set regex engine to avoid compatibility issues on Linux
if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_HAS_POSIX_RE=0")
    # Also try to force C++11 regex
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_HAS_TR1_TUPLE=0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_USE_OWN_TR1_TUPLE=0")
endif()

# Temporarily disable test discovery to avoid regex issues on Linux
# Use platform-specific test discovery
if(WIN32)
    # Windows: Use the standard discovery
    gtest_discover_tests(${TARGET_NAME})
else()
    # Linux/Unix: Skip discovery for now due to regex compatibility issues
    # gtest_discover_tests(${TARGET_NAME} 
    #     DISCOVERY_TIMEOUT 60
    #     PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    # )
    message(STATUS "Skipping test discovery on Linux due to regex compatibility issues")
endif()

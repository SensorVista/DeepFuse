# Define app name
set(TARGET_NAME host-multi)

# Try to find TBB, but make it optional for Linux
find_package(TBB QUIET)
if(NOT TBB_FOUND)
    message(STATUS "TBB not found, using std::thread as fallback")
    set(USE_TBB OFF)
else()
    set(USE_TBB ON)
    message(STATUS "TBB found, using TBB for parallelization")
endif()

# Collect application sources
file(GLOB SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

# Collect application headers
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/dnn/*.cuh
)

# Organize sources and headers in Solution Explorer
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${SOURCES})

# Define executable target
add_executable(${TARGET_NAME} ${SOURCES})
#target_sources(${TARGET_NAME} PRIVATE ${HEADERS})

# Set properties for CUDA linking
set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
)

if(WIN32)
    # Set command line arguments for debugging in Visual Studio and solution folders 
    set_target_properties(${TARGET_NAME} PROPERTIES
        FOLDER "apps"
        VS_DEBUGGER_COMMAND_ARGUMENTS "../../../datasets"
    )
endif()

# Include core DNN headers
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_SOURCE_DIR})

# Link core DNN library
if(USE_TBB)
    target_link_libraries(${TARGET_NAME} PRIVATE dnn TBB::tbb)
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_TBB)
else()
    target_link_libraries(${TARGET_NAME} PRIVATE dnn)
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_STD_THREAD)
endif()

# Enable cuDNN
if(DEFINED USE_CUDNN AND USE_CUDNN)
    target_compile_definitions(dnn PUBLIC ENABLE_CUDNN)
endif()


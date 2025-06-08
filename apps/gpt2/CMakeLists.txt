# Define app name
set(TARGET_NAME gpt2)

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
target_link_libraries(${TARGET_NAME} PRIVATE dnn)

# Enable cuDNN
if(DEFINED USE_CUDNN AND USE_CUDNN)
    target_compile_definitions(dnn PUBLIC ENABLE_CUDNN)
endif()


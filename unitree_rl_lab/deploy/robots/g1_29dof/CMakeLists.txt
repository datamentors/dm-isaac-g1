cmake_minimum_required(VERSION 3.12)
project(g1_29dof_controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

# Detect architecture and set ONNX Runtime path accordingly.
# x86_64: installed system-wide to /usr/local via Dockerfile
# aarch64: pre-built in thirdparty/ directory
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ONNX_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-aarch64-1.22.0/include)
    set(ONNX_LIB ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-aarch64-1.22.0/lib/libonnxruntime.so.1.22.0)
else()
    set(ONNX_INCLUDE_DIR /usr/local/include)
    set(ONNX_LIB /usr/local/lib/libonnxruntime.so)
endif()

include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx # dds
  /usr/local/include/iceoryx/v2.0.2 # dds
  include
  ${ONNX_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/../../include/
)

link_libraries(
  unitree_sdk2 ddsc ddscxx rt pthread # dds
  libboost_program_options.a libyaml-cpp.a fmt
  ${ONNX_LIB}
)

file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(g1_ctrl main.cpp)

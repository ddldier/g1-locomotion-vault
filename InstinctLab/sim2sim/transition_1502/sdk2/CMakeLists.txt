cmake_minimum_required(VERSION 3.16)
project(g1_transition_sdk2_probe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(UNITREE_SDK2_PREFIX "/root/gpufree-data/unitree_sdk2/install" CACHE PATH "SDK2 install prefix")
set(ONNXRUNTIME_ROOT "/root/gpufree-data/unitree_rl_lab_review_20261021/deploy/thirdparty/onnxruntime-linux-x64-1.22.0" CACHE PATH "ONNX Runtime prefix")
list(PREPEND CMAKE_PREFIX_PATH "${UNITREE_SDK2_PREFIX}")
find_package(Threads REQUIRED)
find_package(unitree_sdk2 REQUIRED)
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATHS "${ONNXRUNTIME_ROOT}/include" REQUIRED NO_DEFAULT_PATH)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime libonnxruntime.so.1.22.0 PATHS "${ONNXRUNTIME_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)

add_executable(g1_transition_sdk2_probe main.cpp)
target_include_directories(g1_transition_sdk2_probe PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(g1_transition_sdk2_probe PRIVATE unitree_sdk2 ${ONNXRUNTIME_LIBRARY} Threads::Threads)
set_target_properties(g1_transition_sdk2_probe PROPERTIES
  BUILD_RPATH "${UNITREE_SDK2_PREFIX}/lib;${ONNXRUNTIME_ROOT}/lib")

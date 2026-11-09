cmake_minimum_required(VERSION 3.16)
project(transition_1502_policy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT "/root/gpufree-data/unitree_rl_lab_review_20261021/deploy/thirdparty/onnxruntime-linux-x64-1.22.0")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATHS "${ONNXRUNTIME_ROOT}/include" REQUIRED)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime libonnxruntime.so.1.22.0 PATHS "${ONNXRUNTIME_ROOT}/lib" REQUIRED)

add_executable(transition_1502_policy_server policy_server.cpp)
target_include_directories(transition_1502_policy_server PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(transition_1502_policy_server PRIVATE "${ONNXRUNTIME_LIBRARY}")
set_target_properties(transition_1502_policy_server PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
  INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib")

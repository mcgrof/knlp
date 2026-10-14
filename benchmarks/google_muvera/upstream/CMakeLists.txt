# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.24)
project(knlp_google_fde LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${DEPS_DIR}/abseil" "${CMAKE_BINARY_DIR}/abseil")
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
set(protobuf_ABSL_PROVIDER package CACHE STRING "" FORCE)
set(protobuf_BUILD_LIBUPB OFF CACHE BOOL "" FORCE)
set(protobuf_WITH_ZLIB OFF CACHE BOOL "" FORCE)
add_subdirectory("${DEPS_DIR}/protobuf" "${CMAKE_BINARY_DIR}/protobuf")
set(GOOGLE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/google/graph-mining")
set(PROTO "sketching/point_cloud/fixed_dimensional_encoding_config.proto")
set(GENERATED "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GENERATED}")
add_custom_command(
  OUTPUT "${GENERATED}/sketching/point_cloud/fixed_dimensional_encoding_config.pb.cc"
         "${GENERATED}/sketching/point_cloud/fixed_dimensional_encoding_config.pb.h"
  COMMAND protobuf::protoc --cpp_out=${GENERATED} --proto_path=${GOOGLE_SOURCE}
          "${GOOGLE_SOURCE}/${PROTO}"
  DEPENDS protobuf::protoc "${GOOGLE_SOURCE}/${PROTO}"
  VERBATIM)
add_executable(google_fde google_fde_cli.cc
  "${GOOGLE_SOURCE}/sketching/point_cloud/fixed_dimensional_encoding.cc"
  "${GENERATED}/sketching/point_cloud/fixed_dimensional_encoding_config.pb.cc")
target_include_directories(google_fde PRIVATE "${GOOGLE_SOURCE}" "${GENERATED}"
  "${DEPS_DIR}/eigen")
target_link_libraries(google_fde PRIVATE protobuf::libprotobuf absl::log
  absl::check absl::status absl::statusor absl::strings absl::span)
set_target_properties(google_fde PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")

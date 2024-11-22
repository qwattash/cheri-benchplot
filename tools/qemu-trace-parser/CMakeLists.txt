cmake_minimum_required(VERSION 3.20)
project(qemu-trace-parser LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/qemu_log_entry.proto)

find_package(Boost REQUIRED COMPONENTS
  iostreams log program_options)

set(PARSER_SRCS
  parser.cc
  memcount.cc)

add_executable(qemu-trace-parser ${PARSER_SRCS} ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(qemu-trace-parser Boost::program_options Boost::log Boost::iostreams)
target_link_libraries(qemu-trace-parser ${Protobuf_LIBRARIES})

cmake_minimum_required(VERSION 3.20)
project(fastdwarf LANGUAGES C CXX)

set(CHERISDK "" CACHE STRING "Path to the CHERI SDK directory, used to find LLVM")

set(CMAKE_SHARED_MODULE_PREFIX "")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CHERISDK STREQUAL "")
  message(FATAL_ERROR "Must set the CHERISDK CMake option")
endif ()

enable_testing()

set(CMAKE_PREFIX_PATH "${CHERISDK}/sdk")
find_package(LLVM REQUIRED CONFIG NO_CMAKE_SYSTEM_PATH)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs
  DebugInfoDWARF
  AllTargetsDescs
  AllTargetsInfos
  MC
  Object
  Support)
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})

# find_package(Boost REQUIRED CONFIG COMPONENTS python)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)
message(STATUS "Found PyBind11 ${pybind11_VERSION}")
message(STATUS "Using PyBind11 headers ${pybind11_INCLUDE_DIRS}")
message(STATUS "Using PyBind11 defs ${pybind11_DEFINITIONS}")
message(STATUS "Found Python ${Python3_VERSION}")
message(STATUS "Using Python headers ${Python3_INCLUDE_DIRS}")
find_package(GTest REQUIRED)
include(GoogleTest)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_subdirectory(pydwarf)
add_subdirectory(pychericap)


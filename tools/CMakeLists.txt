cmake_minimum_required(VERSION 3.20)
project(benchplot-tools LANGUAGES C CXX)

set(CHERISDK "" CACHE STRING "Path to the CHERI SDK directory, used to find LLVM")

set(CMAKE_SHARED_MODULE_PREFIX "")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CHERISDK STREQUAL "")
  message(FATAL_ERROR "Must set the CHERISDK CMake option")
endif ()

set(CMAKE_PREFIX_PATH "${CHERISDK}/sdk")

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_subdirectory(dwarf-scraper)



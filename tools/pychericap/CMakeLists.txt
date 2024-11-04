cmake_minimum_required(VERSION 3.20)
project(pychericap LANGUAGES CXX)

set(PYBIND11_FIND_PTYHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(pychericap "pychericap.cc")
target_include_directories(pychericap PUBLIC "${PROJECT_SOURCE_DIR}/../cheri-compressed-cap")
install(TARGETS pychericap DESTINATION .)

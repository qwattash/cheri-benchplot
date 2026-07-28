cmake_minimum_required(VERSION 3.12)
project(CheriMicroBench)

find_package(benchmark REQUIRED)

add_executable(cheri_ubench cheri_micro_benchmark.cc)
target_link_libraries(cheri_ubench PRIVATE benchmark::benchmark-static)

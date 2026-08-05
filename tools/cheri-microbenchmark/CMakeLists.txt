cmake_minimum_required(VERSION 3.12)
project(CheriMicroBench LANGUAGES CXX ASM)

find_package(benchmark REQUIRED)

option(USER_HWPMC_ACCESS "Platform supports user-mode HWPMC access" OFF)
set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Set default build type to RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

set(TARGET_ARCH "${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}")
if(NOT TARGET_ARCH)
  set(TARGET_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(TARGET_ARCH)
  if(TARGET_ARCH MATCHES "^aarch64c")
    add_compile_options(-mabi=purecap-benchmark)
    add_link_options(-mabi=purecap-benchmark)
  elseif(TARGET_ARCH MATCHES "^RISCV")
    # Nothing to do there
  else()
    message(STATUS "Unknown target architecture: ${TARGET_ARCH}")
  endif()
else()
  message(WARNING "Could not detect target architecture, skipping ABI flags")
endif()

add_executable(cheri_ubench cheri_micro_benchmark.cc)
target_sources(cheri_ubench
  PRIVATE
  registry.cc
  copy_loop.cc
  copyinout.cc
  copycommon.S)
target_link_libraries(cheri_ubench PRIVATE benchmark::benchmark)

install(TARGETS cheri_ubench)

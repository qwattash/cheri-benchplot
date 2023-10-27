
add_library(dwarf_lib STATIC
  "dwarf.cc"
  "dwarf_type_layout.cc")
target_include_directories(dwarf_lib PUBLIC ${LLVM_INCLUDE_DIRS})
target_compile_definitions(dwarf_lib PUBLIC ${LLVM_DEFINITIONS_LIST})
target_link_directories(dwarf_lib PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(dwarf_lib ${llvm_libs})
target_compile_options(dwarf_lib PRIVATE "-fPIC" "-fno-rtti")

add_library(pydwarf MODULE "pydwarf.cc")
set_target_properties(pydwarf PROPERTIES OUTPUT_NAME "_pydwarf")
target_include_directories(pydwarf PUBLIC ${pybind11_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_compile_definitions(pydwarf PUBLIC ${pybind11_DEFINITIONS})
target_link_directories(pydwarf PUBLIC ${Python3_LIBRARIES})
target_link_libraries(pydwarf PRIVATE dwarf_lib)

install(TARGETS pydwarf
  LIBRARY
  DESTINATION "pydwarf")

add_executable(test_dwarf_base "dwarf_base_test.cc")
target_link_libraries(test_dwarf_base dwarf_lib GTest::gtest GTest::gtest_main)
target_compile_options(test_dwarf_base PRIVATE "-fPIC"  "-fno-rtti")
gtest_add_tests(TARGET test_dwarf_base TEST_LIST dwarf_unittest)


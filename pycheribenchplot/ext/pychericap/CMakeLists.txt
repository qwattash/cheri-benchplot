
add_library(pychericap MODULE "pychericap.cc")
set_target_properties(pychericap PROPERTIES OUTPUT_NAME "_pychericap")
target_include_directories(pychericap PUBLIC "${PROJECT_SOURCE_DIR}/cheri-compressed-cap")
target_include_directories(pychericap PUBLIC ${pybind11_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_compile_definitions(pychericap PUBLIC ${pybind11_DEFINITIONS})
target_link_directories(pychericap PUBLIC ${Python3_LIBRARIES})
install(TARGETS pychericap
  LIBRARY
  DESTINATION "pychericap")

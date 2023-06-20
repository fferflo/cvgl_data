cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(cvgl_data LANGUAGES CXX)

set(Python3_FIND_STRATEGY LOCATION)
find_package(Python REQUIRED COMPONENTS NumPy)
find_package(pybind11 REQUIRED)
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-python REQUIRED)
find_package(xtensor-io REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(xtensor-interfaces REQUIRED)
find_package(PROJ REQUIRED)
find_package(cosy REQUIRED)
find_package(tiledwebmaps REQUIRED)
find_package(yaml-cpp REQUIRED)

pybind11_add_module(backend backend/main.cpp)
target_compile_features(backend PUBLIC cxx_std_17)
target_link_libraries(backend PUBLIC
  xtensor
  xtensor::optimize
  xtensor::use_xsimd
  xtensor-io
  xtensor-blas
  xtensor-interfaces::opencv
  cosy::cosy
  cosy::proj
  tiledwebmaps::tiledwebmaps
  xtensor-python
  Python::NumPy
  yaml-cpp
)
set_target_properties(backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/cvgl_data")

execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/cvgl_data)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/cvgl_data/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/cvgl_data/__init__.py)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/cvgl_data/prepare.py ${CMAKE_CURRENT_BINARY_DIR}/cvgl_data/prepare.py)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/cvgl_data/paired.py ${CMAKE_CURRENT_BINARY_DIR}/cvgl_data/paired.py)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/cvgl_data/visualize.py ${CMAKE_CURRENT_BINARY_DIR}/cvgl_data/visualize.py)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/cvgl_data/util.py ${CMAKE_CURRENT_BINARY_DIR}/cvgl_data/util.py)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/setup.cfg ${CMAKE_CURRENT_BINARY_DIR}/setup.cfg)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE ${CMAKE_CURRENT_BINARY_DIR}/LICENSE)

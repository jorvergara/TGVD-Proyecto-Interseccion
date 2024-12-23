cmake_minimum_required(VERSION 3.11)
project(octree_module)
set(CMAKE_CXX_STANDARD 17)  

find_package(pybind11 REQUIRED)

find_package(OpenMP REQUIRED)

pybind11_add_module(octree_module octree.cpp)

target_link_libraries(octree_module PRIVATE OpenMP::OpenMP_CXX)

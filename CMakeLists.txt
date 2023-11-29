cmake_minimum_required(VERSION 3.8)
project(_dupin)

set(DEFAULT_BUILD_TYPE Release)

find_package(OpenMP REQUIRED) 

add_subdirectory(extern/pybind11)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/eigen-3.4.0")

pybind11_add_module(_dupin dupin.h dupin.cpp dupininterface.cpp)

set_target_properties(_dupin PROPERTIES
    CXX_STANDARD 17
    CMAKE_CXX_STANDARD_REQUIRED True
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_dupin PUBLIC OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(_dupin PRIVATE VERSION_INFO=${VERSION_INFO})
target_compile_options(_dupin PRIVATE -O2 -march=native -fopenmp)
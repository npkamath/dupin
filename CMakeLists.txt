cmake_minimum_required(VERSION 3.8)
project(_dupin)

set(DEFAULT_BUILD_TYPE Release)

add_subdirectory(extern/pybind11)

pybind11_add_module(_dupin dupin.h dupin.cpp dupininterface.cpp)

set_target_properties(_dupin PROPERTIES
    CXX_STANDARD 17
    CMAKE_CXX_STANDARD_REQUIRED True
)

target_compile_definitions(_dupin PRIVATE VERSION_INFO=${VERSION_INFO})

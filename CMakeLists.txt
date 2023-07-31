cmake_minimum_required(VERSION 3.12)
project(dupin)



include_directories(${PROJECT_SOURCE_DIR}/extern)

find_package(pybind11 CONFIG REQUIRED)

PYBIND11_ADD_MODULE(testinglib MODULE dupin.h dupininterface.cpp)

target_link_libraries(testinglib PRIVATE pybind11::module)

p
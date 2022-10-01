#boilerplate
cmake_minimum_required(VERSION 3.8)
project(AYGALIC VERSION 1.0 LANGUAGES CXX)

# Force compiler to use C++17 standard.
set(CMAKE_CXX11_STANDARD_COMPILE_OPTION -std=c++17)

################### pybind11 ##################################
set(PATH_TO_PYBIND "/root/lifex_mnt/pyfex/pybind11")

include_directories(SYSTEM pybind11/include)
include_directories(SYSTEM $ENV{mkPythonInc}) #need to direct pybind to our docker env python
add_subdirectory(${PATH_TO_PYBIND} ../)
set(PYBIND11_LTO_CXX_FLAGS "OFF" CACHE INTERNAL "")
set(PYBIND11_LTO_LINKER_FLAGS "OFF" CACHE INTERNAL "")

###################### build ################################
pybind11_add_module(lib bindings.cpp)
add_executable(nate original.cpp)
target_include_directories(nate PRIVATE /u/sw/toolchains/gcc-glibc/11/pkgs/boost/1.76.0/include/boost/chrono/)
#include_directories(SYSTEM $ENV{mkBoostInc})

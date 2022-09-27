#boilerplate
cmake_minimum_required(VERSION 3.8)
project(AYGALIC VERSION 1.0 LANGUAGES CXX)

# Force compiler to use C++17 standard.
set(CMAKE_CXX_COMPILER "/u/sw/toolchains/gcc-glibc/11/base/bin/mpicxx")
set(CMAKE_CXX11_STANDARD_COMPILE_OPTION -std=c++17)
#set(NATE_FLAGS "-Wall")
#set(CMAKE_CXX_FLAGS "${NATE_FLAGS}")



################### pybind11 ##################################
set(PATH_TO_PYBIND "/root/lifex_mnt/pyfex/pybind11")

include_directories(SYSTEM pybind11/include)
include_directories(SYSTEM $ENV{mkPythonInc}) #need to direct pybind to our docker env python
add_subdirectory(${PATH_TO_PYBIND} ../)
set(PYBIND11_LTO_CXX_FLAGS "OFF" CACHE INTERNAL "")
set(PYBIND11_LTO_LINKER_FLAGS "OFF" CACHE INTERNAL "")

###################### build ################################
pybind11_add_module(aygalic main.cpp)

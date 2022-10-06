#boilerplate
cmake_minimum_required(VERSION 3.8)
project(AYGALIC VERSION 1.0 LANGUAGES CXX)

# Force compiler to use C++17 standard.
set(CMAKE_CXX11_STANDARD_COMPILE_OPTION -std=c++17)

################### deal.II ##################################
# Locate deal.II and initialize its variables.
find_package(deal.II 9.2.0 REQUIRED HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{mkDealiiPrefix})
deal_ii_initialize_cached_variables()

###################### build ################################
add_executable(linalg basic_equation.cpp)
DEAL_II_SETUP_TARGET(linalg)

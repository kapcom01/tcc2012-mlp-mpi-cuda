cmake_minimum_required(VERSION 2.6)

add_subdirectory(src/arff)

set(CMAKE_CXX_FLAGS "-O2 -std=c++0x")

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_BINARY_DIR}/src/arff)

add_executable(parallel_mlp ${CMAKE_SOURCE_DIR}/src/Main.cpp)
target_link_libraries(parallel_mlp arff)

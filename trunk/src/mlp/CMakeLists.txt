cmake_minimum_required(VERSION 2.6)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

set(CMAKE_CXX_FLAGS "-O2 -std=c++0x")

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(mlp activation/ActivationFunc.cpp activation/HyperbolicFunc.cpp 
	activation/LogisticFunc.cpp BackpropMLP.cpp ExampleSet.cpp Layer.cpp)

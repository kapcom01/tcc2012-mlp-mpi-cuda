cmake_minimum_required(VERSION 2.6)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

set(CMAKE_CXX_FLAGS "-O2 -std=c++0x")

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(mlp activation/ActivationFunction.cpp activation/HyperbolicFunction.cpp activation/LogisticFunction.cpp
	layer/HiddenLayer.cpp layer/Layer.cpp layer/OutputLayer.cpp 
	InputSet.cpp LearningRate.cpp MLPerceptron.cpp Settings.cpp)

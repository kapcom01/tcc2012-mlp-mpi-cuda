CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE(OpenMP)

ADD_LIBRARY(mlp BackpropMLP.cpp ExampleSet.cpp Layer.cpp Neuron.cpp)

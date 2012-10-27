CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(ParallelMLP)

SET(SOURCES
	HostExampleSet.cpp
	HostLayer.cpp
	HostMLP.cpp
	HostOutLayer.cpp
)

ADD_LIBRARY(mlp_serial ${SOURCES})
TARGET_LINK_LIBRARIES(mlp_serial mlp_common)

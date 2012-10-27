CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(ParallelMLP)

SET(SOURCES
	Chronometer.cpp
	ExampleSet.cpp
	Indexes.cpp
	Layer.cpp
	MLP.cpp
	OutLayer.cpp
)

ADD_LIBRARY(mlp_common ${SOURCES})
TARGET_LINK_LIBRARIES(mlp_common rt)

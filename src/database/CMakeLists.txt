CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(ParallelMLP)

SET(SOURCES
	Connection.cpp
	DatabaseException.cpp
	ExampleSetAdapter.cpp
	MLPAdapter.cpp
	RelationAdapter.cpp
)

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
ADD_LIBRARY(database ${SOURCES})
TARGET_LINK_LIBRARIES(database ${Pqxx_LIBRARIES})

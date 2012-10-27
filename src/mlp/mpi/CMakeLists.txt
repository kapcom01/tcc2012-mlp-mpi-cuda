CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(ParallelMLP)

SET(SOURCES
    BalanceInfo.cpp
	RemoteExampleSet.cpp
	RemoteLayer.cpp
	RemoteMLP.cpp
	RemoteOutLayer.cpp
)

INCLUDE_DIRECTORIES(${MPI_CXX_INCLUDE_PATH})
ADD_LIBRARY(mlp_mpi ${SOURCES})
TARGET_LINK_LIBRARIES(mlp_mpi mlp_serial ${MPI_CXX_LIBRARIES})

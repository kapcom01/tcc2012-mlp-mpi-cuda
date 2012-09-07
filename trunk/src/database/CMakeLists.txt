CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE(Pqxx)

ADD_LIBRARY(database BackpropMLPAdapter.cpp Connection.cpp DatabaseException.cpp 
	ExampleSetAdapter.cpp RelationAdapter.cpp)
TARGET_LINK_LIBRARIES(database ${Pqxx_LIBRARIES})

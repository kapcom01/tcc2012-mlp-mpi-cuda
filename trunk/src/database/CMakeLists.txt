cmake_minimum_required(VERSION 2.6)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
find_package(Pqxx)

set(CMAKE_CXX_FLAGS "-O2 -std=c++0x")

include_directories(${CMAKE_SOURCE_DIR}/include ${Pqxx_INCLUDE_DIRS})

add_library(database Connection.cpp DatabaseException.cpp ExampleSetAdapter.cpp
	RelationAdapter.cpp)
target_link_libraries(database ${Pqxx_LIBRARIES})

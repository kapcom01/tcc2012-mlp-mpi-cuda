cmake_minimum_required(VERSION 2.6)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
find_package(FLEX)
find_package(BISON)

set(CMAKE_CXX_FLAGS "-O2 -std=c++0x")

include_directories(${CMAKE_SOURCE_DIR}/include)

BISON_TARGET(PARSER Parser.y ${CMAKE_CURRENT_BINARY_DIR}/Parser.cc)
FLEX_TARGET(SCANNER Scanner.l ${CMAKE_CURRENT_BINARY_DIR}/Scanner.cc)
ADD_FLEX_BISON_DEPENDENCY(SCANNER PARSER)

include_directories(${CMAKE_BINARY_DIR}/src)

add_library(arff Attribute.cpp Driver.cpp ParseException.cpp Relation.cpp
	Scanner.cpp Value.cpp ${BISON_PARSER_OUTPUTS} ${FLEX_SCANNER_OUTPUTS})

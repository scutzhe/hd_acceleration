cmake_minimum_required(VERSION 2.8)
project(demo)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -pthread -Wall -Ofast -Wfatal-errors")

#add wrapper
set(TRT_WRAPPER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tensorRTWrapper/code)
add_subdirectory(${TRT_WRAPPER_DIR})
include_directories(${TRT_WRAPPER_DIR}/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

#add opencv
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES_DIRS})

#build runYolov3
#add_executable(runYolov3 main.cpp eval.cpp dataReader.cpp)
#target_link_libraries(runYolov3 TrtNet ${OpenCV_LIBS})

#build so_file
add_library(yolov3 SHARED main.cpp eval.cpp dataReader.cpp)
target_link_libraries(yolov3 TrtNet ${OpenCV_LIBS})

#install so_file
install(TARGETS yolov3 DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/)

#install exe_file
#install(TARGETS runYolov3 DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/)

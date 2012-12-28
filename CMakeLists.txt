cmake_minimum_required(VERSION 2.6)
project("gist")
add_definitions(-std=c++0x -O3 -Wall -Wno-unused-function -march=native)
find_package(OpenCV REQUIRED)

ADD_LIBRARY(gist STATIC colorport.cc gist_feature.cpp feature.cpp 
  colorgist.cc)

add_executable(extGist gist_feature.cpp feature.cpp extGist.cc)
add_executable(coloronly coloronly.cc colorport.cc)
add_executable(colortool colortool.cc colorport.cc)
add_executable(extColorGist gist_feature.cpp feature.cpp 
  colorport.cc colorgist.cc extColorGist)

target_link_libraries(extGist ${OpenCV_LIBS} )
target_link_libraries(coloronly ${OpenCV_LIBS} gflags glog )
target_link_libraries(colortool ${OpenCV_LIBS} gflags glog )
target_link_libraries(extColorGist ${OpenCV_LIBS} gflags glog )

cmake_minimum_required(VERSION 2.6)
project("gist")
add_definitions(-std=c++0x -O3 -Wall -Wno-unused-function -march=native)
find_package(OpenCV REQUIRED)

add_executable(extGist gist_feature.cpp feature.cpp extGist.cc)
add_executable(coloronly coloronly.cc colorport.cc)
add_executable(colortool colortool.cc colorport.cc)

target_link_libraries(extGist ${OpenCV_LIBS} )
target_link_libraries(coloronly ${OpenCV_LIBS} gflags glog )
target_link_libraries(colortool ${OpenCV_LIBS} gflags glog )

cmake_minimum_required(VERSION 3.15)

project(ME_DSO)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS -O3)

find_package(Eigen3 REQUIRED)
find_package(SophusTempl REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(./src/)

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(TestMEDSO test/test.cpp)
target_link_libraries(TestMEDSO ${OpenCV_LIBS})
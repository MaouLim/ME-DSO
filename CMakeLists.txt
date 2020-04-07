cmake_minimum_required(VERSION 3.15)

project(ME_DSO)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS -O3)

set(FAST_DIR )

find_package(Eigen3 REQUIRED)
find_package(SophusTempl REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fast REQUIRED)

include_directories(./src/)

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${fast_INCLUDE_DIRS})

add_executable(VSLAMTest test/test_vslam.cpp)
target_link_libraries(VSLAMTest ${OpenCV_LIBS} ${fast_LIBRARIES})

add_executable(ThreadingTest test/test_threading.cpp)
target_link_libraries(ThreadingTest ${OpenCV_LIBS} ${fast_LIBRARIES})

add_executable(UtilsTest test/test_utils.cpp)
target_link_libraries(UtilsTest ${OpenCV_LIBS} ${fast_LIBRARIES})
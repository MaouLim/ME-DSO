cmake_minimum_required(VERSION 3.15)

project(ME_DSO)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS -O3)

find_package(Eigen3 REQUIRED)
find_package(SophusTempl REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fast REQUIRED)
find_package(g2o REQUIRED)

include_directories(./src/)

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${fast_INCLUDE_DIRS})
include_directories(${g2o_INCLUDE_DIRS})

add_library(vslam_utils SHARED
    src/utils/config.cpp
    src/utils/blocking_queue.hpp
    src/utils/diff.hpp
    src/utils/messaging.hpp
    src/utils/utils.hpp
    src/utils/patch.hpp
    src/utils/threading.hpp
)

add_library(vslam SHARED
    src/vo/depth_filter.cpp
    src/vo/feature.cpp
    src/vo/frame.cpp
    src/vo/homography.cpp
    src/vo/initializer.cpp
    src/vo/map_point.cpp
    src/vo/matcher.cpp
    src/vo/pose_estimator.cpp
)
target_link_libraries(vslam vslam_utils g2o_core g2o_stuff ${OpenCV_LIBS} ${fast_LIBRARIES})

#add_executable(VSLAMTest test/test_vslam.cpp)
#target_link_libraries(VSLAMTest g2o_core g2o_stuff ${OpenCV_LIBS} ${fast_LIBRARIES})

#add_executable(ThreadingTest test/test_threading.cpp)

#add_executable(UtilsTest test/test_utils.cpp)
#target_link_libraries(UtilsTest ${OpenCV_LIBS})

#add_executable(MatcherTest test/test_matcher.cpp)
#target_link_libraries(MatcherTest ${OpenCV_LIBS} ${fast_LIBRARIES})

add_executable(FeatureTest test/test_feature.cpp)
target_link_libraries(FeatureTest g2o_core g2o_stuff ${OpenCV_LIBS} vslam)

add_executable(PoseEstimatorTest test/test_pose_estimator.cpp)
target_link_libraries(PoseEstimatorTest g2o_core g2o_stuff ${OpenCV_LIBS} vslam)

add_executable(FCFA test/pose_estimation_fcfa.cpp)
target_link_libraries(FCFA ${OpenCV_LIBS})
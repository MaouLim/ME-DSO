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
find_package(PCL REQUIRED)

include_directories(./src/)

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${fast_INCLUDE_DIRS})
include_directories(${g2o_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})

add_definitions(${PCL_DEFINITIONS})

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
    src/vars.cpp
    src/vo/depth_filter.cpp
    src/vo/feature.cpp
    src/vo/frame.cpp
    src/vo/homography.cpp
    src/vo/initializer.cpp
    src/vo/map_point.cpp
    src/vo/matcher.cpp
    src/vo/pose_estimator.cpp
    src/vo/core.cpp
    src/vo/reprojector.cpp
    src/vo/map.cpp
    src/backend/g2o_staff.cpp
    src/backend/bundle_adjustment.cpp
)
target_link_libraries(vslam vslam_utils g2o_core g2o_stuff ${OpenCV_LIBS} ${fast_LIBRARIES})

add_executable(VSLAMTest test/test_vslam.cpp)
target_link_libraries(VSLAMTest pthread g2o_core g2o_stuff ${PCL_LIBRARIES} ${OpenCV_LIBS} vslam)

add_executable(ThreadingTest test/test_threading.cpp)
target_link_libraries(ThreadingTest pthread)

add_executable(UtilsTest test/test_utils.cpp)
target_link_libraries(UtilsTest ${OpenCV_LIBS} vslam_utils)

add_executable(MatcherTest test/test_matcher.cpp)
target_link_libraries(MatcherTest g2o_core g2o_stuff ${OpenCV_LIBS} vslam)

add_executable(FeatureTest test/test_feature.cpp)
target_link_libraries(FeatureTest g2o_core g2o_stuff ${OpenCV_LIBS} vslam)

add_executable(TwoFrameEstimatorTest test/test_twoframe_estimator.cpp)
target_link_libraries(TwoFrameEstimatorTest g2o_core g2o_stuff ${OpenCV_LIBS} vslam)

add_executable(SingleFrameEstTest test/test_singleframe_estimator.cpp)
target_link_libraries(SingleFrameEstTest ${OpenCV_LIBS} vslam)

add_executable(CameraTest test/test_camera.cpp)
target_link_libraries(CameraTest ${OpenCV_LIBS})

add_executable(JaccobianTest test/test_jaccobian.cpp)
target_link_libraries(JaccobianTest ${OpenCV_LIBS})

add_executable(G2OStaffSE3Test test/test_g2o_staff_se3.cpp)
target_link_libraries(G2OStaffSE3Test ${PCL_LIBRARIES} vslam g2o_core g2o_stuff)

add_executable(SophusTest test/test_sophus.cpp)

add_executable(InitializerTest test/test_initializer.cpp)
target_link_libraries(InitializerTest g2o_core g2o_stuff ${PCL_LIBRARIES} ${OpenCV_LIBS} vslam)

add_executable(InitializerV1Test test/test_initializer_v1.cpp)
target_link_libraries(InitializerV1Test g2o_core g2o_stuff ${PCL_LIBRARIES} ${OpenCV_LIBS} vslam)

add_executable(FeatureAlignmentTest test/test_feature_alignment.cpp)
target_link_libraries(FeatureAlignmentTest ${OpenCV_LIBS} vslam)

add_executable(DepthFilterTest test/test_depth_filter.cpp)
target_link_libraries(DepthFilterTest ${OpenCV_LIBS} vslam pthread)
#ifndef _ME_VSLAM_COMMON_HPP_
#define _ME_VSLAM_COMMON_HPP_

/**
 * @cpp_headers stdc++
 */
#include <list>
#include <vector>
#include <memory>

#ifdef _ME_SLAM_DEBUG_INFO_
#include <iostream>
#endif

/**
 * @cpp_headers openCV
 */
#include <opencv2/opencv.hpp>

/**
 * @cpp_headers Eigen3
 */
#include <Eigen/Core>

/**
 * @cpp_headers Sophus
 */
#include <sophus_templ/se3.hpp>
#include <sophus_templ/sim3.hpp>

namespace vslam {

    using Matrix23d = Eigen::Matrix<double, 2, 3>;
    using Matrix32d = Eigen::Matrix<double, 3, 2>;
    using Matrix26d = Eigen::Matrix<double, 2, 6>;
    using Matrix62d = Eigen::Matrix<double, 6, 2>;

    template <typename _Tp>
    using vptr = std::shared_ptr<_Tp>;

    struct feature;
    struct frame;
    struct map_point;
    struct abstract_camera;

    using feature_ptr   = vptr<feature>;
    using frame_ptr     = vptr<frame>;
    using map_point_ptr = vptr<map_point>;
    using camera_ptr    = vptr<abstract_camera>;
}

#endif
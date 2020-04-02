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

/**
 * @cpp_headers fast
 */ 
#include <fast/fast.h>

namespace vslam {

    using Matrix23d = Eigen::Matrix<double, 2, 3>;
    using Matrix32d = Eigen::Matrix<double, 3, 2>;
    using Matrix26d = Eigen::Matrix<double, 2, 6>;
    using Matrix62d = Eigen::Matrix<double, 6, 2>;

    template <typename _Tp>
    using vptr = std::shared_ptr<_Tp>;

    struct config;
    struct feature;
    struct abstract_detector;
    struct frame;
    struct map_point;
    struct abstract_camera;
    struct initializer;
    struct corner;

    /* pointers */
    using config_ptr      = vptr<config>;
    using feature_ptr     = vptr<feature>;
    using detector_ptr    = vptr<abstract_detector>;
    using frame_ptr       = vptr<frame>;
    using map_point_ptr   = vptr<map_point>;
    using camera_ptr      = vptr<abstract_camera>;
    using initializer_ptr = vptr<initializer>;   

    /* sets */
    using corner_set  = std::vector<corner>;
    using feature_set = std::list<feature_ptr>;
    using pyramid_t   = std::vector<cv::Mat>;
}

#endif
#ifndef _ME_VSLAM_COMMON_HPP_
#define _ME_VSLAM_COMMON_HPP_

#define CONST_EPS    1e-8
#define CONST_COS_60 0.5
#define CONST_COS_45 0.70711

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

/**
 * @cpp_headers g2o
 */ 
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

/**
 * @cpp_headers vslam utils
 */
#include <utils/config.hpp>
#include <utils/patch.hpp>
#include <utils/diff.hpp>
#include <utils/threading.hpp>
#include <utils/utils.hpp>

namespace Eigen {

    using Matrix23d = Eigen::Matrix<double, 2, 3>;
    using Matrix32d = Eigen::Matrix<double, 3, 2>;
    using Matrix26d = Eigen::Matrix<double, 2, 6>;
    using Matrix62d = Eigen::Matrix<double, 6, 2>;
}

namespace vslam {

    template <typename _Tp>
    using vptr = std::shared_ptr<_Tp>;

    template <typename _Tp>
    using wptr = std::weak_ptr<_Tp>;

    struct feature;
    struct abstract_detector;
    struct frame;
    struct map_point;
    struct abstract_camera;
    struct initializer;
    struct corner;
    struct patch_matcher;
    struct map;
    struct map_point_seed;
    struct candidate_set;
    struct reprojector;
    struct depth_filter;
    struct twoframe_estimator;
    struct singleframe_estimator;

    /* smart pointers for strong types */
    using feature_ptr      = vptr<feature>;
    using detector_ptr     = vptr<abstract_detector>;
    using frame_ptr        = vptr<frame>;
    using map_point_ptr    = vptr<map_point>;
    using camera_ptr       = vptr<abstract_camera>;
    using initializer_ptr  = vptr<initializer>;   
    using matcher_ptr      = vptr<patch_matcher>;
    using map_ptr          = vptr<map>;
    using reprojector_ptr  = vptr<reprojector>;
    using depth_filter_ptr = vptr<depth_filter>;
    using tf_estimator_ptr = vptr<twoframe_estimator>;
    using sf_estimator_ptr = vptr<singleframe_estimator>;

    /* smart pointers for constant types */
    using feature_cptr      = vptr<const feature>;
    using detector_cptr     = vptr<const abstract_detector>;
    using frame_cptr        = vptr<const frame>;
    using map_point_cptr    = vptr<const map_point>;
    using camera_cptr       = vptr<const abstract_camera>;
    using initializer_cptr  = vptr<const initializer>;   
    using matcher_cptr      = vptr<const patch_matcher>;
    using map_cptr          = vptr<const map>;
    using reprojector_cptr  = vptr<const reprojector>;
    using depth_filter_cptr = vptr<const depth_filter>;

    /* weak pointers */
    using frame_wptr      = wptr<frame>;
    using map_point_wptr  = wptr<map_point>;
    using feature_wptr    = wptr<feature>;


    /* sets */
    using corner_set  = std::vector<corner>;
    using feature_set = std::list<feature_ptr>;
    using pyramid_t   = std::vector<cv::Mat>;

    using patch4b1_uint8_t   = utils::patch2d<uint8_t, 4, 1>;
    using patch4b1_float64_t = utils::patch2d<double, 4, 1>;
    using patch2b1_float64_t = utils::patch2d<double, 2, 1>;

    using patch_t = patch4b1_uint8_t;

#define VSLAM_IN     // readonly variable
#define VSLAM_OUT    // output variable
#define VSLAM_IN_OUT // read and modified variable

    template <typename _ObjTp, typename _ScoreTp>
    using obj_with_score = std::pair<_ObjTp, _ScoreTp>;

    // template <typename _ObjTp, typename _ScoreTp>
    // inline bool operator<(
    //     const obj_with_score<_ObjTp, _ScoreTp>& left, 
    //     const obj_with_score<_ObjTp, _ScoreTp>& right
    // ) {
    //     return left.second < right.second;
    // }

    using frame_with_distance = obj_with_score<frame_ptr, double>;
    using frame_with_overlaps = obj_with_score<frame_ptr, size_t>;
}

namespace utils {

    /**
     * contruct(mk->make) a smart point to indicated type _Tp
     */ 
    template <typename _Tp, typename... _Args>
    vslam::vptr<_Tp> mk_vptr(_Args&&... _args) {
        typedef typename std::remove_cv<_Tp>::type _Tp_nc;
        return std::allocate_shared<_Tp>(
            std::allocator<_Tp_nc>(), std::forward<_Args>(_args)...
        );
    }
}
    

#endif
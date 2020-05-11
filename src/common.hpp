#ifndef _ME_VSLAM_COMMON_HPP_
#define _ME_VSLAM_COMMON_HPP_

#define CONST_EPS    1e-8
#define CONST_COS_60 0.5
#define CONST_COS_45 0.707106781186547524400

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
//#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
//#include <g2o/solvers/csparse/linear_solver_csparse.h>

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

    using Matrix27d = Eigen::Matrix<double, 2, 7>;
    using Matrix72d = Eigen::Matrix<double, 7, 2>;

    using Matrix36d = Eigen::Matrix<double, 3, 6>;
    using Matrix63d = Eigen::Matrix<double, 6, 3>;

    using Matrix37d = Eigen::Matrix<double, 3, 7>;
    using Matrix73d = Eigen::Matrix<double, 7, 3>;
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

namespace vslam::backend {

    struct vertex_se3;
    struct vertex_sim3;
    struct vertex_xyz;
    struct edge_xyz2uv;
    struct edge_xyz2xy1_se3;
    struct edge_se3_to_se3;

    struct g2o_optimizer;
}

namespace config {

    extern const int    height;
    extern const int    width;
    extern const double fx, fy, cx, cy;
    extern const double k1, k2, k3;

    extern const int    pyr_levels;             
    extern const int    cell_sz;      
    extern const int    max_opt_iterations; 
    extern const double opt_converged_thresh_lk;
    extern const double opt_converged_thresh_uv;
    extern const double opt_converged_thresh_xyz;
    extern const double opt_converged_thresh_eps;

    extern const double max_reproj_err_uv;
    extern const double max_reproj_err_xy1;         

    extern const int    min_features_in_first;
    extern const int    min_features_to_tracked;
    extern const double min_init_shift;
    extern const int    min_inliers;
    extern const double min_inlier_ratio;
    extern const int    cv_lk_win_sz;
    extern const double init_scale;

    extern const double min_corner_score;
    extern const double seed_converged_ratio;

    extern const double min_epipolar_search_ncc;   
    extern const int    max_epipolar_search_steps; 
    extern const double max_angle_between_epi_grad;
    extern const double min_len_to_epipolar_search;
    extern const double epipolar_search_step;      


    extern const int    min_reproj_mps;
    extern const int    max_mps_to_local_opt;
    extern const double max_drop_ratio;

    extern const int    max_global_map_frames;
    extern const int    max_local_map_frames;
    extern const double min_key_frame_shift_x;
    extern const double min_key_frame_shift_y;
    extern const double min_key_frame_shift_z;

    extern const int    max_seed_lifetime; // how many key frames in the lifetime of a seed

    extern const int    max_mps_to_reproj;
    extern const int    max_overlaped_key_frames;
    extern const int    max_candidate_mp_fail_reproj;
    extern const int    max_unknown_mp_fail_reproj;
    extern const int    min_good_mp_success_reproj;
}

#define _ME_VSLAM_DEBUG_INFO_ 1

#endif
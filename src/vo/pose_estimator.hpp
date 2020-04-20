#ifndef _ME_VSLAM_POSE_ESTIMATOR_HPP_
#define _ME_VSLAM_POSE_ESTIMATOR_HPP_

#include <common.hpp>

namespace vslam {

    /**
     * @brief two-frame pose only estimator using LK optical-flow. 
     *        given two frames, estimate the coarse SE(3) pose.
     * @note reference frame contains the 2d features and 
     *       corresponding 3d map points, but there is no 
     *       features matched on current frame
     */ 
    struct twoframe_estimator {

        /**
         * @brief the base class of pose estimation algorithm 
         */ 
        struct algo_impl {

            algo_impl() = default;
            virtual ~algo_impl() = default;

            virtual size_t optimize_single_level(
                const vslam::frame_ptr&   ref,
                const vslam::frame_ptr&   cur,
                size_t                    level,
                size_t                    n_iterations,
                Sophus::SE3d&             t_cr
            ) = 0;
        };

        static constexpr int win_half_sz = 2;
        static constexpr int win_sz      = 4;

        enum algorithm { LK_FCFA, LK_ICIA, LK_ICIA_G2O };

        explicit twoframe_estimator(
            size_t    n_iterations = 10, 
            size_t    max_level    = 4, 
            size_t    min_level    = 0, 
            algorithm algo         = LK_ICIA
        );

        ~twoframe_estimator() = default;
        
        void estimate(
            const frame_ptr& ref, 
            const frame_ptr& cur, 
            Sophus::SE3d&    t_cr
        );

    private:
        using algo_ptr = vptr<algo_impl>;

        size_t   _n_iterations;
        size_t   _min_level;
        size_t   _max_level;
        algo_ptr _algo_impl; 
    };

    /**
     * @brief single-frame pose only estimator. given a frame
     *        including initial pose, features and corresponding 
     *        3d map points, estimates (or refines) SE(3) pose
     */ 
    struct singleframe_estimator {

        struct algo_impl {

            algo_impl() = default;
            virtual ~algo_impl() = default;

            virtual size_t estimate(
                const frame_ptr& frame, 
                size_t           n_iterations,
                Sophus::SE3d&    t_cw
            ) = 0;
        };

        enum algorithm { PNP_BA, PNP_G2O/* not implement */, PNP_CV, EPNP_CV, PNP_DLS_CV };

        singleframe_estimator(size_t n_iterations, algorithm algo = PNP_BA);
        virtual ~singleframe_estimator() = default;

        size_t estimate(const frame_ptr& frame, Sophus::SE3d& t_cw) {
            return _algo_impl->estimate(frame, _n_iterations, t_cw);
        }

        static void compute_inliers_and_reporj_err(
            const frame_ptr&          frame, 
            const Sophus::SE3d&       t_cw, 
            double                    reproj_thresh,
            std::vector<feature_ptr>& inliers,
            std::vector<feature_ptr>& outliers,
            double&                   err
        );

    private:

        using algo_ptr = vptr<algo_impl>;

        size_t   _n_iterations;
        algo_ptr _algo_impl;
    };
    
} // namespace vslam

#endif
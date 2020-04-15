#ifndef _ME_VSLAM_POSE_ESTIMATOR_HPP_
#define _ME_VSLAM_POSE_ESTIMATOR_HPP_

#include <common.hpp>

namespace vslam {

    /**
     * @brief the base class of pose estimation algorithm 
     */ 
    struct _pose_est_algo {

        _pose_est_algo() = default;

        virtual ~_pose_est_algo() = default;

        virtual size_t optimize_single_level(
            const vslam::frame_ptr&   ref,
            const vslam::frame_ptr&   cur,
            size_t                    level,
            size_t                    n_iterations,
            Sophus::SE3d&             t_cr
        ) = 0;
    };

    /**
     * @brief pose only estimator. given two frames, 
     *        estimate the coarse SE(3) pose.
     */ 
    struct pose_estimator {

        static constexpr int win_half_sz = 2;
        static constexpr int win_sz      = 4;

        using algo_ptr = vptr<_pose_est_algo>;

        enum algorithm { FCFA, ICIA, ICIA_G2O };

        explicit pose_estimator(
            size_t    n_iterations = 10, 
            size_t    max_level    = 4, 
            size_t    min_level    = 0, 
            algorithm algo         = ICIA
        );

        ~pose_estimator() = default;
        
        void estimate(
            const frame_ptr& ref, 
            const frame_ptr& cur, 
            Sophus::SE3d&    t_cr
        );

    private:
        size_t   _n_iterations;
        size_t   _min_level;
        size_t   _max_level;
        algo_ptr _algo_impl; 
    };
    
} // namespace vslam

#endif
#ifndef _ME_VSLAM_POSE_ESTIMATOR_HPP_
#define _ME_VSLAM_POSE_ESTIMATOR_HPP_

#include <common.hpp>

namespace vslam {

    /**
     * @brief pose only estimator. given two frames, 
     *        estimate the coarse SE(3) pose.
     */ 
    struct pose_estimator {

        pose_estimator() = default;
        ~pose_estimator() = default;
        
        void reset();

        size_t estimate(
            const frame_ptr& ref, 
            const frame_ptr& cur, 
            Sophus::SE3d&    t_cr
        );

    private:
        void _calc_residuals();

        void _solve();

        void _update();

        void _precalc_cache(const cv::Mat& img_level);

        /**
         * @brief since the the jaccobians and the feature 
         *        patches of reference frame will be used 
         *        several times
         * @field caches 
         */
        std::vector<Sophus::Vector6d> _jaccbians_ref;
        std::vector<patch_t>          _patches_ref;
        std::vector<bool>             _visibles;
    };
    
} // namespace vslam

#endif
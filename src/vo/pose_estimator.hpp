#ifndef _ME_VSLAM_POSE_ESTIMATOR_HPP_
#define _ME_VSLAM_POSE_ESTIMATOR_HPP_

#include <common.hpp>

namespace vslam {

    /**
     * @brief pose only estimator. given two frames, 
     *        estimate the coarse SE(3) pose.
     */ 
    struct pose_estimator {

        using patch_type = utils::patch2d<float, 2, 0>;

        pose_estimator();
        ~pose_estimator();
        
        size_t estimate(
            const frame_ptr& ref, 
            const frame_ptr& cur, 
            Sophus::SE3d&    t_cr
        );

    private:
        void _calc_residuals();

        void _precalc_cache(const frame_ptr& ref, size_t level);
        void _clear_cache();
        void _init_graph(const frame_ptr& ref, const frame_ptr& cur, size_t level);

        size_t       _n_iterations;
        size_t       _min_level;
        size_t       _max_level;
        Sophus::SE3d _t_cr;

        /**
         * @brief since the the jaccobians and the feature 
         *        patches of reference frame will be used 
         *        several times
         * @field caches 
         */
        std::vector<Eigen::Matrix<double, patch_type::area, 6>> _jaccobians_ref;
        std::vector<patch_type>                                 _patches_ref;
        std::vector<bool>                                       _visibles_ref;

        /**
         * @field g2o staff
         */ 
        g2o::OptimizationAlgorithm* _algo;
        g2o::SparseOptimizer        _optimizer;
    };
    
} // namespace vslam

#endif
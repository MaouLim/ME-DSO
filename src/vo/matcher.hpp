#ifndef _ME_VSLAM_MATCHER_HPP_
#define _ME_VSLAM_MATCHER_HPP_

#include <common.hpp>

namespace vslam {

    struct patch_matcher {
        
        static const int patch_half_sz;
        static const int path_sz;
        

    };

    namespace affine {
        
        /**
         * @brief calculate the affine matrix from SE(3)
         * @param camera    camera model
         * @param uv_ref    the pixel coordinate of the point on the reference image
         * @param level_ref the pyramid level of the feature
         * @param z_est     the  z(depth) estimation of the point in the ref camera coordinate sys
         * @param t_cr      the SE3 transformation from ref to cur
         */ 
        Eigen::Matrix2d affine_mat(
            const camera_ptr&      camera, 
            const Eigen::Vector2d& uv_ref,
            size_t                 level_ref,
            double                 z_est,
            const Sophus::SE3d&    t_cr
        );

        size_t search_best_level(
            const Eigen::Matrix2d& affine_mat, 
            size_t                 max_level
        );

        /**
         * @brief extract a patch of reference image at 
         *        feature point with (uv_at_level0, level_n)
         * @param image_leveln_ref image at level n
         * @param uv_level0_ref    pixel coordinate at level 0
         * @param level_ref        level n
         * @param affine_rc        affine matrix from cur frame to ref frame
         * @param patch_half_sz    
         * @param patch            the head address of the patch data
         */ 
        bool extract_patch_affine(
            const cv::Mat&         image_leveln_ref,
            const Eigen::Vector2d& uv_level0_ref,
            size_t                 level_ref,
            const Eigen::Matrix2d& affine_rc, 
            size_t                 search_level,
            int                    patch_half_sz,
            uint8_t*               patch
        );
    }

} // namespace vslam

#endif
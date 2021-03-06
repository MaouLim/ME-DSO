#ifndef _ME_VSLAM_MATCHER_HPP_
#define _ME_VSLAM_MATCHER_HPP_

#include <common.hpp>

namespace vslam {

    struct patch_matcher {
        
        bool use_alignment_1d;
        bool edgelet_filtering;

        explicit patch_matcher(bool _use_align1d = false, bool _filter_edgelet = true) : 
            use_alignment_1d(_use_align1d), edgelet_filtering(_filter_edgelet) { }
        ~patch_matcher() = default;

        /**
         * @brief find the matched feature (on ref frame) according to the 
         *        smallest view angle principle and refine the feature on 
         *        the current frame
         * @param mp        VSLAM_IN_OUT map point viewed by cur frame
         * @param cur       VSLAM_IN     current frame
         * @param uv_cur    VSLAM_IN     the pixel coordinate (level 0) of the map point on the 
         *                               current frame, and it will be refined 
         * @param candidate VSLAM_OUT    the feature returned, the association between mp and 
         *                               candidate is built
         */ 
        bool match_direct(
            const map_point_ptr&   mp, 
            const frame_ptr&       cur, 
            const Eigen::Vector2d& uv_cur,
            feature_ptr&           candidate
        );

        /**
         * @brief find the matched feature (on cur frame) using searching along 
         *        the epipolar and estimate the depth of the point described by
         *        the feature on the ref frame
         * @param ref        reference frame
         * @param cur        current frame
         * @param feat_ref   feature on the reference frame
         * @param depth_min  lower-bound of the depth estimation
         * @param depth_max  upper-bound of the depth estimation
         * @param depth_est  initial depth estimation of the feat_ref, if the
         *                   function return true, depth_est will be replace by
         *                   the lastest estimation
         * @param uv_macthed matched uv coordinate of feature on the current frame
         * @return whether the match is found
         */
        bool match_epipolar_search(
            const frame_ptr&   ref,
            const frame_ptr&   cur,
            const feature_ptr& feat_ref,
            double             depth_min,
            double             depth_max,
            double&            depth_est,
            Eigen::Vector2d&   uv_macthed
        );
    
    private:
        //uint8_t _patch[patch_area]                         ;
        //uint8_t _patch_with_border[patch_with_border_area]; //__attribute__ ((aligned (16)));
        static constexpr int _check_sz = patch_t::half_sz + patch_t::border_sz;

        patch_t _patch;
    };

    namespace affine {
        
        /**
         * @brief calculate the affine matrix from SE(3)
         * @param camera        camera model
         * @param uv_ref        the pixel coordinate of the point on the reference image
         * @param level_ref     the pyramid level of the feature
         * @param z_est         the  z(depth) estimation of the point in the ref camera coordinate sys
         * @param patch_half_sz the half size of the patch sampled to calculate
         * @param t_cr          the SE3 transformation from ref to cur
         * @return              the affine transformation form ref to cur
         */ 
        Eigen::Matrix2d affine_mat(
            const camera_ptr&      camera, 
            const Eigen::Vector2d& uv_ref,
            size_t                 level_ref,
            double                 z_est,
            int                    patch_half_sz,
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
         * @param patch            patch data
         */ 
        bool extract_patch_affine(
            const cv::Mat&         image_leveln_ref,
            const Eigen::Vector2d& uv_level0_ref,
            size_t                 level_ref,
            const Eigen::Matrix2d& affine_rc, 
            size_t                 search_level,
            patch_t&               patch
        );
    }

    struct alignment {
        
        /**
         * @brief using ICIA to align the feature along the indicted direction 
         * @cite Lucas-Kanade 20 Years On: A Unifying Framework
         */
        static bool align1d(
            const cv::Mat&         img_cur, 
            const Eigen::Vector2d& orien, 
            const patch_t&         patch, 
            size_t                 n_iterations,
            Eigen::Vector2d&       uv_cur
        );

        /**
         * @brief using ICIA to align the feature
         * @cite Lucas-Kanade 20 Years On: A Unifying Framework
         */ 
        static bool align2d(
            const cv::Mat&   img_cur, 
            const patch_t&   patch, 
            size_t           n_iterations,
            Eigen::Vector2d& uv_cur
        );
    };

} // namespace vslam

#endif
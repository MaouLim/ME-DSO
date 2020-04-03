#ifndef _ME_VSLAM_INITIALIZER_HPP_
#define _ME_VSLAM_INITIALIZER_HPP_

#include <common.hpp>

namespace vslam {

    struct initializer {

        static const int    min_ref_features;
        static const int    min_features_to_tracked;
        static const double min_init_shift;
        static const int    min_inliers;
        static const double map_scale;
        static const double viewport_border;

        enum op_result { 
            FEATURES_NOT_ENOUGH, 
            NO_REF_FRAME, 
            SHIFT_NOT_ENOUGH, 
            INLIERS_NOT_ENOUGH,
            SUCCESS 
        };

        frame_ptr    ref;
        Sophus::SE3d t_cr; // from ref to cur frame

        initializer() : ref(nullptr) { }
        ~initializer() = default;

        op_result set_first(const frame_ptr& first);
        op_result add_frame(const frame_ptr& frame);
        void reset();

    private:
        std::vector<cv::Point2f>     _uvs_ref;
        std::vector<cv::Point2f>     _uvs_cur;
        std::vector<Eigen::Vector3d> _xy1s_ref;
        std::vector<Eigen::Vector3d> _xy1s_cur;

        std::vector<double>          _flow_lens;
        std::vector<int>             _inliers;
        std::vector<Eigen::Vector3d> _xyzs_cur;

        /**
         * @return the number of the features detected
         */ 
        size_t _detect_features(const frame_ptr& target);

        /**
         * @return the number of the features tracked
         */ 
        size_t _rerange_tracked_uvs(const std::vector<uchar>& status);

        /**
         * @brief track the feature points in the reference frame
         * @return the number of the features tracked
         */ 
        size_t initializer::_track_lk(const frame_ptr& cur);

        void initializer::_calc_homography(double focal_len, double reproject_threshold);
    };
}

#endif
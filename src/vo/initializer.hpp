#ifndef _ME_VSLAM_INITIALIZER_HPP_
#define _ME_VSLAM_INITIALIZER_HPP_

#include <common.hpp>

namespace vslam {

    struct initializer {

        static constexpr double border = 1.0;

        enum op_result { 
            FEATURES_NOT_ENOUGH, 
            NO_REF_FRAME, 
            SHIFT_NOT_ENOUGH, 
            FAILED_CALC_HOMOGRAPHY,
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
        size_t _rerange(const std::vector<uchar>& status);

        /**
         * @brief track the feature points in the reference frame
         * @return the number of the features tracked
         */ 
        size_t _track_lk(const frame_ptr& cur);

        /**
         * @return succeed to compute the pose from homography matrix 
         */ 
        bool _calc_homography(double err_mul2, double reproject_threshold);

        /**
         * @brief using the pose to triangulate, recover the position 
         *        of the 3d points (in world coord-sys)
         * @return evalute the pose (sum of reprojection error)
         */ 
        double _compute_inliers_and_triangulate(double reproject_threshold);
    };
}

#endif
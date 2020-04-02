#ifndef _ME_VSLAM_INITIALIZER_HPP_
#define _ME_VSLAM_INITIALIZER_HPP_

#include <common.hpp>

namespace vslam {

    struct initializer {

        static const int min_features_to_init;

        enum init_result { FAILURE, NO_KEYFRAME, SUCCESS };

        frame_ptr    ref;
        Sophus::SE3d t_cr; // from ref to cur frame

        initializer() = default;
        ~initializer() = default;

        init_result set_first(const frame_ptr& first);
        init_result add_frame(const frame_ptr& frame);
        void reset();

    private:
        std::vector<cv::Point2f>     _keypoints_ref;
        std::vector<cv::Point2f>     _tracked_cur;
        std::vector<Eigen::Vector3d> _xy1s_ref;
        std::vector<Eigen::Vector3d> _xy1s_cur;

        std::vector<double>          _disparities;
        std::vector<int>             _inliers;
        std::vector<Eigen::Vector3d> _xyz_cur;

        static void _detect_features(
            const frame_ptr&              target, 
            std::vector<cv::Point2f>&     keypoints, 
            std::vector<Eigen::Vector3d>& xy1s
        );
    };

    inline void initializer::reset() {
        ref.reset();
        t_cr = Sophus::SE3d();

        _keypoints_ref.clear();
        _tracked_cur.clear();
        _xy1s_ref.clear();
        _xy1s_cur.clear();

        _disparities.clear();
        _inliers.clear();
        _xyz_cur.clear();
    }
}

#endif
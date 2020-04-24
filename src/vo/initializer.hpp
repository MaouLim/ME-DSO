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
            FAILED_CALC_POSE,
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
        bool _calc_essential();

        /**
         * @brief using the pose to triangulate, recover the position 
         *        of the 3d points (in world coord-sys)
         * @return evalute the pose (sum of reprojection error)
         */ 
        double _compute_inliers_and_triangulate(double reproject_threshold);
    };

    struct initializer_v2 {

        enum op_result { 
            FEATURES_NOT_ENOUGH,
            NO_FRAME, 
            SHIFT_NOT_ENOUGH, 
            FAILED_CALC_POSE,
            INLIERS_NOT_ENOUGH,
            ACCEPT,
            SUCCESS 
        };

        static constexpr int n_init_frames_require = 10;

        initializer_v2() { 
            _frames.reserve(n_init_frames_require);
            _track_table.reserve(n_init_frames_require);
            _n_feats_tracked.reserve(n_init_frames_require);
        }

        void reset();
        op_result add_frame(const frame_ptr& frame);

    private:
        void _calc_optcal_flow(
            const cv::Mat&            img, 
            std::vector<uchar>&       status, 
            std::vector<cv::Point2f>& uvs
        );

        void _shrink(
            const std::vector<uchar>&       status, 
            const std::vector<cv::Point2f>& uvs,
            std::vector<cv::Point2f>&       shrinked
        );

        void _recover(
            const std::vector<uchar>& status_last, 
            std::vector<uchar>&       status,
            std::vector<cv::Point2f>& uvs
        );

        bool _calc_homography(
            double err_mul2, double reproject_threshold, Sophus::SE3d& t_cr
        );

        double _compute_inliers_and_triangulate(
            double reproject_threshold, const Sophus::SE3d& t_10
        );
    
        std::vector<
            std::pair<std::vector<uchar>, std::vector<cv::Point2f>>
        >                      _track_table;
        std::vector<size_t>    _n_feats_tracked;
        std::vector<frame_ptr> _frames;

        /**
         * @field caches
         */ 
        std::vector<double>          _shift_cache;
        std::vector<cv::Point2f>     _uvs_cache;
        std::vector<Eigen::Vector3d> _xy1s_ref;
        std::vector<Eigen::Vector3d> _xy1s_cur;

        std::vector<uchar>           _inliers_f1;
        std::vector<Eigen::Vector3d> _xyzs_f1;
    };
}

#endif
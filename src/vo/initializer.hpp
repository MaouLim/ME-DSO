#ifndef _ME_VSLAM_INITIALIZER_HPP_
#define _ME_VSLAM_INITIALIZER_HPP_

#include <common.hpp>

namespace vslam {

    struct initializer {

        static constexpr double border             = 1.0;
        static constexpr size_t max_track_failures = 40;

        enum op_result { 
            REF_FRAME_SET = -1,
            SUCCESS       =  0,
            FEATURES_NOT_ENOUGH, 
            SHIFT_NOT_ENOUGH, 
            FAILED_CALC_POSE,
            INLIERS_NOT_ENOUGH
        };

        frame_ptr    ref;
        Sophus::SE3d t_cr; // from ref to cur frame

        initializer() : ref(nullptr), _n_track_failures(0) { }
        ~initializer() = default;

        op_result add_frame(const frame_ptr& frame);
        void reset();

    private:
        size_t _n_track_failures;

        std::vector<cv::Point2f>     _uvs_ref;
        std::vector<cv::Point2f>     _uvs_cur;
        std::vector<Eigen::Vector3d> _xy1s_ref;
        std::vector<Eigen::Vector3d> _xy1s_cur;

        std::vector<double>          _flow_lens;
        std::vector<int>             _inliers;
        std::vector<Eigen::Vector3d> _xyzs_cur;

        /**
         * @brief when fails to track a frame
         */ 
        void _handle_failure();

        /**
         * @return the number of the features detected
         */ 
        size_t _detect_features(const frame_ptr& target);
        size_t _detect_features_v2(const frame_ptr& target);

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
        bool _calc_homography(double err_mul2, Sophus::SE3d& se3) const;
        bool _calc_essential(Sophus::SE3d& se3) const;

        /**
         * @brief using the pose to triangulate, recover the position 
         *        of the 3d points (in world coord-sys)
         * @param se3 the SE(3) transformation from ref to cur
         * @return evalute the pose (mean reprojection error on ref and cur frames of all the inliers)
         */ 
        double _compute_inliers_and_triangulate(
            const Sophus::SE3d&           se3, 
            std::vector<Eigen::Vector3d>& xyzs_cur, 
            std::vector<int>&             inlier_indices
        ) const;
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

        static constexpr int min_init_frames = 10;
        static constexpr int max_trials      = 10;

        std::vector<Sophus::SE3d> _poses_opt;
        Sophus::SE3d t_10, t_21;
        std::vector<Eigen::Vector3d> final_xyzs_f1;
        std::vector<Eigen::Vector2d> final_uvs_f1;

        initializer_v2();
        ~initializer_v2() = default;

        void reset();
        op_result add_frame(const frame_ptr& frame);

    private:
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

        void _reverve_cache(double n_init_feats);

        void _calc_optcal_flow(
            const cv::Mat&            img_last,
            const cv::Mat&            img_cur, 
            const std::vector<uchar>& status_last,
            std::vector<uchar>&       status, 
            std::vector<cv::Point2f>& uvs
        );

        bool _calc_homography(
            const std::vector<Eigen::Vector3d>& xy1s_ref,
            const std::vector<Eigen::Vector3d>& xy1s_cur,
            double                              err_mul2, 
            double                              reproject_threshold, 
            Sophus::SE3d&                       t_cr
        ) const;

        double _triangulate_frame01(
            double                        reproj_threshold,  
            const Sophus::SE3d&           t_10, 
            std::vector<uchar>&           inliers_f1,
            std::vector<Eigen::Vector3d>& xyzs_f1
        ) const;

        void _pose_only_optimize(
            const camera_ptr& cam, const std::vector<Sophus::SE3d>& poses
        );

        double _triangulate_frame012(
            const Sophus::SE3d& t_10, 
            const Sophus::SE3d& t_21, 
            std::vector<uchar>& status
        ) { }
//private:
    public:
        using track_record = 
            std::pair<std::vector<uchar>, std::vector<cv::Point2f>>;

        std::vector<track_record> _track_table;
        size_t                    _n_init_feats;
        size_t                    _n_final_feats;
        size_t                    _count_failures;
        std::vector<frame_ptr>    _frames;

        detector_ptr _det;

        std::vector<Eigen::Vector3d> _xyzs_f1;

        /**
         * @field caches
         */ 
        std::vector<double>          _shift_cache; 
        std::vector<cv::Point2f>     _uvs_cache;   
        std::vector<Eigen::Vector3d> _xy1s_ref;
        std::vector<Eigen::Vector3d> _xy1s_cur;

        /**
         * @field g2o staff
         */ 
        std::vector<backend::vertex_xyz*>  _vs_mp;
        std::vector<backend::vertex_se3*>  _vs_f;
    };
}

#endif
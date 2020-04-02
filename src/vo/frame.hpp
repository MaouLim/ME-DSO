#ifndef _ME_VSLAM_FRAME_HPP_
#define _ME_VSLAM_FRAME_HPP_

#include <common.hpp>

namespace vslam {

    struct frame {

        static const size_t N_GOOD_FEATURES = 5;

        using good_features_t = std::array<feature_ptr, N_GOOD_FEATURES>;

        static double pyr_scale;
        static size_t pyr_levels;

        bool                   key_frame;
        int                    id;
        double                 timestamp;
        Sophus::SE3d           t_cw;          // transform the world coordinate sys to the current frame camera coordinate sys 
        Sophus::SE3d           t_wc;          // the inverse of t_cw 
        camera_ptr             camera;
        std::list<feature_ptr> features;
        size_t                 n_features;
        good_features_t        good_features;
        pyramid_t              pyramid;       // pyramid images, 0 level is the original.

        // std::vector<cv::Mat>    pyr_gx;   // gx = dI/du
        // std::vector<cv::Mat>    pyr_gy;   // gy = dI/dv
        // std::vector<cv::Mat>    pyr_grad; // grad = sqrt(gx^2+gy^2)

        frame(const camera_ptr& _cam, const cv::Mat& _img, double _timestamp, bool _key_frame = false);
        ~frame() = default;

        void as_key_frame() { key_frame = true; _set_good_features(); }
        const cv::Mat& image() const { return pyramid[0]; }
        const Eigen::Vector3d& cam_center() const { return t_wc.translation(); }
        void set_pose(const Sophus::SE3d& _t_cw) { t_cw = _t_cw; t_wc = t_cw.inverse(); }

        bool visible(const Eigen::Vector3d& p_w, double border = 0.0) const;
        void min_and_median_depth(double& min, double& median) const;

        void add_feature(const feature_ptr& _feat) { features.push_front(_feat); ++n_features; }
        bool remove_good_feature(const feature_ptr& _feat);

    private:
        static int _seq_id;

        void _set_good_features() { _remove_useless_features(); _select_good_features(); }
        void _remove_useless_features();
        void _select_good_features();
        void _check_good_feat(const feature_ptr& candidate);
    };
}

#endif

#include <vo/initializer.hpp>
#include <vo/feature.hpp>
#include <vo/frame.hpp>
#include <vo/camera.hpp>
#include <vo/map_point.hpp>
#include <vo/homography.hpp>

#include <utils/config.hpp>
#include <utils/utils.hpp>

namespace vslam {

    initializer::op_result 
    initializer::set_first(const frame_ptr& first) 
    {
        reset();
        size_t n_detected = _detect_features(first);

        if (n_detected < config::min_features_in_first) {
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cerr << "Failed to init, too few keypoints: " 
                      << n_detected << std::endl;
#endif            
            return FEATURES_NOT_ENOUGH;
        }

        ref = first;
        return SUCCESS;
    }

    initializer::op_result 
    initializer::add_frame(const frame_ptr& frame) {
        if (!ref) { 
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "Reference frame not set." << std::endl;
#endif      
            return NO_REF_FRAME;
        }

        const frame_ptr& cur = frame;
        size_t n_tracked = _track_lk(cur);

        if (n_tracked < config::min_features_to_tracked) {
            return FEATURES_NOT_ENOUGH;
        }

        double median_len = *utils::median(_flow_lens.begin(), _flow_lens.end());
        if (median_len < config::min_init_shift) {
            return SHIFT_NOT_ENOUGH;
        }

        if (!_calc_homography(cur->camera->err_mul2(), config::max_reproj_err)) { 
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "Failed to compute pose from homography matrix." << std::endl;
#endif      
            return FAILED_CALC_HOMOGRAPHY;
        }
        double total_err = _compute_inliers_and_triangulate(config::max_reproj_err);
        size_t n_inliers = _inliers.size();

#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "Inliers: " << n_inliers << std::endl;
            std::cout << "Ratio of inliers: " << double(n_inliers) / _xy1s_ref.size() << std::endl;
            std::cout << "Total reprojection error: " << total_err << std::endl;
#endif      
        /**
         * @todo minimun requirement of the number of the inliers 
         *       minimun requirement of the ratio of the inliers
         * 
         * double inliers_ratio = double(n_inliers) / _xy1s_ref.size();
         * if (inliers_ratio < min_inliers_ratio) {
         *     return INLIERS_NOT_ENOUGH;
         * }
         */ 
        if (n_inliers < config::min_inliers) {
            return INLIERS_NOT_ENOUGH;
        }

        std::vector<double> depths; 
        depths.reserve(_xyzs_cur.size());
        for (size_t i = 0; i < _xyzs_cur.size(); ++i) {
            depths.push_back(_xyzs_cur[i].z());
        }

        double median_depth = *utils::median(depths.begin(), depths.end());
        double scale = config::init_scale / median_depth;
        
        /**
         * R1 = R * R0
         * t1 = R * t0 + scale * t
         * 
         * R0 seems to be I
         * t0 seems to be Zero vec
         */
        Sophus::SO3d    rot_cur   = t_cr.so3() * ref->t_cw.so3();
        Eigen::Vector3d trans_cur = t_cr.so3() * ref->t_cw.translation() + scale * t_cr.translation();
        cur->set_pose({ rot_cur, trans_cur });

        for (auto& inlier_idx : _inliers) {

            auto uv_ref = utils::eigen_vec(_uvs_ref[inlier_idx]);
            auto uv_cur = utils::eigen_vec(_uvs_cur[inlier_idx]);

            if (ref->visible(uv_ref, border) && 
                cur->visible(uv_cur, border)) 
            {
                map_point_ptr mp = 
                    utils::mk_vptr<map_point>(cur->t_wc * _xyzs_cur[inlier_idx] * scale);
                feature_ptr feat_ref = 
                    utils::mk_vptr<feature>(ref, uv_ref, 0);
                feature_ptr feat_cur = 
                    utils::mk_vptr<feature>(cur, uv_cur, 0);

                feat_ref->set_describing(mp); feat_ref->use();
                feat_cur->set_describing(mp); feat_cur->use();
            }
        }
        return SUCCESS;
    }

    void initializer::reset() {
        ref.reset();
        t_cr = Sophus::SE3d();

        _uvs_ref.clear(); _uvs_cur.clear();
        _xy1s_ref.clear(); _xy1s_cur.clear();

        _flow_lens.clear();
        _inliers.clear();
        
        _xyzs_cur.clear();
    }

    size_t initializer::_detect_features(const frame_ptr& target) {
        const int h = target->camera->height;
        const int w = target->camera->width;

        feature_set new_features;
        fast_detector detector(h, w, config::cell_sz, config::pyr_levels);
        detector.detect(target, config::min_corner_score, new_features);

        _uvs_ref.clear(); _xy1s_ref.clear();
        for (auto& each : new_features) {
            _uvs_ref.emplace_back(each->uv[0], each->uv[1]);
            _xy1s_ref.emplace_back(each->xy1);
        }
        return _uvs_ref.size();
    }

    size_t initializer::_rerange(const std::vector<uchar>& status) {

        assert(_uvs_ref.size() ==  _uvs_cur.size() && 
               _uvs_ref.size() == _xy1s_ref.size() &&
               _uvs_ref.size() ==    status.size());

        const size_t n_features = status.size();

        size_t i = 0, j = 0;
        while (j < n_features) {
            if (!status[j]) { ++j; }
            if (i != j) {
                _uvs_ref[i] = _uvs_ref[j];
                _uvs_cur[i] = _uvs_cur[j];
                _xy1s_ref[i] = _xy1s_ref[j];
            }
            ++i; ++j;
        }

        _uvs_ref.resize(i);
        _uvs_cur.resize(i);
        _xy1s_ref.resize(i);

        return i;
    }

    size_t initializer::_track_lk(const frame_ptr& cur) {
        /**
         * OpenCV LK optical flow algorithm
         */ 
        std::vector<uchar> status;
        std::vector<float> error;
        cv::TermCriteria criteria(
            cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 
            config::max_opt_iterations, CONST_EPS
        );

        // copy _uvs_ref to _uvs_cur as the initial estimation
        _uvs_cur.resize(_uvs_ref.size());
        std::copy(_uvs_ref.begin(), _uvs_ref.end(), _uvs_cur.begin());

        cv::calcOpticalFlowPyrLK(
            ref->image(), cur->image(), _uvs_ref, _uvs_cur, status, error, 
            cv::Size2i(config::cv_lk_win_sz, config::cv_lk_win_sz), 
            config::pyr_levels, criteria, cv::OPTFLOW_USE_INITIAL_FLOW
        );

        size_t n_tracked = _rerange(status);

        /**
         * calculate the xy1s and disparities
         * count the features tracked in LK 
         */ 

         _xy1s_cur.clear();  _xy1s_cur.reserve(n_tracked);
        _flow_lens.clear(); _flow_lens.reserve(n_tracked);

        for (size_t i = 0; i < n_tracked; ++i) {
            const auto& uv_ref = _uvs_ref[i];
            const auto& uv_cur = _uvs_cur[i];
            _xy1s_cur.push_back(cur->camera->pixel2cam({ uv_cur.x, uv_cur.y }, 1.0));
            _flow_lens.push_back(Eigen::Vector2d(uv_cur.x - uv_ref.x, uv_cur.y - uv_ref.y).norm());
        }

        return n_tracked;
    }

    bool initializer::_calc_homography(
        double err_mul2, double reproject_threshold
    ) {
        homography solver(reproject_threshold, err_mul2, _xy1s_ref, _xy1s_cur);
        return solver.calc_pose_from_matches(t_cr);
    }

    double initializer::_compute_inliers_and_triangulate(
        double reproject_threshold
    ) {
        size_t n_matches = _xy1s_ref.size();

         _inliers.clear();  _inliers.reserve(n_matches);
        _xyzs_cur.clear(); _xyzs_cur.reserve(n_matches);

        double total_err = 0.0;

        for (size_t i = 0; i < n_matches; ++i) {

            //same! Eigen::Vector3d xyz_cur = utils::triangulate(_xy1s_ref[i], _xy1s_cur[i], t_cr);
            Eigen::Vector3d xyz_cur = utils::triangulate_v2(_xy1s_cur[i], _xy1s_ref[i], t_cr);
            _xyzs_cur.push_back(xyz_cur);

            double err_ref = utils::reproject_err(t_cr.inverse() * xyz_cur, _xy1s_ref[i]);
            double err_cur = utils::reproject_err(xyz_cur, _xy1s_cur[i]);

            /**
             * outliers is which reprojection error is too 
             * large or depth is less than zero
             */
            if (reproject_threshold < err_ref || 
                reproject_threshold < err_cur || 
                xyz_cur.z() < 0.0)               
            {
                continue;
            }
            else { _inliers.push_back(i); total_err += (err_ref + err_cur); }
        }
 
        return total_err;
    }
}
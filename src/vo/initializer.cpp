
#include <vo/initializer.hpp>
#include <vo/feature.hpp>
#include <vo/frame.hpp>
#include <vo/camera.hpp>
#include <vo/map_point.hpp>
#include <vo/homography.hpp>

#include <utils/config.hpp>
#include <utils/utils.hpp>

#include <backend/g2o_staff.hpp>

namespace vslam {

    initializer::op_result 
    initializer::add_frame(const frame_ptr& frame) {
        if (!ref) { 
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "[INIT]" << "Reference frame not set, set as a first frame" << std::endl;
#endif      
            //size_t n_detected = _detect_features(frame);
            size_t n_detected = _detect_features_v2(frame);
            if (n_detected < config::min_features_in_first) {
#ifdef _ME_VSLAM_DEBUG_INFO_
                std::cout << "[INIT]" << "Failed to init, too few keypoints: " 
                          << n_detected << std::endl;
#endif            
                return FEATURES_NOT_ENOUGH;
            }

            ref = frame;
            return REF_FRAME_SET;
        }

        const frame_ptr& cur = frame;
        size_t n_tracked = _track_lk(cur);

        if (n_tracked < config::min_features_to_tracked) {
            _handle_failure();
            return FEATURES_NOT_ENOUGH;
        }

        double median_len = *utils::median(_flow_lens.begin(), _flow_lens.end());
        if (median_len < config::min_init_shift) {
            _handle_failure();
            return SHIFT_NOT_ENOUGH;
        }

        /**
         * @note calculate essential
         */
        bool essential_success = false;
        double essential_score = 0.0;
        std::vector<Eigen::Vector3d> xyzs_cur_e;
        std::vector<int> inlier_indices_e;
        Sophus::SE3d t_e;
        {
            essential_success = _calc_essential(t_e);
            double mean_err = _compute_inliers_and_triangulate(t_e, xyzs_cur_e, inlier_indices_e);
            essential_score = double(essential_success) * 
                              double(inlier_indices_e.size()) / (mean_err + config::max_reproj_err_xy1);
        }
        //essential_score = 0;

        /**
         * @note calculate homography
         */ 
        bool homography_success = false;
        double homography_score = 0.0;
        std::vector<Eigen::Vector3d> xyzs_cur_h;
        std::vector<int> inlier_indices_h;
        Sophus::SE3d t_h;
        {
            homography_success = _calc_homography(cur->camera->err_mul2(), t_h);
            double mean_err = _compute_inliers_and_triangulate(t_h, xyzs_cur_h, inlier_indices_h);
            homography_score = double(homography_success) * 
                               double(inlier_indices_h.size()) / (mean_err + config::max_reproj_err_xy1);
        }
        //homography_score = 0;

        if (!essential_success && !homography_success) { 
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "[INIT]" << "Failed to compute pose from matches." << std::endl;
#endif      
            _handle_failure();
            return FAILED_CALC_POSE;
        }
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "[INIT]" << "Homography score: " << homography_score << std::endl;
            std::cout << "[INIT]" << "Essential  score: " << essential_score << std::endl;
#endif              
        if (homography_score < essential_score) {
            _inliers = std::move(inlier_indices_e);
            _xyzs_cur = std::move(xyzs_cur_e);
            t_cr = t_e;
        }
        else {
            _inliers = std::move(inlier_indices_h);
            _xyzs_cur = std::move(xyzs_cur_h);
            t_cr = t_h;
        }
        
        size_t n_inliers = _inliers.size();
        double inliers_ratio = double(n_inliers) / _xy1s_ref.size();
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "[INIT]" << "Inliers: " << n_inliers << std::endl;
            std::cout << "[INIT]" << "Ratio of inliers: " << inliers_ratio << std::endl;
#endif
        if (inliers_ratio < config::min_inlier_ratio || 
                n_inliers < config::min_inliers
        ) {
            _handle_failure();
            return INLIERS_NOT_ENOUGH;
        }

        _n_track_failures = 0;

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

        reset();
        return SUCCESS;
    }

    void initializer::reset() {
        _n_track_failures = 0;
        ref.reset();
        t_cr = Sophus::SE3d();

        _uvs_ref.clear(); _uvs_cur.clear();
        _xy1s_ref.clear(); _xy1s_cur.clear();

        _flow_lens.clear();
        _inliers.clear();
        
        _xyzs_cur.clear();
    }

    void initializer::_handle_failure() { 
        ++_n_track_failures;
        if (max_track_failures < _n_track_failures) { reset(); }
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

    size_t initializer::_detect_features_v2(const frame_ptr& target) {

        size_t max_feats = (config::width / config::cell_sz) * 
                           (config::height / config::cell_sz);
        cv::Ptr<cv::GFTTDetector> det = 
            cv::GFTTDetector::create(max_feats, 0.05, config::cell_sz / 2.);
        std::vector<cv::KeyPoint> kpts;
        det->detect(target->image(), kpts);

        _uvs_ref.clear(); _xy1s_ref.clear();
        for (auto& each : kpts) {
            _uvs_ref.emplace_back(each.pt);
            Eigen::Vector2d uv = { each.pt.x, each.pt.y };
            _xy1s_ref.emplace_back(target->camera->pixel2cam(uv));
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
                _uvs_ref[i]  = _uvs_ref[j];
                _uvs_cur[i]  = _uvs_cur[j];
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
            config::max_opt_iterations, config::opt_converged_thresh_lk
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
        double err_mul2, Sophus::SE3d& se3
    ) const {
        homography solver(config::max_reproj_err_xy1, err_mul2, _xy1s_ref, _xy1s_cur);
        return solver.calc_pose_from_matches(se3);
    }

    bool initializer::_calc_essential(Sophus::SE3d& se3) const {
        cv::Mat mask;
        cv::Mat cam_mat = ref->camera->cv_mat();
        cv::Mat essential = cv::findEssentialMat(_uvs_ref, _uvs_cur, cam_mat, cv::RANSAC, 0.999, 1.0, mask);
        cv::Mat cv_R, cv_t;
        cv::recoverPose(essential, _uvs_ref, _uvs_cur, cam_mat, cv_R, cv_t, mask);
        Eigen::Vector3d t;
        t << cv_t.at<double>(0), cv_t.at<double>(1), cv_t.at<double>(2);
        Eigen::Matrix3d r;
        r << cv_R.at<double>(0, 0), cv_R.at<double>(0, 1), cv_R.at<double>(0, 2), 
             cv_R.at<double>(1, 0), cv_R.at<double>(1, 1), cv_R.at<double>(1, 2), 
             cv_R.at<double>(2, 0), cv_R.at<double>(2, 1), cv_R.at<double>(2, 2);
        se3 = Sophus::SE3d(r, t);
        return true;
    }

    double 
    initializer::_compute_inliers_and_triangulate(
        const Sophus::SE3d&           se3, 
        std::vector<Eigen::Vector3d>& xyzs_cur, 
        std::vector<int>&             inlier_indices
    ) const {
        const Sophus::SE3d& se3_inv = se3.inverse();
        size_t n_matches = _xy1s_ref.size();

        inlier_indices.clear();  inlier_indices.reserve(n_matches);
              xyzs_cur.clear();        xyzs_cur.reserve(n_matches);

        double total_err = 0.0;

        for (size_t i = 0; i < n_matches; ++i) {

            Eigen::Vector3d xyz_cur = utils::triangulate_v2(_xy1s_cur[i], _xy1s_ref[i], se3);
            Eigen::Vector3d xyz_ref = se3_inv * xyz_cur;
            xyzs_cur.push_back(xyz_cur);

            double err_ref = utils::reproject_err(xyz_ref, _xy1s_ref[i]);
            double err_cur = utils::reproject_err(xyz_cur, _xy1s_cur[i]);

            /**
             * outliers is which reprojection error is too 
             * large or depth is less than zero
             */
            if (xyz_cur.z() < 0. || xyz_ref.z() < 0. ||
                config::max_reproj_err_xy1 < err_ref   || 
                config::max_reproj_err_xy1 < err_cur   )               
            {
                continue;
            }
            else { inlier_indices.push_back(i); total_err += (err_ref + err_cur); }
        }

        if (inlier_indices.empty()) { return config::max_reproj_err_xy1 * 2.; }
        return total_err / inlier_indices.size();
    }

    /**
     * initializer v2
     */ 
#ifdef _ME_VSLAM_USE_INITV2_
    initializer_v2::initializer_v2() : 
        _n_init_feats(0), _n_final_feats(0), _count_failures(0)
    { 
        _frames.reserve(min_init_frames);
        _track_table.reserve(min_init_frames);

        _det = utils::mk_vptr<fast_detector>(
             config::height, config::width, 
            config::cell_sz, config::pyr_levels
        ); assert(_det);
    }

    void initializer_v2::reset() {
        _n_init_feats  = 0;
        _n_final_feats = 0;
        _count_failures  = 0;

        _track_table.clear();
        _frames.clear();

        _shift_cache.clear();
        _uvs_cache.clear();
        _xy1s_ref.clear();
        _xy1s_cur.clear();
    }

    initializer_v2::op_result 
    initializer_v2::add_frame(const frame_ptr& frame) {

        /**
         * @note if the first frame is not set
         */ 
        if (_frames.empty()) {

            feature_set new_features;

            size_t n_features = 
                _det->detect(frame, config::min_corner_score, new_features);

            if (n_features < config::min_features_in_first) { 
#ifdef _ME_VSLAM_DEBUG_INFO_
                std::cout << "FEATURES_NOT_ENOUGH" << std::endl;
                std::cout << "Require: " << config::min_features_in_first << std::endl;
                std::cout << "Get: "     << n_features << std::endl;
#endif
                return FEATURES_NOT_ENOUGH;
            }

            _n_init_feats  = n_features;
            _n_final_feats = n_features;

            _reverve_cache(_n_init_feats);
            std::vector<uchar> status(_n_init_feats, 1);
            std::vector<cv::Point2f> uvs;
            uvs.reserve(_n_init_feats);

            for (auto& each : new_features) {
                uvs.emplace_back(each->uv.x(), each->uv.y());
            }
            _uvs_cache = uvs;

            _track_table.emplace_back(std::move(status), std::move(uvs));
            _frames.emplace_back(frame);

            return ACCEPT;
        }

        /**
         * @note the first frame set
         */ 
        const frame_ptr& last = _frames.back();
        const frame_ptr& cur  = frame;

        const auto& status_last = _track_table.back().first;
        std::vector<uchar> status;
        std::vector<cv::Point2f> pts;
        _calc_optcal_flow(
            last->image(), cur->image(), status_last, status, pts
        );

        double median_shift = 
            *utils::median(_shift_cache.begin(), _shift_cache.end());
        if (median_shift < config::min_init_shift) {
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "SHIFT_NOT_ENOUGH" << std::endl;
#endif
            ++_count_failures;
            if (
                max_trials     < _count_failures && 
                _frames.size() < min_init_frames
            ) { 
                reset(); 
#ifdef _ME_VSLAM_DEBUG_INFO_
                std::cout << "Dropped too much frames, initializer reset." << std::endl;
#endif  
            }
            return SHIFT_NOT_ENOUGH;
        }

        size_t n_feats_tracked = _shift_cache.size();
        if (n_feats_tracked < config::min_features_to_tracked) {
            
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "FEATURES_NOT_ENOUGH" << std::endl;
            std::cout << "Require: " << config::min_features_to_tracked << std::endl;
            std::cout << "Get: "     << n_feats_tracked << std::endl;
#endif      
            ++_count_failures;
            if (
                max_trials     < _count_failures && 
                _frames.size() < min_init_frames
            ) { 
                reset(); 
#ifdef _ME_VSLAM_DEBUG_INFO_
                std::cout << "Dropped too much frames, initializer reset." << std::endl;
#endif  
            }
            return FEATURES_NOT_ENOUGH;
        }

        _count_failures = 0;

        _shrink(status, pts, _uvs_cache);
        _track_table.emplace_back(std::move(status), std::move(pts));
        _n_final_feats = n_feats_tracked;
        _frames.emplace_back(frame);
        if (_frames.size() < min_init_frames) { return ACCEPT; }

        /**
         * @note number of frames rearches the requirement
         */ 
        std::vector<Sophus::SE3d> twoview(min_init_frames);
        const camera_ptr& cam = _frames.front()->camera;

        for (size_t i = 1; i < min_init_frames; ++i) {

            _xy1s_ref.clear();
            _xy1s_cur.clear();

            auto& prv = _track_table[i - 1];
            auto& cur = _track_table[i];

            for (size_t j = 0; j < _n_init_feats; ++j) {
                if (!cur.first[j]) { continue; }
                _xy1s_ref.emplace_back(cam->pixel2cam(utils::eigen_vec(prv.second[j])));
                _xy1s_cur.emplace_back(cam->pixel2cam(utils::eigen_vec(cur.second[j])));
            }
            
            if (!_calc_homography(
                _xy1s_ref, _xy1s_cur, cam->err_mul2(), 
                config::max_reproj_err_xy1, twoview[i]
                )
            ) {
#ifdef _ME_VSLAM_DEBUG_INFO_
                std::cout << "FAILED_CALC_POSE" << std::endl;
                std::cout << "Initializer reset." << std::endl;
#endif
                reset();
                return FAILED_CALC_POSE;
            }

            /**
             * @note triangulate using frame 0, 1
             */ 
            if (1 == i) {
                std::vector<uchar>& status_f1 = _track_table[1].first;
                _triangulate_frame01(config::max_reproj_err_xy1, twoview[1], status_f1, _xyzs_f1);
            }
        }

        std::vector<Sophus::SE3d> poses(min_init_frames, Sophus::SE3d());
        for (size_t i = 1; i < min_init_frames; ++i) {
            poses[i] = twoview[i] * poses[i - 1];
        }

        _pose_only_optimize(cam, poses);

        _poses_opt.clear(); _poses_opt.resize(min_init_frames);
        for (size_t i = 1; i < min_init_frames; ++i) {
            _poses_opt[i] = _vs_f[i]->estimate();
        }

        t_10 = _vs_f[1]->estimate();
        Sophus::SE3d t_01 = t_10.inverse();
        t_21 = _vs_f[2]->estimate() * t_01;
        Sophus::SE3d t_12 = t_21.inverse();

        std::vector<uchar>& status_f1 = _track_table[1].first;
        std::vector<Eigen::Vector3d> xy1s_f0, xy1s_f1, xy1s_f2;
        for (size_t i = 0; i < _n_init_feats; ++i) {
            if (!status_f1[i] || !_track_table.back().first[i]) { continue; }
            Eigen::Vector3d xy1_f0 = cam->pixel2cam(utils::eigen_vec(_track_table[0].second[i]));
            Eigen::Vector3d xy1_f1 = cam->pixel2cam(utils::eigen_vec(_track_table[1].second[i]));
            Eigen::Vector3d xy1_f2 = cam->pixel2cam(utils::eigen_vec(_track_table[2].second[i]));
            xy1s_f0.push_back(xy1_f0);
            xy1s_f1.push_back(xy1_f1);
            xy1s_f2.push_back(xy1_f2);
        }
        std::cout << "Before Final inliers: " << xy1s_f0.size() << std::endl;

        std::vector<int> final_inliers;
        
        double total_err = 0;

        for (size_t i = 0; i < xy1s_f0.size(); ++i) {
            
            Eigen::Vector3d xyz_f1_0 = utils::triangulate_v2(xy1s_f1[i], xy1s_f0[i], t_10);
            //Eigen::Vector3d xyz_f1_2 = utils::triangulate_v2(xy1s_f1[i], xy1s_f2[i], t_12);

            Eigen::Vector3d xyz_f1 = (xyz_f1_0 + xyz_f1_0) / 2.0;

            double reproj_err_f0 = utils::reproject_err(t_01 * xyz_f1, xy1s_f0[i]);
            double reproj_err_f1 = utils::reproject_err(xyz_f1, xy1s_f1[i]);
            double reproj_err_f2 = utils::reproject_err(t_21 * xyz_f1, xy1s_f2[i]);

            if (
                xyz_f1.z() < 0 || 
                config::max_reproj_err_xy1 < reproj_err_f0 || 
                config::max_reproj_err_xy1 < reproj_err_f1 || 
                config::max_reproj_err_xy1 < reproj_err_f2
            ) { continue; }
            else { 
                final_inliers.push_back(i); 
                final_xyzs_f1.push_back(xyz_f1);
                final_uvs_f1.push_back(cam->cam2pixel(xy1s_f1[i]));
                total_err += (reproj_err_f0 + reproj_err_f1 + reproj_err_f2) / 3.; 
            }
        }

        std::cout << "Final inliers: " << final_inliers.size() << std::endl;
        return SUCCESS;
    }

    void initializer_v2::_shrink(
        const std::vector<uchar>&       status, 
        const std::vector<cv::Point2f>& uvs,
        std::vector<cv::Point2f>&       shrinked
    ) {
        assert(status.size() == uvs.size());
        shrinked.clear();
        shrinked.reserve(uvs.size());
        for (size_t i = 0; i < uvs.size(); ++i) {
            if (!status[i]) { continue; }
            shrinked.emplace_back(uvs[i]);
        }
    }

    void initializer_v2::_recover(
        const std::vector<uchar>& status_last, 
        std::vector<uchar>&       status,
        std::vector<cv::Point2f>& uvs
    ) {
        assert(status.size() == uvs.size());

        std::vector<uchar>       status_tmp;
        std::vector<cv::Point2f> pts_tmp;

        status_tmp = status_last;
        pts_tmp.resize(status_last.size());

        for (size_t i = 0, j = 0; i < status_last.size(); ++i) {
            if (!status_tmp[i]) { continue; }
            if (!status[j]) { status_tmp[i] = 0; }
            else { pts_tmp[i] = uvs[j]; }
            ++j;
        }

        status = std::move(status_tmp);
        uvs = std::move(pts_tmp);
    }

    void initializer_v2::_reverve_cache(double n_init_feats) {
        _shift_cache.reserve(n_init_feats);
        _xy1s_ref.reserve(n_init_feats);
        _xy1s_cur.reserve(n_init_feats);
    }

    void initializer_v2::_calc_optcal_flow(
        const cv::Mat&            img_last,
        const cv::Mat&            img_cur, 
        const std::vector<uchar>& status_last,
        std::vector<uchar>&       status, 
        std::vector<cv::Point2f>& uvs
    ) {
        uvs = _uvs_cache;
        std::vector<float> error;
        cv::TermCriteria criteria(
            cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 
            config::max_opt_iterations, CONST_EPS
        );

        cv::calcOpticalFlowPyrLK(
            img_last, img_cur, _uvs_cache, uvs, status, error, 
            cv::Size2i(config::cv_lk_win_sz, config::cv_lk_win_sz), 
            config::pyr_levels, criteria, cv::OPTFLOW_USE_INITIAL_FLOW
        );

        assert(status.size() == error.size());
        double max_err = *std::max_element(error.begin(), error.end());

        _shift_cache.clear();
        _shift_cache.reserve(status.size());

        for (size_t i = 0; i < status.size(); ++i) {
            if (!status[i]) { continue; }
            if (max_err * 0.5 < error[i]) { status[i] = 0; continue;  }
            _shift_cache.push_back(cv::norm(_uvs_cache[i] - uvs[i]));
        }
        _recover(status_last, status, uvs);
    }

    bool initializer_v2::_calc_homography(
        const std::vector<Eigen::Vector3d>& xy1s_ref,
        const std::vector<Eigen::Vector3d>& xy1s_cur,
        double                              err_mul2, 
        double                              reproject_threshold, 
        Sophus::SE3d&                       t_cr
    ) const {
        homography solver(reproject_threshold, err_mul2, _xy1s_ref, _xy1s_cur);
        return solver.calc_pose_from_matches(t_cr);
    }

    double initializer_v2::_triangulate_frame01(
        double                        reproj_threshold,  
        const Sophus::SE3d&           t_10, 
        std::vector<uchar>&           status_f1,
        std::vector<Eigen::Vector3d>& xyzs_f1
    ) const {
        xyzs_f1.resize(_n_init_feats);

        double total_err = 0.0;
        size_t j = 0;
        for (size_t i = 0; i < _n_init_feats; ++i) {
            if (!status_f1[i]) { continue; }

            Eigen::Vector3d xyz_f1 = utils::triangulate_v2(_xy1s_cur[j], _xy1s_ref[j], t_10);

            double err_f0 = utils::reproject_err(t_10.inverse() * xyz_f1, _xy1s_ref[j]);
            double err_f1 = utils::reproject_err(xyz_f1, _xy1s_cur[j]);

            /**
             * outliers is which reprojection error is too 
             * large or depth is less than zero
             */
            if (
                xyz_f1.z() < 0.0          ||
                reproj_threshold < err_f0 || 
                reproj_threshold < err_f1
            ) {
                status_f1[i] = 0;
            }
            else { xyzs_f1[i] = xyz_f1; total_err += (err_f1 + err_f0); }
            ++j;
        }
        assert(j == _xy1s_ref.size() && j == _xy1s_cur.size());
        return total_err;
    }

    void initializer_v2::_pose_only_optimize(
        const camera_ptr&                cam, 
        const std::vector<Sophus::SE3d>& poses
    ) {
        /**
         * @note setup optimizer
         */ 
        auto linear_solver = g2o::make_unique<
            g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>
        >();
        auto block_solver = g2o::make_unique<
            g2o::BlockSolver_6_3
        >(std::move(linear_solver));
        
        g2o::OptimizationAlgorithmLevenberg* algo = 
            new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
            
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(algo);
        optimizer.setVerbose(true);

        /**
         * @note create g2o graph
         */ 
        int v_seq = 0;

        _vs_f.clear(); 
        _vs_f.resize(min_init_frames, nullptr);
        
        for (size_t i = 0; i < min_init_frames; ++i) {
            backend::vertex_se3* v = new backend::vertex_se3();
            v->setId(v_seq++);
            v->setEstimate(poses[i]);
            if (0 == i) { v->setFixed(true); }
            optimizer.addVertex(v);
            _vs_f[i] = v;
        }

        const Sophus::SE3d& t_01 = poses[1].inverse();
        const auto&    status_f1 = _track_table[1].first;

        _vs_mp.clear(); 
        _vs_mp.resize(_n_init_feats, nullptr);

        for (size_t i = 0; i < _n_init_feats; ++i) {
            if (!status_f1[i]) { continue; }
            backend::vertex_xyz* v = new backend::vertex_xyz();
            v->setId(v_seq++);
            v->setEstimate(t_01 * _xyzs_f1[i]);
            v->setMarginalized(true);
            optimizer.addVertex(v);
            _vs_mp[i] = v;
        }

        int e_seq = 0;
        for (size_t fid = 0; fid < min_init_frames; ++fid) {
            for (size_t pid = 0; pid < _n_init_feats; ++pid) {
                const auto& record = _track_table[fid];
                if (!status_f1[pid] || !record.first[pid]) { continue; }

                backend::edge_xyz2uv* e = new backend::edge_xyz2uv(cam);
                e->setId(e_seq++);
                e->setVertex(0, _vs_mp[pid]);
                e->setVertex(1, _vs_f[fid]);
                e->setInformation(Eigen::Matrix2d::Identity());
                e->setRobustKernel(new g2o::RobustKernelHuber());
                e->setMeasurement(utils::eigen_vec(record.second[pid]));
                optimizer.addEdge(e);
            }
        }

        /**
         * @note start optimizing
         */ 
        optimizer.initializeOptimization();
        optimizer.optimize(config::max_opt_iterations);
    }
#endif
}
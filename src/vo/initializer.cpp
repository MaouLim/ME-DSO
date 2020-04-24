
#include <vo/initializer.hpp>
#include <vo/feature.hpp>
#include <vo/frame.hpp>
#include <vo/camera.hpp>
#include <vo/map_point.hpp>
#include <vo/homography.hpp>

#include <utils/config.hpp>
#include <utils/utils.hpp>

#include <backend/g2o_staff.hpp>

#define _ME_VSLAM_DEBUG_INFO_ 1

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

        bool ret = _calc_essential();//_calc_homography(cur->camera->err_mul2(), config::max_reproj_err);

        if (!ret) { 
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "Failed to compute pose from matches." << std::endl;
#endif      
            return FAILED_CALC_POSE;
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

    bool initializer::_calc_essential() {
        cv::Mat cam_mat = ref->camera->cv_mat();
        cv::Mat mask;
        cv::Mat essential = cv::findEssentialMat(_uvs_ref, _uvs_cur, cam_mat, cv::RANSAC, 0.999, 1.0, mask);
        cv::Mat cv_R, cv_t;
        auto n_inliers = cv::recoverPose(essential, _uvs_ref, _uvs_cur, cam_mat, cv_R, cv_t, mask);
#ifdef _ME_VSLAM_DEBUG_INFO_
        std::cout << "Inliers: " << n_inliers << std::endl;
#endif
        if (n_inliers < config::min_inliers) { return false; }
        Eigen::Vector3d t;
        t << cv_t.at<double>(0), cv_t.at<double>(1), cv_t.at<double>(2);
        Eigen::Matrix3d r;
        r << cv_R.at<double>(0, 0), cv_R.at<double>(0, 1), cv_R.at<double>(0, 2), 
             cv_R.at<double>(1, 0), cv_R.at<double>(1, 1), cv_R.at<double>(1, 2), 
             cv_R.at<double>(2, 0), cv_R.at<double>(2, 1), cv_R.at<double>(2, 2);
        t_cr = Sophus::SE3d(r, t);
        return true;
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

    void initializer_v2::reset() {
        _frames.clear();
        _track_table.clear();
        _n_feats_tracked.clear();
        _shift_cache.clear();
        _uvs_cache.clear();
        _xy1s_ref.clear();
        _xy1s_cur.clear();
    }

    initializer_v2::op_result 
    initializer_v2::add_frame(const frame_ptr& frame) {

        if (_frames.empty()) {

            // create detector
            fast_detector detector(
                config::height, config::width, 
                config::cell_sz, config::pyr_levels
            );

            feature_set new_features;
            detector.detect(frame, config::min_corner_score, new_features);

            size_t n_features = new_features.size();
            std::cout << "Num of features: " << n_features << std::endl;
            if (n_features < config::min_features_in_first) { 
                // TODO with few features
                std::cout << "FEATURES_NOT_ENOUGH" << std::endl;
                return FEATURES_NOT_ENOUGH;
            }

            std::vector<uchar> status(n_features, 1);
            std::vector<cv::Point2f> uvs;
            uvs.reserve(n_features);

            for (auto& each : new_features) {
                uvs.emplace_back(each->uv.x(), each->uv.y());
            }
            _uvs_cache = uvs;

            _track_table.emplace_back(std::move(status), std::move(uvs));
            _n_feats_tracked.emplace_back(n_features);
            _frames.emplace_back(frame);

            return ACCEPT;
        }

        const frame_ptr& last = _frames.back();
        const frame_ptr& cur  = frame;

        std::vector<uchar>       status;
        std::vector<cv::Point2f> pts;
        _calc_optcal_flow(cur->pyramid[0], status, pts);

        double median_shift = 
            *utils::median(_shift_cache.begin(), _shift_cache.end());
        if (median_shift < config::min_init_shift) {
            std::cout << "SHIFT_NOT_ENOUGH" << std::endl;
            return SHIFT_NOT_ENOUGH;
        }

        size_t n_feats_tracked = _shift_cache.size();
        std::cout << "Num of features: " << n_feats_tracked << std::endl;
        if (n_feats_tracked < config::min_features_to_tracked) {
            std::cout << "FEATURES_NOT_ENOUGH" << std::endl;
            reset();
            return FEATURES_NOT_ENOUGH;
        }

        _shrink(status, pts, _uvs_cache);
        _track_table.emplace_back(std::move(status), std::move(pts));
        _n_feats_tracked.emplace_back(n_feats_tracked);
        _frames.emplace_back(frame);

        if (_frames.size() < n_init_frames_require) {
            return ACCEPT;
        }

        std::vector<Sophus::SE3d> poses(n_init_frames_require);
        const camera_ptr& cam = _frames.front()->camera;
        double err_mul2 = cam->err_mul2();
        size_t n_features = _n_feats_tracked.front();
        for (size_t i = 1; i < n_init_frames_require; ++i) {

            _xy1s_ref.clear();
            _xy1s_cur.clear();

            auto& prv = _track_table[i - 1];
            auto& cur = _track_table[i];

            for (size_t j = 0; j < n_features; ++j) {
                if (!cur.first[j]) { continue; }
                _xy1s_ref.emplace_back(cam->pixel2cam(utils::eigen_vec(prv.second[j])));
                _xy1s_cur.emplace_back(cam->pixel2cam(utils::eigen_vec(cur.second[j])));
            }
            
            if (!_calc_homography(err_mul2, config::max_reproj_err, poses[i])) {
                std::cout << "FAILED_CALC_POSE" << std::endl;
                return FAILED_CALC_POSE;
            }

            if (i == 1) {

                _inliers_f1 = _track_table[1].first;
                _xyzs_f1.resize(_inliers_f1.size());
                auto chi2 = _compute_inliers_and_triangulate(config::max_reproj_err, poses[1]);
                std::cout << "chi2: " << chi2 << std::endl;
            }
            std::cout << "--------------------" << std::endl;
            std::cout << poses[i].matrix3x4() << std::endl;
            std::cout << "--------------------" << std::endl;
        }

        auto linear_solver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
        auto block_solver  = g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
        
        g2o::OptimizationAlgorithmLevenberg* algo = 
            new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
        g2o::SparseOptimizer opt;
        opt.setAlgorithm(algo);
        opt.setVerbose(true);

        int v_seq = 0, e_seq = 0;
        std::vector<backend::vertex_se3*> vs_f(n_init_frames_require, nullptr);
        
        for (size_t i = 0; i < n_init_frames_require; ++i) {
            backend::vertex_se3* v = new backend::vertex_se3();
            v->setId(v_seq++);
            v->setEstimate(poses[i]);
            if (i == 0) { v->setFixed(true); }
            opt.addVertex(v);
            vs_f[i] = v;
        }

        Sophus::SE3d t_01 = poses[1].inverse();
        std::vector<backend::vertex_xyz*> vs_mp(_inliers_f1.size(), nullptr);
        for (size_t i = 0; i < _inliers_f1.size(); ++i) {
            if (!_inliers_f1[i]) { continue; }
            backend::vertex_xyz* v = new backend::vertex_xyz();
            v->setId(v_seq++);
            v->setEstimate(t_01 * _xyzs_f1[i]);
            v->setMarginalized(true);
            opt.addVertex(v);
            vs_mp[i] = v;
        }

        std::vector<std::vector<backend::edge_xyz2uv*>> es(
            n_init_frames_require, std::vector<backend::edge_xyz2uv*>(_inliers_f1.size(), nullptr)
        );

        for (size_t i = 0; i < n_init_frames_require; ++i) {
            for (size_t j = 0; j < _inliers_f1.size(); ++j) {
                if (!_inliers_f1[j] || !_track_table[i].first[j]) { continue; }
                backend::edge_xyz2uv* e = new backend::edge_xyz2uv(cam);
                e->setId(e_seq++);
                e->setVertex(0, vs_mp[j]);
                e->setVertex(1, vs_f[i]);
                e->setInformation(Eigen::Matrix2d::Identity());
                e->setRobustKernel(new g2o::RobustKernelHuber());
                e->setMeasurement(utils::eigen_vec(_track_table[i].second[j]));
                opt.addEdge(e);
            }
        }

        opt.initializeOptimization();
        opt.optimize(10);

        for (size_t i = 0; i < n_init_frames_require; ++i) {
            std::cout << "--------------------" << std::endl;
            std::cout << vs_f[i]->estimate().matrix3x4() << std::endl;
            std::cout << "--------------------" << std::endl;
        }
        
        //TODO triangulate and compute inliers
        //TODO create g2o optimizer
        return SUCCESS;
        
        //TODO select the first and the second frame
        // estimate all the pose between two frames
        // triangulate using the 1st&2nd frames, remove the outliers
        // build g2o graph solve the structure only problem
        // set the first and the last frame as keyframe
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

    void initializer_v2::_calc_optcal_flow(
        const cv::Mat&            img, 
        std::vector<uchar>&       status, 
        std::vector<cv::Point2f>& uvs
    ) {
        const auto& last        = _track_table.back();
        const auto& status_last = last.first;
        const auto& uvs_last    = last.second;
        const cv::Mat& img_last = _frames.back()->pyramid[0];

        uvs = _uvs_cache;

        std::vector<float> error;
        cv::TermCriteria criteria(
            cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 
            config::max_opt_iterations, CONST_EPS
        );

        cv::calcOpticalFlowPyrLK(
            img_last, img, _uvs_cache, uvs, status, error, 
            cv::Size2i(config::cv_lk_win_sz, config::cv_lk_win_sz), 
            config::pyr_levels, criteria, cv::OPTFLOW_USE_INITIAL_FLOW
        );

        //TODO with the error

        _shift_cache.clear(); 
        _shift_cache.reserve(status.size());

        for (size_t i = 0; i < status.size(); ++i) {
            if (!status[i]) { continue; }
            _shift_cache.push_back(cv::norm(_uvs_cache[i] - uvs[i]));
        }

        _recover(status_last, status, uvs);
    }

    bool initializer_v2::_calc_homography(
        double err_mul2, double reproject_threshold, Sophus::SE3d& t_cr
    ) {
        homography solver(reproject_threshold, err_mul2, _xy1s_ref, _xy1s_cur);
        return solver.calc_pose_from_matches(t_cr);
    }

    double initializer_v2::_compute_inliers_and_triangulate(
        double reproject_threshold,  const Sophus::SE3d& t_10
    ) {
        double total_err = 0.0;
        size_t j = 0;
        for (size_t i = 0; i < _inliers_f1.size(); ++i) {
            if (!_inliers_f1[i]) { continue; }

            Eigen::Vector3d xyz_f1 = utils::triangulate_v2(_xy1s_cur[j], _xy1s_ref[j], t_10);
            

            double err_ref = utils::reproject_err(t_10.inverse() * xyz_f1, _xy1s_ref[j]);
            double err_cur = utils::reproject_err(xyz_f1, _xy1s_cur[j]);

            /**
             * outliers is which reprojection error is too 
             * large or depth is less than zero
             */
            if (reproject_threshold < err_ref || 
                reproject_threshold < err_cur || 
                xyz_f1.z() < 0.0)               
            {
                _inliers_f1[i] = 0;
            }
            else { _xyzs_f1[i] = xyz_f1; total_err += (err_cur + err_ref); }
            ++j;
        }
        assert(j == _xy1s_ref.size());
        assert(j == _xy1s_cur.size());
 
        return total_err;
    }
}
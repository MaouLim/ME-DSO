#include <vo/core.hpp>

#include <vo/depth_filter.hpp>
#include <vo/initializer.hpp>
#include <vo/reprojector.hpp>
#include <vo/camera.hpp>
#include <vo/feature.hpp>
#include <vo/frame.hpp>
#include <vo/map.hpp>
#include <vo/pose_estimator.hpp>
#include <vo/map_point.hpp>

#include <backend/bundle_adjustment.hpp>

namespace vslam {
    
    system::system(const camera_ptr& cam) :
        _state(INITIALIZING), _quality(GOOD), _camera(cam), _last(nullptr)
    {
        assert(_camera);
        assert(config::height == _camera->height && 
               config::width == _camera->width);

        _reprojector.reset(
            new reprojector(config::height, config::width, config::cell_sz)
        );
        _initializer.reset(new initializer());

        detector_ptr det =                                                                                                            
            utils::mk_vptr<fast_detector>(config::height, config::width, config::cell_sz, config::pyr_levels);
        auto callback = 
            std::bind(&system::_df_callback, this, std::placeholders::_1, std::placeholders::_2);
        _depth_filter.reset(new depth_filter(det, callback));
        _tf_estimator.reset(new twoframe_estimator(config::max_opt_iterations, 4, 0, twoframe_estimator::LK_ICIA));
        _sf_estimator.reset(new singleframe_estimator(config::max_opt_iterations, vslam::singleframe_estimator::PNP_BA));
    }

    bool system::start() {
        return _depth_filter->start();
    }

    bool system::shutdown() {
        _map.clear();
        _clear_cache();
        return _depth_filter->stop();
    }

    bool system::process_image(const cv::Mat& raw_img, double timestamp) {

        if (raw_img.empty()) { 
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "[SYSTEM]" << "Empty image data." << std::endl;
#endif      
            return false; 
        }

        _clear_cache();
        
        frame_ptr new_frame = _create_frame(raw_img, timestamp);

        switch (_state) {
            case INITIALIZING : {
                _state = track_init_stage(new_frame);
                break;
            }
            case TRACKING : {
                _state = track_frame(new_frame);
                break;
            }
            case RELOCALIZING : {
                _state = relocalize(new_frame);
                break;
            }
            default : { assert(false); }
        }

        _last = new_frame;
        return true;
    }

    system::state_t 
    system::track_init_stage(const frame_ptr& new_frame) {
        auto ret = _initializer->add_frame(new_frame);
        if (initializer::REF_FRAME_SET == ret) {
            new_frame->as_key_frame();
            _map.add_key_frame(new_frame);
            return INITIALIZING;
        }
        if (initializer::SUCCESS == ret) {
            new_frame->as_key_frame();
            _map.add_key_frame(new_frame);
            _depth_filter->commit(new_frame);
            _initializer->reset();
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "[SYSTEM]" << "Initialize successfully." << std::endl; 
#endif
            return TRACKING;
        }
        return INITIALIZING;
    }

    system::state_t 
    system::track_frame(const frame_ptr& new_frame) {
        Sophus::SE3d t_cr;
        _tf_estimator->estimate(_last, new_frame, t_cr);
        new_frame->set_pose(t_cr * _last->t_cw);

        _map.find_covisible_key_frames(new_frame, _kfs_with_dis);
        size_t n_reprojs = 
            _reprojector->reproject_and_match(new_frame, _kfs_with_dis, _candidates, _kfs_with_overlaps);
        
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "[SYSTEM]" << "Reprojected map points: " 
                      << n_reprojs << std::endl; 
#endif
        if (n_reprojs < config::min_reproj_mps) {
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "[SYSTEM]" << "Reprojected map points not enough." << std::endl; 
#endif
            new_frame->set_pose(_last->t_cw);
            _quality = INSUFFICIENT;
            return RELOCALIZING;
        }

        Sophus::SE3d refined_t_cr;
        _sf_estimator->estimate(new_frame, refined_t_cr);

        double reproj_err = 0.0;
        singleframe_estimator::compute_inliers_and_reporj_err(
            new_frame, refined_t_cr, config::max_reproj_err_xy1 * 4., _inliers, _outliers, reproj_err
        );
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "[SYSTEM]" << "Inliers: " 
                      << _inliers.size()  << std::endl; 
#endif

        // for outliers, remove the association with the map point
        for (auto& each : _outliers) { each->reset_describing(); }
        // for inliers, local optimize the map point in the view of new frame
        if (_inliers.size() < config::min_inliers) {
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "[SYSTEM]" << "Inliers not enough." << std::endl; 
#endif
            new_frame->set_pose(_last->t_cw);
            _quality = INSUFFICIENT;
            return RELOCALIZING;
        }
        if (_inliers.size() < _last->n_features) {
            double drop_ratio = 
                double(_last->n_features - _inliers.size()) / _last->n_features;
            if (config::max_drop_ratio < drop_ratio) {
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "[SYSTEM]" << "High drop ratio: " 
                      << drop_ratio << std::endl; 
#endif
                new_frame->set_pose(_last->t_cw);
                _quality = INSUFFICIENT;
                return RELOCALIZING;
            }
        }

        _quality = GOOD;

        auto nth = _inliers.begin() + config::max_mps_to_local_opt;
        std::nth_element(
            _inliers.begin(), nth, _inliers.end(), 
            [](const feature_ptr& a, const feature_ptr& b) { 
                return a->map_point_describing->last_opt < 
                       b->map_point_describing->last_opt; 
            }
        );

        nth = _inliers.begin() + config::max_mps_to_local_opt;
        for (auto itr = _inliers.begin(); itr != nth; ++itr) {
            (*itr)->map_point_describing->local_optimize(config::max_opt_iterations);
            (*itr)->map_point_describing->last_opt = new_frame->id;
        }

        _local_map.insert(new_frame);

        if (!_need_new_kf(new_frame)) {
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "[SYSTEM]" << "Process default frame: " 
                      << new_frame->id << std::endl;
#endif
            _depth_filter->commit(new_frame);
            return TRACKING; 
        }

#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "[SYSTEM]" << "Process key frame: " 
                      << new_frame->id << std::endl;
#endif

        new_frame->as_key_frame();
        // set the latest observation 
        for (auto& each : _inliers) { 
            assert(each->map_point_describing->last_observation() == each);
        }
        _candidates.extract_observed_by(new_frame);

        _build_local_map();
        backend::local_map_ba ba(_local_map, config::max_reproj_err_xy1);
        auto errs = ba.optimize(config::max_opt_iterations);
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "[SYSTEM]" << "Error before BA: " << errs.first
                                    << "Error after BA:  " << errs.second
                      << std::endl;
#endif        
        ba.update();

        _depth_filter->commit(new_frame);
        _reduce_map();
        _map.add_key_frame(new_frame);

        return TRACKING;
    }

    system::state_t 
    system::relocalize(const frame_ptr& new_frame) {
        frame_ptr closest = _map.find_closest_covisible_key_frame(new_frame);
        if (!closest) { 
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "[SYSTEM]" 
                      << "Relocalize failed, failed to find the covisible key frames." 
                      << std::endl;
#endif
            return RELOCALIZING;
        }
        Sophus::SE3d _last_pose = _last->t_cw;
        _last = closest;

        auto ret = track_frame(new_frame);
        if (TRACKING == ret) {
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "[SYSTEM]" << "Relocalize successfully." << std::endl;
#endif
        }
        else { new_frame->set_pose(_last_pose); }
        return ret;
    }

    frame_ptr system::_create_frame(const cv::Mat& raw_img, double timestamp) {
        return utils::mk_vptr<frame>(_camera, raw_img, timestamp);
    }

    void system::_df_callback(const map_point_ptr& new_mp, double cov2) {
#ifdef _ME_VSLAM_DEBUG_INFO_
        std::cout << "[SYSTEM]" << "Depth filter callback: " 
                  << "a new map point created as a candidate." << std::endl; 
        std::cout << "The cov2 of the candidate: " << cov2 << std::endl;               
#endif
        _candidates.add_candidate(new_mp);
    }

    bool system::_need_new_kf(const frame_ptr& frame) const {
        double _, median; 
        assert(vslam::min_and_median_depth_of_frame(frame, _, median));
        for (auto& kf_overlap : _kfs_with_overlaps) {
            Eigen::Vector3d xyz_kf = frame->t_cw * kf_overlap.first->cam_center();
            Eigen::Vector3d xyz_kf_scale = xyz_kf / median;
            if (xyz_kf_scale.x() < config::min_key_frame_shift_x || 
                xyz_kf_scale.y() < config::min_key_frame_shift_y || 
                xyz_kf_scale.z() < config::min_key_frame_shift_z) 
            { return false; }
        }
        return true;
    }

    void system::_build_local_map() {
        size_t n_frames = std::min(config::max_local_map_frames, (int) _kfs_with_overlaps.size()) - 1;
        std::partial_sort(
            _kfs_with_overlaps.begin(), 
            _kfs_with_overlaps.begin() + n_frames, 
            _kfs_with_overlaps.end(),
            [](const frame_with_overlaps& a, const frame_with_overlaps& b) { return a.second > b.second; }
        );
        for (size_t i = 0; i < n_frames; ++i) {
            _local_map.insert(_kfs_with_overlaps[i].first);
        }
    }

    void system::_reduce_map() { 
        //TODO if there is too much key frames in the global map, try to reduce the map
        if (config::max_global_map_frames < _map.n_key_frames()) {

        }
    }

    void system::_clear_cache() {
        _local_map.clear();
        _kfs_with_dis.clear();
        _kfs_with_overlaps.clear();
        _inliers.clear();
        _outliers.clear();
    }

} // namespace vslam

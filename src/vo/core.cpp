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

namespace vslam {
    
    system::system(const camera_ptr& cam) :
        _state(INITIALIZING), _camera(cam), _last(nullptr)
    {
        int height = _camera->height, width = _camera->width;

        const int cell_sz    = 10;/* TODO with config */
        const int pyr_levels = 5;

        _reprojector.reset(
            new reprojector(height, width, cell_sz)
        );
        _initializer.reset(new initializer());

        detector_ptr det =                                                                                                            
            utils::mk_vptr<fast_detector>(height, width, cell_sz, pyr_levels);
        auto callback = 
            std::bind(&system::_df_callback, this, std::placeholders::_1, std::placeholders::_2);
        _depth_filter.reset(new depth_filter(det, callback));
        _tf_estimator.reset(new twoframe_estimator(10, 4, 0, twoframe_estimator::LK_FCFA));
    }

    bool system::process_image(const cv::Mat& raw_img, double timestamp) {

        // TODO cleanup  
        
        frame_ptr new_frame = _create_frame(raw_img, timestamp);

        switch (_state) {
            case INITIALIZING : {
                if (track_init_stage(new_frame)) { _state = DEFAULT_FRAME; }
                break;
            }
            case DEFAULT_FRAME : {
                track_frame(new_frame);
                break;
            }
            case RELOCALIZING : {
                relocalize();
                break;
            }
            default : { assert(false); }
        }

        _last = new_frame;
        return true;
    }

    bool system::track_init_stage(const frame_ptr& new_frame) {
        auto ret = _initializer->add_frame(new_frame);
        if (initializer::NO_REF_FRAME == ret) {
            ret = _initializer->set_first(new_frame);
            if (initializer::SUCCESS == ret) {
                new_frame->as_key_frame();
                _map->add_key_frame(new_frame);
            }
        }
        else if (initializer::SUCCESS == ret) {
            new_frame->as_key_frame();
            _map->add_key_frame(new_frame);
            _depth_filter->commit(new_frame);
            _initializer->reset();
            return true;
        }
        return false;
    }

    bool system::track_frame(const frame_ptr& new_frame) {
        Sophus::SE3d t_cr;
        _tf_estimator->estimate(_last, new_frame, t_cr);
        new_frame->set_pose(t_cr * _last->t_cw);

        _kfs_with_dis.clear();
        _kfs_with_overlaps.clear();

        _map->find_covisible_key_frames(new_frame, _kfs_with_dis);
        size_t n_matches = 
            _reprojector->reproject_and_match(new_frame, _kfs_with_dis, _candidates, _kfs_with_overlaps);
        
        if (n_matches < /*min_reproj_matches*/100) {
            // TODO discard the pose estimation
            return false;
        }

        Sophus::SE3d refined_t_cr;
        _sf_estimator->estimate(new_frame, refined_t_cr);

        double reproj_err = 0.0;
        singleframe_estimator::compute_inliers_and_reporj_err(
            new_frame, refined_t_cr, 0.1/* read from config */, _inliers, _outliers, reproj_err
        );

        // for outliers, remove the association with the map point
        for (auto& each : _outliers) { each->reset_describing(); }

        // for inliers, local optimize the map point in the view of new frame
        if (_inliers.size() < /*min_reproj_inliers*/90) {
            // too few inliers
            // TODO discard the pose estimation
            return false;
        }

        auto nth = _inliers.begin() + /*max_mp_local_opt*/60;
        std::nth_element(
            _inliers.begin(), nth, _inliers.end(), 
            [](const feature_ptr& a, const feature_ptr& b) { 
                return a->map_point_describing->last_opt < 
                       b->map_point_describing->last_opt; 
            }
        );

        nth = _inliers.begin() + /*max_mp_local_opt*/60;
        for (auto itr = _inliers.begin(); itr != nth; ++itr) {
            (*itr)->map_point_describing->local_optimize(/* max_local_opt_iterations */10);
            (*itr)->map_point_describing->last_opt = new_frame->id;
        }

        assert(_inliers.size() <= _last->n_features);
        size_t n_dropped = _last->n_features - _inliers.size();

        if (/*max_kf_feat_dropped*/100 < n_dropped) {
            new_frame->set_pose(_last->t_cw);
            return false;
        }

        //min_and_median_depth_of_frame(new_frame, min, median);
        if (!_need_new_kf()) { _depth_filter->commit(new_frame); return false; }

        new_frame->as_key_frame();
        // set the latest observation 
        for (auto& each : _inliers) { 
            assert(each->map_point_describing->last_observation() == each);
        }
        _candidates.extract_observed_by(new_frame);
        _depth_filter->commit(new_frame);

        //TODO if there is too much key frames in the global map, try to reduce the map
        _reduce_map();

        _map->add_key_frame(new_frame);

        return true;
    }

    frame_ptr system::_create_frame(const cv::Mat& raw_img, double timestamp) {
        return utils::mk_vptr<frame>(_camera, raw_img, timestamp);
    }

} // namespace vslam

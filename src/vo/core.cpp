#include <vo/core.hpp>

#include <vo/depth_filter.hpp>
#include <vo/initializer.hpp>
#include <vo/reprojector.hpp>
#include <vo/camera.hpp>
#include <vo/feature.hpp>
#include <vo/frame.hpp>
#include <vo/map.hpp>
#include <vo/pose_estimator.hpp>

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
        _2frame_estimator.reset(new twoframe_estimator(10, 4, 0, twoframe_estimator::LK_FCFA));
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



                break;
            }

            case RELOCALIZING : {
                break;
            }

            default : { assert(false); }
        }

        _last = new_frame;
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
        _2frame_estimator->estimate(_last, new_frame, t_cr);
        new_frame->set_pose(t_cr * _last->t_cw);

        _kfs_with_dis.clear();
        _kfs_with_overlaps.clear();

        _map->find_covisible_key_frames(new_frame, _kfs_with_dis);
        size_t n_matches = 
            _reprojector->reproject_and_match(new_frame, _kfs_with_dis, _candidates, _kfs_with_overlaps);
        
        if (n_matches < min_reproj_matches) {
            // TODO discard the pose estimation

            return false;
        }


    }

    frame_ptr system::_create_frame(const cv::Mat& raw_img, double timestamp) {
        return utils::mk_vptr<frame>(_camera, raw_img, timestamp);
    }

} // namespace vslam

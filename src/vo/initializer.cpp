#include <utils/config.hpp>
#include <vo/initializer.hpp>
#include <vo/feature.hpp>
#include <vo/frame.hpp>
#include <vo/camera.hpp>

namespace vslam {

    const int initializer::min_features_to_init = config::get<int>("min_features_to_init");

    initializer::init_result 
    initializer::set_first(const frame_ptr& first) 
    {
        reset();
        _detect_features(first, _keypoints_ref, _xy1s_ref);

        if (_keypoints_ref.size() < min_features_to_init) {
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cerr << "Failed to init, too few keypoints: " 
                      << _keypoints_ref.size() << std::endl;
#endif            
            return FAILURE;
        }

        ref = first;
        return SUCCESS;
    }

    void initializer::_detect_features(
        const frame_ptr&              target, 
        std::vector<cv::Point2f>&     keypoints, 
        std::vector<Eigen::Vector3d>& xy1s
    ) {
        const int h = target->camera->height;
        const int w = target->camera->width;

        const size_t pyr_levels       = config::get<int>("pyr_levels");
        const int    cell_sz          = config::get<int>("cell_sz");
        const double min_corner_score = config::get<double>("min_corner_score");

        feature_set new_features;
        fast_detector detector(h, w, cell_sz, pyr_levels);
        detector.detect(target, min_corner_score, new_features);

        keypoints.clear(); xy1s.clear();

        for (auto& each : new_features) {
            keypoints.emplace_back(each->uv[0], each->uv[1]);
            xy1s.emplace_back(each->xy1);
        }
    }

    void _track_lk(
        const frame_ptr& ref, 
        const frame_ptr& cur, 
        std::vector<cv::Point2f>&  
    ) {



    }
}
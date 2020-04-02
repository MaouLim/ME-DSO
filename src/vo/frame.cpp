#include <utils/utils.hpp>
#include <utils/config.hpp>
#include <vo/frame.hpp>
#include <vo/feature.hpp>
#include <vo/camera.hpp>
#include <vo/map_point.hpp>

namespace vslam {

    int    frame::_seq_id    = 0;
    double frame::pyr_scale  = config::get<double>("pyr_scale");
    size_t frame::pyr_levels = config::get<double>("pyr_levels");

    frame::frame(
        const camera_ptr& _cam, 
        const cv::Mat&    _img, 
        double            _timestamp, 
        bool              _key_frame
    ) : id(_seq_id++), timestamp(_timestamp), camera(_cam), 
        n_features(0), key_frame(_key_frame)
    {
        assert(_cam->width == _img.cols && _cam->height == _img.rows);

        for (auto& each : good_features) {
            each = feature_ptr(nullptr);
        }
        pyramid = utils::create_pyramid(_img, pyr_levels, pyr_levels);
    }

    bool frame::visible(const Eigen::Vector3d& p_w, double border) const {
        assert(0.0 <= border);

        Eigen::Vector3d p_c = t_cw * p_w;
        if (p_c.z() < 0.0) { return false; }

        auto uv = camera->cam2pixel(p_c);
        return (border <= uv[0] && int(uv[0] + border) < camera->width && 
                border <= uv[1] && int(uv[1] + border) < camera->height);
    }

    void frame::min_and_median_depth(double& min, double& median) const {
        assert(!features.empty() && 0 < n_features);

        min = std::numeric_limits<double>::max();

        std::vector<double> dvec;
        dvec.reserve(n_features);

        for (auto& each_feat : features) {
            double d = (t_cw * each_feat->host_map_point->position).z();
            dvec.push_back(d);
            min = std::min(min, d);
        }

        auto median_itr = dvec.begin() + size_t((n_features - 1) / 2);
        std::nth_element(dvec.begin(), median_itr, dvec.end());
        median = *median_itr;
    }

    bool frame::remove_good_feature(const feature_ptr& _feat) {
        for (auto& each : good_features) {
            if (_feat == each) {
                each.reset();
                return true;
            }
        }
        _select_good_features();
    }

    void frame::_remove_useless_features() {
        for (auto& each : good_features) {
            if (!each) { continue; }
            if (each->describe_nothing()) { each.reset(); }
        }
    }

    void frame::_select_good_features() {
        for (auto& candidate : features) {
            if (candidate->describe_nothing()) { continue; }
            _check_good_feat(candidate);
        }
    }

    void frame::_check_good_feat(const feature_ptr& candidate) {

        Eigen::Vector2d center = { camera->width / 2., camera->height / 2. };

        if (
            !good_features[0] || 
                utils::distance_l1(candidate->uv, center) < 
                utils::distance_l1(good_features[0]->uv, center)
        ) { good_features[0] = candidate; return; }

        if (center[0] < candidate->uv[0] && center[1] < candidate->uv[1]) {
            if (
                !good_features[1] || 
                    utils::distance_l1(good_features[1]->uv, center) < 
                    utils::distance_l1(candidate->uv, center)
            ) { good_features[1] = candidate; return; }
        }

        if (center[0] < candidate->uv[0] && center[1] >= candidate->uv[1]) {
            if (
                !good_features[2] || 
                    utils::distance_l1(good_features[2]->uv, center) < 
                    utils::distance_l1(candidate->uv, center)
            ) { good_features[2] = candidate; return; }
        }

        if (center[0] >= candidate->uv[0] && center[1] >= candidate->uv[1]) {
            if (
                !good_features[3] || 
                    utils::distance_l1(good_features[3]->uv, center) < 
                    utils::distance_l1(candidate->uv, center)
            ) { good_features[3] = candidate; return; }
        }

        assert(center[0] >= candidate->uv[0] && center[1] < candidate->uv[1]);
        {
            if (
                !good_features[4] || 
                    utils::distance_l1(good_features[4]->uv, center) < 
                    utils::distance_l1(candidate->uv, center)
            ) { good_features[4] = candidate; return; }
        }
    }
}
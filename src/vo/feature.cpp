#include <vo/feature.hpp>

#include <vo/frame.hpp>
#include <vo/map_point.hpp>
#include <vo/camera.hpp>

#include <utils/utils.hpp>

#include <backend/g2o_staff.hpp>

namespace vslam {

    feature::feature(
        const frame_ptr& _host, const Eigen::Vector2d& _uv, size_t _pyr_level
    ) : type(CORNER), host_frame(_host), map_point_describing(nullptr), 
        uv(_uv), grad_orien(1., 0.), level(_pyr_level), good(false), e(nullptr)
    {
        assert(!host_frame.expired());
        xy1 = host_frame.lock()->camera->pixel2cam(uv, 1.0);
    }

    bool feature::set_describing(const map_point_ptr& mp) {
        if (!mp || map_point::REMOVED == mp->type) { return false; }
        if (map_point_describing) { return false; }
        map_point_describing = mp;
        mp->_set_observed_by(shared_from_this());
        return true;
    }

    bool feature::reset_describing(bool cascade) {
        if (!map_point_describing) { return false; }
        if (cascade) {
            map_point_describing->remove_observation(shared_from_this());
        }
        map_point_describing.reset();
        if (good) {
            if (!host_frame.expired()) {
                host_frame.lock()->remove_good_feature(shared_from_this());
            }
            good = false;
        }
        return true;
    }

    bool feature::remove_describing() {
        if (!map_point_describing) { return false; }
        map_point_ptr mp = map_point_describing;
        map_point_describing.reset();
        if (good) {
            if (!host_frame.expired()) {
                host_frame.lock()->remove_good_feature(shared_from_this());
            }
            good = false;
        }
        mp->as_removed();
        return true;
    }

    bool feature::use() {
        if (host_frame.expired()) { assert(false); return false; }
        host_frame.lock()->add_feature(shared_from_this());
        return true;
    }

    backend::edge_xyz2xy1_se3* feature::create_g2o(
        int                  eid, 
        backend::vertex_xyz* v0, 
        backend::vertex_se3* v1, 
        double               weight,
        bool                 robust
    ) {
        if (e) { return e; }
        e = new backend::edge_xyz2xy1_se3();
        e->setVertex(0, v0);
        e->setVertex(1, v1);
        e->setMeasurement(xy1);
        e->setInformation(weight * Eigen::Matrix2d::Identity());
        e->setId(eid);
        if (robust) {
            g2o::RobustKernelHuber* huber = new g2o::RobustKernelHuber();
            e->setRobustKernel(huber);
        }
        return e;
    }

    bool feature::update_from_g2o(double update_thresh) {
        if (!e) { return false; }
        if (update_thresh < e->chi2()) {
            if (map_point_describing->n_obs < 2) {
                remove_describing();
            }
            else { reset_describing(true); }
        }
        e = nullptr;
        return true;
    }

    void fast_detector::detect(
        frame_ptr host, double threshold, feature_set& features
    ) {
        const auto& pyramid  = host->pyramid;
        assert(pyr_levels == pyramid.size());

        const size_t n_corners = grid_rows * grid_cols;
        corner_set corners(n_corners, corner(0, 0, 0, threshold, 0.0));

        for (size_t i = 0; i < pyr_levels; ++i) {
            const int scale = (1 << i);
            std::vector<fast::fast_xy> fast_corners;

#if __SSE2__
            fast::fast_corner_detect_10_sse2(
                (fast::fast_byte*) pyramid[i].data, 
                pyramid[i].cols, pyramid[i].rows, pyramid[i].cols, 20, 
                fast_corners
            );
#elif HAVE_FAST_NEON
            fast::fast_corner_detect_9_neon(
                (fast::fast_byte*) pyramid[i].data, 
                pyramid[i].cols, pyramid[i].rows, pyramid[i].cols, 20, 
                fast_corners
            );
#else
            fast::fast_corner_detect_10(
                (fast::fast_byte*) pyramid[i].data, 
                pyramid[i].cols, pyramid[i].rows, pyramid[i].cols, 20, 
                fast_corners
            );
#endif      
            std::vector<int> scores, nonmax_corners;
            fast::fast_corner_score_10(
                (fast::fast_byte*) pyramid[i].data, pyramid[i].cols, fast_corners, 20, scores
            );
            fast::fast_nonmax_3x3(fast_corners, scores, nonmax_corners);

            for (auto& each : nonmax_corners) {
                fast::fast_xy& xy = fast_corners[each];
                if (!utils::in_image(pyramid[i], xy.x, xy.y, 4)) { continue; }
                int index = cell_index({ xy.x, xy.y }, i);
                if (grid_occupied[index]) { continue; }

                float score = utils::shi_tomasi_score(pyramid[i], xy.x, xy.y);
                if (corners[index].score < score) {
                    corners[index] = corner(xy.x, xy.y, i, score, 0.); 
                }
            }
        }
        
        for (auto& each : corners) {
            if (each.score <= threshold) { continue; }
            features.emplace_front(new feature(host, Eigen::Vector2d(each.x, each.y) * (1 << each.level), each.level));
        }
        
        reset();
    }
}
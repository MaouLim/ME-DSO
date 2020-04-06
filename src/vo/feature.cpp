#include <vo/feature.hpp>
#include <vo/frame.hpp>
#include <vo/map_point.hpp>
#include <vo/camera.hpp>
#include <utils/utils.hpp>

namespace vslam {

    feature::feature(
        const frame_ptr& _host, const Eigen::Vector2d& _uv, size_t _pyr_level
    ) : type(CORNER), host_frame(_host), map_point_describing(nullptr), 
        uv(_uv), grad_orien(1., 0., 0.), level(_pyr_level) 
    {
        assert(!host_frame.expired());
        xy1 = host_frame.lock()->camera->pixel2cam(uv, 1.0);
    }

    feature::feature(
        const frame_ptr&       _host_f, 
        const map_point_ptr&   _mp_desc,
        const Eigen::Vector2d& _uv, 
        size_t                 _pyr_level
    ) : type(CORNER), host_frame(_host_f), map_point_describing(_mp_desc), 
        uv(_uv), grad_orien(1., 0., 0.), level(_pyr_level) 
    {
        assert(!host_frame.expired());
        xy1 = host_frame.lock()->camera->pixel2cam(uv, 1.0);
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
                int index = cell_index({ xy.x, xy.y }, i);
                if (grid_occupied[index]) { continue; }

                float score = utils::shi_tomasi_score(pyramid[i], xy.x, xy.y);
                if (corners[index].score < score) {
                    corners[index] = corner(xy.x, xy.y, i, score, 0.); 
                }
            }
        }

        for (auto& each : corners) {
            if (each.score < threshold) { continue; }
            features.emplace_front(new feature(host, { each.x, each.y }, each.level));
        }
        
        reset();
    }
}
#include <vo/reprojector.hpp>

#include <vo/map_point.hpp>
#include <vo/frame.hpp>
#include <vo/feature.hpp>
#include <vo/camera.hpp>
#include <vo/matcher.hpp>

namespace vslam {

    bool operator<(
        const reprojector::match_type& left, 
        const reprojector::match_type& right
    ) {
        return left.first->type < right.first->type;
    }

    reprojector::reprojector(
        int height, int width, int cell_sz
    ) : n_matches(0), n_trials(0), _cell_sz(cell_sz), 
        _rows(std::ceil(double(height) / _cell_sz)), 
        _cols(std::ceil(double(width)  / _cell_sz))
    {
        const int n_cells =  _rows * _cols;
        _grid.resize(n_cells);
        _cells_order.reserve(n_cells);
        for (int i = 0; i < n_cells; ++i) {
            _cells_order.push_back(i);
        }
        _shuffle();
    }

    void reprojector::clear() {
        n_matches = 0; n_trials = 0;
        for (auto& cell : _grid) { cell.clear(); }
    }

    bool reprojector::_reproject(const map_point_ptr& mp, const frame_ptr& frame) {
        Eigen::Vector2d uv = 
            frame->camera->world2pixel(mp->position, frame->t_cw);
        if (!frame->visible(uv, patch_t::half_sz/* size */)) { return false; }
        _grid[_cell_idx(uv.x(), uv.y())].emplace_back(mp, uv);
        return true;
    }

    bool reprojector::_find_match_in_cell(
        match_set&       cell, 
        const frame_ptr& frame
    ) {
        cell.sort([](const match_type& a, const match_type& b) { return b < a; });
        auto itr = cell.begin();

        while (itr != cell.end()) {
            ++n_trials;

            const map_point_ptr& mp = itr->first;

            if (map_point::REMOVED == mp->type) {
                itr = cell.erase(itr);
                continue;
            }

            feature_ptr candidate;
            bool success = _matcher->match_direct(mp, frame, itr->second, candidate);
            if (!success) {
                ++(mp->n_fail_reproj);
                if (map_point::UNKNOWN == mp->type && 
                    max_unknown_mp_fail_reproj < mp->n_fail_reproj) 
                {
                    mp->as_removed();
                }
                if (map_point::CANDIDATE == mp->type && 
                    max_candidate_mp_fail_reproj < mp->n_fail_reproj)
                {
                    // TODO remove points from the candidates list
                }
            }

            ++(mp->n_success_reproj);
            if (map_point::UNKNOWN == mp->type && 
                min_good_mp_success_reproj < mp->n_success_reproj)
            {
                mp->type = map_point::GOOD;
            }

            frame->add_feature(candidate);
            itr = cell.erase(itr);
        }
    }

}
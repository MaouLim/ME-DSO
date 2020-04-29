#include <vo/reprojector.hpp>

#include <vo/map_point.hpp>
#include <vo/frame.hpp>
#include <vo/feature.hpp>
#include <vo/camera.hpp>
#include <vo/matcher.hpp>
#include <vo/map.hpp>

namespace vslam {

    bool operator<(
        const reprojector::match_type& left, 
        const reprojector::match_type& right
    ) {
        return left.first->type < right.first->type;
    }

    // reprojector::_coarse_match::_coarse_match(
    //     const map_point_ptr&   _mp, 
    //     const Eigen::Vector2d& _uv, 
    //     const handle_t&        _hd
    // ) : map_point(_mp), uv(_uv), handle(_hd) 
    // {
    //     if (map_point::CANDIDATE == map_point->type) {
    //         assert(handle_t(nullptr) != handle);
    //     }
    // }

    reprojector::reprojector(
        int height, int width, int cell_sz
    ) : n_matches(0), n_trials(0), _cell_sz(cell_sz), 
        _rows(std::ceil(double(height) / _cell_sz)), 
        _cols(std::ceil(double(width)  / _cell_sz))
    {
        const int n_cells =  _rows * _cols;
        _grid.resize(n_cells);
        _indices.reserve(n_cells);
        for (int i = 0; i < n_cells; ++i) {
            _indices.push_back(i);
        }
        _matcher = utils::mk_vptr<patch_matcher>();
    }

    size_t reprojector::reproject_and_match(
        const frame_ptr&                  frame,
        std::vector<frame_with_distance>& kfs_with_dis,
        candidate_set&                    candidates,
        std::vector<frame_with_overlaps>& kfs_with_overlaps
    ) {
        clear();
        
        _reproject_covisible_kfs(frame, kfs_with_dis, kfs_with_overlaps);
        _reproject_candidates(frame, candidates);

        _shuffle();
        for (size_t i = 0; i < _grid.size(); ++i) {
            if (config::max_mps_to_reproj <= n_matches) { break; }
            if (_find_match_in_cell(
                    frame, _grid[_indices[i]], candidates
                )
            ) { ++n_matches; }
        }

        return n_matches;
    }

    void reprojector::clear() {
        n_matches = 0; n_trials = 0;
        for (auto& cell : _grid) { cell.clear(); }
    }

    bool reprojector::_reproject_mp(const frame_ptr& frame, const map_point_ptr& mp) {
        Eigen::Vector2d uv = 
            frame->camera->world2pixel(mp->position, frame->t_cw);
        if (!frame->visible(uv, patch_t::half_sz/* size */)) { return false; }
        _grid[_cell_idx(uv.x(), uv.y())].emplace_back(mp, uv);
        return true;
    }

    bool reprojector::_reproject_covisible_kfs(
        const frame_ptr&                  frame, 
        std::vector<frame_with_distance>& kfs_with_dis,
        std::vector<frame_with_overlaps>& kfs_with_overlaps
    ) {
        auto less_comp = [](
            const frame_with_distance& a, 
            const frame_with_distance& b
        ) { return a.second < b.second; };

        std::sort(kfs_with_dis.begin(), kfs_with_dis.end(), less_comp);

        size_t count_kfs = 0;
        kfs_with_overlaps.reserve(config::max_overlaped_key_frames);

        for (auto& kf_with_d : kfs_with_dis) {
            if (config::max_overlaped_key_frames < count_kfs) { break; }

            const frame_ptr& kf = kf_with_d.first;

            kfs_with_overlaps.emplace_back(kf, 0);
            ++count_kfs;

            for (auto& each_feat : kf->features) {
                if (each_feat->describe_nothing()) { continue; }

                const map_point_ptr& mp = each_feat->map_point_describing;

                if (mp->last_proj_kf_id == frame->id) { continue; }

                mp->last_proj_kf_id = frame->id;
                if (_reproject_mp(frame, mp)) { kfs_with_overlaps.back().second += 1; }
            }
        }
        return true;
    }

    bool reprojector::_reproject_candidates(
        const frame_ptr& frame, candidate_set& candidates
    ) {
        auto elem_reproject = [&](candidate_set::candidate_t& candidate) {
            if (!_reproject_mp(frame, candidate.first)) {
                candidate.first->n_fail_reproj += 3;
            }
            return config::max_candidate_mp_fail_reproj < candidate.first->n_fail_reproj;
        };
        candidates.for_each_remove_if(elem_reproject);
        return true;
    }

    bool reprojector::_find_match_in_cell(
        const frame_ptr& frame, 
        cell_type&       cell,
        candidate_set&   candidates
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
                    config::max_unknown_mp_fail_reproj < mp->n_fail_reproj) 
                {
                    mp->as_removed();
                }
                if (map_point::CANDIDATE == mp->type && 
                    config::max_candidate_mp_fail_reproj < mp->n_fail_reproj)
                {
                    candidates.remove_candidate(mp);
                }
                continue;
            }

            ++(mp->n_success_reproj);
            if (map_point::UNKNOWN == mp->type && 
                config::min_good_mp_success_reproj < mp->n_success_reproj)
            {
                mp->type = map_point::GOOD;
            }

            candidate->use();
            itr = cell.erase(itr);
            return true;
        }

        return false;
    }
}
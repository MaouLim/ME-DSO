#include <vo/map.hpp>

#include <vo/map_point.hpp>
#include <vo/feature.hpp>
#include <vo/frame.hpp>

namespace vslam {

    void mp_candidate_set::clear() {
        lock_t lock(_mutex_c);
        _candidates.clear();
    }

    bool mp_candidate_set::add_candidate(const map_point_ptr& mp) {
        auto latest = mp->last_observed();
        if (!latest) { return false; }

        mp->type = map_point::CANDIDATE;
        lock_t lock(_mutex_c);
        _candidates.emplace_back(mp, latest);
        return true;
    }

    bool mp_candidate_set::remove_candidate(const map_point_ptr& mp) {
        lock_t lock(_mutex_c);
        auto itr = _candidates.begin();
        while (itr != _candidates.end()) {
            if (itr->first == mp) {
                _destroy(*itr);
                _candidates.erase(itr);
                return true;
            }
            ++itr;
        }
        return false;
    }

    bool mp_candidate_set::extract_observed_by(const frame_ptr& frame) {
        // TODO
        lock_t lock(_mutex_c);
        auto itr = _candidates.begin();
        while (_candidates.end() != itr) {
            auto host = itr->first->last_observed()->host_frame;
            assert(!host.expired());
            if (host.lock() == frame) {
                itr->first->type = map_point::UNKNOWN;
                itr->first->n_fail_reproj = 0;
                assert(!itr->second->host_frame.expired());
                auto latest_when_created = itr->second->host_frame.lock();
                latest_when_created->add_feature(itr->second);
                itr = _candidates.erase(itr);
            }
            else { ++itr; }
        }
        return true;
    }

    void mp_candidate_set::remove_observed_by(const frame_ptr& frame) {
        lock_t lock(_mutex_c);
        auto itr = _candidates.begin();
        while (itr != _candidates.end()) {
            if (frame == itr->second->host_frame.lock()) {
                _destroy(*itr);
                itr = _candidates.erase(itr);
            }
            else { ++itr; }
        }
    }

    void mp_candidate_set::_destroy(mp_candidate_t& candidate) { 
        candidate.second.reset(); 
        candidate.first->type = map_point::REMOVED;
        _trash_mps.push_back(candidate.first);
    }

    /** 
     * struct map method
     */

    frame_ptr map::key_frame(int frame_id) const {
        for (auto& each : _key_frames) {
            if (each->id == frame_id) { return each; }
        }
        return nullptr;
    }

    // ?? 
    void map::remove_map_point(const map_point_ptr& to_rm) {
        to_rm->type = map_point::REMOVED;
        to_rm->clear_observations();
    }

    frame_ptr 
    map::find_closest_covisible_key_frame(
        const frame_ptr& frame
    ) const {
        frame_ptr closest = nullptr;
        double    min_dis = std::numeric_limits<double>::max();

        for (auto& each_kf : _key_frames) {
            if (frame == each_kf) { continue; }
            for (auto& good_feat : each_kf->good_features) {
                if (!good_feat) { continue; }
                if (frame->visible(good_feat->map_point_describing->position)) {
                    double dis = distance(frame, each_kf);
                    if (dis < min_dis) { 
                        min_dis = dis;
                        closest = each_kf;
                    }
                    break;
                }
            }
        }

        return closest;
    }

    frame_ptr 
    map::find_furthest_key_frame(
        const Eigen::Vector3d& p_w
    ) const {
        frame_ptr furthest = nullptr;
        double    max_dis  = 0.0;

        for (auto& each_kf : _key_frames) {
            double distance = 
                (p_w - each_kf->cam_center()).norm();
            if (max_dis < distance) {
                max_dis = distance;
                furthest = each_kf;
            }
        }

        return furthest;
    }

    void map::find_covisible_key_frames(
        const frame_ptr&                         frame, 
        std::list<std::pair<frame_ptr, double>>& kf_with_dis
    ) {
        for (auto& each_kf : _key_frames) {
            if (frame == each_kf) { continue; }
            for (auto& good_feat : each_kf->good_features) {
                if (!good_feat) { continue; }
                if (frame->visible(good_feat->map_point_describing->position)) {
                    kf_with_dis.emplace_back(
                        each_kf, distance(frame, each_kf)
                    );
                    break;
                }
            }
        }
    }
    
} // namespace vslam

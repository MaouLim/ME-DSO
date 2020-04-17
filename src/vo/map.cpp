#include <vo/map.hpp>

#include <vo/map_point.hpp>
#include <vo/feature.hpp>
#include <vo/frame.hpp>

namespace vslam {

    /** 
     * struct map method
     */

    frame_ptr map::key_frame(int frame_id) const {
        for (auto& each : _key_frames) {
            if (each->id == frame_id) { return each; }
        }
        return nullptr;
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
        const frame_ptr&                  frame, 
        std::vector<frame_with_distance>& kfs_with_dis
    ) {
        kfs_with_dis.clear(); kfs_with_dis.reserve(_n_key_frames);
        for (auto& each_kf : _key_frames) {
            if (frame == each_kf) { continue; }
            for (auto& good_feat : each_kf->good_features) {
                if (!good_feat) { continue; }
                if (frame->visible(good_feat->map_point_describing->position)) {
                    kfs_with_dis.emplace_back(
                        each_kf, distance(frame, each_kf)
                    );
                    break;
                }
            }
        }
    }
    
} // namespace vslam

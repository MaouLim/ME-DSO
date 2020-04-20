#ifndef _ME_VSLAM_MAP_HPP_
#define _ME_VSLAM_MAP_HPP_

#include <common.hpp>

namespace vslam {

    struct map {

        map() : _n_key_frames(0) { }
        
        const frame_ptr& last_key_frame() const { return _key_frames.back(); }
        frame_ptr key_frame(int frame_id) const;
        size_t n_key_frames() const { return _n_key_frames; }

        void add_key_frame(const frame_ptr& kf) { _key_frames.push_back(kf); ++_n_key_frames; }
        void clear() { _key_frames.clear(); _n_key_frames = 0; /*_candidates.clear();*/ }

        /**
         * @brief find the covisible key frame with the frame, 
         *        and return the closest one 
         */ 
        frame_ptr find_closest_covisible_key_frame(const frame_ptr& frame) const;

        /**
         * @brief find the key frame which the camera position is 
         *        furthest away from the given position
         * @param p_w 3d world coordinate
         */ 
        frame_ptr find_furthest_key_frame(const Eigen::Vector3d& p_w) const;

        /**
         * @brief find all the covisible key frame with the frame
         * @param kfs_with_dis VSLAM_OUT covisible key frame and 
         *                     the distance
         */ 
        void find_covisible_key_frames(
            const frame_ptr&                  frame, 
            std::vector<frame_with_distance>& kfs_with_dis
        ) const;

        // TODO
        //void update(const Sophus::SE3d& se3);
        //void update(const Sophus::Sim3d& sim3);

    private:
        std::list<frame_ptr> _key_frames;
        size_t               _n_key_frames;
        //candidate_set        _candidates;
    };

} // namespace vslam

#endif
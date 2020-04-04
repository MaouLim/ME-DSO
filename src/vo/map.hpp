#ifndef _ME_VSLAM_MAP_HPP_
#define _ME_VSLAM_MAP_HPP_

#include <common.hpp>

namespace vslam {

    struct mp_candidates {
        
        
    };

    struct map {

        std::list<frame_ptr> key_frames;
        size_t               n_key_frames;

        map() : n_key_frames(0) { }
        
        const frame_ptr& last_key_frame() const { key_frames.back(); }
        const frame_ptr& key_frame(int frame_id) const;

        void reset();
        void remove_map_point(const map_point_ptr& to_rm);
    };

} // namespace vslam

#endif
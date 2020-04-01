#ifndef _ME_VSLAM_FEATURE_HPP_
#define _ME_VSLAM_FEATURE_HPP_

#include <common.hpp>

namespace vslam {

    struct feature {

        enum type_t { CORNER, EDGELET };

        type_t          type;
        frame_ptr       host_frame;     // in which frame this feature detected 
        map_point_ptr   host_map_point; // map point which described by this feature
        Eigen::Vector2d uv;             // pixel vector 
        Eigen::Vector3d xy1;            // uint-bearing vector
        Eigen::Vector2d grad_orien;     // the orientation of the graditude at this pixel
        size_t          level;          // level of pyramid

        feature(const frame_ptr& _host, const Eigen::Vector2d& _uv, size_t _pyr_level);
        feature(const frame_ptr& _host_f, const map_point_ptr& _host_mp, const Eigen::Vector2d& _uv, size_t _pyr_level);
    };
}

#endif
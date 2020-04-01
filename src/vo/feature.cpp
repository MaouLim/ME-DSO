#include <vo/feature.hpp>
#include <vo/frame.hpp>
#include <vo/map_point.hpp>
#include <vo/camera.hpp>

namespace vslam {

    feature::feature(
        const frame_ptr& _host, const Eigen::Vector2d& _uv, size_t _pyr_level
    ) : type(CORNER), host_frame(_host), host_map_point(nullptr), 
        uv(_uv), grad_orien(1., 0., 0.), level(_pyr_level) 
    {
        xy1 = host_frame->camera->pixel2cam(uv, 1.0);
    }

    feature::feature(
        const frame_ptr&       _host_f, 
        const map_point_ptr&   _host_mp,
        const Eigen::Vector2d& _uv, 
        size_t                 _pyr_level
    ) : type(CORNER), host_frame(_host_f), host_map_point(_host_mp), 
        uv(_uv), grad_orien(1., 0., 0.), level(_pyr_level) 
    {
        xy1 = host_frame->camera->pixel2cam(uv, 1.0);
    }
}
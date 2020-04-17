#include <vo/core.hpp>

#include <vo/depth_filter.hpp>
#include <vo/initializer.hpp>
#include <vo/reprojector.hpp>
#include <vo/camera.hpp>
#include <vo/feature.hpp>

namespace vslam {
    
    system::system(const camera_ptr& cam) :
        _state(NOT_INIT), _camera(cam), _last(nullptr)
    {
        int height = _camera->height, width = _camera->width;

        const int cell_sz    = 10;/* TODO with config */
        const int pyr_levels = 5;

        _reprojector.reset(
            new reprojector(height, width, cell_sz)
        );
        _initializer.reset(new initializer());

        detector_ptr det = 
            utils::mk_vptr<fast_detector>(height, width, cell_sz, pyr_levels);
        auto callback = 
            std::bind(&system::_df_callback, this, std::placeholders::_1, std::placeholders::_2);
        _depth_filter.reset(new depth_filter(det, callback));
    }

} // namespace vslam

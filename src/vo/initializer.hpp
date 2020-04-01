#ifndef _ME_VSLAM_INITIALIZER_HPP_
#define _ME_VSLAM_INITIALIZER_HPP_

#include <common.hpp>

namespace vslam {

    struct initializer {

        using ptr = std::shared_ptr<initializer>;
        
        bool wait_for_first_frame() const { return _wait; }

        void set_first(frame_ptr frame) {
            
            // TODO
            _wait = false;
        }

        bool track(frame_ptr frame) {

        }

    private:
        bool _wait;
    };
}

#endif
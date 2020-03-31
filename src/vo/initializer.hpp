#ifndef _ME_DSO_INITIALIZER_HPP_
#define _ME_DSO_INITIALIZER_HPP_

#include "common.hpp"
#include "frame.hpp"

namespace dso {

    struct initializer {

        using ptr = std::shared_ptr<initializer>;
        
        bool wait_for_first_frame() const { return _wait; }

        void set_first(frame::ptr frame) {
            
            // TODO
            _wait = false;
        }

        bool track(frame::ptr frame) {

        }

    private:
        bool _wait;
    };
}

#endif
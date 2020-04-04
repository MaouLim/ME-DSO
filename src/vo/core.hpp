#ifndef _ME_VSLAM_CORE_HPP_
#define _ME_VSLAM_CORE_HPP_

#include <common.hpp>

namespace vslam {

    struct core_sys {

        struct options {
            
        };

        enum state_t { NOT_INIT, RESET, RUNNING, LOST };

        /* public field */
        state_t          state;
        initializer_ptr  init;
        // tracker_t        tracker; // local map
        // mapper_t         mapper;  // 
        // frames_t         all_key_frames;
        // frames_t         local_map;

        core_sys(const core_sys&) = delete;

        bool start();
        bool shutdown();
        void reset();

        bool process_image(const cv::Mat& raw_img, double timestamp);
        
        //  {
        //     // TODO create new frame for the image.
        //     frame::ptr frame;

        //     if (NOT_INIT == state) {
        //         if (init->wait_for_first_frame()) {
        //             init->set_first(frame);
        //         }
        //         else if (init->track(frame)) {
        //             this->_fill_init_info_into(frame);
        //             this->_deliver_tracked_frame(frame, true);
        //             state = RUNNING;
        //         }
        //         else {
        //             // TODO initializer still tracking
        //         }
        //         return true;
        //     }

        //     // TODO tracker->track(frame); and decide whether the frame is a key frame
        //     tracker->track(frame);
        //     if (frame->key_frame) {
        //         // TODO insert into 
        //     }
        // }

    private:

        // void _fill_init_info_into(frame::ptr frame) {
            

        // }

        // // deliver the tracked frame to the mapper.
        // void _deliver_tracked_frame(frame::ptr frame, bool key_frame) {

        // }
    };
    
} // namespace dso

#endif
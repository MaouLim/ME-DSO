#ifndef _ME_VSLAM_SYSTEM_HPP_
#define _ME_VSLAM_SYSTEM_HPP_

#include <common.hpp>

namespace vslam {

    struct system {

        enum state_t   { NOT_INIT, RESET, RUNNING, LOST };
        enum mode_t    { SUFFIENCY, QUALITY };
        enum op_result {  };

        system(const camera_ptr& cam);
        system(const system&) = delete;

        bool start();
        bool shutdown();
        void reset();

        bool process_image(const cv::Mat& raw_img, double timestamp);

        const camera_ptr& camera() const { return _camera; }
        const frame_ptr&  last_frame() const { return _last; }

    protected:

        virtual op_result track_first(const frame_ptr& frame);
        virtual op_result track_init_stage(const frame_ptr& frame);
        virtual op_result track_frame(const frame_ptr& frame);
        virtual op_result relocalize();

    private:

        frame_ptr _create_frame(const cv::Mat& raw_img);
        void _df_callback(const map_point_ptr& new_mp, double cov2);

        state_t             _state;

        camera_ptr          _camera;
        reprojector_ptr     _reprojector;
        initializer_ptr     _initializer;
        depth_filter_ptr    _depth_filter;
        // TODO backend optimizer

        frame_ptr           _last;
        std::set<frame_ptr> _core_kfs;
        map_ptr             _map;
        candidate_set       _candidates;
    };
    
} // namespace dso

#endif
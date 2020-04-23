#ifndef _ME_VSLAM_SYSTEM_HPP_
#define _ME_VSLAM_SYSTEM_HPP_

#include <vo/map_point.hpp>

namespace vslam {

    struct system {

        enum state_t   { INITIALIZING, DEFAULT_FRAME, RELOCALIZING };
        enum mode_t    { SUFFIENCY, QUALITY };

        system(const camera_ptr& cam);
        system(const system&) = delete;

        bool start();
        bool shutdown();
        void reset();

        bool process_image(const cv::Mat& raw_img, double timestamp);

        const camera_ptr& camera() const { return _camera; }
        const frame_ptr&  last_frame() const { return _last; }

    protected:
        virtual bool track_init_stage(const frame_ptr& new_frame);
        virtual bool track_frame(const frame_ptr& new_frame);
        virtual bool relocalize() { }

    private:

        frame_ptr _create_frame(const cv::Mat& raw_img, double timestamp);
        void _df_callback(const map_point_ptr& new_mp, double cov2) { }
        bool _need_new_kf();
        void _reduce_map();

        state_t             _state;

        camera_ptr          _camera;
        reprojector_ptr     _reprojector;
        depth_filter_ptr    _depth_filter;
        initializer_ptr     _initializer;
        tf_estimator_ptr    _tf_estimator;
        sf_estimator_ptr    _sf_estimator;
        // TODO backend optimizer

        frame_ptr           _last;
        std::set<frame_ptr> _core_kfs;
        map_ptr             _map;
        candidate_set       _candidates;

        /**
         * @field caches
         */ 
        std::vector<frame_with_distance> _kfs_with_dis;
        std::vector<frame_with_overlaps> _kfs_with_overlaps;

        std::vector<feature_ptr> _inliers;
        std::vector<feature_ptr> _outliers;
    };
    
} // namespace vslam

#endif
#ifndef _ME_VSLAM_MAP_POINT_HPP_
#define _ME_VSLAM_MAP_POINT_HPP_

#include <common.hpp>

#define CONST_COS_60 0.5
#define CONST_COS_45 0.70711

namespace vslam {

    // struct depth_info {
    //     double mu;     // the mean of the depth
    //     double sigma2; // the covariance of the depth
    //     double a, b;   // beta distribution parameters
    //     double range;
    // };

    // struct pixel_point {

    //     using dinfo_t = std::unique_ptr<depth_info>;

    //     static double depth_converge_threshold;

    //     Eigen::Vector2d uv;
    //     frame::ptr      host;
    //     bool            converged;
    //     dinfo_t         depth;
        
    //     //pixel_point(double u, double v, )
    // };

    struct map_point {

        enum type_t { REMOVED, CANDIDATE, UNKNOWN, GOOD };

        int                     id;
        Eigen::Vector3d         position; // world coordinate system
        std::list<feature_ptr>  observations;
        size_t                  n_obs;
        int                     last_pub_timestamp;
        int                     last_proj_kf_id;
        int                     last_opt_timestamp;
        int                     n_fail_reproj;
        int                     n_success_reproj;
        type_t                  type;

        explicit map_point(const Eigen::Vector3d& _pos);
        ~map_point() = default;

        void set_observed_by(const feature_ptr& _feat);
        feature_ptr find_observed(const frame_ptr& _frame) const;
        bool remove_obserbed(const frame_ptr& _frame);
        feature_ptr find_closest_observed(const frame_ptr& _frame, const Eigen::Vector3d& _cam_center) const;

        /**
         * @brief minimize the reproject error (on unit-bearing plane)
         * @return the error after iterations
         */ 
        double local_optimize(size_t n_iterations);

    private:
        static int _seq_id; // to generate id;
    };
}

#endif
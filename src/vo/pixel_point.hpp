#ifndef _ME_DSO_PIXEL_POINT_HPP_
#define _ME_DSO_PIXEL_POINT_HPP_

#include <Eigen/Core>

#include "frame.hpp"

namespace dso {

    struct depth_info {
        double mu;     // the mean of the depth
        double sigma2; // the covariance of the depth
        double a, b;   // beta distribution parameters
        double range;
    };

    struct pixel_point {

        using dinfo_t = std::unique_ptr<depth_info>;

        static double depth_converge_threshold;

        Eigen::Vector2d uv;
        frame::ptr      host;
        bool            converged;
        dinfo_t         depth;
        
        //pixel_point(double u, double v, )
    };

}

#endif
#ifndef _ME_VSLAM_JACCOBIAN_HPP_
#define _ME_VSLAM_JACCOBIAN_HPP_

#include <common.hpp>

namespace vslam {

    /**
     * @brief calculate jaccobian mat d(xy1) / d(xyz) 
     *        d(xy1): unit-bearing coordinate (x_u, y_u, 1) in a camera pose(R, t)
     *        d(xyz): the world coordinate of p(x, y, z)
     *        z_c * (x_u, y_u, 1) = (x_c, y_c, z_c) = R * (x, y, z) + t
     * 
     * @param p_c (x_c, y_c, z_c)
     * @param rot R
     * @return J = [1/z_c,     0, -x_c/z_c^2] * R
     *             [    0, 1/z_c, -y_c/z_c^2]
     */            
    inline Eigen::Matrix23d jaccobian_dxy1dxyz(
        const Eigen::Vector3d& p_c, const Eigen::Matrix3d& rot
    ) {
        double x_c = p_c[0], y_c = p_c[1], z_c = p_c[2];
        double zinv = 1. / z_c;
        double zinv2 = zinv * zinv;

        Eigen::Matrix23d j;
        j << zinv,   0., -x_c * zinv2, 
               0., zinv, -y_c * zinv2;
        j = j * rot;

        return j;
    }
    /**
     * @brief calculate jaccobian mat d(xy1) / d(epsillon)
     *        d(xy1): unit-bearing coordinate (x_u, y_u, 1) in a camera pose(R, t)
     *        d(epsillon): epsillon = se(3) -> (R, t)
     *        z_c * (x_u, y_u, 1) = (x_c, y_c, z_c) = exp(epsillon) * (x, y, z)
     * 
     * @param p_c (x_c, y_c, z_c)
     * @return J = [1/z_c,     0, -x_c/z_c^2] * [I, -p_c^]
     *             [    0, 1/z_c, -y_c/z_c^2]
     *           = [1/z_c,     0, -x_c/z_c^2, -x_c*y_c/z_c^2, 1+x_c^2/z_c^2, -y_c/z_c]
     *             [    0, 1/z_c, -y_c/z_c^2, -1-y_c^2/z_c^2, x_c*y_c/z_c^2,  x_c/z_c]
     */
    inline Eigen::Matrix26d jaccobian_dxy1deps(const Eigen::Vector3d& p_c) {
        double x_c = p_c[0], y_c = p_c[1], z_c = p_c[2];
        double zinv = 1. / z_c;
        double zinv2 = zinv * zinv;
        double x2 = x_c * x_c, y2 = y_c * y_c, xy = x_c * y_c;
        Eigen::Matrix26d j;
        j << zinv,   0., -x_c * zinv2,      -xy * zinv2, 1. + x2 * zinv2, -y_c * zinv,
               0., zinv, -y_c * zinv2, -1. - y2 * zinv2,      xy * zinv2,  x_c * zinv;
        return j;
    }
    
} // namespace vslam

#endif
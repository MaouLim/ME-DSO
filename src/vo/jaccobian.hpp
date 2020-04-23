#ifndef _ME_VSLAM_JACCOBIAN_HPP_
#define _ME_VSLAM_JACCOBIAN_HPP_

#include <common.hpp>

namespace vslam {

    /**
     * @brief calculate jaccobian mat d(uv) / d(xy1) 
     *        d(uv): pixel coordinate (u, v, 1)'
     *        d(xy1): unit plane coordinate (x_u, y_u, 1)'
     *        z_c * (u, v, 1)' = K * z_c * (x_u, y_u, 1)' = K * (x_c, y_c, z_c)'
     *        (u, v, 1)' = K * (x_u, y_u, 1)'
     * @param focal_len (fx, fy)'
     * @return J = [fx,  0]
     *             [ 0, fy]
     */  
    inline Eigen::Matrix2d jaccobian_duvdxy1(
        const Eigen::Vector2d focal_len
    ) {
        return focal_len.asDiagonal();
    }

    /**
     * @brief calculate jaccobian mat d(xy1) / d(xyz) 
     *        d(xy1): unit plane coordinate (x_u, y_u, 1)' in a camera pose(R, t, s = 1)
     *        d(xyz): the world coordinate of p(x, y, z)'
     *        z_c * (x_u, y_u, 1)' = (x_c, y_c, z_c)' = sR * (x, y, z)' + t
     * 
     * @param p_c (x_c, y_c, z_c)'
     * @param sxrot sR
     * @return J = [1/z_c,     0, -x_c/z_c^2] * sR
     *             [    0, 1/z_c, -y_c/z_c^2]
     */            
    inline Eigen::Matrix23d jaccobian_dxy1dxyz(
        const Eigen::Vector3d& p_c, const Eigen::Matrix3d& sxrot
    ) {
        double x_c = p_c[0], y_c = p_c[1], z_c = p_c[2];
        double zinv = 1. / z_c;
        double zinv2 = zinv * zinv;

        Eigen::Matrix23d j;
        j << zinv,   0., -x_c * zinv2, 
               0., zinv, -y_c * zinv2;
        j = j * sxrot;

        return j;
    }

    /**
     * @brief BCH fomular on SE(3)
     *        d(Tp): T(R, t) -> SE(3), p(x, y, z)' at world coordinate system, Tp = Rp + t
     *        d(epsillon): epsillon = se(3)
     * @param p_c Tp = Rp + t = (x_c, y_c, z_c)
     * @return J = [I, -(Tp)^]
     */ 
    inline Eigen::Matrix36d jaccobian_dTpdeps(const Eigen::Vector3d& p_c) {
        Eigen::Matrix36d j;
        j.block<3, 3>(0, 0).setIdentity();
        j.block<3, 3>(0, 3) = -1.0 * utils::hat(p_c);
        return j;
    }

    /**
     * @brief calculate jaccobian mat d(xy1) / d(epsillon)
     *        d(xy1): unit plane coordinate (x_u, y_u, 1)' in a camera pose(R, t)
     *        d(epsillon): epsillon = se(3) -> (R, t)
     *        z_c * (x_u, y_u, 1)' = (x_c, y_c, z_c)' = exp(epsillon) * (x, y, z)'
     * 
     * @param p_c (x_c, y_c, z_c)'
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

    /**
     * @brief BCH fomular on Sim(3)
     *        d(Sp): S(R, t, s) -> Sim(3), p(x, y, z)' at world coordinate system, Sp = sRp + t
     *        d(zeta): zeta -> sim(3)
     * @param p_c Sp = sRp + t = (x_c, y_c, z_c)
     * @return J = [I, -(Sp)^, (Sp)]
     */ 
    inline Eigen::Matrix37d jaccobian_dSpdzet(const Eigen::Vector3d& p_c) {
        Eigen::Matrix37d j;
        j.block<3, 3>(0, 0).setIdentity();
        j.block<3, 3>(0, 3) = -1.0 * utils::hat(p_c);
        j.block<3, 1>(0, 6) = p_c;
        return j;
    }

    /**
     * @brief calculate jaccobian mat d(xy1) / d(zeta)
     *        d(xy1): unit plane coordinate (x_u, y_u, 1)' in a camera pose(R, t, s)
     *        d(zeta): zeta = sim(3) -> (R, t, s)
     *        z_c * (x_u, y_u, 1)' = (x_c, y_c, z_c)' = exp(zeta) * (x, y, z)'
     * 
     * @param p_c (x_c, y_c, z_c)'
     * @return J = [1/z_c,     0, -x_c/z_c^2] * [I, -p_c^, p_c]
     *             [    0, 1/z_c, -y_c/z_c^2]
     *           = [1/z_c,     0, -x_c/z_c^2, -x_c*y_c/z_c^2, 1+x_c^2/z_c^2, -y_c/z_c, 0]
     *             [    0, 1/z_c, -y_c/z_c^2, -1-y_c^2/z_c^2, x_c*y_c/z_c^2,  x_c/z_c, 0]
     */
    inline Eigen::Matrix27d jaccobian_dxy1dzet(const Eigen::Vector3d& p_c) {
        double x_c = p_c[0], y_c = p_c[1], z_c = p_c[2];
        double zinv = 1. / z_c;
        double zinv2 = zinv * zinv;
        double x2 = x_c * x_c, y2 = y_c * y_c, xy = x_c * y_c;
        Eigen::Matrix27d j;
        j << zinv,   0., -x_c * zinv2,      -xy * zinv2, 1. + x2 * zinv2, -y_c * zinv, 0.,
               0., zinv, -y_c * zinv2, -1. - y2 * zinv2,      xy * zinv2,  x_c * zinv, 0.;
        return j;
    }
    
} // namespace vslam

#endif
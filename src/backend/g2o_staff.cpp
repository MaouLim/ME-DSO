#include <backend/g2o_staff.hpp>

#include <vo/camera.hpp>
#include <vo/jaccobian.hpp>

namespace vslam::backend {

    void vertex_se3::oplusImpl(const number_t* d) {
        Sophus::Vector6d update;
        update << d[0], d[1], d[2], d[3], d[4], d[5];
        _estimate = Sophus::SE3d::exp(update) * _estimate;
    }

    //TODO sim3

    void vertex_xyz::oplusImpl(const number_t* d) {
        Eigen::Vector3d update;
        update << d[0], d[1], d[2];
        _estimate = _estimate + update;
    }

    void edge_xyz2uv_se3::computeError() {
        const vertex_xyz* v0 = static_cast<const vertex_xyz*>(_vertices[0]);
        const vertex_se3* v1 = static_cast<const vertex_se3*>(_vertices[1]);
        _error = _measurement - camera->cam2pixel(v1->estimate() * v0->estimate());
    }

    void edge_xyz2uv_se3::linearizeOplus() {
        const vertex_xyz* v0 = static_cast<const vertex_xyz*>(_vertices[0]);
        const vertex_se3* v1 = static_cast<const vertex_se3*>(_vertices[1]);
        Eigen::Vector3d xyz = v1->estimate() * v0->estimate();
        Eigen::Matrix2d duvdxy1 = camera->focal_len().asDiagonal();
        _jacobianOplusXi = -1.0 * duvdxy1 * vslam::jaccobian_dxy1dxyz(xyz, v1->estimate().rotationMatrix());
        _jacobianOplusXj = -1.0 * duvdxy1 * vslam::jaccobian_dxy1deps(xyz);
    }

    void edge_se3_to_se3::computeError() {
        const Sophus::SE3d& ti = static_cast<const vertex_se3*>(_vertices[0])->estimate();
        const Sophus::SE3d& tj = static_cast<const vertex_se3*>(_vertices[1])->estimate();
        _error = (_measurement.inverse() * ti.inverse() * tj).log();
    }

    void edge_se3_to_se3::linearizeOplus() {
        const Sophus::SE3d& tj = static_cast<const vertex_se3*>(_vertices[1])->estimate();
        Sophus::Matrix6d tj_inv_adj = tj.inverse().Adj();
        Sophus::Matrix6d jrinv = jr_inv(_error);
        _jacobianOplusXi = -jrinv * tj_inv_adj;
        _jacobianOplusXj =  jrinv * tj_inv_adj;
    }
}
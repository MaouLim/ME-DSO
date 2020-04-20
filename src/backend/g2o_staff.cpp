#include <backend/g2o_staff.hpp>

#include <vo/camera.hpp>
#include <vo/jaccobian.hpp>

namespace backend::g2o_staff {

    void vertex_se3::oplusImpl(const number_t* d) {
        Sophus::Vector6d update;
        update << d[0], d[1], d[2], d[3], d[4], d[5];
        _estimate = Sophus::SE3d::exp(update) * _estimate;
    }

    //TODO 

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
        // TODO check the fomular
        const vertex_xyz* v0 = static_cast<const vertex_xyz*>(_vertices[0]);
        const vertex_se3* v1 = static_cast<const vertex_se3*>(_vertices[1]);
        Eigen::Vector3d xyz = camera->cam2pixel(v1->estimate() * v0->estimate());
        _jacobianOplusXi = -1.0 * camera->eigen_mat() * v1->estimate().rotationMatrix();
        Eigen::Matrix2d duvdxy1 = camera->focal_len().asDiagonal();
        _jacobianOplusXj = -1.0 * duvdxy1 * vslam::jaccobian_dxy1deps(xyz);
    }
}
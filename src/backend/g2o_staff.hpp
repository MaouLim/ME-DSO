#ifndef _ME_VSLAM_G2O_STAFF_HPP_
#define _ME_VSLAM_G2O_STAFF_HPP_

#include <common.hpp>

namespace backend::g2o_staff {

    struct vertex_se3 : 
        g2o::BaseVertex<6, Sophus::SE3d> {

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        vertex_se3() = default;

        bool read(std::istream& is) { return false; }
        bool write(std::ostream& os) const { return false; }

        void setToOriginImpl() override { _estimate = Sophus::SE3d(); }
        void oplusImpl(const number_t* d) override;
    };

    struct vertex_sim3 : 
        g2o::BaseVertex<7, Sophus::Sim3d> {
        
        //TODO
    };

    struct vertex_xyz : 
        g2o::BaseVertex<3, Eigen::Vector3d> {

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        vertex_xyz() = default;

        bool read(std::istream& is) { return false; }
        bool write(std::ostream& os) const { return false; }

        void setToOriginImpl() override { _estimate.setZero(); }
        void oplusImpl(const number_t* d) override;
    };

    struct edge_xyz2uv_se3 : 
        g2o::BaseBinaryEdge<2, Eigen::Vector2d, vertex_xyz, vertex_se3> {

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        edge_xyz2uv_se3(const vslam::camera_ptr& _cam) : camera(_cam) { }

        bool read(std::istream& is) { return false; }
        bool write(std::ostream& os) const { return false; }

        void computeError() override;
        void linearizeOplus() override;

        vslam::camera_ptr camera;
    };
    
} // namespace backend::g2o_staff

#endif
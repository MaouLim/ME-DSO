#ifndef _ME_VSLAM_G2O_STAFF_HPP_
#define _ME_VSLAM_G2O_STAFF_HPP_

#include <common.hpp>

namespace vslam::backend {

    struct vertex_se3 : 
        g2o::BaseVertex<6, Sophus::SE3d> {

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        vertex_se3() = default;

        bool read(std::istream& is) override { return false; }
        bool write(std::ostream& os) const override { return false; }

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

        bool read(std::istream& is) override { return false; }
        bool write(std::ostream& os) const override { return false; }

        void setToOriginImpl() override { _estimate.setZero(); }
        void oplusImpl(const number_t* d) override;
    };

    /**
     * @brief error edge between pose nodes and landmark nodes
     * @param measurement the uv pixel coordinate to represent
     *                    map points on the frames
     */ 
    struct edge_xyz2uv_se3 : 
        g2o::BaseBinaryEdge<2, Eigen::Vector2d, vertex_xyz, vertex_se3> {

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        edge_xyz2uv_se3(const vslam::camera_ptr& _cam) : camera(_cam) { }

        bool read(std::istream& is) override { return false; }
        bool write(std::ostream& os) const override { return false; }

        void computeError() override;
        void linearizeOplus() override;

        vslam::camera_ptr camera;
    };

    /**
     * @brief error edge between two pose nodes
     * @param measurement se3 transformation between two frame 
     *                    t_v1v0 from v0 to v1
     */ 
    struct edge_se3_to_se3 : 
        g2o::BaseBinaryEdge<6, Sophus::SE3d, vertex_se3, vertex_se3> {
        
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        edge_se3_to_se3() = default;

        bool read(std::istream& is) override { return false; }
        bool write(std::ostream& os) const override { return false; }

        void computeError() override;
        void linearizeOplus() override;

        /**
         * @note se3 err has to be normalized
         */ 
        static Sophus::Matrix6d jr_inv(const Sophus::Vector6d& err) {
            Sophus::Matrix6d res;
            res.block<3, 3>(0, 0) = Sophus::SO3d::hat(err.tail<3>());
            res.block<3, 3>(0, 3) = Sophus::SO3d::hat(err.head<3>());
            res.block<3, 3>(3, 0).setZero();
            res.block<3, 3>(3, 3) = res.block<3, 3>(0, 0);
            res = 0.5 * res + Sophus::Matrix6d::Identity();
            return res;
        }
    };

    struct g2o_optimizer {

        explicit g2o_optimizer(bool verbose = true, size_t n_trials = 5);
        virtual ~g2o_optimizer() = default;

        virtual void create_graph() { }
        virtual void update() { }

        std::pair<double, double> optimize(size_t n_iterations) {
            _optimizer.clear();

            this->create_graph();

            _optimizer.initializeOptimization();
            _optimizer.computeActiveErrors();
            double err_before = _optimizer.activeChi2();
            _optimizer.optimize(n_iterations);
            double err_after = _optimizer.activeChi2();

            return std::make_pair(err_before, err_after);
        }

    protected:
        g2o::SparseOptimizer _optimizer; 
    };
    
} // namespace backend

#endif
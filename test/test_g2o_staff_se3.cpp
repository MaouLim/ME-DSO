#include <fstream>

#include <common.hpp>
#include <backend/g2o_staff.hpp>

#include <pcl/point_types.h>
#include <pcl/pcl_base.h>
#include <pcl/visualization/cloud_viewer.h>

const std::string data_file = "data/sphere.g2o";

typedef pcl::PointXYZRGB                point_t;
typedef pcl::PointCloud<point_t>        point_cloud_t;
typedef pcl::visualization::CloudViewer viewer_t;

struct v_se3 : 
    vslam::backend::vertex_se3 {
    
    bool read(std::istream& is) override {
        double data[7];
        for (int i = 0; i < 7; i++) {
            is >> data[i];
        }
        setEstimate(Sophus::SE3d(
            Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
            Sophus::Vector3d(data[0], data[1], data[2])
        ));
        return true;
    }

    bool write(std::ostream& os) const override {
        os << id() << " ";
        Eigen::Quaterniond q = _estimate.unit_quaternion();
        os << _estimate.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << std::endl;
        return true;
    }
};

struct e_se3_to_se3 : 
    vslam::backend::edge_se3_to_se3 {

    bool read(std::istream& is) override {
        double data[7];
        for (int i = 0; i < 7; i++) { is >> data[i]; }
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        q.normalize();
        setMeasurement(Sophus::SE3d(q, Sophus::Vector3d(data[0], data[1], data[2])));
        for (int i = 0; i < information().rows() && is.good(); ++i) {
            for (int j = i; j < information().cols() && is.good(); ++j) {
                is >> information()(i, j);
                if (i != j) { information()(j, i) = information()(i, j); }
            }
        }
        return true;
    }

    bool write(std::ostream &os) const override {
        auto v0 = (const vslam::backend::vertex_se3*) _vertices[0];
        auto v1 = (const vslam::backend::vertex_se3*) _vertices[1];
        os << v0->id() << " " << v1->id() << " ";
        const Sophus::SE3d& m = _measurement;
        const Eigen::Quaterniond& q = m.unit_quaternion();
        os << m.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << " ";

        // information matrix 
        for (int i = 0; i < information().rows(); ++i) {
            for (int j = i; j < information().cols(); ++j) {
                os << information()(i, j) << " ";
            }
        }
        os << std::endl;
        return true;
    }
};

int main(int argc, char** argv) {

    std::ifstream stream(data_file, std::ios_base::in);
    if (!stream.good()) { return -1; }

    using block_solver_t  = g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>>;
    using linear_solver_t = g2o::LinearSolverEigen<block_solver_t::PoseMatrixType>;

    auto algo = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<block_solver_t>(
            g2o::make_unique<linear_solver_t>()
        )
    );

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algo);
    optimizer.setVerbose(true);
    
    int count_vertex = 0, count_edge = 0;
    while (!stream.eof()) {

        std::string name;
        stream >> name;
        if ("VERTEX_SE3:QUAT" == name) {
            v_se3* v = new v_se3();
            int id = 0;
            stream >> id;
            v->setId(id);
            v->read(stream);
            if (0 == id) { v->setFixed(true); }
            optimizer.addVertex(v);
            ++count_vertex;
        }
        else if ("EDGE_SE3:QUAT" == name) {
            e_se3_to_se3* e = new e_se3_to_se3();
            int id1 = 0, id2 = 0;
            stream >> id1 >> id2;
            e->setId(count_edge++);
            e->setVertex(0, optimizer.vertices()[id1]);
            e->setVertex(1, optimizer.vertices()[id2]);
            e->read(stream);
            optimizer.addEdge(e);
        }
    } 

    optimizer.initializeOptimization();
    optimizer.optimize(1);

    /**
     * @brief show the point cloud after optimize 1
     */ 
    {
        point_cloud_t::Ptr cloud(new point_cloud_t());
        for (auto v : optimizer.vertices()) {

            Sophus::SE3d pose = ((v_se3*) v.second)->estimate();
            Eigen::Vector3d t = pose.translation();

            point_t p_xyzrgb;
            p_xyzrgb.x = t[0];
            p_xyzrgb.y = t[1];
            p_xyzrgb.z = t[2];

            p_xyzrgb.r = 255;
            p_xyzrgb.g = 255;
            p_xyzrgb.b = 255;

            cloud->push_back(p_xyzrgb);
        }

        viewer_t viewer("after opt 1");
        viewer.showCloud(cloud);
        while (!viewer.wasStopped()) { }
    }


    optimizer.initializeOptimization();
    optimizer.optimize(10);

    /**
     * @brief show the point cloud after optimize 10
     */ {
        point_cloud_t::Ptr cloud(new point_cloud_t());
        for (auto v : optimizer.vertices()) {
        
            Sophus::SE3d pose = ((v_se3*) v.second)->estimate();
            Eigen::Vector3d t = pose.translation();

            point_t p_xyzrgb;
            p_xyzrgb.x = t[0];
            p_xyzrgb.y = t[1];
            p_xyzrgb.z = t[2];

            p_xyzrgb.r = 255;
            p_xyzrgb.g = 255;
            p_xyzrgb.b = 255;

            cloud->push_back(p_xyzrgb);
        }

        viewer_t viewer("after opt 10");
        viewer.showCloud(cloud);
        while (!viewer.wasStopped()) { }
    }

    return 0;
}
#include <iostream>
#include <fstream>
#include <sstream>

#include <vo/initializer.hpp>
#include <vo/camera.hpp>
#include <vo/frame.hpp>
#include <vo/feature.hpp>
#include <vo/map_point.hpp>
#include <backend/g2o_staff.hpp>

#include <pcl/point_types.h>
#include <pcl/pcl_base.h>
#include <pcl/visualization/cloud_viewer.h>

const std::string dataset_dir = "data/fr1_floor/";
const std::string association_file = dataset_dir + "rgb-gt.txt";
const size_t      max_images = 500;

cv::Mat find_associate_depth(double timestamp) {
    std::ifstream fin(dataset_dir + "rgb-depth.txt");
    assert(fin.good());

    while (!fin.eof()) {
        std::string line;
        std::getline(fin, line);
        std::stringstream stream(line);
        double ts;
        stream >> ts;
        if (ts == timestamp) {
            std::string _1, _2, _3;
            stream >> _1 >> _2 >> _3;
            return cv::imread(dataset_dir + _3, cv::IMREAD_UNCHANGED);
        }
    }
    assert(false);
    return cv::Mat();
}

# ifdef _OPT_
using namespace vslam;

void optimize(const frame_ptr& ref, const frame_ptr& cur) {

    auto linear_solver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    auto block_solver  = g2o::make_unique<g2o::BlockSolverX>(std::move(linear_solver));
        
    g2o::OptimizationAlgorithmLevenberg* algo = 
        new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
        
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algo);
    optimizer.setVerbose(true);

    int vid = 0, e_id = 0;

    auto v_f0 = ref->create_g2o(vid++, true);
    auto v_f1 = cur->create_g2o(vid++, false);

    optimizer.addVertex(v_f0);
    optimizer.addVertex(v_f1);

    for (auto& feat : ref->features) {
        assert(!feat->describe_nothing());
        auto v_p = feat->map_point_describing->create_g2o(vid++);
        auto e = feat->create_g2o(e_id, v_p, v_f0);
        optimizer.addVertex(v_p);
        optimizer.addEdge(e);
    }

    for (auto& feat : cur->features) {
        assert(!feat->describe_nothing());
        auto v_p = feat->map_point_describing->v;
        assert(v_p);
        auto e = feat->create_g2o(e_id++, v_p, v_f1);
        optimizer.addEdge(e);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(20);

    ref->update_from_g2o();
    cur->update_from_g2o();

    for (auto& feat : ref->features) {
        assert(!feat->describe_nothing());
        feat->map_point_describing->update_from_g2o();
    }
}
#endif

double calc_scale(
    const std::vector<Eigen::Vector3d>& vs_est, 
    const std::vector<Eigen::Vector3d>& vs_gt
) {
    assert(vs_est.size() == vs_gt.size());
    assert(!vs_est.empty());

    double a = 0, b = 0;
    for (size_t i = 0; i < vs_est.size(); ++i) {
        a +=  vs_gt[i].dot(vs_est[i]);
        b += vs_est[i].dot(vs_est[i]);
    }

    return a / b;
}

int main(int argc, char** argv) {

    std::vector<std::pair<double, cv::Mat>> images;
    std::vector<Sophus::SE3d>               gts;

    std::ifstream fin(association_file);
    assert(fin.good());

    size_t count = 0;

    while (!fin.eof()) {
        if (max_images <= count) { break; }

        std::string line;
        std::getline(fin, line);
        std::stringstream stream(line);

        double timestamp0, timestamp1;
        std::string path;
        Eigen::Vector3d t; // tx ty tz 
        double q[4];       // qx qy qz qw

        stream >> timestamp0 >> path 
               >> timestamp1 >> t[0] >> t[1] >> t[2] 
                             >> q[0] >> q[1] >> q[2] >> q[3];

        Eigen::Quaterniond r(q[3], q[0], q[1], q[2]);            
        cv::Mat img = cv::imread(dataset_dir + path, cv::IMREAD_GRAYSCALE);
        assert(img.data);

        images.emplace_back(timestamp0, img);
        gts.emplace_back(r, t);
        ++count;
    }

    vslam::camera_ptr cam = utils::mk_vptr<vslam::pinhole_camera>(
            config::height, config::width, 517.3, 516.5, 325.1, 249.7
    );

    vslam::initializer init;
    int start = 0;
    vslam::frame_ptr first_frame = 
        utils::mk_vptr<vslam::frame>(cam, images[start].second, images[start].first);

    std::cout << "set_first: " << init.add_frame(first_frame) << std::endl;;
    vslam::frame_ptr f = nullptr;

    size_t i = start + 1;
    for (; i < max_images; ++i) {
        auto& pair = images[i];
        f = utils::mk_vptr<vslam::frame>(cam, pair.second, pair.first);
        auto ret = init.add_frame(f);
        std::cout << i << ".add_frame: " << ret << std::endl;
        if (vslam::initializer::SUCCESS == ret) { break; }
    }

    typedef pcl::PointXYZRGB                point_t;
    typedef pcl::PointCloud<point_t>        point_cloud_t;
    typedef pcl::visualization::CloudViewer viewer_t;

    point_cloud_t::Ptr cloud(new point_cloud_t());

    cv::Mat depth = find_associate_depth(first_frame->timestamp);
    assert(depth.data);

    std::vector<Eigen::Vector3d> vs_est, vs_gt;

    for (auto& feat : first_frame->features) {
        if (feat->describe_nothing()) { continue; }
        const auto& mp  = feat->map_point_describing;
        const auto& xy1 = feat->xy1;
        const auto& uv  = feat->uv;
        uint16_t d = depth.at<uint16_t>(uv.y(), uv.x());
        if (d <= 0) { continue; }
        double z = double(d) / 5000.0;

        point_t p_gt; {
            p_gt.x = xy1.x() * z;
            p_gt.y = xy1.y() * z;
            p_gt.z = z;
            p_gt.r = 255;
            p_gt.g = 255;
            p_gt.b = 255;
            vs_gt.emplace_back(p_gt.x, p_gt.y, p_gt.z);
        }

        point_t p_est; {
            p_est.x = mp->position.x();
            p_est.y = mp->position.y();
            p_est.z = mp->position.z();
            p_est.r = 255;
            p_est.g = 0;
            p_est.b = 0;
            vs_est.emplace_back(p_est.x, p_est.y, p_est.z);
        }
        
        cloud->push_back(p_gt);
        cloud->push_back(p_est);
    }
    viewer_t viewer("pts");
    viewer.showCloud(cloud);
    while (!viewer.wasStopped()) { }

    double k = calc_scale(vs_est, vs_gt);
    cloud->clear();
    for (auto& feat : first_frame->features) {
        if (feat->describe_nothing()) { continue; }
        const auto& mp  = feat->map_point_describing;
        const auto& xy1 = feat->xy1;
        const auto& uv  = feat->uv;
        uint16_t d = depth.at<uint16_t>(uv.y(), uv.x());
        if (d <= 0) { continue; }
        double z = double(d) / 5000.0;

        point_t p_gt; {
            p_gt.x = xy1.x() * z;
            p_gt.y = xy1.y() * z;
            p_gt.z = z;
            p_gt.r = 255;
            p_gt.g = 255;
            p_gt.b = 255;
            vs_gt.emplace_back(p_gt.x, p_gt.y, p_gt.z);
        }

        point_t p_est; {
            p_est.x = mp->position.x() * k;
            p_est.y = mp->position.y() * k;
            p_est.z = mp->position.z() * k;
            p_est.r = 255;
            p_est.g = 0;
            p_est.b = 0;
            vs_est.emplace_back(p_est.x, p_est.y, p_est.z);
        }
        
        cloud->push_back(p_gt);
        cloud->push_back(p_est);
    }
    viewer_t viewer2("pts2");
    viewer2.showCloud(cloud);
    while (!viewer2.wasStopped()) { }


#ifdef _OPT_
    std::cout << "start BA..." << std::endl;
    optimize(first_frame, f);

    cloud->clear();

    for (auto& feat : first_frame->features) {
        if (feat->describe_nothing()) { continue; }
        const auto& mp  = feat->map_point_describing;
        const auto& xy1 = feat->xy1;
        const auto& uv  = feat->uv;
        uint16_t d = depth.at<uint16_t>(uv.y(), uv.x());
        if (d <= 0) { continue; }
        double z = double(d) / 5000.0;

        point_t p_gt; {
            p_gt.x = xy1.x() * z;
            p_gt.y = xy1.y() * z;
            p_gt.z = z;
            p_gt.r = 255;
            p_gt.g = 255;
            p_gt.b = 255;
        }

        point_t p_est; {
            p_est.x = mp->position.x();
            p_est.y = mp->position.y();
            p_est.z = mp->position.z();
            p_est.r = 255;
            p_est.g = 0;
            p_est.b = 0;
        }
        
        cloud->push_back(p_gt);
        cloud->push_back(p_est);
    }
    viewer_t viewer2("after BA");
    viewer2.showCloud(cloud);
    while (!viewer2.wasStopped()) { }
#endif
    return 0;
}


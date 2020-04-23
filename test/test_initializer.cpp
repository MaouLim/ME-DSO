#include <iostream>
#include <fstream>
#include <sstream>

#include <vo/initializer.hpp>
#include <vo/camera.hpp>
#include <vo/frame.hpp>
#include <vo/feature.hpp>
#include <vo/map_point.hpp>

const std::string dataset_dir = "data/fr1_xyz/";
const std::string association_file = dataset_dir + "rgb-gt.txt";
const size_t      max_images = 50;

void draw_level0_features(const vslam::frame_ptr& frame) {
    cv::Mat img = frame->pyramid[0].clone();
    for (auto& each : frame->features) {
        if (0 != each->level) { continue; }
        cv::circle(img, cv::Point2i(each->uv.x(), each->uv.y()), 3, 255, 2);
    }

    cv::imshow(std::to_string(frame->id), img);
    cv::waitKey();
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

    vslam::camera_ptr cam = 
        utils::mk_vptr<vslam::pinhole_camera>(config::height, config::width, 517.3, 516.5, 325.1, 249.7);

    vslam::initializer init;

    vslam::frame_ptr first_frame = 
        utils::mk_vptr<vslam::frame>(cam, images.front().second, images.front().first);
    cv::imshow("ref", images.front().second);

    std::cout << "set_first: " << init.set_first(first_frame) << std::endl;;
    vslam::frame_ptr f = nullptr;

    size_t i = 1;
    for (; i < max_images; ++i) {
        auto& pair = images[i];
        f = utils::mk_vptr<vslam::frame>(cam, pair.second, pair.first);
        auto ret = init.add_frame(f);
        std::cout << i << ".add_frame: " << ret << std::endl;
        cv::imshow("cur", pair.second);
        cv::waitKey();
        if (vslam::initializer::SUCCESS == ret) { break; }
    }

    {
        Sophus::SE3d gt_cr = gts[i] * gts[0].inverse();
        std::cout << "GT:" << std::endl;
        std::cout << gt_cr.rotationMatrix() << std::endl;
        std::cout << gt_cr.translation().normalized().transpose() << std::endl;
    }

    {
        std::cout << "EST:" << std::endl;
        std::cout << f->t_cw.rotationMatrix() << std::endl;
        std::cout << f->t_cw.translation().normalized().transpose() << std::endl;
    }

    draw_level0_features(first_frame);
    draw_level0_features(f);

    std::vector<cv::Point2f> uv0;
    std::vector<cv::Point2f> uv1;

    auto itr_ref = first_frame->features.begin();
    auto itr_cur = f->features.begin();
    while (itr_cur != f->features.end()) {
        const auto& mp0 = (*itr_ref)->map_point_describing;
        const auto& mp1 = (*itr_cur)->map_point_describing;
        assert(mp0 == mp1);

        Eigen::Vector2d p0 = (*itr_ref)->uv;
        Eigen::Vector2d p1 = (*itr_cur)->uv;

        uv0.emplace_back(p0.x(), p0.y());
        uv1.emplace_back(p1.x(), p1.y());

        // std::cout << p0.transpose() << ", " 
        //           << p1.transpose() << std::endl;

        ++itr_cur; ++itr_ref;
    }

    std::cout << "Size of matches: " << uv0.size() << std::endl;

    cv::Mat cv_cam_mat = cam->cv_mat();
    cv::Mat essential = cv::findEssentialMat(uv0, uv1, cv_cam_mat);
    cv::Mat R, cv_t;
    auto n_inliers = cv::recoverPose(essential, uv0, uv1, cv_cam_mat, R, cv_t);

    Eigen::Vector3d t;
    t << cv_t.at<double>(0), cv_t.at<double>(1), cv_t.at<double>(2);

    std::cout << "EST by OpenCV:" << std::endl;
    std::cout << "n_inliers: " << n_inliers << std::endl;
    std::cout << R << std::endl;
    std::cout << t.normalized().transpose() << std::endl;

    //TODO using the feature-based method to evalute the pose

    return 0;
}
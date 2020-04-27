#include <iostream>

#include <vo/initializer.hpp>
#include <vo/pose_estimator.hpp>
#include <vo/matcher.hpp>
#include <vo/camera.hpp>
#include <vo/frame.hpp>
#include <vo/feature.hpp>

const std::string dataset_dir = "data/fr1_xyz/";
const std::string association_file = dataset_dir + "rgb-gt.txt";
const size_t      max_images = 500;

void get_imgs_and_gts(
    std::vector<std::pair<double, cv::Mat>>& images, 
    std::vector<Sophus::SE3d>&               gts
) {
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
}

cv::Mat draw_feats(const vslam::frame_ptr& f) {

    cv::Scalar_<int> r = { 0, 0, 255 };
    cv::Scalar_<int> g = { 0, 255, 0 };
    cv::Scalar_<int> blu = { 255, 0, 0 };
    cv::Scalar_<int> bla = { 0, 0, 0 };
    cv::Scalar_<int> y = { 0, 255, 255 };

    cv::Scalar color[5] = { r, g, blu, bla, y };

    cv::Mat f_img;
    cv::cvtColor(f->image(), f_img, cv::COLOR_GRAY2BGR);
    for (auto& feat : f->features) {
        cv::circle(f_img, cv::Point2f{ feat->uv.x(), feat->uv.y() }, 2, color[feat->level]);
    }

    return f_img;
}

std::vector<std::pair<double, cv::Mat>> images;
std::vector<Sophus::SE3d>               gts;

int main(int argc, char** argv) {

    get_imgs_and_gts(images, gts);

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

    size_t test_img_idx = i + 2;
    vslam::frame_ptr test_frame = 
        utils::mk_vptr<vslam::frame>(cam, images[test_img_idx].second, images[test_img_idx].first);

    cv::imshow("f", draw_feats(f));
    cv::imshow("test_frame", test_frame->image());
    cv::waitKey();

    //vslam::twoframe_estimator(config::max_opt_iterations, 4, 0, );

    return 0;
}

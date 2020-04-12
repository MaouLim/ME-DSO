#include <common.hpp>

#include <vo/camera.hpp>
#include <vo/frame.hpp>
#include <vo/feature.hpp>

int main(int argc, char** argv) {

    cv::Mat img_ref = cv::imread("data/01.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img_cur = cv::imread("data/02.png", cv::IMREAD_GRAYSCALE);

    assert(img_ref.data && img_cur.data);

    vslam::detector_ptr detector 
        = utils::mk_vptr<vslam::fast_detector>(480, 640, 10, 5);
    vslam::camera_ptr cam = 
        utils::mk_vptr<vslam::pinhole_camera>(480, 640, 517.3, 516.5, 325.1, 249.7);

    vslam::frame_ptr ref = utils::mk_vptr<vslam::frame>(cam, img_ref, 0);
    vslam::frame_ptr cur = utils::mk_vptr<vslam::frame>(cam, img_cur, 0);

    vslam::feature_set a, b;

    detector->detect(ref, 100.0, a);
    detector->detect(cur, 100.0, b);

    int level_counter[5] = { 0 };

    for (auto each_feat : a) {
        if (each_feat->level == 0) {
            auto uv = each_feat->uv;
            cv::circle(img_ref, cv::Point2i(uv.x(), uv.y()), 3, 255, 2);
        }

        ++level_counter[each_feat->level];
    }

    for (int i = 0; i < 5; ++i) {
        std::cout << "level " << i << ": " << level_counter[i] << std::endl;
    }

    std::cout << "a n_features :" << a.size() << std::endl;
    std::cout << "b n_features :" << b.size() << std::endl;

    cv::imshow("ref", img_ref);
    cv::waitKey();
    
    return 0;
}
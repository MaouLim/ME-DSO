#include <iostream>

#include <common.hpp>

#include <vo/camera.hpp>
#include <vo/pose_estimator.hpp>
#include <vo/frame.hpp>
#include <vo/feature.hpp>

int main(int argc, char** argv) {

    cv::Mat img_ref = cv::imread("data/01.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img_cur = cv::imread("data/02.png", cv::IMREAD_GRAYSCALE);

    assert(img_ref.data && img_cur.data);
//fast_detector(int _h, int _w, int _cell_sz, size_t _n_levels);
    vslam::detector_ptr detector 
        = utils::mk_vptr<vslam::fast_detector>(480, 640, 10, 5);
    vslam::camera_ptr cam = 
        utils::mk_vptr<vslam::pinhole_camera>(480, 640, 517.3, 516.5, 325.1, 249.7);

    vslam::frame_ptr ref = utils::mk_vptr<vslam::frame>(cam, img_ref, 0);
    vslam::frame_ptr cur = utils::mk_vptr<vslam::frame>(cam, img_cur, 0);

    detector->detect(ref, 10.0, ref->features);
    detector->detect(cur, 10.0, cur->features);

    Sophus::SE3d t_cr;

    vslam::pose_estimator est(10, 0, 4);
    est.estimate(ref, cur, t_cr);

    std::cout << "R:\n" << t_cr.rotationMatrix() << std::endl;
    std::cout << "t:\n" << t_cr.translation() << std::endl;
    
    return 0;
}
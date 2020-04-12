#include <iostream>

#include <common.hpp>

#include <vo/camera.hpp>
#include <vo/pose_estimator.hpp>
#include <vo/frame.hpp>

#include <utils/utils.hpp>

int main(int argc, char** argv) {

    cv::Mat img_ref = cv::imread("../data/01.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img_cur = cv::imread("../data/02.png", cv::IMREAD_GRAYSCALE);

    //vslam::abstract_camera* cam = new vslam::pinhole_camera(480, 640, 517.3, 516.5, 325.1, 249.7);
    vslam::camera_ptr cam = 
        utils::mk_vptr<vslam::pinhole_camera>(480, 640, 517.3, 516.5, 325.1, 249.7);

    vslam::frame_ptr ref = utils::mk_vptr<vslam::frame>(cam, img_ref, 0);
    vslam::frame_ptr cur = utils::mk_vptr<vslam::frame>(cam, img_cur, 0);

    Sophus::SE3d t_cr;

    vslam::pose_estimator est;
    est.estimate(ref, cur, t_cr);

    std::cout << "R:\n" << t_cr.rotationMatrix() << std::endl;
    std::cout << "t:\n" << t_cr.translation() << std::endl;
    
    return 0;
}
#include <iostream>

#include <common.hpp>

#include <vo/camera.hpp>
#include <vo/pose_estimator.hpp>
#include <vo/frame.hpp>
#include <vo/map_point.hpp>
#include <vo/feature.hpp>

double depth_scale = 1. / 5000.0;

void set_ref_frame(
    const vslam::frame_ptr&   ref, 
    const vslam::feature_set& candidates,
    const cv::Mat&            depth_img
) {
    ref->set_pose(Sophus::SE3d());
    for (auto& each_feat : candidates) {
        const Eigen::Vector2d& uv_level0 = each_feat->uv;
        uint16_t d = depth_img.at<uint16_t>(uv_level0.y(), uv_level0.x());
        if (d <= 0) { continue; }
        double z = (double) d * depth_scale;
        Eigen::Vector3d xyz_ref = ref->camera->pixel2cam(uv_level0, z);
        vslam::map_point_ptr mp = utils::mk_vptr<vslam::map_point>(xyz_ref);
        each_feat->set_describing(mp);
        each_feat->use();
        //std::cout << uv_level0.transpose() << std::endl;
    }
    std::cout << "n_features: " << ref->n_features << std::endl;
}

int main(int argc, char** argv) {

    cv::Mat img_ref = cv::imread("data/01.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img_cur = cv::imread("data/04.png", cv::IMREAD_GRAYSCALE);

    cv::Mat depth_ref = cv::imread("data/01d.png", cv::IMREAD_UNCHANGED);

    assert(img_ref.data && img_cur.data && depth_ref.data);

    vslam::detector_ptr detector 
        = utils::mk_vptr<vslam::fast_detector>(480, 640, 10, 5);
    vslam::camera_ptr cam = 
        utils::mk_vptr<vslam::pinhole_camera>(480, 640, 517.3, 516.5, 325.1, 249.7);

    vslam::frame_ptr ref = utils::mk_vptr<vslam::frame>(cam, img_ref, 0);
    vslam::frame_ptr cur = utils::mk_vptr<vslam::frame>(cam, img_cur, 0);

    vslam::feature_set ref_feats, cur_feats;

    detector->detect(ref, 1000.0, ref_feats);
    detector->detect(cur, 1000.0, cur_feats);

    set_ref_frame(ref, ref_feats, depth_ref);

    {
        Sophus::SE3d t_cr;
        vslam::twoframe_estimator est(10, 4, 0, vslam::twoframe_estimator::LK_ICIA);
        est.estimate(ref, cur, t_cr);

        std::cout << "ICIA:" << std::endl;
        std::cout << "R:\n" << t_cr.rotationMatrix() << std::endl;
        std::cout << "t:\n" << t_cr.translation() << std::endl;
    }
    {
        Sophus::SE3d t_cr;
        vslam::twoframe_estimator est(10, 4, 0, vslam::twoframe_estimator::LK_FCFA);
        est.estimate(ref, cur, t_cr);

        std::cout << "FCFA:" << std::endl;
        std::cout << "R:\n" << t_cr.rotationMatrix() << std::endl;
        std::cout << "t:\n" << t_cr.translation() << std::endl;
    }
    {
        Sophus::SE3d t_cr;
        vslam::twoframe_estimator est(10, 4, 0, vslam::twoframe_estimator::LK_ICIA_G2O);
        est.estimate(ref, cur, t_cr);

        std::cout << "ICIA_G2O:" << std::endl;
        std::cout << "R:\n" << t_cr.rotationMatrix() << std::endl;
        std::cout << "t:\n" << t_cr.translation() << std::endl;
    }

    return 0;
}
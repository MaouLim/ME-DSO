#include <iostream>

#include <common.hpp>

#include <vo/camera.hpp>
#include <vo/feature.hpp>
#include <vo/frame.hpp>
#include <vo/pose_estimator.hpp>
#include <vo/map_point.hpp>

#include <opencv2/features2d.hpp>

void set_frames(
    const cv::Mat&          img_ref, 
    const cv::Mat&          img_cur, 
    const cv::Mat&          depth_ref,
    const vslam::frame_ptr& ref,
    const vslam::frame_ptr& cur
) {
    std::vector<cv::KeyPoint> kpts_ref, kpts_cur;
    cv::Mat desc_ref, desc_cur;
    std::vector<cv::DMatch> matches;

    auto cv_det = cv::ORB::create(1000);
    cv_det->detectAndCompute(img_ref, cv::Mat(), kpts_ref, desc_ref);
    cv_det->detectAndCompute(img_cur, cv::Mat(), kpts_cur, desc_cur);

    cv::Ptr<cv::BFMatcher> cv_matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    cv_matcher->match(desc_ref, desc_cur, matches);

    double min_dis = UINT32_MAX, max_dis = 0.;

	for (auto i = 0; i < matches.size(); ++i) {
		double dis = matches[i].distance;
		if (dis < min_dis) { min_dis = dis; }
		if (max_dis < dis) { max_dis = dis; }
	}

    double threshold = min_dis + 0.2 * (max_dis - min_dis);

    ref->set_pose(Sophus::SE3d());

    size_t count_points = 0;

    for (auto& match : matches) {
        if (threshold < match.distance) { continue; }
        const cv::KeyPoint& kpt = kpts_ref[match.queryIdx];
        uint16_t d = depth_ref.at<uint16_t>(kpt.pt);
        if (d <= 0) { continue; }

        ++count_points;

        double z = double(d) / 5000.0;
        const cv::KeyPoint& kpt_cur = kpts_cur[match.trainIdx];
        Eigen::Vector2d uv_ref = { kpt.pt.x, kpt.pt.y };

        Eigen::Vector2d uv_cur = { kpt_cur.pt.x, kpt_cur.pt.y };
        Eigen::Vector3d xyz_ref = ref->camera->pixel2cam(uv_ref, z);
        
        vslam::feature_ptr feat_ref = utils::mk_vptr<vslam::feature>(ref, uv_ref, 5);
        vslam::feature_ptr feat_cur = utils::mk_vptr<vslam::feature>(cur, uv_cur, 5);
        vslam::map_point_ptr mp = utils::mk_vptr<vslam::map_point>(xyz_ref);

        feat_ref->map_point_describing = mp;
        feat_cur->map_point_describing = mp;
        mp->set_observed_by(feat_ref);
        mp->set_observed_by(feat_cur);

        ref->add_feature(feat_ref);
        cur->add_feature(feat_cur);
    }

    std::cout << "Count Points: " << count_points << std::endl;
}

int main(int argc, char** argv) {

    cv::Mat img_ref = cv::imread("data/01.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img_cur = cv::imread("data/04.png", cv::IMREAD_GRAYSCALE);

    cv::Mat depth_ref = cv::imread("data/01d.png", cv::IMREAD_UNCHANGED);

    assert(img_ref.data && img_cur.data && depth_ref.data);

    vslam::camera_ptr cam = 
        utils::mk_vptr<vslam::pinhole_camera>(480, 640, 517.3, 516.5, 325.1, 249.7);

    vslam::frame_ptr ref = utils::mk_vptr<vslam::frame>(cam, img_ref, 0);
    vslam::frame_ptr cur = utils::mk_vptr<vslam::frame>(cam, img_cur, 0);

    set_frames(img_ref, img_cur, depth_ref, ref, cur);

    Sophus::SE3d coarse_t_cr;
    vslam::twoframe_estimator tf_est(10, 4, 0, vslam::twoframe_estimator::LK_FCFA);
    tf_est.estimate(ref, cur, coarse_t_cr);

    std::cout << "Coarse pose: " << std::endl;
    std::cout << "R:\n" << coarse_t_cr.rotationMatrix() << std::endl;
    std::cout << "t:\n" << coarse_t_cr.translation() << std::endl;

    cur->set_pose(coarse_t_cr);

    {
        Sophus::SE3d refined_t_cr;
        vslam::singleframe_estimator sf_est(10, vslam::singleframe_estimator::PNP_BA);
        sf_est.estimate(cur, refined_t_cr);

        std::cout << "PNP_BA Refined pose: " << std::endl;
        std::cout << "R:\n" << refined_t_cr.rotationMatrix() << std::endl;
        std::cout << "t:\n" << refined_t_cr.translation() << std::endl;
    }
    {
        Sophus::SE3d refined_t_cr;
        vslam::singleframe_estimator sf_est(10, vslam::singleframe_estimator::PNP_G2O);
        sf_est.estimate(cur, refined_t_cr);
    
        std::cout << "PNP_G2O Refined pose: " << std::endl;
        std::cout << "R:\n" << refined_t_cr.rotationMatrix() << std::endl;
        std::cout << "t:\n" << refined_t_cr.translation() << std::endl;
    }
    {
        Sophus::SE3d refined_t_cr;
        vslam::singleframe_estimator sf_est(10, vslam::singleframe_estimator::PNP_CV);
        sf_est.estimate(cur, refined_t_cr);
    
        std::cout << "PNP_CV Refined pose: " << std::endl;
        std::cout << "R:\n" << refined_t_cr.rotationMatrix() << std::endl;
        std::cout << "t:\n" << refined_t_cr.translation() << std::endl;
    }
    {
        Sophus::SE3d refined_t_cr;
        vslam::singleframe_estimator sf_est(10, vslam::singleframe_estimator::EPNP_CV);
        sf_est.estimate(cur, refined_t_cr);
    
        std::cout << "EPNP_CV Refined pose: " << std::endl;
        std::cout << "R:\n" << refined_t_cr.rotationMatrix() << std::endl;
        std::cout << "t:\n" << refined_t_cr.translation() << std::endl;
    }
    {
        Sophus::SE3d refined_t_cr;
        vslam::singleframe_estimator sf_est(10, vslam::singleframe_estimator::PNP_DLS_CV);
        sf_est.estimate(cur, refined_t_cr);
    
        std::cout << "PNP_DLS_CV Refined pose: " << std::endl;
        std::cout << "R:\n" << refined_t_cr.rotationMatrix() << std::endl;
        std::cout << "t:\n" << refined_t_cr.translation() << std::endl;
    }

    return 0;
}
#include <iostream>

#include <vo/initializer.hpp>
#include <vo/pose_estimator.hpp>
#include <vo/matcher.hpp>
#include <vo/camera.hpp>
#include <vo/frame.hpp>
#include <vo/feature.hpp>
#include <vo/map_point.hpp>

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

    // Sophus::SE3d gt = gts[i] * gts[0].inverse();
    // std::cout << "GT R:\n" << gt.rotationMatrix() << std::endl;
    // std::cout << "GT t:\n" << gt.translation().normalized().transpose() << std::endl;
    std::cout << "EST R:\n" << f->t_cw.rotationMatrix() << std::endl;
    std::cout << "EST t:\n" << f->t_cw.translation().normalized().transpose() << std::endl;

    size_t test_img_idx = i + 10;
    vslam::frame_ptr test_frame = 
        utils::mk_vptr<vslam::frame>(cam, images[test_img_idx].second, images[test_img_idx].first);

    cv::imshow("first_frame", draw_feats(first_frame));
    cv::imshow("f", draw_feats(f));
    cv::imshow("test_frame", test_frame->image());
    cv::waitKey();

#ifdef _TEST_MATCH_DIRECT_
#ifdef _TEST_A_SINGLE_SAMPLE_
    {
        cv::Mat f0, f2;
        cv::cvtColor(first_frame->image(), f0, cv::COLOR_GRAY2BGR);
        cv::circle(f0, cv::Point2i{ 421, 410 }, 2, 0);
        cv::imshow("f0", f0);
        cv::cvtColor(test_frame->image(), f2, cv::COLOR_GRAY2BGR);
        cv::circle(f2, cv::Point2f{ 459.6146236935906f, 375.44258665888685f }, 2, 0);
        cv::imshow("f2", f2);
        cv::waitKey();

        Eigen::Vector2d uv_ref = { 421, 410 };
        Eigen::Vector2d uv_cur = { 459.6146236935906, 375.44258665888685 };

        vslam::patch_t patch_ref;
        uint8_t* ptr = patch_ref.data;

        double center_x = uv_ref.x();
        double center_y = uv_ref.y();

        std::cout << "patch:" << std::endl;

        for (auto r = 0; r < 10; ++r) {
            for (auto c = 0; c < 10; ++c) {
                double x = center_x - 5. + c;
                double y = center_y - 5. + r;
                double v = utils::bilinear_interoplate<uint8_t>(first_frame->image(), x, y);
                assert(v < 255.);
                *ptr = (uint8_t) v;
                std::cout << int(*ptr) << " ";
                ++ptr;
            }
            std::cout << std::endl;
        }

        std::cout << "Align success: " 
                  << vslam::alignment::align2d(test_frame->image(), patch_ref, 10, uv_cur) 
                  << std::endl;
        cv::circle(f2, cv::Point2f{ uv_cur.x(), uv_cur.y() }, 2, { 0, 0, 255 });
        cv::imshow("f2-align", f2);
        cv::waitKey();
    }
#endif

    {
        vslam::twoframe_estimator tf_est(
            config::max_opt_iterations, 4, 0, vslam::twoframe_estimator::LK_ICIA
        );
        Sophus::SE3d t_21;
        tf_est.estimate(f, test_frame, t_21);
        test_frame->set_pose(t_21 * f->t_cw);

        cv::Mat f2_clone;
        cv::cvtColor(test_frame->image(), f2_clone, cv::COLOR_GRAY2BGR);

        std::vector<std::pair<vslam::map_point_ptr, Eigen::Vector2d>> candidates;

        for (auto& feat : f->features) {
            if (feat->describe_nothing()) { continue; }
            const auto& mp = feat->map_point_describing;
            Eigen::Vector3d xyz_f = f->t_cw * mp->position;
            Eigen::Vector3d xyz_2 = t_21 * xyz_f;
            Eigen::Vector2d uv_2  = test_frame->camera->cam2pixel(xyz_2);
            cv::circle(f2_clone, cv::Point2f{ uv_2.x(), uv_2.y() }, 2, { 0, 0, 255 });
            candidates.emplace_back(mp, uv_2);
        }

        cv::imshow("test_frame_ICIA", f2_clone);
        cv::waitKey();

        vslam::patch_matcher matcher(true);

        size_t count_matches = 0;

        for (auto& each : candidates) {
            vslam::feature_ptr new_feat;
            bool success = matcher.match_direct(each.first, test_frame, each.second, new_feat);
            std::cout << "Succeed to matched: " << success << std::endl;
            if (success && new_feat) { new_feat->use(); ++count_matches; } 
        }

        std::cout << "Total candidates: " << candidates.size() << std::endl;
        std::cout << "Matches: " << count_matches << std::endl;
        
        cv::imshow("After match", draw_feats(test_frame));
        cv::waitKey();
    }

#ifdef _TEST_MATCHER_USING_LK_FCFA_
    {
        vslam::twoframe_estimator tf_est(
            config::max_opt_iterations, 4, 0, vslam::twoframe_estimator::LK_FCFA
        );
        Sophus::SE3d t_21;
        tf_est.estimate(f, test_frame, t_21);

        cv::Mat f2_clone;
        cv::cvtColor(test_frame->image(), f2_clone, cv::COLOR_GRAY2BGR);

        for (auto& feat : f->features) {
            if (feat->describe_nothing()) { continue; }
            const auto& mp = feat->map_point_describing;
            Eigen::Vector3d xyz_f = f->t_cw * mp->position;
            Eigen::Vector3d xyz_2 = t_21 * xyz_f;
            Eigen::Vector2d uv_2  = test_frame->camera->cam2pixel(xyz_2);
            cv::circle(f2_clone, cv::Point2f{ uv_2.x(), uv_2.y() }, 2, { 0, 0, 255 });
        }

        cv::imshow("test_frame_FCFA", f2_clone);
        cv::waitKey();
    }
#endif   
#endif

#define _TEST_MATCH_EPIPOLAR_SEARCH_ 1
#ifdef _TEST_MATCH_EPIPOLAR_SEARCH_

    {
        vslam::twoframe_estimator tf_est(
            config::max_opt_iterations, 4, 0, vslam::twoframe_estimator::LK_ICIA
        );
        Sophus::SE3d t_21;
        tf_est.estimate(f, test_frame, t_21);
        test_frame->set_pose(t_21 * f->t_cw);

        cv::Mat f2_clone;
        cv::cvtColor(test_frame->image(), f2_clone, cv::COLOR_GRAY2BGR);

        size_t count_reproj_success = 0;

        for (auto& feat : f->features) {
            if (feat->describe_nothing()) { continue; }
            const auto& mp = feat->map_point_describing;
            Eigen::Vector3d xyz_2 = test_frame->t_cw * mp->position;
            Eigen::Vector2d uv_2  = test_frame->camera->cam2pixel(xyz_2);
            if (!test_frame->visible(uv_2)) { continue; }
            ++count_reproj_success;
            cv::circle(f2_clone, cv::Point2f{ uv_2.x(), uv_2.y() }, 2, { 0, 0, 255 });
        }

        std::cout << "Total features: " << f->features.size() << std::endl;
        std::cout << "Reproj: " << count_reproj_success << std::endl;;
        cv::imshow("test_frame_LK_ICIA", f2_clone);
        cv::waitKey();

        vslam::patch_matcher matcher(false);

        size_t count_epi_search = 0;
        double d_mean_err = 0;

        for (auto& feat : first_frame->features) {
            if (feat->describe_nothing()) { continue; }
            const auto& mp = feat->map_point_describing;
            double d_gt = (mp->position - first_frame->cam_center()).norm();

            double d_est = d_gt * 1.2;
            double d_min = d_gt * 0.8;
            double d_max = d_gt * 2.0;

            bool success = 
                matcher.match_epipolar_search(first_frame, test_frame, feat, d_min, d_max, d_est);
            count_epi_search += success;
            std::cout << "Match Epipolar: " << success << std::endl;
            if (success) {
                std::cout << "depth GT: " << d_gt  << std::endl;
                std::cout << "depth EST: " << d_est << std::endl;

                d_mean_err += (d_est - d_gt) * (d_est - d_gt);

                Eigen::Vector3d xyz_est = mp->position / d_gt * d_est;
                Eigen::Vector3d xyz_est_f2 = test_frame->t_cw * xyz_est;
                Eigen::Vector2d uv_est = test_frame->camera->cam2pixel(xyz_est_f2);
                cv::circle(f2_clone, cv::Point2f{ uv_est.x(), uv_est.y() }, 2, { 0, 255, 255 });
            }
        }
        std::cout << "Total features: " << first_frame->features.size() << std::endl;
        std::cout << "Epipolar search matches: " << count_epi_search << std::endl;
        std::cout << "Mean error: " << std::sqrt(d_mean_err / count_epi_search) << std::endl;
        cv::imshow("epipolar search f2", f2_clone);
        cv::waitKey();
    }

#endif

    return 0;
}

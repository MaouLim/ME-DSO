#include <iostream>
#include <fstream>
#include <sstream>

#include <vo/initializer.hpp>
#include <vo/camera.hpp>
#include <vo/frame.hpp>
#include <vo/feature.hpp>
#include <vo/map_point.hpp>

#include <pcl/point_types.h>
#include <pcl/pcl_base.h>
#include <pcl/visualization/cloud_viewer.h>

const std::string dataset_dir = "data/fr1_xyz/";
const std::string association_file = dataset_dir + "rgb-gt.txt";
const size_t      max_images = 500;

void draw_level0_features(const vslam::frame_ptr& frame) {
    cv::Mat img = frame->pyramid[0].clone();
    for (auto& each : frame->features) {
        if (0 != each->level) { continue; }
        cv::circle(img, cv::Point2i(each->uv.x(), each->uv.y()), 3, 255, 2);
    }

    cv::imshow(std::to_string(frame->id), img);
    cv::waitKey();
}

void extract_keypoints(
    const vslam::frame_ptr&    frame, 
    std::vector<cv::KeyPoint>& kpts, 
    cv::Mat&                   descriptor
) {
    kpts.clear(); 
    cv::Mat img = frame->pyramid[0].clone();
    cv::Ptr<cv::Feature2D> det = cv::ORB::create(600);
    det->detectAndCompute(img, cv::Mat(), kpts, descriptor);
}

void calc_pose(
    const std::vector<cv::Point2f>& uv0, 
    const std::vector<cv::Point2f>& uv1, 
    const cv::Mat&                  cam_mat
) {
    cv::Mat essential = cv::findEssentialMat(uv0, uv1, cam_mat);
    cv::Mat R, cv_t;
    auto n_inliers = cv::recoverPose(essential, uv0, uv1, cam_mat, R, cv_t);
    Eigen::Vector3d t;
    t << cv_t.at<double>(0), cv_t.at<double>(1), cv_t.at<double>(2);
    std::cout << "n_inliers: " << n_inliers << std::endl;
    std::cout << R << std::endl;
    std::cout << t.normalized().transpose() << std::endl;
}

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

#ifdef _TEST_SINGLE_
    vslam::initializer init;

    vslam::frame_ptr first_frame = 
        utils::mk_vptr<vslam::frame>(cam, images.front().second, images.front().first);
    cv::imshow("ref", images.front().second);

    std::cout << "set_first: " << init.add_frame(first_frame) << std::endl;;
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

    /**
     * @test OpenCV findEssentialMat, recoverPose
     *       almostly same as the initializer
     */ {
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

            ++itr_cur; ++itr_ref;
        }
        std::cout << "Size of matches: " << uv0.size() << std::endl;
        std::cout << "EST by OpenCV using features provided by initializer" << std::endl;
        calc_pose(uv0, uv1, cam->cv_mat());
    }

    /**
     * @test OpenCV feature-based method
     * 
     */ {
        
        std::vector<cv::KeyPoint> kpts0, kpts1;
        cv::Mat desc0, desc1;
        extract_keypoints(first_frame, kpts0, desc0);
        extract_keypoints(f, kpts1, desc1);

        std::vector<cv::DMatch> matches;
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        matcher.match(desc0, desc1, matches);

        double min_dis = std::numeric_limits<double>::max();
        double max_dis = 0.0;

        for (auto& match : matches) {
            if (match.distance < min_dis) { min_dis = match.distance; }
            if (max_dis < match.distance) { max_dis = match.distance; }
        }

        /**
         * @test 2d
         */ {
            std::vector<cv::Point2f> uv0;
            std::vector<cv::Point2f> uv1;

            double thresh = 4 * min_dis;// + 0.5 * (max_dis - min_dis);
            for (auto& match : matches) {
                if (thresh < match.distance) { continue; }
                uv0.emplace_back(kpts0[match.queryIdx].pt);
                uv1.emplace_back(kpts1[match.trainIdx].pt);
            }

            std::cout << "Size of matches: " << uv0.size() << std::endl;
            calc_pose(uv0, uv1, cam->cv_mat());
        }

        /**
         * @test PNP
         */ {
            //std::vector<cv::Point2f> uv0;
            std::vector<cv::Point3f> xyz0;
            std::vector<cv::Point2f> uv1;

            cv::Mat depth = find_associate_depth(images[0].first);
            
            double thresh = 5 * min_dis;// + 0.5 * (max_dis - min_dis);
            for (auto& match : matches) {
                if (thresh < match.distance) { continue; }

                cv::Point2f p = kpts0[match.queryIdx].pt;
                uint16_t d = depth.at<uint16_t>(p);
                if (d <= 0) { continue; }
                float z = float(d) / 5000.f;
                Eigen::Vector3d xyz = cam->pixel2cam(Eigen::Vector2d(p.x, p.y), z);
                xyz0.emplace_back(xyz[0], xyz[1], xyz[2]);
                uv1.emplace_back(kpts1[match.trainIdx].pt);
            }
            
            std::cout << "Size of matches: " << uv1.size() << std::endl;
            cv::Mat rvec, tvec;
            cv::solvePnP(xyz0, uv1, cam->cv_mat(), cv::Mat(), rvec, tvec);

            Sophus::SO3d r = Sophus::SO3d::exp({ rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2) });
            Eigen::Vector3d t;
            t << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);
            //std::cout << "n_inliers: " << n_inliers << std::endl;
            std::cout << r.matrix() << std::endl;
            std::cout << t.normalized().transpose() << std::endl;
        }
    }
#endif
#ifdef _TEST_CONTINUOUS_
    /**
     * @test continuous test
     * 
     */ {

        cv::viz::Viz3d visualizer("VO");
	    cv::viz::WCoordinateSystem world_sys(1.0), camera_sys(0.5);
	    cv::Vec3d camera_pos(0., -1., -1.), camera_center(0., 0., 0.), camera_yaxis(0., 1., 0.);
	    cv::Affine3d cam_pose = cv::viz::makeCameraPose(camera_pos, camera_center, camera_yaxis);
	    visualizer.setViewerPose(cam_pose);

	    world_sys.setRenderingProperty(cv::viz::LINE_WIDTH, 2.);
	    camera_sys.setRenderingProperty(cv::viz::LINE_WIDTH, 2.);
	    visualizer.showWidget("world", world_sys);
	    visualizer.showWidget("camera", camera_sys);

        vslam::initializer init;

        for (auto i = 0; i < max_images - 1; ++i) {
            
            init.reset();

            vslam::frame_ptr f0 = 
                utils::mk_vptr<vslam::frame>(cam, images[i].second, images[i].first);
            vslam::frame_ptr f1 = 
                utils::mk_vptr<vslam::frame>(cam, images[i + 1].second, images[i + 1].first);

            // if (vslam::initializer::SUCCESS != init.set_first(f0)) { 
            //     std::cout << "failed set first" << std::endl; 
            //     continue; 
            // }
            // auto ret = init.add_frame(f1);
            // if (vslam::initializer::SUCCESS != ret) { 
            //     std::cout << "failed add_frame: " << ret << std::endl; 
            //     continue; 
            // }

            std::cout << f1->t_cw.matrix3x4() << std::endl;

		    auto t_cam2world = gts[0] * gts[i].inverse(); //f1->t_wc;
		    auto r = t_cam2world.rotationMatrix();
		    auto t = t_cam2world.translation();

		    cv::Affine3d::Mat3 rot(
		    	r(0, 0), r(0, 1), r(0, 2),
		    	r(1, 0), r(1, 1), r(1, 2),
		    	r(2, 0), r(2, 1), r(2, 2)
		    );
		    cv::Affine3d::Vec3 trans(t[0], t[1], t[2]);
		    cv::Affine3d transform(rot, trans);

		    cv::imshow("color", f1->pyramid[0]);
		    cv::waitKey(100);
		    visualizer.setWidgetPose("camera", transform);
		    visualizer.spinOnce(1);
	    }
    }
#endif

    {
        vslam::initializer_v2 init;
        auto start = 80;
        auto idx = start;
        for (;idx < max_images; ++idx) {
            
            vslam::frame_ptr f0 = 
                utils::mk_vptr<vslam::frame>(cam, images[idx].second, images[idx].first);

            auto ret = init.add_frame(f0);
            std::cout << ret << std::endl;
            if (ret == vslam::initializer_v2::SUCCESS) {  break; }
	    }
#define _TEST_PCL_ 1
#ifdef _TEST_PCL_
        const auto& pts = init.final_xyzs_f1;
        const auto& uvs = init.final_uvs_f1;


        typedef pcl::PointXYZRGB                point_t;
        typedef pcl::PointCloud<point_t>        point_cloud_t;
        typedef pcl::visualization::CloudViewer viewer_t;

        point_cloud_t::Ptr cloud(new point_cloud_t());
        for (auto v : pts) {
            point_t p_xyzrgb;
            p_xyzrgb.x = v[0];
            p_xyzrgb.y = v[1];
            p_xyzrgb.z = v[2];

            p_xyzrgb.r = 255;
            p_xyzrgb.g = 0;
            p_xyzrgb.b = 0;

            cloud->push_back(p_xyzrgb);
        }

        cv::Mat depth = find_associate_depth(init._frames[1]->timestamp);
        assert(depth.data);
        for (auto uv : uvs) {
            uint16_t d = depth.at<uint16_t>(uv.y(), uv.x());
            if (d <= 0) { continue; }
            double z = double(d) / 5000.0;
            Eigen::Vector3d xyz = cam->pixel2cam(uv, z);

            point_t p_xyzrgb;
            p_xyzrgb.x = xyz[0];
            p_xyzrgb.y = xyz[1];
            p_xyzrgb.z = xyz[2];

            p_xyzrgb.r = 255;
            p_xyzrgb.g = 255;
            p_xyzrgb.b = 255;
            cloud->push_back(p_xyzrgb);
        }

        viewer_t viewer("pts");
        viewer.showCloud(cloud);
        while (!viewer.wasStopped()) { }
#endif

#ifdef _TEST_CV_VIZ_
        cv::viz::Viz3d visualizer("VO");
	    cv::viz::WCoordinateSystem world_sys(1.0), camera_sys(0.2), gt_cam_sys(0.4);
	    cv::Vec3d camera_pos(0., -1., -1.), camera_center(0., 0., 0.), camera_yaxis(0., 1., 0.);
	    cv::Affine3d cam_pose = cv::viz::makeCameraPose(camera_pos, camera_center, camera_yaxis);
        cv::Affine3d gt_cam_pose = cv::viz::makeCameraPose(camera_pos, camera_center, camera_yaxis);
	    visualizer.setViewerPose(cam_pose);
        visualizer.setViewerPose(gt_cam_pose);

	    //world_sys.setRenderingProperty(cv::viz::LINE_WIDTH, 1.);
	    camera_sys.setRenderingProperty(cv::viz::LINE_WIDTH, 1.);
        gt_cam_sys.setRenderingProperty(cv::viz::LINE_WIDTH, 2.);
	    //visualizer.showWidget("world", world_sys);
	    visualizer.showWidget("camera", camera_sys);
        visualizer.showWidget("gt", gt_cam_sys);

        int step = (idx - start) /  vslam::initializer_v2::min_init_frames;

        for (size_t i = 0, j = 0; i < vslam::initializer_v2::min_init_frames; ++i, j += step) {
            {
                auto t_cam2world = init._poses_opt[i].inverse();//f1->t_wc;
		        auto r = t_cam2world.rotationMatrix();
		        auto t = t_cam2world.translation() * 2.0;

		        cv::Affine3d::Mat3 rot(
		        	r(0, 0), r(0, 1), r(0, 2),
		        	r(1, 0), r(1, 1), r(1, 2),
		        	r(2, 0), r(2, 1), r(2, 2)
		        );
		        cv::Affine3d::Vec3 trans(t[0], t[1], t[2]);
		        cv::Affine3d transform(rot, trans);
		        visualizer.setWidgetPose("camera", transform);
            }
            {
                auto t_cam2world = gts[start] * gts[start + j].inverse();
		        auto r = t_cam2world.rotationMatrix();
		        auto t = t_cam2world.translation();

		        cv::Affine3d::Mat3 rot(
		        	r(0, 0), r(0, 1), r(0, 2),
		        	r(1, 0), r(1, 1), r(1, 2),
		        	r(2, 0), r(2, 1), r(2, 2)
		        );
		        cv::Affine3d::Vec3 trans(t[0], t[1], t[2]);
		        cv::Affine3d transform(rot, trans);
		        visualizer.setWidgetPose("gt", transform);
            }

            cv::imshow("color", init._frames[i]->pyramid[0]);
		    cv::waitKey();
		    visualizer.spinOnce(1);
        }
#endif
    }

    return 0;
}
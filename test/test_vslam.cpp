#include <iostream>
#include <opencv2/opencv.hpp>

#include <vslam.hpp>

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
    size_t count = 0;
    for (auto& feat : f->features) {
        if (!feat->describe_nothing()) { 
            ++count; 
            cv::circle(f_img, cv::Point2f{ feat->uv.x(), feat->uv.y() }, 2, color[feat->level]);
        }
        else {
            cv::circle(f_img, cv::Point2f{ feat->uv.x(), feat->uv.y() }, 2, { 255, 155, 0 });
        }
    }
    std::cout << "Num of features: " << f->n_features << std::endl;
    std::cout << "Num of desc features: " << count << std::endl;

    return f_img;
}

std::vector<std::pair<double, cv::Mat>> images;
std::vector<Sophus::SE3d>               gts;

int main(int argc, char** argv) {

    get_imgs_and_gts(images, gts);

    vslam::camera_ptr cam = utils::mk_vptr<vslam::pinhole_camera>(
        config::height, config::width, 517.3, 516.5, 325.1, 249.7
    );

    vslam::system slam_sys(cam);
    slam_sys.start();

    cv::viz::Viz3d visualizer("VO");
	cv::viz::WCoordinateSystem world_sys(1.0), camera_sys(0.2);
	cv::Vec3d camera_pos(0., -1., -1.), camera_center(0., 0., 0.), camera_yaxis(0., 1., 0.);
	cv::Affine3d cam_pose = cv::viz::makeCameraPose(camera_pos, camera_center, camera_yaxis);
	visualizer.setViewerPose(cam_pose);

	camera_sys.setRenderingProperty(cv::viz::LINE_WIDTH, 1.);
	visualizer.showWidget("camera", camera_sys);

    for (auto& data : images) {
        slam_sys.process_image(data.second, data.first);
        const auto& last_frame = slam_sys.last_frame();
        cv::Mat f_vis = draw_feats(last_frame);
        // todo visualize the pose
        // try to accelerate the depth_filter converging 

        auto t_cam2world = last_frame->t_wc;//f1->t_wc;
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
        visualizer.spinOnce(1);

        cv::imshow("VIS", f_vis);
        cv::waitKey(60);
    }
    slam_sys.shutdown();
    return 0;
}
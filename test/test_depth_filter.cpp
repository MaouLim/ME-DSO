#include <iostream>

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
    for (auto& feat : f->features) {
        cv::circle(f_img, cv::Point2f{ feat->uv.x(), feat->uv.y() }, 2, color[feat->level]);
    }

    return f_img;
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

std::vector<std::pair<double, cv::Mat>> images;
std::vector<Sophus::SE3d>               gts;

void df_callback(const vslam::map_point_ptr& mp, double cov2) {
    std::cout << "Seed upgrading..." << std::endl;
    std::cout << "Position: " << mp->position.transpose() << std::endl;
    std::cout << "Cov:      " << cov2 << std::endl;
}

int main(int argc, char** argv) {

    get_imgs_and_gts(images, gts);

    vslam::camera_ptr cam = utils::mk_vptr<vslam::pinhole_camera>(
        config::height, config::width, 517.3, 516.5, 325.1, 249.7
    );

    vslam::depth_filter filter(&df_callback);
    filter.start();

    std::vector<vslam::frame_ptr> all_frames;
    vslam::twoframe_estimator tf_est;
    vslam::frame_ptr f;
    vslam::frame_ptr last;

    int start = 0;

    // initialize cheating.
    f = utils::mk_vptr<vslam::frame>(cam, images[start].second, images[start].first);
    cv::Mat depth_mat = find_associate_depth(images[start].first);
    assert(depth_mat.data);

    size_t max_feats = (config::width / config::cell_sz) * 
                       (config::height / config::cell_sz);
    cv::Ptr<cv::GFTTDetector> det = 
        cv::GFTTDetector::create(max_feats, 0.05, config::cell_sz / 2.);
    std::vector<cv::KeyPoint> kpts;
    det->detect(f->image(), kpts);

    for (auto& each : kpts) {
        uint16_t d = depth_mat.at<uint16_t>(each.pt);
        if (d <= 0) { continue; }
        double z = double(d) / 5000.0;
        Eigen::Vector2d uv(each.pt.x, each.pt.y);
        vslam::feature_ptr feat = 
            utils::mk_vptr<vslam::feature>(f, uv, 0);
        vslam::map_point_ptr mp = 
            utils::mk_vptr<vslam::map_point>(cam->pixel2cam(uv, z));
        feat->set_describing(mp);
        feat->use();
    }
    f->as_key_frame();
    filter.commit(f);
    all_frames.push_back(f);

    cv::imshow("init", draw_feats(f));
    cv::waitKey();

    int key_frame_step = 100;

    for (size_t i = start + 1; i < max_images; ++i) {
        f = utils::mk_vptr<vslam::frame>(cam, images[i].second, images[i].first);
        last = all_frames.back();

        Sophus::SE3d f_pose;
        tf_est.estimate(last, f, f_pose);
        f->set_pose(f_pose * last->t_cw);

        if (0 == i % key_frame_step) { 
            f->as_key_frame();
        }
        filter.commit(f);
        all_frames.push_back(f);

        cv::imshow("frame", f->image());
        cv::waitKey();
    }

    return 0;
}
/**
 * Created by Maou Lim on 2019/12/26.
 * @note pose estimation using sparse direct method
 */

#include <chrono>

#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <sophus_templ/se3.hpp>

namespace chrono = std::chrono;

typedef Eigen::Vector2d             vec2d;
typedef Eigen::Vector3d             vec3d;
typedef Eigen::Matrix3d             mat3d;
typedef Eigen::Matrix<double, 6, 6> mat6d;
typedef Eigen::Matrix<double, 2, 6> mat2x6d;
typedef Eigen::Matrix<double, 6, 1> vec6d;

inline double _val(const cv::Mat& img, double x, double y) {
	int x_ = std::floor(x);
	int y_ = std::floor(y);

	if (x_ < 0) { x_ = 0; }
	if (x_ >= img.cols) { x_ = img.cols - 1; }
	if (y_ < 0) { y_ = 0; }
	if (y_ >= img.rows) { y_ = img.rows - 1; }

	double v00 = (double) img.at<uint8_t>(y_, x_);
	double v01 = (double) img.at<uint8_t>(y_, x_ + 1);
	double v10 = (double) img.at<uint8_t>(y_ + 1, x_);
	double v11 = (double) img.at<uint8_t>(y_ + 1, x_ + 1);

	return v11 * (x - x_) * (y - y_) +
	       v10 * (x_ + 1 - x) * (y - y_) +
	       v01 * (x - x_) * (y_ + 1 - y) +
	       v00 * (x_ + 1 - x) * (y_ + 1 - y);
}

inline vec2d _grad(const cv::Mat& img, double x, double y) {
	int x_ = std::floor(x);
	int y_ = std::floor(y);

	if (x_ < 0) { x_ = 0; }
	if (x_ >= img.cols) { x_ = img.cols - 1; }
	if (y_ < 0) { y_ = 0; }
	if (y_ >= img.rows) { y_ = img.rows - 1; }

	double v00 = (double) img.at<uint8_t>(y_, x_);
	double v01 = (double) img.at<uint8_t>(y_, x_ + 1);
	double v10 = (double) img.at<uint8_t>(y_ + 1, x_);
	double v11 = (double) img.at<uint8_t>(y_ + 1, x_ + 1);

	return vec2d(
		(y_ + 1 - y) * (v01 - v00) + (y - y_) * (v11 - v10),
		(x_ + 1 - x) * (v10 - v00) + (x - x_) * (v11 - v01)
	);
}

inline mat2x6d jaccobian_dxy1deps(const Eigen::Vector3d& p_c) {
	double x_c = p_c[0], y_c = p_c[1], z_c = p_c[2];
	double zinv = 1. / z_c;
	double zinv2 = zinv * zinv;
	double x2 = x_c * x_c, y2 = y_c * y_c, xy = x_c * y_c;
	mat2x6d j;
	j << zinv,   0., -x_c * zinv2,      -xy * zinv2, 1. + x2 * zinv2, -y_c * zinv,
		0., zinv, -y_c * zinv2, -1. - y2 * zinv2,      xy * zinv2,  x_c * zinv;
	return j;
}

void direct_method_single_level(
	const cv::Mat&            prev,
	const cv::Mat&            next,
	const std::vector<vec3d>& points,
	const mat3d&              cam_mat,
	std::vector<bool>&        status,
//	mat3d&                    rotation,
//	vec3d&                    translation,
Sophus::SE3d&    T,
	size_t                    win_sz = 3
) {
	assert(1 == win_sz % 2);

	const int half_w = win_sz / 2;
	const int max_iterations = 10;
	const double fx = cam_mat(0, 0);
	const double fy = cam_mat(1, 1);

	double prev_loss = 0.;
	auto iter = 0;

	while (iter < max_iterations) {

		mat6d h = mat6d::Zero();
		vec6d g = vec6d::Zero();
		double loss = 0.;

		for (auto i = 0; i < points.size(); ++i) {
			if (!status[i]) { continue; }

			const vec3d& p = points[i];

			vec3d p1_homo = cam_mat * p;
			vec2d p1 = vec2d(p1_homo[0] / p1_homo[2], p1_homo[1] / p1_homo[2]);
			//vec3d p_trans = rotation * p + translation;
			vec3d p_trans = T * p;
			vec3d p2_homo = cam_mat * p_trans;
			vec2d p2 = vec2d(p2_homo[0] / p2_homo[2], p2_homo[1] / p2_homo[2]);

			status[i] = !(p2.x() < 0 || next.cols <= p2.x() ||
			              p2.y() < 0 || next.rows <= p2.y());

			double x = p_trans.x(), y = p_trans.y(), z = p_trans.z();
			double x2 = x * x, y2 = y * y, z2 = z * z;

			/**
			 * @note delta x is Lie algebra
			 */
			mat2x6d dudx = cam_mat.block<2, 2>(0, 0) * jaccobian_dxy1deps(p_trans);
//			dudx <<
//			     fx / z,    0.f,    -fx * x / z2,   -fx * x * y / z2,   fx + fx * x2 / z2,  -fx * y / z,
//				    0.f, fy / z,    -fy * y / z2, -fy - fy * y2 / z2,     fy * x * y / z2,   fy * x / z;


			for (auto dx = -half_w; dx <= half_w; ++dx) {
				for (auto dy = -half_w; dy <= half_w; ++dy) {
					vec2d delta = vec2d(dx, dy);
					vec2d q1 = p1 + delta;
					vec2d q2 = p2 + delta;

					double err = _val(prev, q1.x(), q1.y()) - _val(next, q2.x(), q2.y());
					loss += 0.5 * err * err;
					/**
					 * @var i: image
					 *      u: pixel
					 */
					vec2d didu = _grad(next, q2.x(), q2.y());
					double gx = 0.5 * (_val(next, q2.x() + 1, q2.y()) - _val(next, q2.x() - 1, q2.y()));
					double gy = 0.5 * (_val(next, q2.x(), q2.y() + 1) - _val(next, q2.x(), q2.y() - 1));
					didu << gx, gy;
					/**
					 * @note linear part of loss function (扰动模型).
					 */
					vec6d jacobian = -didu.transpose() * dudx;
//					std::cout << "point:\n" << p.transpose() << std::endl;
//					std::cout << "j:\n" << jacobian.transpose() << std::endl;
					h += jacobian * jacobian.transpose();
					g += jacobian * -err;
				}
			}
		}

		//std::cout << "h:\n" << h << std::endl;
		//std::cout << "g:\n" << g.transpose() << std::endl;

		vec6d dx = h.ldlt().solve(g);

		if (std::isnan(dx[0])) { break; }
		if (0 != iter && prev_loss < loss) { std::cerr << "loss inc: " << iter << std::endl; break; }
		if (dx.norm() < 1e-8) { std::cerr << "converge: " << iter << std::endl; break; }

		// update
//		mat3d delta_r = rodriguez(dx.tail<3>());
//		vec3d delta_t = jacobian(dx);
//
//		rotation = delta_r * rotation;
//		translation = delta_r * translation + delta_t;
		T = Sophus::SE3d::exp(dx) * T;
		prev_loss = loss;
		++iter;
	}
//	rotation = T.rotationMatrix();
//	translation = T.translation();
}

void direct_method_pyramid(
	const cv::Mat&            prev,
	const cv::Mat&            next,
	const std::vector<vec3d>& points,
	const mat3d&              cam_mat,
	std::vector<bool>&        status,
//	mat3d&                    rotation,
//	vec3d&                    translation,
Sophus::SE3d&               T,
	size_t                    win_sz = 3,
	size_t                    level  = 4
) {
	const double pyr_scale = 0.5;
	std::vector<mat3d> scales(level, mat3d::Identity());
	for (auto i = 1; i < level; ++i) {
		scales[i](0, 0) = scales[i - 1](0, 0) * pyr_scale;
		scales[i](1, 1) = scales[i - 1](1, 1) * pyr_scale;
	}

	status.resize(points.size());
	/**
	 * @note create image pyramid
	 */
	std::vector<cv::Mat> pyr_prev, pyr_next;
	std::vector<mat3d> pyr_cam_mat;
	pyr_prev.reserve(level) ; pyr_next.reserve(level) ; pyr_cam_mat.reserve(level);
	pyr_prev.push_back(prev); pyr_next.push_back(next); pyr_cam_mat.push_back(cam_mat);
	for (auto i = 1; i < level; ++i) {
	 	cv::Mat tmp_prev, tmp_next;
	 	cv::resize(pyr_prev[i - 1], tmp_prev, cv::Size(), pyr_scale, pyr_scale);
	 	cv::resize(pyr_next[i - 1], tmp_next, cv::Size(), pyr_scale, pyr_scale);
	 	pyr_prev.push_back(tmp_prev);
	 	pyr_next.push_back(tmp_next);
	}

	for (size_t i = level; 0 < i; --i) {
		size_t current_level = i - 1;
		status.assign(status.size(), true);
		direct_method_single_level(
		/* readonly */	 pyr_prev[current_level], pyr_next[current_level], points, scales[current_level] * cam_mat,
		/* read&write */ status, /*rotation, translation*/T, win_sz
		);
	}
}

void _vis_direct_method(
	const cv::Mat&                  prev,
	const cv::Mat&                  next,
	const std::vector<cv::Point2f>& pts_prev,
	const std::vector<cv::Point2f>& pts_next,
	const std::vector<bool>&        status
) {
	assert(pts_next.size() == pts_prev.size() &&
	       pts_next.size() == status.size());

	const cv::Scalar_<int> WHITE(255, 255, 255);
	const cv::Scalar_<int> BLACK(0, 0, 0);
	const int n_points = pts_next.size();

	cv::Mat prev_clone = prev.clone();
	cv::Mat next_clone = next.clone();

	for (auto i = 0; i < n_points; ++i) {
		if (status[i]) {
			cv::circle(prev_clone, pts_prev[i], 3, WHITE);
			cv::circle(next_clone, pts_next[i], 3, WHITE);
		}
		else {
			cv::circle(prev_clone, pts_prev[i], 3, BLACK);
		}
	}
	cv::Mat concat;
	cv::hconcat(prev_clone, next_clone, concat);
	cv::cvtColor(concat, concat, cv::COLOR_GRAY2RGB);
	for (auto i = 0; i < n_points; ++i) {
		if (!status[i]) { continue; }
		cv::line(concat, pts_prev[i], pts_next[i] + cv::Point2f(prev.cols, 0.f), WHITE);
	}

	cv::imshow("visualization", concat);
	cv::waitKey();
}

int main(int argc, char** argv) {

	const int    n_images    = 500;
	const double depth_scale = 1. / 5000;

	const std::string depth_path = "pose3d/1_depth.png";//"tum_dataset/000_depth.png";
	mat3d cam_mat;
	cam_mat <<
//		525.0,   0.0, 319.5,
//		  0.0, 525.0, 239.5,
//		  0.0,   0.0,   1.0;
            517.3,   0.0, 325.1,
		0.0, 516.5, 249.7,
		0.0,   0.0,   1.0;
	mat3d cam_mat_inv = cam_mat.inverse();

	std::vector<cv::Mat> images(n_images);
	images[0] = cv::imread("pose3d/1.png", cv::IMREAD_GRAYSCALE);
	images[1] = cv::imread("pose3d/2.png", cv::IMREAD_GRAYSCALE);
//	for (auto i = 0; i < n_images; ++i) {
//		char tmp[16];
//		sprintf(tmp, "tum_dataset/%.3d.png", i);
//		images[i] = cv::imread(std::string(tmp), cv::IMREAD_GRAYSCALE);
//	}

	cv::Mat origin_depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
	const cv::Mat& origin = images.front();

	std::vector<cv::KeyPoint> key_points;
	cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(300, 0.01, 20);
	detector->detect(origin, key_points);

	std::vector<vec3d>       points3d;
	std::vector<cv::Point2f> points2d;
	points3d.reserve(key_points.size());
	points2d.reserve(key_points.size());

	for (auto& each : key_points) {
		uint16_t d = origin_depth.at<uint16_t>(each.pt.y, each.pt.x);
		if (d <= 0) { continue; }
		double z = (double) d * depth_scale;
		vec3d p = z * cam_mat_inv * vec3d(each.pt.x, each.pt.y, 1.);
		points3d.push_back(p);
		points2d.push_back(each.pt);
	}

	const int n_points = points2d.size();

	/**
	 * @brief 第一帧与后续帧的光流追踪
	 */ {
//	 	std::vector<bool> status;
//		std::vector<cv::Point2f> points_prev(n_points);
//		std::vector<cv::Point2f> points_next(n_points);
//
//		mat3d r = mat3d::Identity();
//		vec3d t = vec3d::Zero();
//
//		for (auto i = 0; i < n_images; ++i) {
//			status.clear();
//			direct_method_pyramid(images.front(), images[i], points3d, cam_mat, status, r, t, 3);
//
//			std::cout << "rotation:\n" << r << std::endl;
//			std::cout << "translation:\n" << t << std::endl;
//
//			points_next.swap(points_prev);
//
//			for (auto j = 0; j < n_points; ++j) {
//				if (!status[j]) { continue; }
//				vec3d p_trans = r * points3d[j] + t;
//				vec3d p_proj = cam_mat * p_trans;
//				points3d[j] = p_trans;
//				points_next[j] = cv::Point2f(p_proj[0] / p_proj[2], p_proj[1] / p_proj[2]);
//			}
//
//			_vis_direct_method(images.front(), images[i], points2d, points_next, status);
//		}
	}

	/**
	 * @brief 相邻两帧的光流追踪
	 */ {
		std::vector<bool>        status;
		std::vector<cv::Point2f> points_prev(n_points);
		std::vector<cv::Point2f> points_next(points2d);

//		mat3d r = mat3d::Identity();
//		vec3d t = vec3d::Zero();
		Sophus::SE3d T;

		for (auto i = 0; i + 1 < n_images; ++i) {

			chrono::steady_clock::time_point start = chrono::steady_clock::now();
			direct_method_pyramid(images[i], images[i + 1], points3d, cam_mat, status, T, 3, 5);
			chrono::steady_clock::time_point end = chrono::steady_clock::now();

			std::cout << "rotation:\n" << T.rotationMatrix() << std::endl;
			std::cout << "translation:\n" << T.translation() << std::endl;
			std::cout << "time used(ms):" << chrono::duration_cast<chrono::milliseconds>(end - start).count()
			          << std::endl;

			points_next.swap(points_prev);

			for (auto j = 0; j < n_points; ++j) {
				if (!status[j]) { continue; }
				vec3d p_trans = T * points3d[j];
				vec3d p_proj = cam_mat * p_trans;
				points3d[j] = p_trans;
				points_next[j] = cv::Point2f(p_proj[0] / p_proj[2], p_proj[1] / p_proj[2]);
			}

			_vis_direct_method(images[i], images[i + 1], points_prev, points_next, status);
		}
	}

	return 0;
}















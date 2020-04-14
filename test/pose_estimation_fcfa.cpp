/*
 * Created by Maou Lim on 2020/4/14.
 */

#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <sophus_templ/se3.hpp>

namespace Eigen {
	using Vector6d  = Matrix<double, 6, 1>;
	using Matrix6d  = Matrix<double, 6, 6>;
	using Matrix26d = Matrix<double, 2, 6>;
	using Matrix62d = Matrix<double, 6, 2>;
}

inline Eigen::Vector2d _grad(const cv::Mat& img, double x, double y) {
	int x_ = std::floor(x);
	int y_ = std::floor(y);

	double v00 = (double) img.at<uint8_t>(y_, x_);
	double v01 = (double) img.at<uint8_t>(y_, x_ + 1);
	double v10 = (double) img.at<uint8_t>(y_ + 1, x_);
	double v11 = (double) img.at<uint8_t>(y_ + 1, x_ + 1);

	return {
		(y_ + 1 - y) * (v01 - v00) + (y - y_) * (v11 - v10),
		(x_ + 1 - x) * (v10 - v00) + (x - x_) * (v11 - v01)
	};
}

inline double _val(const cv::Mat& img, double x, double y) {
	int x_ = std::floor(x);
	int y_ = std::floor(y);

	double v00 = (double) img.at<uint8_t>(y_, x_);
	double v01 = (double) img.at<uint8_t>(y_, x_ + 1);
	double v10 = (double) img.at<uint8_t>(y_ + 1, x_);
	double v11 = (double) img.at<uint8_t>(y_ + 1, x_ + 1);

	return v11 * (x - x_) * (y - y_) +
	       v10 * (x_ + 1 - x) * (y - y_) +
	       v01 * (x - x_) * (y_ + 1 - y) +
	       v00 * (x_ + 1 - x) * (y_ + 1 - y);
}

inline bool in_img(
	const cv::Mat&         img,
	const Eigen::Vector2d& uv,
	double                 border = 0.0,
	int                    level  = 0
) {
	assert(0.0 <= border);
	double scale = (1 << level);
	double x = uv.x() / scale;
	double y = uv.y() / scale;
	return border <= x && int(border + x) < img.cols &&
	       border <= y && int(border + y) < img.rows;
}

inline Eigen::Matrix26d jaccobian_dxy1deps(const Eigen::Vector3d& p_c) {
	double x_c = p_c[0], y_c = p_c[1], z_c = p_c[2];
	double zinv = 1. / z_c;
	double zinv2 = zinv * zinv;
	double x2 = x_c * x_c, y2 = y_c * y_c, xy = x_c * y_c;
	Eigen::Matrix26d j;
	j << zinv,   0., -x_c * zinv2,      -xy * zinv2, 1. + x2 * zinv2, -y_c * zinv,
		   0., zinv, -y_c * zinv2, -1. - y2 * zinv2,      xy * zinv2,  x_c * zinv;
	return j;
}

inline Eigen::Matrix2d bilinear_w(double u, double v) {
	int x = std::floor(u);
	int y = std::floor(v);
	double dx = u - x;
	double dy = v - y;
	Eigen::Matrix2d w;
	w << (1. - dx) * (1. - dy),
	            dx * (1. - dy),
	     (1. - dx) * dy,
	            dx * dy;
	return w;
}

inline Eigen::Matrix2d bilinear_w(const Eigen::Vector2d& uv) {
	return bilinear_w(uv.x(), uv.y());
}

constexpr int half_sz = 2;
constexpr int sz = half_sz * 2;
constexpr int area = sz * sz;

std::vector<cv::Mat> build_pyr(const cv::Mat& img, int levels = 5) {
	std::vector<cv::Mat> pyr(levels);
	pyr[0] = img;
	for (int i = 1; i < levels; ++i) {
		cv::resize(pyr[i - 1], pyr[i], cv::Size(), 0.5, 0.5);
	}
	return pyr;
}

int count_out = 0;

int main(int argc, char** argv) {

	Eigen::Matrix3d cam_mat;
	cam_mat <<
	        517.3,   0.0, 325.1,
		0.0, 516.5, 249.7,
		0.0,   0.0,   1.0;
	Eigen::Matrix3d cam_mat_inv = cam_mat.inverse();
	constexpr double depth_scale = 1. / 5000.;

	cv::Mat ref       = cv::imread("data/01.png", cv::IMREAD_GRAYSCALE);
	cv::Mat cur       = cv::imread("data/04.png", cv::IMREAD_GRAYSCALE);
	cv::Mat depth_img = cv::imread("data/01d.png", cv::IMREAD_UNCHANGED);

	assert(ref.data && cur.data && depth_img.data);

	auto pyr_ref = build_pyr(ref);
	auto pyr_cur = build_pyr(cur);

	std::vector<cv::KeyPoint> key_points;
	cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(300, 0.01, 20);
	detector->detect(ref, key_points);

	std::vector<Eigen::Vector3d> xyzs_ref;
	std::vector<Eigen::Vector2d> uvs_ref;

	for (auto& each : key_points) {
		uint16_t d = depth_img.at<uint16_t>(each.pt.y, each.pt.x);
		if (d <= 0) { continue; }
		double z = (double) d * depth_scale;
		Eigen::Vector3d p = cam_mat_inv * (z * Eigen::Vector3d(each.pt.x, each.pt.y, 1.));
		xyzs_ref.push_back(p);
		uvs_ref.emplace_back(each.pt.x, each.pt.y);
	}

	constexpr int max_iterations = 10;
	Sophus::SE3d t_cr;
	std::vector<bool> visible;

	for (int level = 3; 0 <= level; --level) {

		const double scale = (1 << level);
		visible.clear();
		visible.resize(uvs_ref.size(), true);

		double last_chi2 = std::numeric_limits<double>::max();
		Sophus::SE3d last_t_cr;

		for (int itr = 0; itr < max_iterations; ++itr) {

			double chi2 = 0.0;
			Eigen::Vector6d jres    = Eigen::Vector6d::Zero();
			Eigen::Matrix6d hessian = Eigen::Matrix6d::Zero();
			int count_valid = 0;

			for (int i = 0; i < uvs_ref.size(); ++i) {
				if (!visible[i]) { continue; }

				const cv::Mat& Iref = pyr_ref[level];
				const cv::Mat& Icur = pyr_cur[level];

				Eigen::Vector2d uv_ref = uvs_ref[i] / scale;
				Eigen::Vector2i iuv_ref = uv_ref.cast<int>();

				Eigen::Vector3d p = cam_mat * xyzs_ref[i];
				assert((p.head<2>() / p[2] - uvs_ref[i]).norm() < 1e-10);

				if (!in_img(Iref, uv_ref, half_sz + 1)) {
					visible[i] = false;
					continue;
				}

				auto& xyz_ref = xyzs_ref[i];
				Eigen::Vector3d xyz_cur = t_cr * xyz_ref;
				Eigen::Vector3d uv_homo = cam_mat * xyz_cur;
				Eigen::Vector2d uv_0_cur = uv_homo.head<2>() / uv_homo[2];
				Eigen::Vector2d uv_cur = uv_0_cur / scale;
				Eigen::Vector2i iuv_cur = uv_cur.cast<int>();

				if (!in_img(Icur, uv_cur, half_sz + 1)) {
					visible[i] = false;
					continue;
				}

				++count_valid;

				Eigen::Matrix26d duv0deps = cam_mat.block<2, 2>(0, 0) * jaccobian_dxy1deps(xyz_cur);

				auto w_ref = bilinear_w(uv_ref);
				auto w_cur = bilinear_w(uv_cur);

				int ref_step = Iref.step.p[0];
				uint8_t* ref_ptr = Iref.data + (iuv_ref.y() - half_sz) * ref_step + (iuv_ref.x() - half_sz);

				int cur_step = Icur.step.p[0];
				uint8_t* cur_ptr = Icur.data + (iuv_cur.y() - half_sz) * cur_step + (iuv_cur.x() - half_sz);

// 				Eigen::Matrix6d _hessian = hessian;
// 				Eigen::Vector6d _jres = jres;
// 				double _chi2 = chi2;
// 				std::vector<Eigen::Vector6d> _js, js;


// 				for (auto dy = -half_sz; dy < half_sz; ++dy) {
// 					for (auto dx = -half_sz; dx < half_sz; ++dx) {
// 						Eigen::Vector2d delta(dx, dy);
// 						Eigen::Vector2d q1 = uv_ref + delta;
// 						Eigen::Vector2d q2 = uv_cur + delta;

// 						double err = _val(Iref, q1.x(), q1.y()) - _val(Icur, q2.x(), q2.y());
// 						_chi2 += 0.5 * err * err;
// 						/**
// 						 * @var i: image
// 						 *      u: pixel
// 						 */
// 						Eigen::Vector2d didu = _grad(Icur, q2.x(), q2.y());
// //						double gx = 0.5 * (_val(Icur, q2.x() + 1, q2.y()) - _val(Icur, q2.x() - 1, q2.y()));
// //						double gy = 0.5 * (_val(Icur, q2.x(), q2.y() + 1) - _val(Icur, q2.x(), q2.y() - 1));
// //						didu << gx, gy;
// 						/**
// 						 * @note linear part of loss function (扰动模型).
// 						 */
// 						Eigen::Vector6d jacobian = -didu.transpose() * duv0deps;
// 						_js.push_back(jacobian);
// 						_hessian += jacobian * jacobian.transpose();
// 						_jres += jacobian * -err;
// 					}
// 				}

				for (int r = 0; r < sz; ++r) {
					for (int c = 0; c < sz; ++c) {
						double intensity_ref =
							w_ref(0, 0) * double(ref_ptr[0])        + w_ref(0, 1) * double(ref_ptr[1]) +
							w_ref(1, 0) * double(ref_ptr[ref_step]) + w_ref(1, 1) * double(ref_ptr[ref_step + 1]);
						assert(ref_ptr[0]            == Iref.at<uint8_t>(iuv_ref.y() - half_sz + r, iuv_ref.x() - half_sz + c));
						assert(ref_ptr[ref_step + 1] == Iref.at<uint8_t>(iuv_ref.y() - half_sz + r + 1, iuv_ref.x() - half_sz + c + 1));
						double intensity_cur =
							w_cur(0, 0) * cur_ptr[0]        + w_cur(0, 1) * cur_ptr[1] +
							w_cur(1, 0) * cur_ptr[cur_step] + w_cur(1, 1) * cur_ptr[cur_step + 1];
						assert(cur_ptr[1]        == Icur.at<uint8_t>(iuv_cur.y() - half_sz + r, iuv_cur.x() - half_sz + c + 1));
						assert(cur_ptr[cur_step] == Icur.at<uint8_t>(iuv_cur.y() - half_sz + r + 1, iuv_cur.x() - half_sz + c));

						double err = intensity_ref - intensity_cur;
						err = _val(Iref, uv_ref.x() - half_sz + c, uv_ref.y() - half_sz + r) -
							  _val(Icur, uv_cur.x() - half_sz + c, uv_cur.y() - half_sz + r);
						chi2 += 0.5 * err * err;
						auto g = _grad(Icur, uv_cur.x() - half_sz + c, uv_cur.y() - half_sz + r);

						Eigen::Vector6d jacc = -g.transpose() * duv0deps / scale;
						//js.push_back(jacc);
						jres.noalias() +=    jacc * -err;
						hessian.noalias() += jacc * jacc.transpose();

						if (count_out < 100) {
							std::cout << "[0]jres:" << jres.transpose() << std::endl;
							count_out++;
						}

						//std::cout << "point:\n" << xyzs_ref[i].transpose() << std::endl;
						//std::cout << "j:\n" << jacc.transpose() << std::endl;

						++ref_ptr;
						++cur_ptr;
					}
					ref_ptr += (ref_step - sz);
					cur_ptr += (cur_step - sz);
				}
				// std::cout << _chi2 << ", " << chi2 << std::endl;
				// assert(_chi2 - chi2 < 1e-8);
//				for (size_t k = 0; k < 4; ++k) {
//					std::cout << "js" << std::endl;
//					std::cout << js[k].transpose() << std::endl;
//					std::cout << _js[k].transpose() << std::endl;
//					assert((js[k] - _js[k]).norm() < 1e-10);
//				}
//
				// std::cout << "diff:" << (jres - _jres).transpose() << std::endl;


				// hessian = _hessian;
				// jres = _jres;
				// chi2 = _chi2;
			}
//			std::cout << "hessian:" << hessian << std::endl;
			std::cout << "jres:" << jres.transpose() << std::endl;
			std::cout << "valid:" << count_valid << std::endl;

			Eigen::Vector6d update = hessian.ldlt().solve(jres);
			if (update.hasNaN()) { assert(false); return -1; }

			if (0 < itr && last_chi2 < chi2) {
				std::cout << "inc: " << itr << std::endl;
				t_cr = last_t_cr;
				break;
			}

			if (update.norm() < 1e-8) {
				std::cout << "jres:" << jres.transpose() << std::endl;
				std::cout << "chi2: " << chi2 << std::endl;
				std::cout << "converged at " << itr << std::endl;
				break;
			}

			last_chi2 = chi2;
			last_t_cr = t_cr;
			t_cr = Sophus::SE3d::exp(update) * t_cr;
		}

		std::cout << "R:\n" << t_cr.rotationMatrix() << std::endl;
		std::cout << "t:\n" << t_cr.translation() << std::endl;
	}

	return 0;
}

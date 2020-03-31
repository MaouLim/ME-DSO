#ifndef _ME_DSO_UTILS_HPP_
#define _ME_DSO_UTILS_HPP_

#include <opencv2/opencv.hpp>

namespace utils {

    static const cv::Mat dx_kernel = (cv::Mat_<double>(2, 2) << -0.25, +0.25, -0.25, +0.25);
    static const cv::Mat dy_kernel = (cv::Mat_<double>(2, 2) << -0.25, -0.25, +0.25, +0.25);

    inline void calc_dx(const cv::Mat& src, cv::Mat& dst) {
        cv::filter2D(src, dst, src.depth(), dx_kernel);
    }

    inline void calc_dy(const cv::Mat& src, cv::Mat& dst) {
        cv::filter2D(src, dst, src.depth(), dy_kernel);
    }

    inline void calc_grad(const cv::Mat& dx, const cv::Mat& dy, cv::Mat& dst) {
        cv::sqrt(dx.mul(dx) + dy.mul(dy), dst);
    }

    inline void down_sample(cv::Mat& target, double scale) {
        cv::resize(target, target, cv::Size(), scale, scale, cv::INTER_CUBIC);
    }
}

#endif
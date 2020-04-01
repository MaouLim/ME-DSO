#ifndef _ME_VSLAM_UTILS_HPP_
#define _ME_VSLAM_UTILS_HPP_

#include <common.hpp>

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

    inline void down_sample(const cv::Mat& src, cv::Mat& dst, double scale) {
        cv::resize(src, dst, cv::Size(), scale, scale, cv::INTER_CUBIC);
    }

    inline std::vector<cv::Mat> 
    create_pyramid(const cv::Mat& img_level0, size_t n_levels, double scale) {
        assert(CV_64FC1 == img_level0.type());

        std::vector<cv::Mat> pyramid(n_levels, cv::Mat());
        pyramid[0] = img_level0;
        for (size_t i = 1; i < n_levels; ++i) {
            down_sample(pyramid[i - 1], pyramid[i], scale);
        }

        return pyramid;
    }

    template <typename _EigenVecTp>
    double distance_l1(const _EigenVecTp& left, const _EigenVecTp& right) {
        return (left - right).lpNorm<1>();
    }

    template <typename _EigenVecTp>
    double distance_l2(const _EigenVecTp& left, const _EigenVecTp& right) {
        return (left - right).lpNorm<2>();
    }

    template <typename _EigenVecTp>
    double distance_inf(const _EigenVecTp& left, const _EigenVecTp& right) {
        return (left - right).lpNorm<-1>();
    }
}

#endif
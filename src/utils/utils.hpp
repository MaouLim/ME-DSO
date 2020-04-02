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

    inline float
    shi_tomasi_score(const cv::Mat& img, int u, int v, int patch_half_sz = 4) {
        assert(CV_64FC1 == img.type());

        float dx2 = 0.f, dy2 = 0.f, dxy = 0.f;

        const int patch_size = patch_half_sz * 2;
        const int patch_area = patch_size * patch_size;
        const int x_min = u - patch_half_sz;
        const int x_max = u + patch_half_sz;
        const int y_min = v - patch_half_sz;
        const int y_max = v + patch_half_sz;

        if (x_min < 1 || img.cols <= x_max + 1 || 
            y_max < 1 || img.rows <= y_max + 1) 
        {
            // patch is too close to the boundary
            return 0.f;
        }

        for (int x = x_min; x < x_max; ++x) {
            for (int y = y_min; y < y_max; ++y) {
                float dx = img.at<double>(y, x + 1) - img.at<double>(y, x - 1);
                float dy = img.at<double>(y + 1, x) - img.at<double>(y - 1, x);
                dx2 += dx * dx;
                dy2 += dy * dy;
            }
        }

        // const int stride = img.step.p[0];
        // for (int y = y_min; y < y_max; ++y) {
        //     const uint8_t* ptr_left   = img.data + stride*y + x_min - 1;
        //     const uint8_t* ptr_right  = img.data + stride*y + x_min + 1;
        //     const uint8_t* ptr_top    = img.data + stride*(y-1) + x_min;
        //     const uint8_t* ptr_bottom = img.data + stride*(y+1) + x_min;
        //     for (int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom) {
        //         float dx = *ptr_right - *ptr_left;
        //         float dy = *ptr_bottom - *ptr_top;
        //         dx2 += dx*dx;
        //         dy2 += dy*dy;
        //         dxy += dx*dy;
        //     }
        // }

        // Find and return smaller eigenvalue:
        dx2 = dx2 / (2.0 * patch_area);
        dy2 = dy2 / (2.0 * patch_area);
        dxy = dxy / (2.0 * patch_area);
        return 0.5f * (dx2 + dy2 - sqrt((dx2 + dy2) * (dx2 + dy2) - 4.f * (dx2 * dy2 - dxy * dxy)));
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
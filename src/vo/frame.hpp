#ifndef _ME_DSO_FRAME_HPP_
#define _ME_DSO_FRAME_HPP_

#include <opencv2/opencv.hpp>

#include "common.hpp"
#include "utils/utils.hpp"

namespace dso {

    struct frame {

        using ptr = std::shared_ptr<frame>;
    
        frame(const cv::Mat& img, bool key_frame = false);

        static uint64_t      seq_id;
        static double        pyr_scale;
        static uint64_t      pyr_levels;

        bool     key_frame;
        uint64_t id;
        
        // cv::Mat -> (H, W, Dtype=CV_64F)
        std::vector<cv::Mat>    pyr_img;  // origin image 
        std::vector<cv::Size2i> pyr_size;
        std::vector<cv::Mat>    pyr_gx;   // gx = dI/du
        std::vector<cv::Mat>    pyr_gy;   // gy = dI/dv
        std::vector<cv::Mat>    pyr_grad; // grad = sqrt(gx^2+gy^2)
        
        std::vector<pixel_point> points;
    };

    uint64_t frame::seq_id     = 0;
    double   frame::pyr_scale  = 0.5;
    uint64_t frame::pyr_levels = 5;

    inline frame::frame(const cv::Mat& _raw_img, bool _kf) {
        assert(CV_8UC1 == _raw_img.type());
        key_frame = _kf;

        cv::Mat _processed;
        if (CV_64F != _processed.depth()) {
            _processed.convertTo(_processed, CV_64F, 1./255);
        }

        pyr_img.resize(pyr_levels);
        pyr_gx.resize(pyr_levels);
        pyr_gy.resize(pyr_levels);
        pyr_grad.resize(pyr_levels);

        for (auto i = 0; i < pyr_levels; ++i) {
            // fill the current layer of pyramids
            pyr_img[i] = _processed.clone();
            pyr_size[i] = _processed.size();
            utils::calc_dx(_processed, pyr_gx[i]);
            utils::calc_dy(_processed, pyr_gy[i]);
            utils::calc_grad(pyr_gx[i], pyr_gy[i], pyr_grad[i]);

            // 
            utils::down_sample(_processed, pyr_scale);
        }
    }
}

#endif
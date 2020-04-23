#ifndef _ME_VSLAM_DATA_HPP_
#define _ME_VSLAM_DATA_HPP_

#include <opencv2/opencv.hpp>

namespace utils {

    struct dataset_reader {

        dataset_reader(const std::string& association_file) { }

        std::pair<cv::Mat, double> next() { return { cv::Mat(), 0. }; }

    private:
        std::vector<std::string> _file_list;
    };
    
} // namespace utils


#endif
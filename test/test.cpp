#include <iostream>
#include <opencv2/opencv.hpp>

#include "utils/utils.hpp"

int main(int argc, char** argv) {

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    assert(img.depth() == CV_8U);

    cv::Mat tmp;
    img.convertTo(tmp, CV_64F, 1./255.);
    assert(tmp.depth() == CV_64F);

    cv::imshow("1", tmp);
    cv::waitKey();

    utils::down_sample(tmp, 0.6);
    cv::imshow("1", tmp);
    cv::waitKey();


    // cv::Mat gx, gy, g;
    // utils::calc_dx(tmp, gx);
    // utils::calc_dy(tmp, gy);
    // utils::calc_grad(gx, gy, g);
    // cv::imshow("dx", gx);
    // cv::imshow("dy", gy);
    // cv::imshow("grad", g);
    // cv::waitKey();
    return 0;
}
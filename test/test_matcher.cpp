#include <opencv2/opencv.hpp>
#include <iostream>


int main(int argc, char** argv) {

    cv::Mat img = cv::imread("../data/test.png", cv::IMREAD_GRAYSCALE);
    cv::Mat col3 = img.col(3);

    std::cout << "img rows: " << col3.rows << std::endl;
    std::cout << "img cols: " << col3.cols << std::endl;
    std::cout << "img.step.p[0]" << col3.step.p[0] << std::endl;
    std::cout << "img.step.p[1]" << col3.step.p[1] << std::endl;
    return 0;
}

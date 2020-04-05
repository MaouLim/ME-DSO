#include <iostream>
#include <opencv2/opencv.hpp>

#include <common.hpp>
#include <utils/config.hpp>
#include <utils/utils.hpp>

struct A;
struct B;

struct A { 
    A() { } 
    ~A() { std::cout << "A" << std::endl; }  
    std::shared_ptr<B> b; 
};

struct B { 
    B() { } 
    ~B() { std::cout << "B" << std::endl; } 
    std::weak_ptr<A> a; 
};

int main(int argc, char** argv) {

    auto pa = std::make_shared<A>();
    auto pb = std::make_shared<B>();

    pa->b = pb;
    pb->a = pa;

    std::cout << "PA count: " << pa.use_count() << std::endl;
    std::cout << "PB count: " << pb.use_count() << std::endl;
    pa.reset();
    std::cout << "PA count: " << pa.use_count() << std::endl;
    std::cout << "PB count: " << pb.use_count() << std::endl;
    pb.reset();
    std::cout << "PA count: " << pa.use_count() << std::endl;
    std::cout << "PB count: " << pb.use_count() << std::endl;

    //Eigen::Vector3d a = { -1, -2, 3 }; 
    //Eigen::Vector3d b = { 2, 3, 4 };
    //std::cout << utils::distance_inf(a, b) << std::endl;

    //std::cout << b.cwiseAbs().maxCoeff() << std::endl;

    //std::vector<int> v = { 6, 5, 4, 3, 2};
    //std::cout << *utils::median(&v[0], &v[5]) << std::endl;

    // auto i = v.begin() + int((v.size() - 1) / 2);
    // std::nth_element(v.begin(), i, v.end());
    // std::cout << *i << std::endl;

    // for (auto& each : v) {
    //     std::cout << each << " ";
    // }

    //vslam::config::load_configuration("../conf/default.yaml");
    //std::cout << vslam::config::get<double>("abc", 123.0);

    // cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    // assert(img.depth() == CV_8U);

    // cv::Mat tmp;
    // img.convertTo(tmp, CV_64F, 1./255.);
    // assert(tmp.depth() == CV_64F);

    // cv::imshow("1", tmp);
    // cv::waitKey();

    // utils::down_sample(tmp, 0.6);
    // cv::imshow("1", tmp);
    // cv::waitKey();
//cv::triangulatePoints();

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
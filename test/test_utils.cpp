#include <utils/utils.hpp>
#include <iostream>


int main(int argc, char** argv) {

    // Eigen::Vector3d x0 = { 2, 3, 1 };
    // Eigen::Vector3d x1 = { 3, 4, 1 };

    // // Sophus::SE3d t_10(Eigen::Quaterniond(1, 2, 3, 1), { 10, 20, 30 });
    // // std::cout << utils::triangulate(x0, x1, t_10) << std::endl;
    // // std::cout << utils::triangulate_v2(x1, x0, t_10) << std::endl;

    // //std::shared_ptr<const Eigen::Vector3d> p(new Eigen::Vector3d(1, 2, 3));
    // auto q = std::make_shared<Eigen::Vector3d>(3, 4, 5);
    // auto p = std::make_shared<const Eigen::Vector3d>(1, 2, 3);

    // p = q;

    // const std::shared_ptr<const Eigen::Vector3d>& cr_p = q;
    
    // std::cout << *p << std::endl;
    Sophus::Vector6d delta;
    delta << 1., 2., 3., 4., 5., 6.;

    Sophus::SE3d se3 = Sophus::SE3d::exp(delta);
    std::cout << "R: \n" << se3.rotationMatrix() << std::endl;
    std::cout << "t: \n" << se3.translation().transpose() << std::endl;
    std::cout << "delta: \n" << Sophus::Vector6d(se3.log()).transpose() << std::endl;

    cv::Mat phi = (cv::Mat_<double>(3, 1) << 4, 5, 6);
    cv::Mat R_cvmat;
    cv::Rodrigues(phi, R_cvmat);
    std::cout << "R CV: \n" << R_cvmat << std::endl;
    cv::Mat phi_hat;
    cv::Rodrigues(R_cvmat, phi_hat);
    std::cout << "phi CV: \n" << phi_hat << std::endl;

    return 0;
}

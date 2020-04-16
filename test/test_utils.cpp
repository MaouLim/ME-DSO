#include <utils/utils.hpp>
#include <iostream>


int main(int argc, char** argv) {

    Eigen::Vector3d x0 = { 2, 3, 1 };
    Eigen::Vector3d x1 = { 3, 4, 1 };

    // Sophus::SE3d t_10(Eigen::Quaterniond(1, 2, 3, 1), { 10, 20, 30 });
    // std::cout << utils::triangulate(x0, x1, t_10) << std::endl;
    // std::cout << utils::triangulate_v2(x1, x0, t_10) << std::endl;

    //std::shared_ptr<const Eigen::Vector3d> p(new Eigen::Vector3d(1, 2, 3));
    auto q = std::make_shared<Eigen::Vector3d>(3, 4, 5);
    auto p = std::make_shared<const Eigen::Vector3d>(1, 2, 3);

    p = q;

    const std::shared_ptr<const Eigen::Vector3d>& cr_p = q;
    
    std::cout << *p << std::endl;

    return 0;
}

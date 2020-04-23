#include <iostream>
#include <sophus_templ/sim3.hpp>

int main(int argc, char** argv) {

    /**
     * @test test Sophus::Sim3d 
     *
     */{
        Eigen::Quaterniond q(1, 2, 3, 4);
        Eigen::Vector3d    t(10, 20, 30);
        std::cout << "q to matrix:\n"  << q.toRotationMatrix() << std::endl;
        std::cout << "q square norm: " << q.squaredNorm() << std::endl;

        Sophus::Sim3d T(q, t);

        std::cout << T.scale() << std::endl;
        std::cout << T.rotationMatrix() << std::endl;
        std::cout << T.translation() << std::endl;
        std::cout << T.scale() * T.rotationMatrix() << std::endl;
        std::cout << T.rxso3().matrix() << std::endl;

        Eigen::Vector3d p(5, 6, 7);

        std::cout << "Tp:" << T * p << std::endl;
        std::cout << "sRp + t:" << T.scale() * T.rotationMatrix() * p + T.translation() << std::endl;
    }


    return 0;
}
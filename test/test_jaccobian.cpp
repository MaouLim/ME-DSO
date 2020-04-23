#include <iostream>
#include <vo/jaccobian.hpp>

int main() {

    const Eigen::Vector3d q = { 1, 2, 3 };

    /**
     * @test test dTpdesp
     */ {
        std::cout << vslam::jaccobian_dTpdeps(q) << std::endl;
    }

    /**
     * @test test dSpdzet
     */ {
        std::cout << vslam::jaccobian_dSpdzet(q) << std::endl;
    }
    return 0;
}
#include <iostream>

#include <vo/camera.hpp>

int main(int argc, char** argv) {

    vslam::camera_ptr cam = 
        utils::mk_vptr<vslam::pinhole_camera>(480, 640, 517.3, 516.5, 325.1, 249.7);

    std::cout << "cv mat:\n"    << cam->cv_mat() << std::endl;
    std::cout << "eigen mat:\n" << cam->eigen_mat() << std::endl;

    return 0;
}
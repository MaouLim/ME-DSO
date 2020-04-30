#include <utils/utils.hpp>
#include <iostream>
#include <set>

int main(int argc, char** argv) {

    /**
     * @brief test std::set<std::shared_ptr<_Tp>>
     */ 
    {
        std::set<std::shared_ptr<int>> s;

        auto i0 = std::make_shared<int>(1);
        auto i1 = std::make_shared<int>(0);
        auto i2 = std::make_shared<int>(2);
        auto i3 = std::make_shared<int>(3);
        auto i4 = std::make_shared<int>(4);
        auto i5 = std::make_shared<int>(5);

        std::cout << "i0 < i1 : " << (i0 < i1) << std::endl;

        s.insert(i0);
        s.insert(i1);
        s.insert(i2);
        s.insert(i3);
        s.insert(i4);
        s.insert(i5);

        for (auto& each : s) { std::cout << *each << " "; }
        std::cout << std::endl;

        auto ret = s.insert(i0);
        std::cout << "insert i0: " << *ret.first << ", " << ret.second << std::endl;
        ret = s.emplace(new int(7));
        std::cout << "insert 7 : " << *ret.first << ", " << ret.second << std::endl;
    }
    
    /**
     * @brief test calc_depth_cov
     */ 
    {
        Eigen::Vector3d xyz = {
            -0.22364371963846993, 0.36487313637321739, 0.89613397497273795
        };

        Eigen::Vector3d trans = {
            -0.0023678912818245719, 0.005247017625799915, -0.0084210397622856425
        };

        double focal = 517.3;
        double tau = utils::calc_depth_cov(xyz, trans, focal);
        std::cout << tau << std::endl;
    }

    return 0;
}

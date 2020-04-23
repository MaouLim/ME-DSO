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
    
    return 0;
}

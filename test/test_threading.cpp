#include <utils/threading.hpp>
#include <iostream>

void fooba(int a, int b, double* p) {
    std::cout << "hello" << std::endl;
    std::cout << a << " " << b << " " << *p << std::endl;
}

int main(int argc, char** argv) {

    double d = 5;

    utils::atomic_flag f;
    f.do_and_exchange_if(false, &fooba, 1, 2, &d);

    return 0;
}
#include <utils/threading.hpp>
#include <iostream>
#include <unistd.h>

void fooba(int a, int b, double* p) {
    std::cout << "hello" << std::endl;
    std::cout << a << " " << b << " " << *p << std::endl;
}

struct msg : utils::message_base {
    msg(int a) : val(a) { }
    int val;
};


int main(int argc, char** argv) {

    utils::async_executor<msg> executor;
    executor.add_handler([](msg& m) { sleep(1); std::cout << m.val << std::endl; });
    executor.start();

    while (true) {
        executor.commit(msg { 1 });
        sleep(1); 
        std::cout << 100 << std::endl;
    }
    

    return 0;
}
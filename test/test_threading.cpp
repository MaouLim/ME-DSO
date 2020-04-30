#include <utils/threading.hpp>
#include <iostream>
#include <unistd.h>

struct msg : 
    utils::async_executor::message_type {
    
    msg(int a) : val(a) { }
    int val;

    typename utils::async_executor::task_catagory::item_type 
    catagory() const override { return utils::async_executor::task_catagory::NORMAL; }
};

void func(utils::async_executor::message_type& m) {
    sleep(1);
    msg& p = dynamic_cast<msg&>(m);
    std::cout << m.catagory() << std::endl;
    std::cout << p.val << std::endl;
}

int main(int argc, char** argv) {

    utils::async_executor executor;
    executor.add_handler(&func);
    executor.start();

    while (true) {
        executor.commit(msg(200));
        sleep(1); 
        std::cout << 100 << std::endl;
    }

    return 0;
}
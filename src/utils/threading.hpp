#ifndef _ME_VSLAM_THREADING_HPP_
#define _ME_VSLAM_THREADING_HPP_

#include <utils/messaging.hpp>

namespace utils {

    // template <typename _Parameter>
    // struct paramter_msg : message_base { 

    //     using param_type = _Parameter;

    //     param_type data;

    //     paramter_msg(const param_type& _data) : data(_data) { }
    //     virtual ~paramter_msg() = default;
    //     message_catagory catagory() const override { return DATA; }
    // };

    template <
        typename _ParameterMessage, 
        typename _ResultMessage
    >
    struct async_executor {

        using param_type  = _ParameterMessage;
        using result_type = _ResultMessage;

        virtual ~async_executor() { /* thread join into the parent thread */ }
        bool start() { _thread = std::thread(&_main_loop, this); }
        bool stop()  { _tasks.wait_and_push(stop_signal()); }
        void commit(const param_type& param) { _tasks.wait_and_push(param); }

    protected:

        virtual result_type process(param_type& tast_param) = 0;
        virtual void handle_result(result_type& result) { }

    private:
        void _main_loop() {

            while (true) {
                auto msg_ptr = _tasks.wait_and_pop();
                stop_signal* stop = dynamic_cast<stop_signal*>(msg_ptr.get());
                if (!stop) { break; }
                param_type* param = dynamic_cast<param_type*>(msg_ptr.get());
                auto res = process(*param);
                handle_result(res);
            }
        }

        std::thread   _thread;
        message_queue _tasks;
    };
    
} // namespace utils

#endif
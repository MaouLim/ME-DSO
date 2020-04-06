#ifndef _ME_VSLAM_THREADING_HPP_
#define _ME_VSLAM_THREADING_HPP_

#include <functional>
#include <thread>

#include <utils/messaging.hpp>

namespace utils {

    template <
        typename _Parameter, 
        typename _Callable = std::function<void(_Parameter&)>
    >
    struct async_executor {

        using param_type   = _Parameter;
        using handler_type = _Callable;

        async_executor() = default;
        explicit async_executor(size_t queue_sz) : _queue(queue_sz) { }
        virtual ~async_executor() { stop(); join(); }

        bool start();
        bool stop();
        bool join();

        virtual bool commit(const param_type& param) { _queue.wait_and_push(param); }
        void add_handler(handler_type&& handler) { _handlers.push_back(handler); }

    protected:
        bool                      _running;
        std::thread               _thread;
        message_queue             _queue;
        std::vector<handler_type> _handlers;

    private:
        void _main_loop();
    };

    template <
        typename _Parameter, typename _Callable
    >
    bool async_executor<_Parameter, _Callable>::start() {
        if (_running) { return false; }
        _thread = std::thread(&_main_loop, this);
        _running = true;
        return true;
    }

    template <
        typename _Parameter, typename _Callable
    >
    bool async_executor<_Parameter, _Callable>::stop() {
        if (!_running) { return false; }
        _queue.wait_and_push(stop_signal());
        _running = false;
        return true;
    }

    template <
        typename _Parameter, typename _Callable
    >
    bool async_executor<_Parameter, _Callable>::join() {
        if (!_thread.joinable()) { return false; }
        _thread.join();
        _thread = nullptr;
        return true;
    }

    template <
        typename _Parameter, typename _Callable
    >
    void async_executor<_Parameter, _Callable>::_main_loop() {

        while (_running) {
            auto msg_ptr = _queue.wait_and_pop();
            stop_signal* stop = dynamic_cast<stop_signal*>(msg_ptr.get());
            if (!stop) { break; }
            param_type* param = dynamic_cast<param_type*>(msg_ptr.get());
            for (auto& func_call : _handlers) {
                func_call(*param);
            }
        }

        // clean up the rest parameters those are not been executed
        _queue.clear();
    }

    struct atomic_flag {

        atomic_flag() : _flag(false) { }
        explicit atomic_flag(bool val) : _flag(val) {  }

        atomic_flag(const atomic_flag&) = delete;
        atomic_flag& operator=(bool) = delete;
        atomic_flag& operator=(const atomic_flag&) = delete;

        void store(bool val);
        void reset();

        template <typename _Predicate, typename... _Args>
        void do_if(bool condition, _Predicate&& callable, _Args&&... args);

        template <typename _IfPredicate, typename _ElsePredicate>
        void do_if_else(bool condition, _IfPredicate&& if_do, _ElsePredicate&& else_do);

        template <typename _Predicate, typename... _Args>
        void do_and_reset_if(bool condition, _Predicate&& callable, _Args&&... args);

        template <typename _Predicate, typename... _Args>
        void do_and_exchange_if(bool condition, _Predicate&& callable, _Args&&... args);

    private:
        bool               _flag;
        mutable std::mutex _mutex;
    };

    void atomic_flag::store(bool val) {
        std::lock_guard<std::mutex> lock(_mutex);
        _flag = val;
    }

    void atomic_flag::reset() {
        std::lock_guard<std::mutex> lock(_mutex);
        _flag = false;
    }

    template <typename _Predicate, typename... _Args>
    void atomic_flag::do_if(
        bool condition, _Predicate&& callable, _Args&&... args
    ) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_flag == condition) { callable(std::forward<_Args>(args)...); }
    }

    template <typename _IfPredicate, typename _ElsePredicate>
    void atomic_flag::do_if_else(bool condition, _IfPredicate&& if_do, _ElsePredicate&& else_do) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_flag == condition) { if_do(); }
        else { else_do(); }
    }

    template <typename _Predicate, typename... _Args>
    void atomic_flag::do_and_reset_if(
        bool condition, _Predicate&& callable, _Args&&... args
    ) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_flag == condition) { callable(std::forward<_Args>(args)...); _flag = false; }
    }

    template <typename _Predicate, typename... _Args>
    void atomic_flag::do_and_exchange_if(
        bool condition, _Predicate&& callable, _Args&&... args
    ) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_flag == condition) { callable(std::forward<_Args>(args)...); _flag = !_flag; }
    }
    
} // namespace utils

#endif
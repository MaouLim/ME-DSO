#ifndef _ME_VSLAM_THREADING_HPP_
#define _ME_VSLAM_THREADING_HPP_

#include <vector>
#include <functional>
#include <thread>

#include <utils/messaging.hpp>

namespace utils {

    struct async_executor {

        struct task_catagory {

            using item_type = int;

            static constexpr item_type CTRL_STOP = 0;
            static constexpr item_type NORMAL    = 1;

            static item_type null() { return -1; }
        };

        using message_type  = message_base<task_catagory>;
        using message_ptr   = std::shared_ptr<message_type>;
        using message_queue = bounded_blocking_queue<message_ptr>;
        using handler_type  = std::function<void(message_type&)>;

        async_executor() : _running(false) { }
        explicit async_executor(size_t queue_sz) : _running(false), _queue(queue_sz) { }
        virtual ~async_executor() { stop(); join(); }

        bool start();
        bool stop();
        bool join();

        template <typename _MessageTypeDerived>
        bool commit(const _MessageTypeDerived& msg);
        template <typename _MessageTypeDerived>
        bool commit(_MessageTypeDerived&& msg);

        void add_handler(handler_type&& handler) { _handlers.push_back(handler); }

    protected:

        bool commit(const message_ptr& msg) { _queue.wait_and_push(msg); return true; }

        struct _stop_signal : message_type {

            _stop_signal() = default;
            virtual ~_stop_signal() = default;

            typename task_catagory::item_type 
            catagory() const override { return task_catagory::CTRL_STOP; }
        };

        bool                      _running;
        std::thread               _thread;
        message_queue             _queue;
        std::vector<handler_type> _handlers;

    private:
        void _main_loop();
    };

    inline bool async_executor::start() {
        if (_running) { return false; }
        _running = true;
        _thread = std::thread(&async_executor::_main_loop, this);
        return true;
    }

    inline bool async_executor::stop() {
        if (!_running) { return false; }
        _running = false;
        _queue.wait_and_push(std::make_shared<_stop_signal>());
        return true;
    }

    inline bool async_executor::join() {
        if (!_thread.joinable()) { return false; }
        _thread.join();
        _running = false;
        return true;
    }

    template <typename _MessageTypeDerived>
    inline bool async_executor::commit(const _MessageTypeDerived& msg) {
        static_assert(
            std::is_same   <message_type, _MessageTypeDerived>::value || 
            std::is_base_of<message_type, _MessageTypeDerived>::value
        );
        auto ptr = std::make_shared<_MessageTypeDerived>(msg);
        _queue.wait_and_push(ptr);
        return true;
    }

    template <typename _MessageTypeDerived>
    bool async_executor::commit(_MessageTypeDerived&& msg) {
        static_assert(
            std::is_same   <message_type, _MessageTypeDerived>::value || 
            std::is_base_of<message_type, _MessageTypeDerived>::value
        );
        auto ptr = std::make_shared<_MessageTypeDerived>(std::move(msg));
        _queue.wait_and_push(ptr);
        return true;
    }

    inline void async_executor::_main_loop() {

        while (_running) {
            auto msg_ptr = _queue.wait_and_pop();
            if (task_catagory::CTRL_STOP == msg_ptr->catagory()) { _running = false; break; }
            for (auto& func_call : _handlers) {
                func_call(*msg_ptr);
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

    inline void atomic_flag::store(bool val) {
        std::lock_guard<std::mutex> lock(_mutex);
        _flag = val;
    }

    inline void atomic_flag::reset() {
        std::lock_guard<std::mutex> lock(_mutex);
        _flag = false;
    }

    template <typename _Predicate, typename... _Args>
    inline void atomic_flag::do_if(
        bool condition, _Predicate&& callable, _Args&&... args
    ) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_flag == condition) { callable(std::forward<_Args>(args)...); }
    }

    template <typename _IfPredicate, typename _ElsePredicate>
    inline void atomic_flag::do_if_else(bool condition, _IfPredicate&& if_do, _ElsePredicate&& else_do) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_flag == condition) { if_do(); }
        else { else_do(); }
    }

    template <typename _Predicate, typename... _Args>
    inline void atomic_flag::do_and_reset_if(
        bool condition, _Predicate&& callable, _Args&&... args
    ) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_flag == condition) { callable(std::forward<_Args>(args)...); _flag = false; }
    }

    template <typename _Predicate, typename... _Args>
    inline void atomic_flag::do_and_exchange_if(
        bool condition, _Predicate&& callable, _Args&&... args
    ) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_flag == condition) { callable(std::forward<_Args>(args)...); _flag = !_flag; }
    }
    
} // namespace utils

#endif
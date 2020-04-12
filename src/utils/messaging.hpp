#ifndef _ME_VSLAM_MESSAGING_HPP_
#define _ME_VSLAM_MESSAGING_HPP_

#include <utils/blocking_queue.hpp>

namespace utils {

    /**
     * @brief message base type passed by message queue
     */ 
    enum message_catagory { META, CONTROL, DATA };

	struct message_base {
		virtual ~message_base() = default;
		virtual message_catagory catagory() const { return DATA; }
	};

    struct stop_signal : message_base {
        virtual ~stop_signal() = default;
        message_catagory catagory() const override { return CONTROL; }
    };

    struct start_signal : message_base {
        virtual ~start_signal() = default;
        message_catagory catagory() const override { return CONTROL; }
    };

    struct pause_signal : message_base {
        virtual ~pause_signal() = default;
        message_catagory catagory() const override { return CONTROL; }
    };

	using message_queue = bounded_blocking_queue<
		message_base,
		std::deque<message_base>
	>;
    
} // namespace utils

#endif
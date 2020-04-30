#ifndef _ME_VSLAM_MESSAGING_HPP_
#define _ME_VSLAM_MESSAGING_HPP_

#include <utils/blocking_queue.hpp>

namespace utils {

    /**
     * @brief message base type passed by message queue
     */ 

    template <typename _Catagory>
	struct message_base {
        using catagory_type = _Catagory;
        using item_type     = typename catagory_type::item_type;
        
		virtual ~message_base() = default;
		virtual item_type catagory() const { return catagory_type::null(); }
	};
    
} // namespace utils

#endif
#ifndef _ME_VSLAM_DEPTH_FILTER_HPP_
#define _ME_VSLAM_DEPTH_FILTER_HPP_

#include <common.hpp>

namespace vslam {

    /**
     * @brief implement the depth filter used by SVO, 
     *        ref 'Video-based, Real-Time Multi View Stereo'
     */
    struct depth_filter : utils::async_executor {

        using base_type          = utils::async_executor;
        using message_type       = typename base_type::message_type;
        using message_ptr        = typename base_type::message_ptr;
        using handler_type       = typename base_type::handler_type;
        using converged_callback = std::function<void(const map_point_ptr&, double)>;
        using lock_t             = std::lock_guard<std::mutex>;

        static constexpr size_t max_queue_sz = 5;

        explicit depth_filter(const converged_callback& _cb);
        depth_filter(const detector_ptr& _det, const converged_callback& _cb);
        virtual ~depth_filter() = default;

        bool commit(const frame_ptr& frame) { return commit(std::make_shared<_df_frame_msg>(frame)); }
        
    protected:

        bool commit(const message_ptr& msg);

        struct _df_frame_msg : 
            utils::async_executor::message_type {

            frame_ptr frame;

            explicit _df_frame_msg(const frame_ptr& _frame) : frame(_frame) { }
            virtual ~_df_frame_msg() = default;

            typename task_catagory::item_type 
            catagory() const override { return task_catagory::NORMAL; }
        };

    private:
        using seed_iterator = std::list<map_point_seed>::iterator;

        void _handle_message(message_type& msg);
        void _initialize_seeds(const frame_ptr& kf);
        void _update_seeds(const frame_ptr& frame);
        void _handle_seed_itr(const frame_ptr& cur, seed_iterator& itr);

        static void _update(double x, double tau2, map_point_seed& seed);

        converged_callback        _callback;
        detector_ptr              _detector;
        utils::atomic_flag        _new_key_frame;
        size_t                    _count_key_frames;

        mutable std::mutex        _mutex_seeds;
        std::list<map_point_seed> _seeds;

        // matcher
        matcher_ptr               _matcher;
    };
}

#endif
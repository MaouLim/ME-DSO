#ifndef _ME_VSLAM_MAP_HPP_
#define _ME_VSLAM_MAP_HPP_

#include <common.hpp>

namespace vslam {

    /**
     * @brief a container stores the <map point, feature> pairs
     */ 
    struct mp_candidate_set {
        
        using mp_candidate_t = std::pair<map_point_ptr, feature_ptr>;

        mp_candidate_set() = default;
        ~mp_candidate_set() = default;

        /**
         * @brief remove all the candidates
         * @note trash will be saved
         */ 
        void clear();

        /**
         * @brief create a candidate and add it into the set
         * @note if the map point has no describing features, 
         *       the add operation will return a failure
         */ 
        bool add_candidate(const map_point_ptr& mp);

        /**
         * @brief search and remove a candidate associated 
         *        with a map point
         * @note the removed map point will be add into the
         *       trash list
         * @param mp map point to search
         */ 
        bool remove_candidate(const map_point_ptr& mp);

        /**
         * @brief extract candidates and add the feature associated
         *        into the host frame
         * @note if the map point of candidate observed by the frame, 
         *       the feature associated will be extracted to the its 
         *       host frame, then the candidate will be removed
         */ 
        bool extract_observed_by(const frame_ptr& frame);

        /**
         * @brief remove the candidates which the feature of the candidate
         *       is extracted by the frame
         */ 
        void remove_observed_by(const frame_ptr& frame);
    
    private:
        void _destroy(mp_candidate_t& candidate);
        void _clear_trash() { _trash_mps.clear(); }

        using lock_t = std::lock_guard<std::mutex>;

        mutable std::mutex        _mutex_c;
        std::list<mp_candidate_t> _candidates;
        std::list<map_point_ptr>  _trash_mps;
    };

    struct map {

        map() : _n_key_frames(0) { }
        
        const frame_ptr& last_key_frame() const { _key_frames.back(); }
        frame_ptr key_frame(int frame_id) const;
        size_t n_key_frames() const { return _n_key_frames; }

        void add_key_frame(const frame_ptr& kf) { _key_frames.push_back(kf); ++_n_key_frames; }
        void clear() { _key_frames.clear(); _n_key_frames = 0; _candidates.clear(); }

        // maybe move to map_point.hpp
        void remove_map_point(const map_point_ptr& to_rm);

        /**
         * @brief find the covisible key frame with the frame, 
         *        and return the closest one 
         */ 
        frame_ptr find_closest_covisible_key_frame(const frame_ptr& frame) const;

        /**
         * @brief find the key frame which the camera position is 
         *        furthest away from the given position
         * @param p_w 3d world coordinate
         */ 
        frame_ptr find_furthest_key_frame(const Eigen::Vector3d& p_w) const;

        /**
         * @brief find all the covisible key frame with the frame
         * @param kf_with_dis the output, covisible key frame and 
         *                    the distance
         */ 
        void find_covisible_key_frames(
            const frame_ptr&                         frame, 
            std::list<std::pair<frame_ptr, double>>& kf_with_dis
        );

        // TODO
        void update(const Sophus::SE3d& se3);
        void update(const Sophus::Sim3d& sim3);

    private:
        std::list<frame_ptr> _key_frames;
        size_t               _n_key_frames;
        mp_candidate_set     _candidates;
    };

} // namespace vslam

#endif
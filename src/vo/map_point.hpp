#ifndef _ME_VSLAM_MAP_POINT_HPP_
#define _ME_VSLAM_MAP_POINT_HPP_

#include <common.hpp>

namespace vslam {

    struct map_point_seed {

        static constexpr double conveged_ratio = 0.005;

        int         id;
        int         generation_id;
        int         count_updates;
        feature_ptr host_feature;

        /**
         * @field mu     the mean of depth_inv
         * @field sigma2 the covariance of depth_inv
         */ 
        double mu, sigma2; 
        double a, b;
        double dinv_range;

        map_point_seed(int _gen_id, const feature_ptr& host, double d_mu, double d_min);
        ~map_point_seed();

        bool converged(double ratio = conveged_ratio) const { return std::sqrt(sigma2) < ratio * dinv_range; }

        /**
         * @brief upgrade a converged seed to a map point
         * @param t_wc transform coordinates from camera sys to world sys
         */ 
        map_point_ptr upgrade(const Sophus::SE3d& t_wc) const;

    private:
        static int _seed_seq;
    };

    struct map_point {

        enum type_t { REMOVED, CANDIDATE, UNKNOWN, GOOD };

        int                     id;
        Eigen::Vector3d         position; // world coordinate system
        std::list<feature_wptr> observations;
        size_t                  n_obs;
        int                     last_pub_timestamp;
        int                     last_proj_kf_id;
        int                     last_opt_timestamp;
        int                     n_fail_reproj;
        int                     n_success_reproj;
        type_t                  type;

        explicit map_point(const Eigen::Vector3d& _pos);
        ~map_point() = default;

        void set_observed_by(const feature_ptr& _feat);
        void clear_observations() { observations.clear(); n_obs = 0; }

        /**
         * @brief set the map point as a REMOVED point
         */ 
        void as_removed() { type = REMOVED; clear_observations(); }

        /**
         * @brief find the latest observing feature
         * @note if the observing feature is weak (unable to 
         *       upgrade as a ptr), it will be remove, until 
         *       a strong feature is found.  
         */
        feature_ptr last_observed();

        /**
         * @brief sequential access the observations list to 
         *        find the observing feature in the frame
         * @note if the observing feature is weak (unable to 
         *       upgrade as a ptr), it will be remove.  
         */ 
        feature_ptr find_observed(const frame_ptr& _frame);
        bool remove_observed_by(const frame_ptr& _frame);

        /**
         * @brief find the feature that observes the map point 
         *        and the view angle between observing from _cam_center and
         *        the feature is least. 
         * @param _cam_center view point
         */ 
        feature_ptr find_closest_observed(const Eigen::Vector3d& _cam_center);

        /**
         * @brief minimize the reproject error (on unit-bearing plane)
         * @return the error after iterations
         */ 
        double local_optimize(size_t n_iterations);

    private:
        static int _seq_id; // to generate id;

        static frame_ptr _get_frame(const feature_wptr& ob);
    };

    /**
     * @brief a container stores the <map point, feature> pairs
     */ 
    struct candidate_set {
        
        using candidate_t = std::pair<map_point_ptr, feature_ptr>;

        candidate_set() = default;
        ~candidate_set() = default;

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

        /**
         * @brief thread-safe travesal
         * @note it can change the element of candidates, but will 
         *       not change the size of the candidates
         * @param pred callable object, ex. void(const candidate_t&) 
         */ 
        template <typename _Predicate>
        void for_each(_Predicate&& _pred);

        /**
         * @brief thread-safe travesal
         * @param cond callable object, requires to return a bool value
         *             to determined whether to remove the current element
         *             ex. bool(const candidate_t&) 
         * @return the number of elements removed
         */
        template <typename _Condition>
        size_t for_each_remove_if(_Condition&& cond);
    
    private:
        void _destroy(candidate_t& candidate);
        void _clear_trash() { _trash_mps.clear(); }

        using lock_t = std::lock_guard<std::mutex>;

        mutable std::mutex        _mutex_c;
        std::list<candidate_t>    _candidates;
        std::list<map_point_ptr>  _trash_mps;
    };
}

#endif
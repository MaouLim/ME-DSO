#ifndef _ME_VSLAM_REPROJECTOR_HPP_
#define _ME_VSLAM_REPROJECTOR_HPP_

#include <common.hpp>

namespace vslam {

    /**
     * @brief reproject the map points into the (key) frames to find the 
     *        features matched to the points
     */ 
    struct reprojector {

        static const size_t max_matches                  = 400;
        static const size_t max_overlaped_kfs            = 20;
        static const size_t max_candidate_mp_fail_reproj = 30;
        static const size_t max_unknown_mp_fail_reproj   = 15;
        static const size_t min_good_mp_success_reproj   = 10;

        size_t n_matches;
        size_t n_trials;

        reprojector(int height, int width, int cell_sz);

        /**
         * @brief clear and reset the grid
         */ 
        void clear();

        /**
         * @brief 
         * @param frame         
         * @param kfs_with_dis      VSLAM_IN_OUT the covisible key frames with
         *                          the distance from reference frame 
         * @param candidates        VSLAM_IN_OUT set of candidate map points, a 
         *                          candidate will be removed when failed to 
         *                          reproject certain times
         * @param kfs_with_overlaps VSLAM_OUT the covisible key frames and 
         *                          the successful times of reprojection each
         *                          frame 
         * @return n_matches
         */
        size_t reproject_and_match(
            const frame_ptr&                  frame,
            std::vector<frame_with_distance>& kfs_with_dis,
            candidate_set&                    candidates,
            std::vector<frame_with_overlaps>& kfs_with_overlaps
        );

    private:

        // struct _coarse_match {
        //     using handle_t = std::list<map_point_ptr>::const_iterator;

        //     map_point_ptr   map_point;
        //     Eigen::Vector2d uv;
        //     handle_t        handle;

        //     _coarse_match(
        //         const map_point_ptr&   _mp, 
        //         const Eigen::Vector2d& _uv, 
        //         const handle_t&        _hd = handle_t(nullptr)
        //     );
        // };

        using match_type = std::pair<map_point_ptr, Eigen::Vector2d>;//_coarse_match;
        using cell_type  = std::list<match_type>;

        friend bool operator<(const match_type& left, const match_type& right);
        
        size_t _cell_idx(double x, double y) const;
        void _shuffle() { std::random_shuffle(_indices.begin(), _indices.end()); }

        /**
         * @brief reproject map point into the grid on frame
         * @return whether reprojection is success
         */ 
        bool _reproject_mp(const frame_ptr& frame, const map_point_ptr& mp);

        /**
         * @brief find the covisible key frames with frame in the 
         *        global map, and reproject all the covisible map 
         *        point into the grid on frame
         * @param frame 
         * @param kfs_with_dis      VSLAM_IN_OUT the covisible key frames with
         *                          the distance from reference frame          
         * @param kfs_with_overlaps VSLAM_OUT the covisible key frames and 
         *                          the successful times of reprojection each
         *                          frame 
         */ 
        bool _reproject_covisible_kfs(
            const frame_ptr&                  frame, 
            std::vector<frame_with_distance>& kfs_with_dis,
            std::vector<frame_with_overlaps>& kfs_with_overlaps
        );

        /**
         * @brief reproject canidate map points into the grid on frame
         * @param candidates VSLAM_IN_OUT set of candidate map points, a 
         *                   candidate will be removed when failed to 
         *                   reproject certain times
         */ 
        bool _reproject_candidates(const frame_ptr& frame, candidate_set& candidates);

        bool _find_match_in_cell(
            const frame_ptr& frame, 
            cell_type&       cell, 
            candidate_set&   candidates
        );

        int                    _cell_sz;
        std::vector<cell_type> _grid;
        size_t                 _rows;
        size_t                 _cols;
        std::vector<int>       _indices;

        matcher_ptr            _matcher;
    };

    inline size_t reprojector::_cell_idx(double x, double y) const { 
        return int(y / _cell_sz) * _cols + int(x / _cell_sz); 
    }
    
} // namespace vslam


#endif
#ifndef _ME_VSLAM_REPROJECTOR_HPP_
#define _ME_VSLAM_REPROJECTOR_HPP_

#include <common.hpp>

namespace vslam {

    /**
     * @brief reproject the map points into the (key) frames to find the 
     *        features matched to the points
     */ 
    struct reprojector {

        size_t n_matches;
        size_t n_trials;

        reprojector(int height, int width, int cell_sz);

        void clear();
        //void reproject
    private:
        using match_type = std::pair<map_point_ptr, Eigen::Vector2d>;
        using match_set  = std::list<match_type>;

        friend bool operator<(const match_type& left, const match_type& right);

        //void _initialize_grid(int _height, int _width);
        
        size_t _cell_idx(double x, double y) const;
        void _shuffle() { std::random_shuffle(_cells_order.begin(), _cells_order.end()); }

        bool _reproject(const map_point_ptr& mp, const frame_ptr& frame);
        bool _find_match_in_cell(match_set& cell, const frame_ptr& frame);

        int                    _cell_sz;
        std::vector<match_set> _grid;
        size_t                 _rows;
        size_t                 _cols;
        std::vector<int>       _cells_order;

        matcher_ptr            _matcher;
    };

    inline size_t reprojector::_cell_idx(double x, double y) const { 
        return int(y / _cell_sz) * _cols + int(x / _cell_sz); 
    }
    
} // namespace vslam


#endif
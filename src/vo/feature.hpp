#ifndef _ME_VSLAM_FEATURE_HPP_
#define _ME_VSLAM_FEATURE_HPP_

#include <common.hpp>

namespace vslam {

    /**
     * @brief interesting pixel point on the frame 
     *        which can describe a 3d map point (or landmark) 
     */ 
    struct feature {

        enum type_t { CORNER, EDGELET };

        type_t          type;
        frame_wptr      host_frame;           // in which frame this feature detected 
        map_point_ptr   map_point_describing; // map point which described by this feature
        Eigen::Vector2d uv;                   // pixel vector 
        Eigen::Vector3d xy1;                  // uint-bearing vector
        Eigen::Vector2d grad_orien;           // the orientation of the graditude at this pixel
        size_t          level;                // level of pyramid

        feature(const frame_ptr& _host, const Eigen::Vector2d& _uv, size_t _pyr_level);
        feature(const frame_ptr& _host_f, const map_point_ptr& _mp_desc, const Eigen::Vector2d& _uv, size_t _pyr_level);

        bool describe_nothing() const { return !map_point_describing; }
    };

    /**
     * @brief the struct for corner detect by fast
     */ 
    struct corner {

        int    x; 
        int    y;
        int    level;
        double score;
        double angle;

        corner(int _x, int _y, int _level, double _score, double _angle) : 
            x(_x), y(_y), level(_level), score(_score), angle(_angle) { }
    };

    struct abstract_detector {

        static const int BORDER = 8;

        abstract_detector(int _h, int _w, int _cell_sz, size_t _n_levels);
        virtual ~abstract_detector() = default;

        virtual void detect(frame_ptr host, double threshold, feature_set& features) = 0;

        void set_grid_occupied(int x, int y);

    protected:
        int               cell_sz;
        int               grid_cols;
        int               grid_rows;
        size_t            pyr_levels;
        std::vector<bool> grid_occupied;

        void reset();
        int cell_index(int x, int y, size_t level);
    };

    abstract_detector::abstract_detector(
        int _h, int _w, int _cell_sz, size_t _n_levels
    ) : cell_sz(_cell_sz), pyr_levels(_n_levels), 
        grid_cols(std::ceil(double(_w) / _cell_sz)), 
        grid_rows(std::ceil(double(_h) / _cell_sz)) 
    {
        grid_occupied.resize(grid_rows * grid_cols, false);
    }

    inline void abstract_detector::set_grid_occupied(int x, int y) {
        grid_occupied[cell_index(x, y, 0)] = true;
    }

    inline void abstract_detector::reset() {
        std::fill(grid_occupied.begin(), grid_occupied.end(), false);
    }
 
    inline int 
    abstract_detector::cell_index(int x, int y, size_t level = 0) {
        int scale = (1 << level);
        return int(scale * y / cell_sz) * grid_cols + int(scale * x / cell_sz);
    }

    struct fast_detector : abstract_detector {

        fast_detector(int _h, int _w, int _cell_sz, size_t _n_levels);
        virtual ~fast_detector() = default;

        void detect(frame_ptr host, double threshold, feature_set& features) override;
    };

    fast_detector::fast_detector(int _h, int _w, int _cell_sz, size_t _n_levels) : 
        abstract_detector(_h, _w, _cell_sz, _n_levels) { }
}

#endif
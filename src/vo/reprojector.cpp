#include <vo/reprojector.hpp>

#include <vo/map_point.hpp>
#include <vo/frame.hpp>
#include <vo/camera.hpp>

namespace vslam {

    bool operator<(
        const reprojector::match_type& left, 
        const reprojector::match_type& right
    ) {
        return left.first->type < right.first->type;
    }

    reprojector::reprojector(
        int height, int width, int cell_sz
    ) : n_matches(0), n_trials(0), _cell_sz(cell_sz), 
        _rows(std::ceil(double(height) / _cell_sz)), 
        _cols(std::ceil(double(width)  / _cell_sz))
    {
        const int n_cells =  _rows * _cols;
        _grid.resize(n_cells);
        _cells_order.reserve(n_cells);
        for (int i = 0; i < n_cells; ++i) {
            _cells_order.push_back(i);
        }
        _shuffle();
    }

    void reprojector::clear() {
        n_matches = 0; n_trials = 0;
        for (auto& cell : _grid) { cell.clear(); }
    }

    bool reprojector::_reproject(const map_point_ptr& mp, const frame_ptr& frame) {
        Eigen::Vector2d uv = 
            frame->camera->world2pixel(mp->position, frame->t_cw);
        if (!frame->visible(uv, patch_t::half_sz/* size */)) { return false; }
        _grid[_cell_idx(uv.x(), uv.y())].emplace_back(mp, uv);
        return true;
    }

}
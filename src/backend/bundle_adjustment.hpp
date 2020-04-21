#ifndef _ME_VSLAM_BUNDLE_ADJUSTMENT_HPP_
#define _ME_VSLAM_BUNDLE_ADJUSTMENT_HPP_

#include <common.hpp>
#include <backend/g2o_staff.hpp>

namespace vslam::backend {

    struct twoframe_ba : g2o_optimizer {

        twoframe_ba(const frame_ptr& f0, const frame_ptr& f1, const map_ptr& m, double reproj_thresh);
        virtual ~twoframe_ba() = default;

        void create_graph() override;
        void update() override;

    protected:
        edge_xyz2uv_se3* create_g2o_edge(
            vertex_xyz*        v0, 
            vertex_se3*        v1, 
            const feature_ptr& feat, 
            double             weight = 1.0,
            bool               robust_kernel = true
        );

    private:
        frame_ptr  _frame0;
        frame_ptr  _frame1;
        camera_ptr _cam;
        double     _reproj_thresh2;

        std::vector<
            std::pair<feature_ptr, edge_xyz2uv_se3*>
        >       _edges;
        map_ptr _map;
    };

    struct local_map_ba : g2o_optimizer {

        local_map_ba();

        void create_graph() override;
        void update() override;

    private:
        frame_ptr            _main_kf;
        std::set<frame_ptr>& _core_kfs;
        camera_ptr           _cam;
        double               _reproj_thresh2;

        std::set<map_point_ptr> _mps;
        std::list<frame_ptr>    _covisibles;
    };

    struct global_map_ba : g2o_optimizer {

        void create_graph() override;
        void update() override;
    };
    
} // namespace backend

#endif
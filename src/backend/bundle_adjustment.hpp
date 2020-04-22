#ifndef _ME_VSLAM_BUNDLE_ADJUSTMENT_HPP_
#define _ME_VSLAM_BUNDLE_ADJUSTMENT_HPP_

#include <common.hpp>
#include <backend/g2o_staff.hpp>

namespace vslam::backend {

    struct twoframe_ba : g2o_optimizer {

        twoframe_ba(const frame_ptr& f0, const frame_ptr& f1, double reproj_thresh);
        virtual ~twoframe_ba() = default;

        void create_graph() override;
        void update() override;

    private:
        frame_ptr                _frame0;
        frame_ptr                _frame1;
        double                   _reproj_thresh2;
        
        /**
         *@field caches 
         */
        std::vector<feature_ptr> _feats;
    };

    struct local_map_ba : g2o_optimizer {

        local_map_ba();

        void create_graph() override;
        void update() override;

    private:
        std::set<frame_ptr>& _core_kfs;
        double               _reproj_thresh2;

        /**
         *@field caches 
         */
        std::vector<map_point_ptr> _mps;
        std::list<frame_ptr>       _covisibles;
        std::vector<feature_ptr>   _feats;
    };

    struct global_map_ba : g2o_optimizer {

        void create_graph() override;
        void update() override;

    private:
        map_ptr _map;
        double  _reproj_thresh;

        /**
         *@field caches 
         */
        std::vector<map_point_ptr> _mps;
        std::vector<feature_ptr>   _feats;
    };
    
} // namespace backend

#endif
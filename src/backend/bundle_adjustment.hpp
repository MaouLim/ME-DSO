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

        void set_f0(const frame_ptr& f0) { _frame0 = f0; }
        void set_f1(const frame_ptr& f1) { _frame1 = f1; }

    private:
        frame_ptr _frame0;
        frame_ptr _frame1;
        double    _reproj_thresh2;
        
        /**
         *@field caches 
         */
        std::vector<feature_ptr> _feats;
    };

    struct local_map_ba : g2o_optimizer {

        local_map_ba(std::set<frame_ptr>& core_kfs, double reproj_thresh) : 
            _core_kfs(core_kfs), _reproj_thresh2(reproj_thresh * reproj_thresh) { }

        void create_graph() override;
        void update() override;

        void set_core_kfs(std::set<frame_ptr>& core_kfs) { _core_kfs = core_kfs; }

    private:
        std::set<frame_ptr>& _core_kfs;
        double               _reproj_thresh2;

        /**
         *@field caches 
         */
        std::vector<map_point_ptr> _mps;
        std::vector<frame_ptr>     _covisibles;
        std::vector<feature_ptr>   _feats;

        void _clear_cache() {
            _mps.clear(); _covisibles.clear(); _feats.clear();
        }
    };

    struct global_map_ba : g2o_optimizer {

        global_map_ba(const map_ptr& m, double reproj_thresh) : 
            _map(m), _reproj_thresh(reproj_thresh) 
        { 
            assert(_map);
        }

        void create_graph() override;
        void update() override;

    private:
        map_ptr _map;
        double  _reproj_thresh;

        /**
         *@field caches 
         */
        std::vector<map_point_ptr> _mps;
        std::vector<feature_ptr>   _feats_bad;
        std::vector<feature_ptr>   _feats_opt;

        void _clear_cache() {
            _mps.clear(); _feats_bad.clear(); _feats_opt.clear();
        }
    };
    
} // namespace backend

#endif
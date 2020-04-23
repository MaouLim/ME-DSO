#include <backend/bundle_adjustment.hpp>

#include <vo/frame.hpp>
#include <vo/feature.hpp>
#include <vo/map_point.hpp>
#include <vo/map.hpp>

namespace vslam::backend {

    twoframe_ba::twoframe_ba(
        const frame_ptr& f0, 
        const frame_ptr& f1, 
        double           reproj_thresh
    ) : _frame0(f0), _frame1(f1),
        _reproj_thresh2(reproj_thresh * reproj_thresh) 
    { 
        assert(_frame0 && _frame1);
    }
    
    void twoframe_ba::create_graph() {
        _feats.clear();
        _feats.reserve(_frame0->n_features * 2);

        int v_seq = 0;

        auto v0 = _frame0->create_g2o(v_seq++, true);
        auto v1 = _frame1->create_g2o(v_seq++, false, false);

        _optimizer.addVertex(v0);
        _optimizer.addVertex(v1);

        for (auto each_feat : _frame0->features) {
            if (each_feat->describe_nothing()) { continue; }
            auto v = each_feat->map_point_describing->create_g2o(v_seq++, false, true);
            _optimizer.addVertex(v);

            auto e = each_feat->create_g2o(_feats.size(), v, v0);
            _feats.emplace_back(each_feat);
            _optimizer.addEdge(e);

            feature_ptr ob = each_feat->map_point_describing->find_observed_by(_frame1);
            if (!ob) { continue; }
            e = ob->create_g2o(_feats.size(), v, v1);
            _feats.emplace_back(ob);
            _optimizer.addEdge(e);
        }
    }

    void twoframe_ba::update() {
        _frame0->update_from_g2o();
        _frame1->update_from_g2o();

        for (auto& each_feat : _feats) {
            each_feat->update_from_g2o(_reproj_thresh2);
        }

        for (auto each_feat : _frame0->features) {
            if (each_feat->describe_nothing()) { continue; }
            each_feat->map_point_describing->update_from_g2o();
        }
    }

    void local_map_ba::create_graph() {
        int v_seq = 0;
        _clear_cache();

        for (auto& kf : _core_kfs) {
            auto v = kf->create_g2o(v_seq++);
            _optimizer.addVertex(v);

            for (auto& feat : kf->features) {
                if (feat->describe_nothing()) { continue; }

                const auto& mp = feat->map_point_describing;

                if (!mp->v) {
                    auto v_mp = mp->create_g2o(v_seq++);
                    _optimizer.addVertex(v_mp);
                    _mps.emplace_back(mp);

                    auto visit_ob = [&](const feature_ptr& ob) {
                        frame_ptr ob_frame = ob->host_frame.lock(); assert(ob_frame);
                        if (!ob_frame->v) {
                            auto v_f = ob_frame->create_g2o(v_seq++, true);
                            _optimizer.addVertex(v_f);
                            _covisibles.emplace_back(ob_frame);
                        }

                        // create edges
                        auto e = ob->create_g2o(_feats.size(), mp->v, ob_frame->v, (1 << ob->level));
                        _feats.emplace_back(ob);
                        _optimizer.addEdge(e);
                    };

                    mp->for_each_observation(visit_ob);
                }
            }
        }
    }

    void local_map_ba::update() {
        for (auto& kf : _core_kfs) { kf->update_from_g2o(); }
        for (auto& covisible : _covisibles) { covisible->shutdown_g2o(); }
        for (auto& mp : _mps) { mp->update_from_g2o(); }
        for (auto& feat : _feats) { feat->update_from_g2o(_reproj_thresh2); }
    }

    void global_map_ba::create_graph() {
        int v_seq = 0;
        _clear_cache();

        for (auto& kf : _map->key_frames()) {
            auto v_kf = kf->create_g2o(v_seq++);
            _optimizer.addVertex(v_kf);

            for (auto& feat : kf->features) {
                if (feat->describe_nothing()) { continue; }
                const auto& mp = feat->map_point_describing;

                if (!mp->v) {
                    auto v_mp = mp->create_g2o(v_seq++);
                    _optimizer.addVertex(v_mp);
                    _mps.emplace_back(mp);
                }

                // calculate reproject error, add it into bad features if the error is large
                double err = utils::reproject_err(kf->t_cw * mp->position, feat->xy1, 0.5);
                if (_reproj_thresh < err) { _feats_bad.emplace_back(feat); continue; }
                
                auto e = feat->create_g2o(_feats_opt.size(), mp->v, v_kf);
                _optimizer.addEdge(e);
                _feats_opt.emplace_back(feat);
            }
        }
    }

    void global_map_ba::update() {
        for (auto& kf : _map->key_frames()) { kf->update_from_g2o(); }
        for (auto& mp : _mps) { mp->update_from_g2o(); }

        for (auto& feat : _feats_bad) { 
            if (feat->map_point_describing->n_obs < 2) {
                feat->remove_describing();
            }
            else { feat->reset_describing(true); }
        }

        const double reproj_thresh2 = _reproj_thresh * _reproj_thresh;
        for (auto& feat : _feats_opt) { feat->update_from_g2o(reproj_thresh2); }
    }
}
#include <backend/bundle_adjustment.hpp>

#include <vo/frame.hpp>
#include <vo/feature.hpp>
#include <vo/map_point.hpp>
#include <vo/map.hpp>

namespace vslam::backend {

    twoframe_ba::twoframe_ba(
        const frame_ptr& f0, 
        const frame_ptr& f1, 
        const map_ptr&   m,
        double           reproj_thresh
    ) : _frame0(f0), _frame1(f1), _map(m), 
        _reproj_thresh2(reproj_thresh * reproj_thresh) 
    { 
        assert(_frame0 && _frame1 && _map);
        assert(_frame0->camera == _frame1->camera);
        _cam = _frame0->camera;
    }
    
    void twoframe_ba::create_graph() {
        _edges.clear();
        _edges.reserve(_frame0->n_features);

        int v_seq = 0;

        auto v0 = _frame0->create_g2o_staff(v_seq++, true);
        auto v1 = _frame1->create_g2o_staff(v_seq++, false, false);

        _optimizer.addVertex(v0);
        _optimizer.addVertex(v1);

        for (auto each_feat : _frame0->features) {
            if (each_feat->describe_nothing()) { continue; }
            auto v = each_feat->map_point_describing->create_g2o_staff(v_seq++, false, true);
            _optimizer.addVertex(v);

            auto e = create_g2o_edge(v, v0, each_feat);
            _optimizer.addEdge(e);

            feature_ptr ob = each_feat->map_point_describing->find_observed(_frame1);
            if (!ob) { continue; }
            e = create_g2o_edge(v, v1, ob);
            _optimizer.addEdge(e);
        }
    }

    void twoframe_ba::update() {
        _frame0->update_from_g2o();
        _frame1->update_from_g2o();

        for (auto each_feat : _frame0->features) {
            if (each_feat->describe_nothing()) { continue; }
            each_feat->map_point_describing->update_from_g2o();
        }

        for (auto& pair : _edges) {
            if (_reproj_thresh2 < pair.second->chi2()) {
                assert(!pair.first->describe_nothing());
                pair.first->map_point_describing->as_removed();
                pair.first->map_point_describing = nullptr;
            }
        }
    }

    edge_xyz2uv_se3* twoframe_ba::create_g2o_edge(
        vertex_xyz*        v0, 
        vertex_se3*        v1, 
        const feature_ptr& feat, 
        double             weight,
        bool               robust_kernel
    ) {
        edge_xyz2uv_se3* e = new edge_xyz2uv_se3(_cam);
        e->setVertex(0, v0);
        e->setVertex(1, v1);
        e->setMeasurement(feat->uv);
        e->setInformation(weight * Eigen::Matrix2d::Identity());
        e->setId(_edges.size());
        if (robust_kernel) {
            g2o::RobustKernelHuber* huber = new g2o::RobustKernelHuber();
            e->setRobustKernel(huber);
        }
        _edges.emplace_back(feat, e);
        return e;
    }

    void local_map_ba::create_graph() {
        int v_seq = 0;

        for (auto& kf : _core_kfs) {
            auto v = kf->create_g2o_staff(v_seq++);
            _optimizer.addVertex(v);

            for (auto& feat : kf->features) {
                if (feat->describe_nothing()) { continue; }
                _mps.insert(feat->map_point_describing);
            }
        }

        for (auto& mp : _mps) {
            auto v_mp = mp->create_g2o_staff(v_seq++);
            _optimizer.addVertex(v_mp);

            for (auto& each_ob : mp->observations) {
                if (each_ob.expired()) { continue; }
                feature_ptr ob = each_ob.lock();
                frame_ptr ob_frame = ob->host_frame.lock();
                assert(ob_frame);
                if (!ob_frame->v) {
                    auto v_f = ob_frame->create_g2o_staff(v_seq++, true);
                    _optimizer.addVertex(v_f);
                    _covisibles.push_back(ob_frame);
                }

                //TODO create edges
            }
        }
    }

    void local_map_ba::update() {

    }
}
#include <vo/map_point.hpp>

#include <vo/jaccobian.hpp>
#include <vo/frame.hpp>
#include <vo/feature.hpp>

#include <backend/g2o_staff.hpp>

namespace vslam {

    int map_point_seed::_seed_seq = 0;

    map_point_seed::map_point_seed(
        int _gen_id, const feature_ptr& host, double d_mu, double d_min
    ) : generation_id(_gen_id), id(_seed_seq++), count_updates(0), 
        host_feature(host), a(10.), b(10.) 
    { 
        mu = 1. / d_mu;
        dinv_range = 1. / d_min;
        sigma2 = dinv_range * dinv_range / 36.;
    }
    
    map_point_seed::~map_point_seed() {
#ifdef _ME_VSLAM_DEBUG_INFO_
        // std::cout << "Map point seed: " << id 
        //           << " live time: "     << count_updates 
        //           << std::endl;
#endif
    }

    map_point_ptr 
    map_point_seed::upgrade(const Sophus::SE3d& t_wc) const {
        Eigen::Vector3d xyz_unit = host_feature->xy1.normalized();
        Eigen::Vector3d xyz      = t_wc * (xyz_unit / mu);
        auto mp = utils::mk_vptr<map_point>(xyz);
        assert(host_feature->set_describing(mp));
        return mp;
    }

    /**
     * map point methods
     */ 

    int map_point::_seq_id = 0;

    map_point::map_point(const Eigen::Vector3d& _pos) : 
        id(_seq_id++), position(_pos), n_obs(0), 
        last_pub_timestamp(0), last_proj_kf_id(-1), last_opt(0), 
        n_fail_reproj(0), n_success_reproj(0), type(UNKNOWN), v(nullptr)
    { }

    void map_point::as_removed() { 
        if (type == REMOVED) { return; }
        for (auto& each : observations) {
            if (_expired(each)) { continue; }
            each.lock()->reset_describing();
        }
        _clear_observations();
        type = REMOVED; 
    }

    feature_ptr map_point::last_observation() {
        auto itr = observations.begin();
        while (itr != observations.end()) {
            if (_expired(*itr)) { 
                itr = observations.erase(itr); --n_obs;
                continue;
            }
            return itr->lock();
        }
        return nullptr;
    }

    bool map_point::remove_observation(const feature_ptr& _feat) {
        assert(_feat);
        auto itr = observations.begin();
        while (itr != observations.end()) {
            if (_expired(*itr)) { 
                itr = observations.erase(itr); --n_obs;
                continue;
            }
            if (_feat == itr->lock()) {
                observations.erase(itr); --n_obs;
                return true;
            }
            ++itr;
        }
        return false;
    }

    feature_ptr map_point::find_observed_by(const frame_ptr& _frame) {
        assert(_frame);
        auto itr = observations.begin();
        while (itr != observations.end()) {
            if (_expired(*itr)) { 
                itr = observations.erase(itr); --n_obs;
                continue;
            }
            auto exist_ob = itr->lock();
            if (_frame == exist_ob->host_frame.lock()) {
                return exist_ob;
            }
            else { ++itr; }
        }
        return nullptr;
    }

    bool map_point::remove_observed_by(const frame_ptr& _frame) {
        assert(_frame);
        auto itr = observations.begin();
        while (itr != observations.end()) {
            if (_expired(*itr)) { 
                itr = observations.erase(itr); --n_obs;
                continue;
            }
            if (_frame == _get_frame(*itr)) {
                observations.erase(itr); --n_obs;
                return true;
            }
            ++itr;
        }
        return false;
    }

    feature_ptr 
    map_point::find_closest_observed(
        const Eigen::Vector3d& _cam_center
    ) {
        Eigen::Vector3d view_orien = _cam_center - position;
        view_orien.normalize();

        double max_cos   = 0.0;
        feature_ptr res = nullptr;

        auto itr = observations.begin();
        while (itr != observations.end()) {
            if (_expired(*itr)) {
                itr = observations.erase(itr); --n_obs;
                continue;
            }
            auto exist_ob = itr->lock();
            auto exist_host_frame = exist_ob->host_frame.lock();
            if (!exist_host_frame) { ++itr; continue; }
            Eigen::Vector3d orien = exist_host_frame->cam_center() - position;
            orien.normalize();
            double cos_theta = view_orien.dot(orien);
            if (max_cos < cos_theta) {
                max_cos = cos_theta;
                res = exist_ob;
            }
            ++itr;
        }
        return max_cos < CONST_COS_60 ? nullptr : res;
    }

    double map_point::local_optimize(size_t n_iterations) {
        Eigen::Vector3d old_pos = position;

        double last_chi2 = 0.0;
        Eigen::Matrix3d H;
        Eigen::Vector3d b;

        for (size_t i = 0; i < n_iterations; ++i) {
            
            double chi2 = 0.0;
            H.setZero(); b.setZero();

            for (auto& feature_observed : observations) {
                if (_expired(feature_observed)) { continue; }
                
                auto exist_ob = feature_observed.lock();
                auto host_frame = exist_ob->host_frame.lock();
                if (!host_frame) { continue; }

                Eigen::Vector3d p_c = host_frame->t_cw * position;
                Eigen::Matrix23d jacc   = -1.0 * jaccobian_dxy1dxyz(p_c, host_frame->t_cw.rotationMatrix());
                Eigen::Matrix32d jacc_t = jacc.transpose();
                Eigen::Vector2d p_xy1 = utils::project(p_c);
                Eigen::Vector2d err = exist_ob->xy1.head<2>() - p_xy1;
                H.noalias() +=  jacc_t * jacc;
                b.noalias() += -jacc_t * err;
                chi2 += err.squaredNorm();
            }

            Eigen::Vector3d delta = H.ldlt().solve(b);
            if (delta.hasNaN()) { assert(false); return -1.; }

            if (0 < i && last_chi2 < chi2) {
#ifdef _ME_VSLAM_DEBUG_INFO_OPT_
                std::cout << "[MP]" << "Loss increased at " << i << std::endl;
#endif
                position = old_pos;
                break;
            }

            old_pos = position;
            position += delta;
            last_chi2 = chi2;

            if (delta.norm() < config::opt_converged_thresh_xyz) {
#ifdef _ME_VSLAM_DEBUG_INFO_OPT_
                std::cout << "[MP]" << "Converged at" << i << std::endl;
#endif 
                break;
            }
        }

        return last_chi2;
    }

    backend::vertex_xyz* 
    map_point::create_g2o(
        int vid, bool fixed, bool marg
    ) {
        if (v) { return v; }
        v = new backend::vertex_xyz();
        v->setId(vid);
        v->setFixed(fixed);
        if (!fixed) { v->setMarginalized(marg); }
        v->setEstimate(position);
        return v;
    }

    bool map_point::update_from_g2o() {
        if (!v) { return false; }
        position = v->estimate();
        v = nullptr;
        return true;
    }

    void map_point::_set_observed_by(const feature_ptr& _feat) {
        assert(_feat);
        observations.emplace_front(_feat);
        ++n_obs;
    }

    bool map_point::_expired(const feature_wptr& ob) const {
        return ob.expired() || (this != ob.lock()->map_point_describing.get());
    }

    frame_ptr map_point::_get_frame(const feature_wptr& ob) {
        auto exist_ob = ob.lock();
        if (!exist_ob) { return nullptr; }
        return exist_ob->host_frame.lock();
    }

    /**
     * candidate set methods
     */ 

    void candidate_set::clear() {
        lock_t lock(_mutex_c);
        _candidates.clear();
    }

    bool candidate_set::add_candidate(const map_point_ptr& mp) {
        if (!mp || map_point::REMOVED == mp->type) { return false; }
        auto latest = mp->last_observation();
        if (!latest) { return false; }

        mp->type = map_point::CANDIDATE;
        lock_t lock(_mutex_c);
        _candidates.emplace_back(mp, latest);
        return true;
    }

    bool candidate_set::remove_candidate(const map_point_ptr& mp) {
        if (!mp) { return false; }
        lock_t lock(_mutex_c);
        auto itr = _candidates.begin();
        while (itr != _candidates.end()) {
            if (itr->first == mp) {
                _destroy(*itr);
                _candidates.erase(itr);
                return true;
            }
            ++itr;
        }
        return false;
    }

    size_t candidate_set::extract_observed_by(const frame_ptr& frame) {
        lock_t lock(_mutex_c);
        size_t count = 0;
        auto itr = _candidates.begin();
        while (_candidates.end() != itr) {
            auto host = itr->first->last_observation()->host_frame;
            assert(!host.expired());
            if (host.lock() == frame) {
                itr->first->type = map_point::UNKNOWN;
                itr->first->n_fail_reproj = 0;
                itr->second->use();
                itr = _candidates.erase(itr);
                ++count;
            }
            else { ++itr; }
        }
        return count;
    }

    void candidate_set::remove_observed_by(const frame_ptr& frame) {
        if (!frame) { return; }
        lock_t lock(_mutex_c);
        auto itr = _candidates.begin();
        while (itr != _candidates.end()) {
            if (frame == itr->second->host_frame.lock()) {
                _destroy(*itr);
                itr = _candidates.erase(itr);
            }
            else { ++itr; }
        }
    }

    void candidate_set::_destroy(candidate_t& candidate) { 
        candidate.second.reset(); 
        candidate.first->as_removed();
        _trash_mps.push_back(candidate.first);
    }
}
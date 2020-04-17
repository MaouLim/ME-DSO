#include <vo/map_point.hpp>
#include <vo/jaccobian.hpp>
#include <vo/frame.hpp>
#include <vo/feature.hpp>

namespace vslam {

    int map_point_seed::_seed_seq = 0;

    map_point_seed::map_point_seed(
        int _gen_id, const feature_ptr& host, double d_mu, double d_min
    ) : generation_id(_gen_id), id(_seed_seq++), count_updates(0), host_feature(host), mu(1. / d_mu), 
        dinv_range(1. / d_min), sigma2(dinv_range * dinv_range / 36.), a(10.), b(10.) { }
    
    map_point_seed::~map_point_seed() {
#ifdef _ME_VSLAM_DEBUG_INFO_
        std::cout << "Map point seed: " << id 
                  << " live time: "     << live_time 
                  << std::endl;
#endif
    }

    map_point_ptr 
    map_point_seed::upgrade(const Sophus::SE3d& t_wc) const {
        Eigen::Vector3d xyz_unit = host_feature->xy1.normalized();
        Eigen::Vector3d xyz      = t_wc * (xyz_unit / mu);
        auto mp = utils::mk_vptr<map_point>(xyz);
        mp->set_observed_by(host_feature);
        assert(host_feature->describe_nothing());
        host_feature->map_point_describing = mp;
        return mp;
    }

    /**
     * map point methods
     */ 

    int map_point::_seq_id = 0;

    map_point::map_point(const Eigen::Vector3d& _pos) : 
        id(_seq_id++), position(_pos), n_obs(0), 
        last_pub_timestamp(0), last_proj_kf_id(-1), last_opt_timestamp(0), 
        n_fail_reproj(0), n_success_reproj(0), type(UNKNOWN) 
    { }

    void map_point::set_observed_by(const feature_ptr& _feat) {
        observations.emplace_front(_feat);
        ++n_obs;
    }

    feature_ptr map_point::last_observed() {
        auto itr = observations.begin();
        while (itr != observations.end()) {
            if (!itr->expired()) { return itr->lock(); }
            else { itr = observations.erase(itr); --n_obs; }
        }
        return nullptr;
    }

    feature_ptr map_point::find_observed(const frame_ptr& _frame) {
        assert(_frame);
        auto itr = observations.begin();
        while (itr != observations.end()) {
            auto exist_ob = itr->lock();
            if (!exist_ob) {  
                itr = observations.erase(itr); 
                --n_obs; 
                continue;
            }
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
            if (itr->expired()) { 
                itr = observations.erase(itr); 
                --n_obs;
                continue;
            }
            if (_frame == _get_frame(*itr)) {
                observations.erase(itr);
                --n_obs;
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
            auto exist_ob = itr->lock();
            if (!exist_ob) { 
                itr = observations.erase(itr); 
                --n_obs;
                continue; 
            }
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
        static const double converge_eps = 1e-10;

        Eigen::Vector3d old_pos = position;

        double last_chi2 = 0.0;
        Eigen::Matrix3d H;
        Eigen::Vector3d b;

        for (size_t i = 0; i < n_iterations; ++i) {
            
            double chi2 = 0.0;
            H.setZero(); b.setZero();

            for (auto& feature_observed : observations) {

                auto exist_ob = feature_observed.lock();
                if (!exist_ob) { continue; }
                auto host_frame = exist_ob->host_frame.lock();
                if (!host_frame) { continue; }

                Eigen::Vector3d p_c = host_frame->t_cw * position;
                Eigen::Matrix23d jacc   = -1.0 * jaccobian_dxy1dxyz(p_c, host_frame->t_cw.rotationMatrix());
                Eigen::Matrix32d jacc_t = jacc.transpose();
                Eigen::Vector2d p_xy1 = (p_c / p_c[2]).head<2>();
                Eigen::Vector2d err = exist_ob->xy1.head<2>() - p_xy1;
                H.noalias() +=  jacc_t * jacc;
                b.noalias() += -jacc_t * err;
                chi2 += err.squaredNorm();
            }

            Eigen::Vector3d delta = H.ldlt().solve(b);
            if (!std::isnan(delta[0])) { assert(false); return -1.; }

            if (0 < i && last_chi2 < chi2) {
#ifdef _ME_SLAM_DEBUG_INFO_
                std::cout << "loss increased, roll back." << std::endl;
#endif
                position = old_pos;
                break;
            }

            old_pos = position;
            position += delta;
            last_chi2 = chi2;

            if (delta.norm() < converge_eps) {
#ifdef _ME_SLAM_DEBUG_INFO_
                std::cout << "converged." << std::endl;
#endif 
                break;
            }
        }

        return last_chi2;
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
        auto latest = mp->last_observed();
        if (!latest) { return false; }

        mp->type = map_point::CANDIDATE;
        lock_t lock(_mutex_c);
        _candidates.emplace_back(mp, latest);
        return true;
    }

    bool candidate_set::remove_candidate(const map_point_ptr& mp) {
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

    bool candidate_set::extract_observed_by(const frame_ptr& frame) {
        // TODO
        lock_t lock(_mutex_c);
        auto itr = _candidates.begin();
        while (_candidates.end() != itr) {
            auto host = itr->first->last_observed()->host_frame;
            assert(!host.expired());
            if (host.lock() == frame) {
                itr->first->type = map_point::UNKNOWN;
                itr->first->n_fail_reproj = 0;
                assert(!itr->second->host_frame.expired());
                auto latest_when_created = itr->second->host_frame.lock();
                latest_when_created->add_feature(itr->second);
                itr = _candidates.erase(itr);
            }
            else { ++itr; }
        }
        return true;
    }

    void candidate_set::remove_observed_by(const frame_ptr& frame) {
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

    template <typename _Predicate>
    void candidate_set::for_each(_Predicate&& _pred) {
        lock_t lock(_mutex_c);
        for (auto& each : _candidates) { _pred(each); }
    }

    template <typename _Condition>
    size_t candidate_set::for_each_remove_if(_Condition&& cond) {
        size_t count_rm = 0;
        lock_t lock(_mutex_c);
        auto itr = _candidates.begin();
        while (itr != _candidates.end()) {
            if (cond(*itr)) {
                ++count_rm;
                _destroy(*itr);
                itr = _candidates.erase(itr);
            }
            else { ++itr; }
        }
        return count_rm;
    }

    void candidate_set::_destroy(candidate_t& candidate) { 
        candidate.second.reset(); 
        candidate.first->as_removed();
        _trash_mps.push_back(candidate.first);
    }
}
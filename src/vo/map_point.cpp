#include <vo/map_point.hpp>
#include <vo/jaccobian.hpp>
#include <vo/frame.hpp>
#include <vo/feature.hpp>

namespace vslam {

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

    feature_ptr map_point::find_observed(const frame_ptr& _frame) const {
        assert(_frame);
        for (auto& each_ob : observations) {
            auto exist_ob = each_ob.lock();
            if (!exist_ob) { continue; }
            if (_frame == exist_ob->host_frame.lock()) {
                return exist_ob;
            }
        }
        return nullptr;
    }

    bool map_point::remove_observed_by(const frame_ptr& _frame) {
        assert(_frame);
        auto itr = observations.begin();
        while (itr != observations.end()) {
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
    ) const {
        Eigen::Vector3d view_orien = _cam_center - position;
        view_orien.normalize();

        double max_cos   = 0.0;
        feature_ptr res = nullptr;

        auto itr = observations.begin();
        while (itr != observations.end()) {
            auto exist_ob = itr->lock();
            if (!exist_ob) { ++itr; continue; }
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
                Matrix23d jacc   = -1.0 * jaccobian_dxy1dxyz(p_c, host_frame->t_cw.rotationMatrix());
                Matrix32d jacc_t = jacc.transpose();
                Eigen::Vector2d p_xy1 = (p_c / p_c[2]).head<2>();
                Eigen::Vector2d err = exist_ob->xy1.head<2>() - p_xy1;
                H.noalias() +=  jacc_t * jacc;
                b.noalias() += -jacc_t * err;
                chi2 += err.squaredNorm();
            }

            Eigen::Vector3d delta = H.ldlt().solve(b);
            if (!std::isnan(delta[0])) { assert(false); return; }

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

    inline frame_ptr map_point::_get_frame(const feature_wptr& ob) {
        auto exist_ob = ob.lock();
        if (!exist_ob) { return nullptr; }
        return exist_ob->host_frame.lock();
    }
}
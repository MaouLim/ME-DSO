#include <vo/depth_filter.hpp>
#include <vo/frame.hpp>
#include <vo/feature.hpp>
#include <vo/camera.hpp>
#include <vo/map_point.hpp>
#include <vo/matcher.hpp>

#include <utils/config.hpp>
#include <utils/utils.hpp>

namespace vslam {

    depth_filter::depth_filter(const converged_callback& _cb) : 
        base_type(max_queue_sz), 
        _callback(_cb), _count_key_frames(0)
    {
        _detector = utils::mk_vptr<fast_detector>(
            config::height, config::width, config::cell_sz, config::pyr_levels
        );
        add_handler(std::bind(&depth_filter::_handle_param, this, std::placeholders::_1));
    }

    depth_filter::depth_filter(
        const detector_ptr& _det, const converged_callback& _cb
    ) : base_type(max_queue_sz), 
        _detector(_det), _callback(_cb), _count_key_frames(0) 
    { 
        add_handler(std::bind(&depth_filter::_handle_param, this, std::placeholders::_1));
    }

    bool depth_filter::commit(const param_type& param) {
        if (param.frame->key_frame) {
            _new_key_frame.do_and_exchange_if(false, [&]() {
                _queue.force_to_push(param);
            });
        }
        else { 
            _new_key_frame.do_if_else(
                true, 
                [&]() { _queue.wait_and_push(param); },
                [&]() { _queue.force_to_push(param); }
            );
        }
        return true;
    }

    void depth_filter::_handle_param(param_type& param) {
        if (param.frame->key_frame) {
            _queue.clear();
            _initialize_seeds(param.frame);
            _new_key_frame.reset();
            ++_count_key_frames;
        }
        _update_seeds(param.frame);
    }

    void depth_filter::_initialize_seeds(const frame_ptr& kf) {
        
        feature_set features;
        _detector->set_grid_occupied(kf->features);
        size_t n_seeds = _detector->detect(kf, config::min_corner_score, features);
#ifdef _ME_VSLAM_DEBUG_INFO_
        std::cout << "[DF]" << "Num of seeds created: " << n_seeds << std::endl;
#endif
        double d_min = 0., d_median = 0.;
        min_and_median_depth_of_frame(kf, d_min,  d_median);

        lock_t lock(_mutex_seeds);
        for (const auto& each_feat : features) {
            _seeds.emplace_back(_count_key_frames, each_feat, d_median, d_min);
        }
    }

    void depth_filter::_update_seeds(const frame_ptr& frame) {

        lock_t lock(_mutex_seeds);

        auto itr = _seeds.begin();
        while (itr != _seeds.end()) {
            auto callbable = std::bind(
                &depth_filter::_handle_seed_itr, this, std::cref(frame), std::ref(itr)
            );
            _new_key_frame.do_if(false, callbable);
        }
    }

    void depth_filter::_handle_seed_itr(
        const frame_ptr& cur, seed_iterator& itr
    ) {
        if (
            config::max_seed_lifetime < _count_key_frames - itr->generation_id
        ) {
            itr = _seeds.erase(itr);
            return;
        }

        auto ref = itr->host_feature->host_frame.lock();
        if (!ref) { assert(false); return; }

        Eigen::Vector3d xyz_unit_ref = itr->host_feature->xy1.normalized(); 
        Eigen::Vector3d xyz_ref = xyz_unit_ref / itr->mu;
        Eigen::Vector3d xyz_world = ref->t_wc * xyz_ref;
        if (!cur->visible(xyz_world)) { ++itr; return; }

        double dinv_max = itr->mu + std::sqrt(itr->sigma2);
        double dinv_min = std::max(itr->mu - std::sqrt(itr->sigma2), CONST_EPS);
        double depth_est = 1. / itr->mu;
        Eigen::Vector2d uv_matched;
        if (!_matcher->match_epipolar_search(
                ref, cur, itr->host_feature,
                1. / dinv_max, 1. / dinv_min,  depth_est
            )
        ) {
            itr->b += 1.; ++itr;
            return;
        }

        double focal_len = cur->camera->err_mul2();
        Eigen::Vector3d trans_cr = (ref->t_cw * cur->t_wc /* t_rc */).translation();
        // tau: depth uncertainty
        double tau = utils::calc_depth_cov(xyz_ref, trans_cr, focal_len);
        // tau_inv: inversed depth uncertainty
        double tau_inv = 0.5 * (1.0 / std::max(depth_est - tau, CONST_EPS) - 1.0 / (depth_est + tau));

        _update(1. / depth_est, tau * tau, *itr);
        ++itr->count_updates;

        if (cur->key_frame) {
            _detector->set_grid_occupied(uv_matched);
        }

        if (itr->converged()) {
            map_point_ptr new_mp = itr->upgrade(ref->t_wc);
            _callback(new_mp, itr->sigma2);
            itr = _seeds.erase(itr);
            return;
        }

        if (std::isnan(dinv_max)) {
            itr = _seeds.erase(itr);
            return;
        }

        ++itr;
    }

    void depth_filter::_update(
        double x, double tau2, map_point_seed& seed
    ) {
        double sig2 = sqrt(tau2 + seed.sigma2);
        assert(!std::isnan(sig2));

        //std::normal_distribution<double> norm_dist(seed.mu, sig2);
        double s2 = 1.0 / (1.0 / seed.sigma2 + 1.0 / tau2);
        double m = s2 * (seed.mu / seed.sigma2 + x / tau2);

        double C1 = seed.a / (seed.a + seed.b) * utils::normal_pdf(seed.mu, std::sqrt(sig2), x);
        double C2 = seed.b / (seed.a + seed.b) / seed.dinv_range;
        double norm_factor = C1 + C2;

        C1 /= norm_factor;
        C2 /= norm_factor;

        double f = C1 * (seed.a + 1.) / (seed.a + seed.b + 1.) + 
                   C2 * seed.a / (seed.a + seed.b + 1.);
        double e = C1 * (seed.a + 1.) * (seed.a + 2.) / ((seed.a + seed.b + 1.) * (seed.a + seed.b + 2.)) + 
                   C2 * seed.a * (seed.a + 1.) / ((seed.a + seed.b + 1.) * (seed.a + seed.b + 2.));

        // update parameters
        double new_mu = C1 * m + C2 * seed.mu;
        seed.sigma2 = C1 * (s2 + m * m) + C2 * (seed.sigma2 + seed.mu * seed.mu) - new_mu * new_mu;
        seed.mu = new_mu;
        seed.a = (e - f) / (f - e / f);
        seed.b = seed.a * (1. - f) / f;
    }

} // namespace vslam

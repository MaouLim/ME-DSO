#include <vo/depth_filter.hpp>
#include <vo/frame.hpp>
#include <vo/feature.hpp>
#include <vo/camera.hpp>
#include <vo/map_point.hpp>
#include <utils/utils.hpp>
#include <utils/config.hpp>

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

    const size_t depth_filter::max_queue_sz = 
        config::get<int>("max_queue_sz");

    const double depth_filter::min_corner_score = 
        config::get<double>("min_corner_score");

    const size_t depth_filter::max_seed_lifetime = 
        config::get<int>("max_seed_lifetime");

    depth_filter::depth_filter(
        const detector_ptr& _det, const result_handler& _callable
    ) : base_type(max_queue_sz) , _detector(_det), _count_key_frames(0) {
        set_result_handler(_callable);
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
    }

    depth_filter::result_type 
    depth_filter::process(param_type& param) {
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
        _detector->detect(kf, min_corner_score, features);

        double d_min = 0., d_median = 0.;
        kf->min_and_median_depth(d_min,  d_median);

        lock_t lock(_mutex_seeds);
        for (const auto& each_feat : features) {
            _seeds.emplace_back(_count_key_frames, each_feat, d_median, d_min);
        }
    }

    void depth_filter::_update_seeds(const frame_ptr& frame) {

        lock_t lock(_mutex_seeds);

        auto itr = _seeds.begin();
        while (itr != _seeds.end()) {
            _new_key_frame.do_if(
                false, &_handle_seed_itr, this, std::cref(frame), std::ref(itr)
            );
        }
    }

    void depth_filter::_handle_seed_itr(
        const frame_ptr& cur, seed_iterator& itr
    ) {
        if (
            max_seed_lifetime < _count_key_frames - itr->generation_id
        ) {
            itr = _seeds.erase(itr);
            return;
        }

        auto ref = itr->host_feature->host_frame.lock();
        if (!ref) { assert(false); return; }

        Eigen::Vector3d xyz_ref = itr->host_feature->xy1 / itr->mu;
        Eigen::Vector3d xyz_world = ref->t_wc * xyz_ref;
        if (!cur->visible(xyz_world)) {
            ++itr; return;
        }

        double dinv_max = itr->mu + std::sqrt(itr->sigma2);
        double dinv_min = std::max(itr->mu - std::sqrt(itr->sigma2), CONST_EPS);
        double d = 0.;
        Eigen::Vector2d uv_matched;
        if (!_matcher->epipolar_search(/* TODO */uv_matched)) {
            itr->b += 1.; ++itr;
            return;
        }

        double focal_len = cur->camera->err_mul2();
        Eigen::Vector3d trans_cr = (ref->t_cw * cur->t_wc /* t_rc */).translation();
        // tau: depth uncertainty
        double tau = utils::calc_depth_cov(xyz_ref, trans_cr, focal_len);
        // tau_inv: inversed depth uncertainty
        double tau_inv = 0.5 * (1.0 / std::max(d - tau, CONST_EPS) - 1.0 / (d + tau));

        _update(1. / d, tau * tau, *itr);
        ++itr->count_updates;

        if (cur->key_frame) {
            _detector->set_grid_occupied(uv_matched);
        }

        if (itr->converged(/**/)) {
            assert(itr->host_feature->describe_nothing());
            auto new_mp = utils::mk_vptr<map_point>(xyz_world);
            new_mp->set_observed_by(itr->host_feature);
            itr->host_feature->map_point_describing = new_mp;

            _handler(_df_result_msg(new_mp, itr->sigma2));
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

        std::normal_distribution<double> norm_dist(seed.mu, sig2);
        double s2 = 1.0 / (1.0 / seed.sigma2 + 1.0 / tau2);
        double m = s2 * (seed.mu / seed.sigma2 + x / tau2);

        double C1 = seed.a / (seed.a + seed.b) * norm_dist(x);
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

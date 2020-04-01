#ifndef _ME_VSLAM_DEPTH_FILTER_HPP_
#define _ME_VSLAM_DEPTH_FILTER_HPP_

#include <common.hpp>

namespace vslam {

    /**
     * @brief implement the depth filter used by SVO, 
     *        ref 'Video-based, Real-Time Multi View Stereo'
     */

    struct depth_filter {
        using ptr = std::shared_ptr<depth_filter>;

        void update_some() const;
        void update(double x, double tau2, depth_info& seed) const;
        double calc_tau2();
    };

    inline void depth_filter::update(
        double x, double tau2, depth_info& seed
    ) const {
        double sig2 = sqrt(tau2 + seed.sigma2);
        assert(!std::isnan(sig2));

        std::normal_distribution<double> norm_dist(seed.mu, sig2);
        double s2 = 1.0 / (1.0 / seed.sigma2 + 1.0 / tau2);
        double m = s2 * (seed.mu / seed.sigma2 + x / tau2);

        double C1 = seed.a / (seed.a + seed.b) * norm_dist(x);
        double C2 = seed.b / (seed.a + seed.b) / seed.range;
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
}

#endif
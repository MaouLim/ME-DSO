#include <vo/depth_filter.hpp>

namespace vslam {

    int map_point_seed::_batch_seq = 0;
    int map_point_seed::_seed_seq  = 0;

    map_point_seed::map_point_seed(
        const feature_ptr& host, double d_mu, double d_min
    ) : batch_id(_batch_seq), id(_seed_seq++), live_time(0), host_feature(host), mu(1. / d_mu), 
       dinv_range(1. / d_min), sigma2(dinv_range * dinv_range / 36.), a(10.), b(10.) { }
    
    map_point_seed::~map_point_seed() {
#ifdef _ME_VSLAM_DEBUG_INFO_
        std::cout << "Map point seed: " << id 
                  << " live time: "     << live_time 
                  << std::endl;
#endif
    }

    

} // namespace vslam

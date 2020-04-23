#include <common.hpp>

namespace config {

    /**
     * @global_var
     */ 
    const int    height                  = 480;
    const int    width                   = 640;
    const double fx                      = 517.3;
    const double fy                      = 516.5;
    const double cx                      = 325.1;
    const double cy                      = 249.7;
    const double k1                      = 0.0;
    const double k2                      = 0.0;
    const double k3                      = 0.0;

    const int    pyr_levels              = 5;
    const int    cell_sz                 = 10;
    const int    max_opt_iterations      = 10;

    const double max_reproj_err          = 0.7 / (fx + fy);

    const int    min_features_in_first   = 500;
    const int    min_features_to_tracked = 450;
    const double min_init_shift          = 30.0;
    const int    min_inliers             = 400;
    const int    cv_lk_win_sz            = 30;
    const double init_scale              = 1.0;
 
    const double min_corner_score        = 500.0;
    
} // namespace config

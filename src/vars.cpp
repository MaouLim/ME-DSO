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

    const double max_reproj_err_uv       = 0.4;
    const double max_reproj_err_xy1      = max_reproj_err_uv * 2. / (fx + fy);

    const int    min_features_in_first   = 400;
    const int    min_features_to_tracked = 250;
    const double min_init_shift          = 30.0;
    const int    min_inliers             = 100;
    const double min_inlier_ratio        = 0.75;
    const int    cv_lk_win_sz            = 21;
    const double init_scale              = 1.0;
 
    const double min_corner_score        = 300.0;

    const double min_epipolar_search_ncc    = 0.85;
    const int    max_epipolar_search_steps  = 80;
    const double max_angle_between_epi_grad = M_PI / 4;
    const double min_len_to_epipolar_search = 2.0;
    const double epipolar_search_step       = CONST_COS_45;
    
} // namespace config

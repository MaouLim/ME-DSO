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
    const int    max_epipolar_search_steps  = 1000;
    const double max_angle_between_epi_grad = M_PI / 4;
    const double min_len_to_epipolar_search = 2.0;
    const double epipolar_search_step       = CONST_COS_45;

    /**
     * @note constants for core system
     */ 
    const int    min_reproj_mps       = 200;
    const int    max_mps_to_local_opt = 100;
    const double max_drop_ratio       = 0.4;


    const int    max_global_map_frames = 500;
    const int    max_local_map_frames  = 10;
    const double min_key_frame_shift_x = 0.3;
    const double min_key_frame_shift_y = 0.25;
    const double min_key_frame_shift_z = 0.4;

    const int    max_seed_lifetime     = 10;

    const int    max_mps_to_reproj            = 800;
    const int    max_overlaped_key_frames     = 10;
    const int    max_candidate_mp_fail_reproj = 30;
    const int    max_unknown_mp_fail_reproj   = 15;
    const int    min_good_mp_success_reproj   = 10;
} // namespace config

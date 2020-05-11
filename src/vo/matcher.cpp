#include <vo/matcher.hpp>
#include <vo/camera.hpp>
#include <vo/feature.hpp>
#include <vo/map_point.hpp>
#include <vo/frame.hpp>

#include <utils/config.hpp>
#include <utils/utils.hpp>
#include <utils/diff.hpp>

namespace vslam {

    bool patch_matcher::match_direct(
        const map_point_ptr&   mp, 
        const frame_ptr&       cur, 
        const Eigen::Vector2d& uv_cur,
        feature_ptr&           candidate
    ) {
        auto feat_ref = mp->find_closest_observed(cur->cam_center());
        if (!feat_ref) { return false; }

        auto ref = feat_ref->host_frame.lock();
        assert(ref);
        if (!ref->visible(
                feat_ref->uv, 
                _check_sz, 
                feat_ref->level
            )
        ) { return false; }

        Eigen::Vector3d xyz_ref = ref->t_cw * mp->position;
        Eigen::Matrix2d affine_cr = affine::affine_mat(
            ref->camera, feat_ref->uv, feat_ref->level, 
            xyz_ref.z(), _check_sz, cur->t_cw * ref->t_wc
        );
#ifdef _ME_VSLAM_DEBUG_INFO_MATCHER_
        std::cout << "[MACTHER]"
                  << "Affine transformation from frame[" 
                  << ref->id << "] to frame[" << cur->id << "]:" << std::endl;
        std::cout << "ref uv: " << feat_ref->uv.transpose() << std::endl;
        std::cout << "cur uv: " << uv_cur.transpose() << std::endl;
        std::cout << affine_cr << std::endl;
#endif
        size_t max_level = config::pyr_levels - 1; assert(0 < config::pyr_levels);
        size_t search_level = affine::search_best_level(affine_cr, max_level);

        affine::extract_patch_affine(
            ref->pyramid[feat_ref->level], feat_ref->uv, feat_ref->level, 
            affine_cr.inverse(), search_level, _patch
        );

        bool success = false;
        double scale = (1 << search_level);
        Eigen::Vector2d uv_leveln = uv_cur / scale;
        if (feature::EDGELET == feat_ref->type) {
            Eigen::Vector2d grad_orien_cur = 
                (affine_cr * feat_ref->grad_orien).normalized();
            success = alignment::align1d(
                cur->pyramid[search_level], grad_orien_cur, _patch, 
                config::max_opt_iterations, uv_leveln
            );
            if (success) {
                candidate.reset(new feature(cur, uv_leveln * scale, search_level));
                candidate->type = feature::EDGELET;
                candidate->grad_orien = grad_orien_cur;
                candidate->set_describing(mp);
            }
        }
        else if (feature::CORNER == feat_ref->type) {
            success = alignment::align2d(
                cur->pyramid[search_level], _patch, 
                config::max_opt_iterations, uv_leveln
            );
            if (success) {
                candidate.reset(new feature(cur, uv_leveln * scale, search_level));
                candidate->type = feature::CORNER;
                candidate->set_describing(mp);
            }
        }
        else { assert(false); }
        return success;
    }

    bool patch_matcher::match_epipolar_search(
        const frame_ptr&   ref,
        const frame_ptr&   cur,
        const feature_ptr& feat_ref,
        double             depth_min,
        double             depth_max,
        double&            depth_est,
        Eigen::Vector2d&   uv_macthed
    ) {
        Sophus::SE3d t_cr = cur->t_cw * ref->t_wc;
        //double best_zmssd = 1e10 /* threshold */;

        // unit vector point to the 3d point
        Eigen::Vector3d xyz_unit_ref = feat_ref->xy1.normalized();

        Eigen::Vector3d xyz_min_cur = t_cr * (depth_min * xyz_unit_ref);
        Eigen::Vector3d xyz_max_cur = t_cr * (depth_max * xyz_unit_ref);

        Eigen::Vector2d xy_min_cur = utils::project(xyz_min_cur);
        Eigen::Vector2d xy_max_cur = utils::project(xyz_max_cur);
        Eigen::Vector2d uv_min_cur = cur->camera->cam2pixel(xyz_min_cur);
        Eigen::Vector2d uv_max_cur = cur->camera->cam2pixel(xyz_max_cur);

        Eigen::Vector2d epipolar_pixel_plane = uv_max_cur - uv_min_cur;
        Eigen::Vector2d epipolar_unit_plane  = xy_max_cur - xy_min_cur;
        double epipolar_pixel_len = epipolar_pixel_plane.norm();
        Eigen::Vector2d epipolar_orien = epipolar_unit_plane.normalized();

        Eigen::Matrix2d affine_cr = affine::affine_mat(
            ref->camera, feat_ref->uv, feat_ref->level, 
            xyz_unit_ref[2] * depth_est, _check_sz, t_cr
        );

        if (feature::EDGELET == feat_ref->type && edgelet_filtering) {
            Eigen::Vector2d grad_orien_cur = 
                (affine_cr * feat_ref->grad_orien).normalized();
            double cos_angle = std::abs(grad_orien_cur.dot(epipolar_orien));
            if (cos_angle < std::cos(config::max_angle_between_epi_grad)) { return false; }
        }
        size_t max_level = config::pyr_levels - 1;
        size_t level_cur = affine::search_best_level(affine_cr, max_level);

        affine::extract_patch_affine(
            ref->pyramid[feat_ref->level], feat_ref->uv, feat_ref->level, 
            affine_cr.inverse(), level_cur, _patch
        );

        double scale = (1 << level_cur);

        if (epipolar_pixel_len < config::min_len_to_epipolar_search) {
            Eigen::Vector2d uv_cur = (uv_min_cur + uv_max_cur) * 0.5;
            Eigen::Vector2d uv_leveln_cur = uv_cur / scale;
            bool success = false;
            if (use_alignment_1d) {
                success = alignment::align1d(
                    cur->pyramid[level_cur], epipolar_pixel_plane.normalized(), 
                    _patch, config::max_opt_iterations, uv_leveln_cur
                );
            }
            else {
                success = alignment::align2d(
                    cur->pyramid[level_cur], _patch, 
                    config::max_opt_iterations, uv_leveln_cur
                );
            }
            if (!success) { return false; }
            uv_cur = uv_leveln_cur * scale;
            double depth_cur = 0.;
            bool ret = utils::depth_from_triangulate_v2(
                cur->camera->pixel2cam_unit(uv_cur), 
                xyz_unit_ref, t_cr, depth_cur, depth_est
            );

            if (ret) { uv_macthed = uv_cur; }
            return ret;
        }

        size_t n_steps = epipolar_pixel_len / config::epipolar_search_step;
        Eigen::Vector2d step = epipolar_unit_plane / n_steps;
        if (config::max_epipolar_search_steps < n_steps) { 
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cout << "[MACTHER]"
                      << "The epipolar is too long: " 
                      << epipolar_pixel_len 
                      << std::endl;
#endif
            return false; 
        }

        // epipolar search
        //double best_ssd = std::numeric_limits<double>::max();
        double best_ncc = 0.;
        Eigen::Vector2d best_xy;
        Eigen::Vector2d xy = xy_min_cur + step;
        Eigen::Vector2i last_uv(0, 0);
        const cv::Mat& img_leveln = cur->pyramid[level_cur];
        
        for (size_t i = 0; i < n_steps; ++i, xy += step) {

            Eigen::Vector2d uv = cur->camera->cam2pixel(utils::homogenize(xy));
            Eigen::Vector2i uv_leveln = (uv / scale).cast<int>();

            if (last_uv == uv_leveln) { continue; }
            last_uv = uv_leveln;
            
            if (!utils::in_image(
                    img_leveln, 
                    uv_leveln.x(), uv_leveln.y(), 
                    patch_t::half_sz
                )
            ) { continue; }

            const int cur_stride = img_leveln.step.p[0];
            uint8_t* patch_cur = 
                img_leveln.data + (uv_leveln.y() - patch_t::half_sz) * cur_stride + uv_leveln.x() - patch_t::half_sz;
            double ncc = 
                utils::diff_2d<uint8_t>::zm_ncc(
                    _patch.start(), patch_t::stride(), 
                    patch_cur,      cur_stride
                );
            if (best_ncc < ncc) {
                best_ncc = ncc;
                best_xy = xy;
            }
        }

        if (best_ncc < config::min_epipolar_search_ncc) { 
#ifdef _ME_VSLAM_DEBUG_INFO_MATCHER_
            std::cout << "[MACTHER]"
                      << "No similar patch."
                      << "Best Zero-Mean-NCC: " << best_ncc << std::endl;
#endif
            return false;
        }

        // subpixel refinement
        Eigen::Vector2d best_uv = cur->camera->cam2pixel(utils::homogenize(best_xy));

#ifdef _ME_VSLAM_DEBUG_INFO_
//#define _ME_VSLAM_DEBUG_DRAW_EPIPOLAR_ 1
#ifdef _ME_VSLAM_DEBUG_DRAW_EPIPOLAR_
        cv::Mat f0, f1;
        cv::cvtColor(ref->image(), f0, cv::COLOR_GRAY2BGR);
        cv::cvtColor(cur->image(), f1, cv::COLOR_GRAY2BGR);
        cv::Point2f p  = { feat_ref->uv.x(), feat_ref->uv.y() };
        cv::Point2f p0 = {   uv_min_cur.x(),   uv_min_cur.y() };
        cv::Point2f p1 = {   uv_max_cur.x(),   uv_max_cur.y() };
        cv::Point2f c  = {      best_uv.x(),      best_uv.y() };
        cv::line(f1, p0, p1, { 0, 0, 255 }, 1);
        cv::circle(f0, p, 2, { 0, 255, 255 }, 2);
        cv::circle(f1, c, 2, { 0, 255, 255 }, 2);
        cv::Mat f_concat;
        cv::hconcat(f0, f1, f_concat);
        cv::imshow("vis epipolar search", f_concat);
        cv::waitKey();
#endif
#endif

        Eigen::Vector2d best_uv_leveln = best_uv / scale;
        bool success = false;
        if (use_alignment_1d) {
            success = alignment::align1d(
                cur->pyramid[level_cur], epipolar_pixel_plane.normalized(), 
                _patch, config::max_opt_iterations, best_uv_leveln
            );
        }
        else {
            success = alignment::align2d(
                cur->pyramid[level_cur], _patch, 
                config::max_opt_iterations, best_uv_leveln
            );
        }
        if (!success) { return false; }
        Eigen::Vector2d uv_refined = best_uv_leveln * scale;
        double depth_cur = 0.;
        bool ret = utils::depth_from_triangulate_v2(
            cur->camera->pixel2cam_unit(uv_refined), 
            xyz_unit_ref, t_cr, depth_cur, depth_est
        );
        if (ret) { uv_macthed = uv_refined; }
        return ret;
    }
    
    Eigen::Matrix2d affine::affine_mat(
        const camera_ptr&      camera, 
        const Eigen::Vector2d& uv_ref,
        size_t                 level_ref,
        double                 z_est,
        int                    patch_half_sz,
        const Sophus::SE3d&    t_cr
    ) {
        //const int half_sz = patch_half_sz + border_sz;
        const int scale = (1 << level_ref); 
        Eigen::Vector2d uv_plus_du = uv_ref + Eigen::Vector2d(patch_half_sz * scale, 0.);
        Eigen::Vector2d uv_plus_dv = uv_ref + Eigen::Vector2d(0., patch_half_sz * scale);

        Eigen::Vector3d xyz = camera->pixel2cam(uv_ref, z_est);
        Eigen::Vector3d xyz_plus_du =  camera->pixel2cam(uv_plus_du, z_est);
        Eigen::Vector3d xyz_plus_dv =  camera->pixel2cam(uv_plus_dv, z_est);

        auto uv_cur = camera->world2pixel(xyz, t_cr);
        auto uv_plus_du_cur = camera->world2pixel(xyz_plus_du, t_cr);
        auto uv_plus_dv_cur = camera->world2pixel(xyz_plus_dv, t_cr);

        Eigen::Matrix2d res;
        res.col(0) = (uv_plus_du_cur - uv_cur) / patch_half_sz;
        res.col(1) = (uv_plus_dv_cur - uv_cur) / patch_half_sz;

        return res;
    }

    size_t 
    affine::search_best_level(
        const Eigen::Matrix2d& affine_mat, size_t max_level
    ) {
        double determinant = affine_mat.determinant();
        if (determinant <= 3.) { return 0; }
        size_t level = size_t(std::log2(determinant / 3.) / 2.) + 1;
        return max_level < level ? max_level : level;
    }

    bool affine::extract_patch_affine(
        const cv::Mat&         image_leveln_ref,
        const Eigen::Vector2d& uv_level0_ref,
        size_t                 level_ref,
        const Eigen::Matrix2d& affine_rc, 
        size_t                 search_level,
        patch_t&               patch
    ) {
        if (affine_rc.hasNaN()) { return false; }

        constexpr int half = patch_t::size_with_border / 2;
        uint8_t* ptr = patch.data;
        Eigen::Vector2d center_ref = uv_level0_ref / double(1 << level_ref);

        for (int y = 0; y < patch_t::size_with_border; ++y) {
            for (int x = 0; x < patch_t::size_with_border; ++x) {
                Eigen::Vector2d uv_cur(x - half, y - half);
                uv_cur *= double(1 << search_level);
                // uv: uv at leveln
                Eigen::Vector2d uv = affine_rc * uv_cur + center_ref;
                if (!utils::in_image(image_leveln_ref, uv[0], uv[1])) { *ptr = 0; }
                else {
                    *ptr = utils::bilinear_interoplate<uint8_t>(image_leveln_ref, uv[0], uv[1]);
                }
                ++ptr;
            }
        }
        return true;
    }

    bool alignment::align1d(
        const cv::Mat&         img_cur, 
        const Eigen::Vector2d& orien, 
        const patch_t&         patch, 
        size_t                 n_iterations,
        Eigen::Vector2d&       uv_cur
    ) {
        Eigen::Vector2d search_dir = orien.normalized();
        // derivative along the orien
        double __attribute__((__aligned__(16))) patch_dl_ref[patch_t::area];

        Eigen::Matrix2d hessian = Eigen::Matrix2d::Zero();

        double* dl_ptr = patch_dl_ref;
        const int ref_stride = patch_t::stride();
        const uint8_t* ref_ptr = patch.start();
        
        for (int r = 0; r < patch_t::size; ++r) {
            for (int c = 0; c < patch_t::size; ++c) {
                double dx = (         double(ref_ptr[1]) -          double(ref_ptr[-1])) / 2.;
                double dy = (double(ref_ptr[ref_stride]) - double(ref_ptr[-ref_stride])) / 2.;
                double dl = Eigen::Vector2d(dx, dy).dot(search_dir);

                *dl_ptr = dl;
                Eigen::Vector2d jacc;
                jacc << dl, 1.;
                hessian += jacc * jacc.transpose();

                ++dl_ptr;
                ++ref_ptr;
            }
            ref_ptr += (patch_t::border_sz * 2);
        }

        const int cur_stride = img_cur.step.p[0];

        double intensity_mean_diff = 0.;
        double u = uv_cur.x(), v = uv_cur.y();

        double last_chi2 = std::numeric_limits<double>::max();
        double last_u = u, last_v = v;
        double last_intensity_mean_diff = 0.;
        bool converged = false;

        for (size_t i = 0; i < n_iterations; ++i) {

            if (!utils::in_image(img_cur, u, v, patch_t::half_sz)) { break; }

            int x = std::floor(u), y = std::floor(v);
            double dx = u - x, dy = v - y;

            double w00 = (1. - dx) * (1. - dy);
            double w01 = dx * (1. - dy);
            double w10 = (1. - dx) * dy;
            double w11 = dx * dy;

            uint8_t* cur_ptr = 
                img_cur.data + (y - patch_t::half_sz) * cur_stride + (x - patch_t::half_sz);
            ref_ptr = patch.start();
            dl_ptr = patch_dl_ref;

            double chi2 = 0.0;
            Eigen::Vector2d b = Eigen::Vector2d::Zero();
            
            for (int r = 0; r < patch_t::size; ++r) {
                for (int c = 0; c < patch_t::size; ++c) {
                    double intensity_cur = 
                        w00 * cur_ptr[0] + 
                        w01 * cur_ptr[1] + 
                        w10 * cur_ptr[cur_stride] + 
                        w11 * cur_ptr[cur_stride + 1];
                    // res = I(W(x;p)) - T(x)
                    double err = intensity_cur - *ref_ptr + intensity_mean_diff;
                    chi2 += 0.5 * err * err;

                    // jres = sum(dTdW*dWdp * res)
                    b[0] += err * (*dl_ptr);
                    b[1] += err * 1.;

                    ++dl_ptr;
                    ++ref_ptr; ++cur_ptr;
                }
                cur_ptr += (cur_stride - patch_t::size);
                ref_ptr += (patch_t::border_sz * 2);
            }

            Eigen::Vector2d delta = hessian.ldlt().solve(b);
            if (delta.hasNaN()) { assert(false); return false; }
            if (0 < i && last_chi2 < chi2) { 
#ifdef _ME_VSLAM_DEBUG_INFO_OPT_
                std::cout << "[ALIGN1D]" << "Loss increased at " << i << std::endl;
#endif
#ifdef _ME_VSLAM_ALIGN_CONSERVATIVE_
                u = last_u; v = last_v;  
                intensity_mean_diff = last_intensity_mean_diff;
                break; 
#endif
            }

            last_u = u;
            last_v = v;
            last_intensity_mean_diff = intensity_mean_diff;
            last_chi2 = chi2;
            
            // p := p - delta_p
            u -= delta[0] * search_dir[0];
            v -= delta[0] * search_dir[1];
            intensity_mean_diff -= delta[1];

            if (std::abs(delta[0]) < config::opt_converged_thresh_uv) {
#ifdef _ME_VSLAM_DEBUG_INFO_OPT_
                std::cout << "[ALIGN1D]" << "Converged at " << i << std::endl;
#endif
                converged = true;
                break;
            }
        }

        uv_cur << u, v;
        return converged;
    }

    bool alignment::align2d(
        const cv::Mat&   img_cur, 
        const patch_t&   patch, 
        size_t           n_iterations,
        Eigen::Vector2d& uv_cur
    ) {
        double __attribute__((__aligned__(16))) patch_dx_ref[patch_t::area];
        double __attribute__((__aligned__(16))) patch_dy_ref[patch_t::area];

        Eigen::Matrix3d hessian = Eigen::Matrix3d::Zero();

        double* dx_ptr = patch_dx_ref;
        double* dy_ptr = patch_dy_ref;

        const int ref_stride = patch_t::stride();
        const uint8_t* ref_ptr = patch.start();

        for (int r = 0; r < patch_t::size; ++r) {
            for (int c = 0; c < patch_t::size; ++c) {
                *dx_ptr = (         double(ref_ptr[1]) -          double(ref_ptr[-1])) / 2.;
                *dy_ptr = (double(ref_ptr[ref_stride]) - double(ref_ptr[-ref_stride])) / 2.;
                Eigen::Vector3d jacc;
                jacc << *dx_ptr, *dy_ptr, 1.;
                hessian += jacc * jacc.transpose();
                ++dx_ptr; ++dy_ptr; 
                ++ref_ptr;
            }
            ref_ptr += (patch_t::border_sz * 2);
        }

        const int cur_stride = img_cur.step.p[0];
        double u = uv_cur.x(), v = uv_cur.y();
        double intensity_mean_diff = 0.;

        double last_chi2 = std::numeric_limits<double>::max();
        double last_u = uv_cur.x(), last_v = uv_cur.y(); 
        double last_intensity_mean_diff = 0.;
        bool converged = false;

        for (size_t i = 0; i < n_iterations; ++i) {

            if (!utils::in_image(img_cur, u, v, patch_t::half_sz)) { break; }

            int x = std::floor(u), y = std::floor(v);
            double dx = u - x, dy = v - y;

            double w00 = (1. - dx) * (1. - dy);
            double w01 = dx * (1. - dy);
            double w10 = (1. - dx) * dy;
            double w11 = dx * dy;

            uint8_t* cur_ptr = 
                img_cur.data + (y - patch_t::half_sz) * cur_stride + (x - patch_t::half_sz);
            ref_ptr = patch.start();
            dx_ptr = patch_dx_ref;
            dy_ptr = patch_dy_ref;

            double chi2 = 0.0;
            Eigen::Vector3d b = Eigen::Vector3d::Zero();
            
            for (int r = 0; r < patch_t::size; ++r) {
                for (int c = 0; c < patch_t::size; ++c) {
                    double intensity_cur = 
                        w00 * cur_ptr[0] + 
                        w01 * cur_ptr[1] + 
                        w10 * cur_ptr[cur_stride] + 
                        w11 * cur_ptr[cur_stride + 1];
                    double err = intensity_cur - *ref_ptr + intensity_mean_diff;
                    chi2 += 0.5 * err * err;
                    b[0] += err * (*dx_ptr);
                    b[1] += err * (*dy_ptr);
                    b[2] += err * 1.;

                    ++dx_ptr; ++dy_ptr;
                    ++ref_ptr; ++cur_ptr;
                }
                cur_ptr += (cur_stride - patch_t::size);
                ref_ptr += (patch_t::border_sz * 2);
            }

            Eigen::Vector3d delta = hessian.ldlt().solve(b);
            if (delta.hasNaN()) { assert(false); return false; }
            if (0 < i && last_chi2 < chi2) { 
#ifdef _ME_VSLAM_DEBUG_INFO_OPT_
                std::cout << "[ALIGN2D]" << "Loss increased at " << i << std::endl;
#endif
#ifdef _ME_VSLAM_ALIGN_CONSERVATIVE_
                u = last_u; v = last_v; 
                intensity_mean_diff = last_intensity_mean_diff;
                break;
#endif
            }

            last_u = u;
            last_v = v;
            last_intensity_mean_diff = intensity_mean_diff;
            last_chi2 = chi2;

            u -= delta[0];
            v -= delta[1];
            intensity_mean_diff -= delta[2];

            if (delta.head<2>().norm() < config::opt_converged_thresh_uv) {
#ifdef _ME_VSLAM_DEBUG_INFO_OPT_
                std::cout << "[ALIGN2D]" << "Converged at " << i << std::endl;
#endif
                converged = true;
                break;
            }
        }

        uv_cur << u, v;
        return converged;
    }

} // namespace vslam

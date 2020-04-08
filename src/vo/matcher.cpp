#include <vo/matcher.hpp>
#include <vo/camera.hpp>
#include <vo/feature.hpp>
#include <vo/map_point.hpp>
#include <vo/frame.hpp>
#include <utils/config.hpp>
#include <utils/utils.hpp>

namespace vslam {

    void patch_matcher::_create_patch_from_patch_with_border() {
        uint8_t* p = _patch;
        uint8_t* q = _patch_with_border + patch_with_border_sz + border_sz;
        for (int r = 0; r < patch_sz; ++r) {
            for (int c = 0; c < patch_sz; ++c) {
                p[c] = q[c];
            }
            p += patch_sz;
            q += patch_with_border_sz;
        }
    }

    bool patch_matcher::find_match_and_align(
        const map_point_ptr& mp, 
        const frame_ptr&     cur, 
        Eigen::Vector2d&     uv_cur
    ) {
        auto feat_ref = mp->find_closest_observed(cur->cam_center());
        if (!feat_ref) { return false; }

        auto ref = feat_ref->host_frame.lock();
        assert(ref);
        if (!ref->visible(
                feat_ref->uv, 
                patch_half_sz + border_sz + 1, 
                feat_ref->level
            )
        ) { return false; }

        Eigen::Vector3d xyz_ref = ref->t_cw * mp->position;
        Eigen::Matrix2d affine_cr = affine::affine_mat(
            ref->camera, feat_ref->uv, feat_ref->level, 
            xyz_ref.z(), cur->t_cw * ref->t_wc
        );

        size_t max_level = config::get<int>("pyr_levels") - 1;
        size_t search_level = affine::search_best_level(affine_cr, max_level);
        affine::extract_patch_affine(
            ref->pyramid[feat_ref->level], feat_ref->uv, 
            feat_ref->level, affine_cr.inverse(), search_level, 
            patch_half_sz + border_sz, _patch_with_border
        );

        _create_patch_from_patch_with_border();

        bool success = false;
        double scale = (1 << search_level);
        Eigen::Vector2d uv_leveln = uv_cur / scale;
        if (feature::EDGELET == feat_ref->type) {
            Eigen::Vector2d grad_orien_cur = 
                (affine_cr * feat_ref->grad_orien).normalized();
            success = alignment::align1d(/*todo*/);
        }
        else if (feature::CORNER == feat_ref->type) {
            success = alignment::align2d(/*todo*/);
        }
        else { assert(false); }

        uv_cur = uv_leveln * scale;
        return success;
    }

    bool patch_matcher::find_match_epipolar_and_align(
        const frame_ptr&   ref,
        const frame_ptr&   cur,
        const feature_ptr& feat_ref,
        double             depth_est,
        double             depth_min,
        double             depth_max,
        double&            depth_ref
    ) {
        Sophus::SE3d t_cr = cur->t_cw * ref->t_wc;
        //double best_zmssd = 1e10 /* threshold */;

        Eigen::Vector3d xyz_unit_ref = feat_ref->xy1.normalized();

        Eigen::Vector3d xyz_min_cur = t_cr * (depth_min * xyz_unit_ref);
        Eigen::Vector3d xyz_max_cur = t_cr * (depth_max * xyz_unit_ref);
        Eigen::Vector2d uv_min_cur = cur->camera->cam2pixel(xyz_min_cur);
        Eigen::Vector2d uv_max_cur = cur->camera->cam2pixel(xyz_max_cur);

        Eigen::Vector2d epipolar_vec = uv_min_cur - uv_max_cur;
        double epipolar_len = epipolar_vec.norm();
        Eigen::Vector2d epipolar_orien = epipolar_vec / epipolar_len;

        Eigen::Matrix2d affine_cr = affine::affine_mat(
            ref->camera, feat_ref->uv, feat_ref->level, 
            xyz_unit_ref[2] * depth_est, t_cr
        );

        if (feature::EDGELET == feat_ref->type && edgelet_filtering) {
            Eigen::Vector2d grad_orien_cur = 
                (affine_cr * feat_ref->grad_orien).normalized();
            double cos_angle = std::abs(grad_orien_cur.dot(epipolar_orien));
            if (cos_angle < std::cos(max_angle_between_epi_grad)) { return false; }
        }
        size_t max_level = config::get<int>("pyr_levels") - 1;
        size_t level_cur = affine::search_best_level(affine_cr, max_level);

        affine::extract_patch_affine(
            ref->pyramid[feat_ref->level], feat_ref->uv, 
            feat_ref->level, affine_cr.inverse(), level_cur, 
            patch_half_sz + border_sz, _patch_with_border
        );

        _create_patch_from_patch_with_border();

        if (epipolar_len < min_len_to_epipolar_search) {
            Eigen::Vector2d uv_cur = (uv_min_cur + uv_max_cur) * 0.5;
            double scale = (1 << level_cur);
            Eigen::Vector2d uv_leveln_cur = uv_cur / scale;
            if (!alignment::align1d()) { return false; }
            uv_cur = uv_leveln_cur * scale;
            double depth_cur = 0.;
            return utils::depth_from_triangulate_v2(
                cur->camera->pixel2cam_unit(uv_cur), 
                xyz_unit_ref, t_cr, depth_cur, depth_ref
            );
        }

        size_t n_steps = epipolar_len / epipolar_search_step;
        if (max_epipolar_search_steps < n_steps) { return false; }

        //TODO
    }
    
    inline Eigen::Matrix2d affine::affine_mat(
        const camera_ptr&      camera, 
        const Eigen::Vector2d& uv_ref,
        size_t                 level_ref,
        double                 z_est,
        const Sophus::SE3d&    t_cr
    ) {
        const int patch_half_sz = patch_matcher::patch_half_sz + 1;
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

    inline size_t 
    affine::search_best_level(
        const Eigen::Matrix2d& affine_mat, size_t max_level
    ) {
        double determinant = affine_mat.determinant();
        if (determinant <= 3.) { return 0; }
        size_t level = size_t(std::log2(determinant / 3.) / 2.) + 1;
        return max_level < level ? max_level : level;
    }

    inline bool extract_patch_affine(
        const cv::Mat&         image_leveln_ref,
        const Eigen::Vector2d& uv_level0_ref,
        size_t                 level_ref,
        const Eigen::Matrix2d& affine_rc, 
        size_t                 search_level,
        int                    patch_half_sz,
        uint8_t*               patch
    ) {
        if (affine_rc.hasNaN()) { return false; }

        const int patch_sz = patch_half_sz * 2;

        uint8_t* ptr = patch;
        Eigen::Vector2d center_ref = uv_level0_ref / double(1 << level_ref);

        for (int y = 0; y < patch_sz; ++y) {
            for (int x = 0; x < patch_sz; ++x) {
                Eigen::Vector2d uv_cur(x - patch_half_sz, y - patch_half_sz);
                uv_cur *= double(1 << search_level);
                // uv: uv at leveln
                Eigen::Vector2d uv = affine_rc * uv_cur + center_ref;
                if (utils::in_image(image_leveln_ref, uv[0], uv[1], 1.)) { *ptr = 0; }
                else {
                    *ptr = utils::bilinear_interoplate<uint8_t>(image_leveln_ref, uv[0], uv[1]);
                }
            }
        }
    }

} // namespace vslam

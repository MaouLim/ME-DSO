#include <vo/matcher.hpp>
#include <vo/camera.hpp>
#include <utils/utils.hpp>

namespace vslam {

    const int patch_matcher::patch_half_sz = 4;
    const int patch_matcher::path_sz       = patch_matcher::patch_half_sz * 2;
    
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
        const Eigen::Matrix2d& affine_cr, 
        size_t                 search_level,
        int                    patch_half_sz,
        uint8_t*               patch
    ) {
        const int patch_sz = patch_half_sz * 2;
        Eigen::Matrix2d affine_rc = affine_cr.inverse();

        if (std::isnan(affine_rc(0, 0))) {
            return false;
        }

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

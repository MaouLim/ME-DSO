#ifndef _ME_VSLAM_UTILS_HPP_
#define _ME_VSLAM_UTILS_HPP_

#include <common.hpp>

namespace utils {

    /**
     * some utils related to OpenCV
     */ 

    static const cv::Mat dx_kernel = (cv::Mat_<double>(2, 2) << -0.25, +0.25, -0.25, +0.25);
    static const cv::Mat dy_kernel = (cv::Mat_<double>(2, 2) << -0.25, -0.25, +0.25, +0.25);

    /**
     * @note require image type CV_32FC1 or CV_64FC1
     */ 
    inline void calc_dx(const cv::Mat& src, cv::Mat& dst) {
        cv::filter2D(src, dst, src.depth(), dx_kernel);
    }

    /**
     * @note require image type CV_32FC1 or CV_64FC1
     */ 
    inline void calc_dy(const cv::Mat& src, cv::Mat& dst) {
        cv::filter2D(src, dst, src.depth(), dy_kernel);
    }

    /**
     * @note require image type CV_32FC1 or CV_64FC1
     */ 
    inline void calc_grad(const cv::Mat& dx, const cv::Mat& dy, cv::Mat& dst) {
        cv::sqrt(dx.mul(dx) + dy.mul(dy), dst);
    }

    inline void down_sample(const cv::Mat& src, cv::Mat& dst, double scale) {
        cv::resize(src, dst, cv::Size(), scale, scale, cv::INTER_CUBIC);
    }

    inline bool in_image(
        const cv::Mat& img, double x, double y, double border = 0.
    ) {
        assert(0.0 <= border);
        return border <= x && int(x + border) < img.cols && 
               border <= y && int(y + border) < img.rows;
    }

    /**
     * @brief bilinear interpolate algorithm
     */ 
    template <typename _DType>
    double bilinear_interoplate(
        const cv::Mat& img, double u, double v
    ) {
        int x = std::floor(u);
        int y = std::floor(v);
        double dx = u - x;
        double dy = v - y;

        //         dx
        // (v00)-************-(v01)
        //    dy *   |      *
        //       *---*-(u,v)*
        //       *          *
        //       *          *
        // (v10)-**********-(v11)

        double w00 = (1. - dx) * (1. - dy);
        double w01 = dx * (1. - dy);
        double w10 = (1. - dx) * dy;
        double w11 = dx * dy;

        _DType v00 = img.at<_DType>(y, x);
        _DType v10 = img.at<_DType>(y + 1, x);
        _DType v01 = img.at<_DType>(y, x + 1);
        _DType v00 = img.at<_DType>(y + 1, x + 1);

        return w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
    }

    /**
     * @note require image type CV_8UC1
     */ 
    inline float
    shi_tomasi_score(const cv::Mat& img, int u, int v, int patch_half_sz = 4) {
        assert(CV_8UC1 == img.type());

        float dx2 = 0.f, dy2 = 0.f, dxy = 0.f;

        const int patch_size = patch_half_sz * 2;
        const int patch_area = patch_size * patch_size;
        const int x_min = u - patch_half_sz;
        const int x_max = u + patch_half_sz;
        const int y_min = v - patch_half_sz;
        const int y_max = v + patch_half_sz;

        if (x_min < 1 || img.cols <= x_max + 1 || 
            y_max < 1 || img.rows <= y_max + 1) 
        {
            // patch is too close to the boundary
            return 0.f;
        }

        for (int x = x_min; x < x_max; ++x) {
            for (int y = y_min; y < y_max; ++y) {
                float dx = (float) img.at<uchar>(y, x + 1) - (float) img.at<uchar>(y, x - 1);
                float dy = (float) img.at<uchar>(y + 1, x) - (float) img.at<uchar>(y - 1, x);
                dx2 += dx * dx;
                dy2 += dy * dy;
            }
        }

        // const int stride = img.step.p[0];
        // for (int y = y_min; y < y_max; ++y) {
        //     const uint8_t* ptr_left   = img.data + stride*y + x_min - 1;
        //     const uint8_t* ptr_right  = img.data + stride*y + x_min + 1;
        //     const uint8_t* ptr_top    = img.data + stride*(y-1) + x_min;
        //     const uint8_t* ptr_bottom = img.data + stride*(y+1) + x_min;
        //     for (int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom) {
        //         float dx = *ptr_right - *ptr_left;
        //         float dy = *ptr_bottom - *ptr_top;
        //         dx2 += dx*dx;
        //         dy2 += dy*dy;
        //         dxy += dx*dy;
        //     }
        // }

        // Find and return smaller eigenvalue:
        dx2 = dx2 / (2.0 * patch_area);
        dy2 = dy2 / (2.0 * patch_area);
        dxy = dxy / (2.0 * patch_area);
        return 0.5f * (dx2 + dy2 - sqrt((dx2 + dy2) * (dx2 + dy2) - 4.f * (dx2 * dy2 - dxy * dxy)));
    }

    inline std::vector<cv::Mat> 
    create_pyramid(const cv::Mat& img_level0, size_t n_levels, double scale = 0.5) {
        std::vector<cv::Mat> pyramid(n_levels, cv::Mat());
        pyramid[0] = img_level0;
        for (size_t i = 1; i < n_levels; ++i) {
            down_sample(pyramid[i - 1], pyramid[i], scale);
        }

        return pyramid;
    }

    /**
     * convert OpenCV data types to Eigen3 types
     */ 
    
    inline Eigen::Vector2d eigen_vec(const cv::Point2f& cv_pt) {
        return { double(cv_pt.x), double(cv_pt.y) };
    }


    /**
     * some utils related to Eigen3 math
     */ 

    inline Eigen::Vector2d project(
        const Eigen::Vector3d& homogeneous
    ) {
        return homogeneous.head<2> / homogeneous[2];
    }

    inline Eigen::Vector3d project(
        const Eigen::Vector4d& homogeneous
    ) {
        return homogeneous.head<3>() / homogeneous[3];
    }

    inline Eigen::Vector3d homogenize(
        const Eigen::Vector2d& xy
    ) {
        return { xy.x(), xy.y(), 1. };
    }

    inline Eigen::Vector4d homogenize(
        const Eigen::Vector3d& xyz
    ) {
        return { xyz.x(), xyz.y(), xyz.z(), 1. };
    }

    inline Eigen::Matrix3d hat(const Eigen::Vector3d& v) {
        Eigen::Matrix3d v_hat;
        v_hat <<  0, -v[2],  v[1],
               v[2],     0, -v[0],
              -v[1],  v[0],     0;
        return v_hat;
    }

    template <typename _EigenVecTp>
    double distance_l1(const _EigenVecTp& left, const _EigenVecTp& right) {
        return (double) (left - right).cwiseAbs().sum();
    }

    template <typename _EigenVecTp>
    double distance_l2(const _EigenVecTp& left, const _EigenVecTp& right) {
        return (double) (left - right).squareNorm();
    }

    template <typename _EigenVecTp>
    double distance_inf(const _EigenVecTp& left, const _EigenVecTp& right) {
        return (double) (left - right).cwiseAbs().maxCoeff();
    }

    /**
     * @brief triangulation
     *        s1*(x1, y1, 1)' = R*s0*(x0, y0, 1)' + t
     *    <=> [-R*(x0, y0, 1)', (x1, y1, 1)'] * (s0, s1)' = t
     *    <=> A * (s0, s1)' = t
     *    <=> (s0, s1)' = (A' * A)^(-1) * A' * t
     * @param xy1_ref (x0, y0, 1)
     * @param xy1_cur (x1, y1, 1)
     * @param t_cr pose from ref to cur -> (R, t)
     * @return (s1*(x1, y1, 1)' + R*s0*(x0, y0, 1)' + t) * 0.5
     */       
    inline Eigen::Vector3d triangulate(
        const Eigen::Vector3d& xy1_ref, 
        const Eigen::Vector3d& xy1_cur, 
        const Sophus::SE3d&    t_cr    
    ) {
        const auto& rot   = t_cr.rotationMatrix();
        const auto& trans = t_cr.translation();
        vslam::Matrix23d a_t;
        a_t.block<1, 3>(0, 0) = -(rot * xy1_ref).transpose();
        a_t.block<1, 3>(1, 0) = xy1_cur.transpose();
        Eigen::Vector2d s = (a_t * a_t.transpose()).inverse() * a_t * trans;
        // return the mid point
        return (rot * xy1_ref * s[0] + trans + xy1_cur * s[1]) * 0.5;
    }

    /**
     * @brief triangulation
     *        s0*(x0, y0, 1)' = R*s1*(x1, y1, 1)' + t
     *    <=> [ p0'p0,      -p0'Rp1][s0, s1]' = [p0't, (Rp1)'t]'
     *        [p0'Rp1, -(Rp1)'(Rp1)]
     * @param xy1_ref p0 = (x0, y0, 1) 
     * @param xy1_cur p1 = (x1, y1, 1)
     * @param t_rc pose from cur to ref -> (R, t)
     * @return (s0*(x0, y0, 1)' + R*s1*(x1, y1, 1)' + t) * 0.5
     */   
    inline Eigen::Vector3d triangulate_v2(
        const Eigen::Vector3d& xy1_ref, 
        const Eigen::Vector3d& xy1_cur, 
        const Sophus::SE3d&    t_rc
    ) {
        Eigen::Vector3d trans = t_rc.translation();
        Eigen::Matrix3d rot   = t_rc.rotationMatrix();

        Eigen::Vector3d f2 = rot * xy1_cur;
        Eigen::Vector2d b  = { xy1_ref.dot(trans), f2.dot(trans) };
        Eigen::Matrix2d a;
        a(0, 0) = xy1_ref.dot(xy1_ref);
        a(0, 1) = -xy1_ref.dot(f2);
        a(1, 0) = -a(0, 1);
        a(1, 1) = -f2.dot(f2);
        Eigen::Vector2d s = a.inverse() * b;
        Eigen::Vector3d xm = s[0] * xy1_ref;
        Eigen::Vector3d xn = s[1] * f2 + trans;
        return 0.5 * (xm + xn);
    }

    /**
     * @warning this function will not validate the depth
     * @note    when the parameters xy1_ref(cur) replaced 
     *          by the unit sphere coordinate, the z_ref(cur)
     *          return the depth(length of p_ref)
     */ 
    inline bool 
    depth_from_triangulate_v2(
        const Eigen::Vector3d& xy1_ref, 
        const Eigen::Vector3d& xy1_cur, 
        const Sophus::SE3d&    t_rc,
        double&                z_ref,
        double&                z_cur
    ) {
        Eigen::Vector3d trans = t_rc.translation();
        Eigen::Matrix3d rot   = t_rc.rotationMatrix();

        Eigen::Vector3d f2 = rot * xy1_cur;
        Eigen::Vector2d b  = { xy1_ref.dot(trans), f2.dot(trans) };
        Eigen::Matrix2d a;
        a(0, 0) = xy1_ref.dot(xy1_ref);
        a(0, 1) = -xy1_ref.dot(f2);
        a(1, 0) = -a(0, 1);
        a(1, 1) = -f2.dot(f2);
        if (a.determinant() < CONST_EPS) { return false; }
        auto s = a.inverse() * b;
        z_ref = s[0]; z_cur = s[1];
        return true;
    }

    /**
     * @param xyz coordinate in camera coordinate system 
     * @param xy1 coordinate in unit-bearing coordinate system 
     */
    inline double reproject_err(
        const Eigen::Vector3d& xyz, 
        const Eigen::Vector3d& xy1, 
        double                 alpha = 1.0, 
        double                 beta  = 0.0
    ) {
        return alpha * (xyz / xyz.z() - xy1).norm() + beta;
    }

    inline double sampsonus_err(
        const Eigen::Vector3d& xy1_ref, 
        const Eigen::Matrix3d& essential, 
        const Eigen::Vector3d& xy1_cur
    ) {
        Eigen::Vector3d exy1_cur = essential * xy1_cur;
        Eigen::Vector3d etxy1_ref = essential.transpose() * xy1_ref;
        double err = xy1_ref.transpose() * exy1_cur;
        return err * err / (exy1_cur.head<2>().squaredNorm() + etxy1_ref.head<2>().squaredNorm());
    }

    /**
     * @brief compute the depth uncertainty
     * @cite  REMODE: Probabilistic, Monocular Dense Reconstruction in Real Time
     * @param p        the coordinate in the ref
     * @param trans_cr the translation from ref to cur:
     *                 trans_cr = translation_component(t_rc)
     *                          = t_cr^(-1).translation()
     */ 
    inline double calc_depth_cov(
        const Eigen::Vector3d& p, 
        const Eigen::Vector3d& trans_cr, 
        double focal_len
    ) {
        double p_norm = p.norm();
        double t_norm = trans_cr.norm();
        Eigen::Vector3d f = p / p_norm;
        Eigen::Vector3d a = p - trans_cr;
        double alpha = std::acos(f.dot(trans_cr) / t_norm);
        double beta  = std::acos(-a.dot(f) / (a.norm() * t_norm));
        double beta_plus = beta + 2.0 * atan(0.5 / focal_len);
        double gamma = M_PI - alpha - beta_plus;
        double p_plus_norm = t_norm * std::sin(beta_plus) / std::sin(gamma);
        return std::abs(p_plus_norm - p_norm);
    }

    /**
     * other utils
     */ 

    template <typename _RandItr>
    _RandItr median(_RandItr first, _RandItr last) {
        ptrdiff_t diff = last - first;
        _RandItr nth = first + ptrdiff_t((diff - 1) / 2);
        std::nth_element(first, nth, last);
        return nth;
    }

    /**
     * contruct(mk->make) a smart point to indicated type _Tp
     */ 
    template <typename _Tp, typename... _Args>
    vslam::vptr<_Tp> mk_vptr(_Args&&... _args) {
        typedef typename std::remove_cv<_Tp>::type _Tp_nc;
        return std::allocate_shared<_Tp>(
            std::allocator<_Tp_nc>(), std::forward<_Args>(_args)...
        );
    }
}

#endif
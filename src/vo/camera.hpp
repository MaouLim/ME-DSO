#ifndef _ME_VSLAM_CAMERA_HPP_
#define _ME_VSLAM_CAMERA_HPP_

#include <common.hpp>

namespace vslam {

    struct abstract_camera {

        int height, width;

        abstract_camera(int _h, int _w) : height(_h), width(_w) { }
        virtual ~abstract_camera() = default;
        virtual Eigen::Vector2d cam2pixel(const Eigen::Vector3d&) const = 0;
        virtual Eigen::Vector3d pixel2cam(const Eigen::Vector2d&, double) const = 0;

        /**
         * @param p_w  3d point in world
         * @param t_cw camera pose, convert world coordinate 
         *             system to camera coordinate system
         */ 
        Eigen::Vector2d world2pixel(const Eigen::Vector3d& p_w, const Sophus::SE3d& t_cw) const;

        /**
         * @param p_p  2d pixel point
         * @param t_wc camera pose inverse, convert camera coordinate system
         *             to world coordinate system
         */ 
        Eigen::Vector3d pixel2world(
            const Eigen::Vector2d& p_p, const Sophus::SE3d& t_wc, double depth = 1.0) const;
    };

    inline Eigen::Vector2d 
    abstract_camera::world2pixel(
        const Eigen::Vector3d& p_w, const Sophus::SE3d& t_cw
    ) const {
        return this->cam2pixel(t_cw * p_w);
    }

    inline Eigen::Vector3d 
    abstract_camera::pixel2world(
        const Eigen::Vector2d& p_p, const Sophus::SE3d& t_wc, double depth
    ) const {
        return t_wc * this->pixel2cam(p_p, depth);
    }

    /**
     * @brief non-distortion pinhole camera model
     */ 
    struct pinhole_camera : abstract_camera {

        using ptr = vptr<pinhole_camera>;

        pinhole_camera(int _h, int _w);
        virtual ~pinhole_camera() = default;

        Eigen::Vector2d cam2pixel(const Eigen::Vector3d& p_c) const override;
        Eigen::Vector3d pixel2cam(const Eigen::Vector2d& p_p, double depth = 1.0) const override;

        Eigen::Matrix3d eigen_mat() const;
        cv::Mat cv_mat() const;
        Eigen::Vector4d eigen_vec() const;

        double fx, fy, cx, cy;
    };

    pinhole_camera::pinhole_camera(int _h, int _w) : 
        abstract_camera(_h, _w),
        fx(0.), fy(0.), cx(0.), cy(0.) { }

    inline Eigen::Vector2d 
    pinhole_camera::cam2pixel(
        const Eigen::Vector3d& p_c
    ) const {
        return { (fx * p_c[0] + cx) / p_c[2], (fy * p_c[0] + cy) / p_c[2] };
    }

    inline Eigen::Vector3d 
    pinhole_camera::pixel2cam(
        const Eigen::Vector2d& p_p, double depth
    ) const {
        return { (p_p[0] * depth - cx) / fx,  (p_p[1] * depth - cy) / fy, depth };
    }

    inline Eigen::Matrix3d pinhole_camera::eigen_mat() const {
        Eigen::Matrix3d cam_mat;
        cam_mat << (fx, 0, 0, 0, fy, 0, 0, 0, 1);
        return cam_mat;
    }

    inline cv::Mat pinhole_camera::cv_mat() const {
        return cv::Mat_<double>(3, 3) << (fx, 0, 0, 0, fy, 0, 0, 0, 1);
    }

    inline Eigen::Vector4d pinhole_camera::eigen_vec() const {
        return { fx, fy, cx, cy };
    }
}

#endif
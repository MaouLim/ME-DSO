#ifndef _ME_VSLAM_HOMOGRAPHY_HPP_
#define _ME_VSLAM_HOMOGRAPHY_HPP_

#include <common.hpp>

namespace vslam {

    struct homography_decomposition {

        Eigen::Vector3d trans;
        Eigen::Matrix3d rot;
        double          d;
        Eigen::Vector3d n;
        Sophus::SE3d    t_cr;
        int             score;
    };

    inline bool operator<(
        const homography_decomposition& left, 
        const homography_decomposition& right
    ) {
        return left.score < right.score;
    }

    struct homography {

        double                                threshold;
        double                                err_mul2;
        const std::vector<Eigen::Vector3d>&   xy1s_ref;
        const std::vector<Eigen::Vector3d>&   xy1s_cur;
        std::vector<bool>                     inliers;
        Eigen::Matrix3d                       h_cr;
        std::vector<homography_decomposition> decompositions;
        
        homography(
            double                              _thresh,
            double                              _err_mul2,
            const std::vector<Eigen::Vector3d>& _xy1s_ref,
            const std::vector<Eigen::Vector3d>& _xy1s_cur
        ) : threshold(_thresh), err_mul2(_err_mul2), 
            xy1s_ref(_xy1s_ref), xy1s_cur(_xy1s_cur) 
        { 
            assert(xy1s_ref.size() == xy1s_cur.size());
        }

        bool calc_pose_from_matches(Sophus::SE3d& t_cr);

    private:
        void _calc_from_matches();
        size_t _compute_inliers();
        bool _decompose();
        void _find_best_decomposition();
    };
}

#endif
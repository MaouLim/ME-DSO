#include <vo/homography.hpp>
#include <utils/utils.hpp>

namespace vslam {

    bool homography::calc_pose_from_matches(Sophus::SE3d& t_cr) {
        _calc_from_matches();
        if (!_decompose()) { return false; }
        _compute_inliers();
        _find_best_decomposition();
        assert(!decompositions.empty());
        t_cr = decompositions.front().t_cr;
        return true;
    }

    void homography::_calc_from_matches() {

        size_t n_matches = xy1s_ref.size();

        std::vector<cv::Point2f> src_pts(n_matches);
        std::vector<cv::Point2f> dst_pts(n_matches);

        for (size_t i = 0; i < n_matches; ++i) {
            src_pts[i] = cv::Point2f(xy1s_ref[i][0], xy1s_ref[i][1]);
            dst_pts[i] = cv::Point2f(xy1s_cur[i][0], xy1s_cur[i][1]);
        }

        cv::Mat cv_h = 
            cv::findHomography(src_pts, dst_pts, cv::RANSAC, 2. / err_mul2);
        
        h_cr(0,0) = cv_h.at<double>(0,0);
        h_cr(0,1) = cv_h.at<double>(0,1);
        h_cr(0,2) = cv_h.at<double>(0,2);
        h_cr(1,0) = cv_h.at<double>(1,0);
        h_cr(1,1) = cv_h.at<double>(1,1);
        h_cr(1,2) = cv_h.at<double>(1,2);
        h_cr(2,0) = cv_h.at<double>(2,0);
        h_cr(2,1) = cv_h.at<double>(2,1);
        h_cr(2,2) = cv_h.at<double>(2,2);
    }

    size_t homography::_compute_inliers() {
        size_t n_matches = xy1s_ref.size();
        size_t count_inliers = 0;

        inliers.clear(); 
        inliers.resize(n_matches);

        for (size_t i = 0; i < n_matches; ++i) {
            Eigen::Vector3d xyz = h_cr * xy1s_ref[i];
            Eigen::Vector2d xy = xyz.head<2>() / xyz.z();
            Eigen::Vector2d err_vec = xy1s_cur[i].head<2>() - xy;
            double err = err_mul2 * err_vec.norm();
            inliers[i] = (err < threshold);
            count_inliers += inliers[i];
        }
        return count_inliers;
    }
        
    bool homography::_decompose() {
        decompositions.clear();

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(h_cr, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Vector3d singular_values = svd.singularValues();

        double d1 = fabs(singular_values[0]); // The paper suggests the square of these (e.g. the evalues of AAT)
        double d2 = fabs(singular_values[1]); // should be used, but this is wrong. c.f. Faugeras' book.
        double d3 = fabs(singular_values[2]);

        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();    
                        
        double s = U.determinant() * V.determinant();  // VT^T
        double dPrime_PM = d2;

        int the_case = 0;
        if (d1 != d2 && d2 != d3) { the_case = 1; }
        else if (d1 == d2 && d2 == d3) { the_case = 3; }
        else { the_case = 2; }

        if (the_case != 1) {
#ifdef _ME_VSLAM_DEBUG_INFO_
            std::cerr << "FATAL homography initialization: " 
                      << "This motion case is not implemented or is degenerate." 
                      << std::endl;
#endif
            return false;
        }
        double x1_PM;
        double x2;
        double x3_PM;
        // All below deals with the case = 1 case.
        // Case 1 implies (d1 != d3)
        {   // Eq. 12
            x1_PM = sqrt((d1*d1 - d2*d2) / (d1*d1 - d3*d3));
            x2    = 0;
            x3_PM = sqrt((d2*d2 - d3*d3) / (d1*d1 - d3*d3));
        };

        double e1[4] = {1.0,-1.0, 1.0,-1.0};
        double e3[4] = {1.0, 1.0,-1.0,-1.0};

        Eigen::Vector3d np;
        homography_decomposition decomp;

        // Case 1, d' > 0:
        decomp.d = s * dPrime_PM;
        for (size_t signs = 0; signs < 4; ++signs) {
            // Eq 13
            decomp.rot = Eigen::Matrix3d::Identity();
            double dSinTheta = (d1 - d3) * x1_PM * x3_PM * e1[signs] * e3[signs] / d2;
            double dCosTheta = (d1 * x3_PM * x3_PM + d3 * x1_PM * x1_PM) / d2;
            decomp.rot(0,0) = dCosTheta;
            decomp.rot(0,2) = -dSinTheta;
            decomp.rot(2,0) = dSinTheta;
            decomp.rot(2,2) = dCosTheta;
            // Eq 14
            decomp.trans[0] = (d1 - d3) * x1_PM * e1[signs];
            decomp.trans[1] = 0.0;
            decomp.trans[2] = (d1 - d3) * -x3_PM * e3[signs];
            np[0] = x1_PM * e1[signs];
            np[1] = x2;
            np[2] = x3_PM * e3[signs];
            decomp.n = V * np;
            decompositions.push_back(decomp);
        }

        // Case 1, d' < 0:
        decomp.d = s * -dPrime_PM;
        for (size_t signs = 0; signs < 4; ++signs) {
            // Eq 15
            decomp.rot = -1. * Eigen::Matrix3d::Identity();
            double dSinPhi = (d1 + d3) * x1_PM * x3_PM * e1[signs] * e3[signs] / d2;
            double dCosPhi = (d3 * x1_PM * x1_PM - d1 * x3_PM * x3_PM) / d2;
            decomp.rot(0,0) = dCosPhi;
            decomp.rot(0,2) = dSinPhi;
            decomp.rot(2,0) = dSinPhi;
            decomp.rot(2,2) = -dCosPhi;
            // Eq 16
            decomp.trans[0] = (d1 + d3) * x1_PM * e1[signs];
            decomp.trans[1] = 0.0;
            decomp.trans[2] = (d1 + d3) * x3_PM * e3[signs];
            np[0] = x1_PM * e1[signs];
            np[1] = x2;
            np[2] = x3_PM * e3[signs];
            decomp.n = V * np;
            decompositions.push_back(decomp);
        }

        // Save rotation and translation of the decomposition
        for (size_t i = 0; i < decompositions.size(); ++i) {
            Eigen::Matrix3d R = s * U * decompositions[i].rot * V.transpose();
            Eigen::Vector3d t = U * decompositions[i].trans;
            decompositions[i].t_cr = Sophus::SE3(R, t);
        }

        return true;
    }

    void homography::_find_best_decomposition() {
        assert(decompositions.size() == 8);
        for (size_t i = 0; i < decompositions.size(); ++i) {
            auto& decom = decompositions[i];
            size_t n_positives = 0;
            for (size_t j = 0; j < xy1s_ref.size(); ++j) {
                if (!inliers[j]) { continue; }
                double visibility_test = h_cr.row(2).dot(xy1s_ref[j]) / decom.d;
                //const Eigen::Vector3d& v2 = xy1s_ref[j];
                //double visibility_test = (h_cr(2, 0) * v2[0] + h_cr(2, 1) * v2[1] + h_cr(2, 2)) / decom.d;
                if (0.0 < visibility_test) { ++n_positives; }
            }
            decom.score = -n_positives;
        }

        sort(decompositions.begin(), decompositions.end());
        decompositions.resize(4);

        for (size_t i = 0; i  <decompositions.size(); ++i) {
            auto& decom = decompositions[i];
            int n_positives = 0;
            for (size_t j = 0; j < xy1s_ref.size(); ++j) {
                if (!inliers[j]) { continue; }
                double visibility_test = decom.n.dot(xy1s_ref[j]) / decom.d;
                if (0.0 < visibility_test) { ++n_positives; }
            };
            decom.score = -n_positives;
        }

        sort(decompositions.begin(), decompositions.end());
        decompositions.resize(2);

        // According to Faugeras and Lustman, ambiguity exists if the two scores are equal
        // but in practive, better to look at the ratio!
        double ratio = (double) decompositions[1].score / (double) decompositions[0].score;
        if (ratio < 0.9) { decompositions.erase(decompositions.begin() + 1); }// no ambiguity!
        else {
             // two-way ambiguity. Resolve by sampsonus score of all points.
            double err_squared_limit = threshold * threshold * 4.0;
            //double adSampsonusScores[2];
            double sampsonus_score[2] = { 0.0, 0.0 };
            
            for (size_t i = 0; i < 2; ++i) {
                const auto& t_cr = decompositions[i].t_cr;
                Eigen::Matrix3d essential = t_cr.rotationMatrix() * utils::hat(t_cr.translation());
                double total_err = 0;
                for (size_t j = 0; j < xy1s_ref.size(); ++j) {
                    double err = utils::sampsonus_err(xy1s_ref[j], essential, xy1s_cur[j]);
                    if (err_squared_limit < err) { err = err_squared_limit; }
                    total_err += err;
                }
                sampsonus_score[i] = total_err;
            }
            if (sampsonus_score[0] <= sampsonus_score[1]) {
                decompositions.erase(decompositions.begin() + 1);
            }
            else { decompositions.erase(decompositions.begin()); }
        }
    }
} // namespace vslam

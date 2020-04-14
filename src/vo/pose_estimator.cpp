#include <vo/pose_estimator.hpp>
#include <vo/frame.hpp>
#include <vo/feature.hpp>
#include <vo/map_point.hpp>
#include <vo/camera.hpp>
#include <vo/jaccobian.hpp>

#include <utils/utils.hpp>

// namespace icia {

//     struct _vertex_pose_only : 
//         g2o::BaseVertex<6, Sophus::SE3d> {

//         _vertex_pose_only() = default;
//         ~_vertex_pose_only() = default;
        
//         bool read(std::istream& is) override { return false; }
//         bool write(std::ostream& os) const override { return false; }

//         void setToOriginImpl() override { _estimate = Sophus::SE3d(); }

//         void oplusImpl(const double* u) override {
//             Sophus::Vector6d update;
//             update << u[0], u[1], u[2], u[3], u[4], u[5];
//             _estimate = _estimate * Sophus::SE3d::exp(-update);
//         }
//     };

//     struct _uedge_patch_photometric_err : 
//         g2o::BaseUnaryEdge<1, cv::Mat, _vertex_pose_only> {

//         using patch_type = vslam::pose_estimator::patch_type;

//         _uedge_patch_photometric_err(
//             const Eigen::Vector3d&          xyz_ref, 
//             const patch_type&               patch_ref, 
//             size_t                          level,
//             const vslam::camera_ptr&        camera,
//             const Eigen::Matrix<
//                 double, patch_type::area, 6
//             >&                              jacc_caches
//         ) : _xyz_ref(xyz_ref), _patch_ref(patch_ref),
//             _level(level), _camera(camera), _jacc_caches(jacc_caches) 
//         { }

//         bool read(std::istream& is) override { return false; }
//         bool write(std::ostream& os) const override { return false; }

//         void computeError() override {
//             auto v = (const _vertex_pose_only*) _vertices[0];
//             auto& t_cr = v->estimate();
//             const cv::Mat& img_leveln_cur = _measurement;

//             Eigen::Vector3d xyz_cur   = t_cr * _xyz_ref;
//             Eigen::Vector2d uv_cur    = _camera->cam2pixel(xyz_cur);
//             Eigen::Vector2d uv_leveln = uv_cur / (1 << _level);

//             _error.setZero();

//             if (!utils::in_image(
//                     img_leveln_cur, uv_leveln.x(), uv_leveln.y(), _check_sz
//                 )
//             ) { this->setLevel(1); return; }
            
//             auto w = utils::bilinear_weights(uv_leveln);

//             const int    ref_stride = patch_type::size;
//             const float* ref_ptr    = _patch_ref.data;
            
//             const int      cur_stride = img_leveln_cur.step.p[0];
//             const uint8_t* cur_ptr    = img_leveln_cur.data + 
//                                         (int(uv_leveln.y()) - patch_type::half_sz) * cur_stride + 
//                                         (int(uv_leveln.x()) - patch_type::half_sz);
//             size_t count_pixels = 0;

//             for (int r = 0; r < patch_type::size; ++r) {
//                 for (int c = 0; c < patch_type::size; ++c) {
//                     float intensity_cur = w(0, 0) * cur_ptr[0] + 
//                                           w(0, 1) * cur_ptr[1] + 
//                                           w(1, 0) * cur_ptr[cur_stride] + 
//                                           w(1, 1) * cur_ptr[cur_stride + 1];
//                     _error(0, 0) += intensity_cur - *ref_ptr;
//                     ++ref_ptr; ++cur_ptr;
//                 }
//                 cur_ptr += (cur_stride - patch_type::size);
//             }
//         }

//         void linearizeOplus() override {
//             _jacobianOplusXi.setZero();
//             for (int i = 0; i < patch_type::area; ++i) {
//                 _jacobianOplusXi -= _jacc_caches.row(i);
//             }
//         }

//     private:
//         static constexpr int _check_sz = patch_type::half_sz + 1;

//         const Eigen::Vector3d    _xyz_ref;
//         const patch_type&        _patch_ref;
//         size_t                   _level;
//         const vslam::camera_ptr& _camera;

//         const Eigen::Matrix<double, patch_type::area, 6>& _jacc_caches;
//     };

// } // namespace icia

namespace fcfa {

    size_t _optimize_single_level(
        const vslam::frame_ptr&   ref,
        const vslam::frame_ptr&   cur,
        size_t                    level,
        size_t                    n_iterations,
        Sophus::SE3d&             t_cr
    ) {
        constexpr int check_sz = vslam::pose_estimator::win_half_sz + 1;
        constexpr int half_sz  = vslam::pose_estimator::win_half_sz;
        constexpr int sz       = vslam::pose_estimator::win_sz;

        const double scale = (1 << level);
        const Eigen::Vector2d focal_len = cur->camera->focal_len();
        Eigen::Matrix2d duv0dxy1;
        duv0dxy1 << focal_len[0], 0., 0., focal_len[1];
        
        const cv::Mat& ref_img = ref->pyramid[level];
        const cv::Mat& cur_img = cur->pyramid[level];

        std::vector<bool> visibles;
		visibles.resize(ref->n_features, true);

		double last_chi2 = std::numeric_limits<double>::max();
		Sophus::SE3d last_t_cr = t_cr;

        size_t itr = 0;
        for (; itr < n_iterations; ++itr) {

            double chi2 = 0.0;
			Sophus::Vector6d jres    = Sophus::Vector6d::Zero();
			Sophus::Matrix6d hessian = Sophus::Matrix6d::Zero();

            size_t feat_idx = 0;
            for (auto& feat_ref : ref->features) {
                if (!visibles[feat_idx] || 
                    level != feat_ref->level || 
                    feat_ref->describe_nothing()
                ) { ++feat_idx; continue; }

                Eigen::Vector2d uv_ref = feat_ref->uv / scale;
                if (!utils::in_image(ref_img, uv_ref.x(), uv_ref.y(), check_sz)) {
                    visibles[feat_idx] = false;
                    ++feat_idx; continue;
                }

                Eigen::Vector3d xyz_w = feat_ref->map_point_describing->position;
                Eigen::Vector3d xyz_ref = ref->t_cw * xyz_w;
                Eigen::Vector3d xyz_cur = t_cr * xyz_ref;

				Eigen::Vector2d uv_0_cur = cur->camera->cam2pixel(xyz_cur);
				Eigen::Vector2d uv_cur = uv_0_cur / scale;

                if (!utils::in_image(cur_img, uv_cur.x(), uv_cur.y(), check_sz)) {
                    visibles[feat_idx] = false;
                    ++feat_idx; continue;
                }

                Eigen::Vector2i iuv_ref = uv_ref.cast<int>();
				Eigen::Vector2i iuv_cur = uv_cur.cast<int>();

                Eigen::Matrix26d duv0deps = duv0dxy1 * vslam::jaccobian_dxy1deps(xyz_cur);

                auto w_ref = utils::bilinear_weights(uv_ref);
				auto w_cur = utils::bilinear_weights(uv_cur);

				int ref_step = ref_img.step.p[0];
				uint8_t* ref_ptr = ref_img.data + (iuv_ref.y() - half_sz) * ref_step + (iuv_ref.x() - half_sz);

				int cur_step = cur_img.step.p[0];
				uint8_t* cur_ptr = cur_img.data + (iuv_cur.y() - half_sz) * cur_step + (iuv_cur.x() - half_sz);

                for (int r = 0; r < sz; ++r) {
					for (int c = 0; c < sz; ++c) {
						double intensity_ref =
							w_ref(0, 0) * ref_ptr[0]        + w_ref(0, 1) * ref_ptr[1] +
							w_ref(1, 0) * ref_ptr[ref_step] + w_ref(1, 1) * ref_ptr[ref_step + 1];
						double intensity_cur =
							w_cur(0, 0) * cur_ptr[0]        + w_cur(0, 1) * cur_ptr[1] +
							w_cur(1, 0) * cur_ptr[cur_step] + w_cur(1, 1) * cur_ptr[cur_step + 1];

						double err = intensity_ref - intensity_cur;
						chi2 += 0.5 * err * err;
						Eigen::Vector2d g = utils::gradient(
                            cur_img, uv_cur.x() - half_sz + c, uv_cur.y() - half_sz + r
                        );

						Sophus::Vector6d jacc = -g.transpose() * duv0deps / scale;

						jres    += jacc * -err;
						hessian += jacc * jacc.transpose();

						++ref_ptr;
						++cur_ptr;
					}
					ref_ptr += (ref_step - sz);
					cur_ptr += (cur_step - sz);
				}

                ++feat_idx;
            }

            Sophus::Vector6d update = hessian.ldlt().solve(jres);
			if (update.hasNaN()) { assert(false); return itr; }

			if (0 < itr && last_chi2 < chi2) {
				std::cout << "loss increased at iteration: " << itr << std::endl;
				t_cr = last_t_cr;
				break;
			}

			if (update.norm() < CONST_EPS) {
				std::cout << "converged at iteration: " << itr << std::endl;
				break;
			}

			last_chi2 = chi2;
			last_t_cr = t_cr;
			t_cr = Sophus::SE3d::exp(update) * t_cr;
        }

        return itr;
    }
    
} // namespace fcfa


namespace vslam {

    pose_estimator::pose_estimator(
        size_t       n_iterations,
        size_t       min_level,
        size_t       max_level
    ) : _n_iterations(n_iterations), _min_level(min_level), _max_level(max_level) 
    {  }

    void pose_estimator::estimate(
        const frame_ptr& ref, 
        const frame_ptr& cur, 
        Sophus::SE3d&    t_cr
    ) {
        auto l = _max_level + 1;
        while (_min_level < l) {
            fcfa::_optimize_single_level(ref, cur, --l, _n_iterations, t_cr);
            // _init_graph_and_optimize(ref, cur, idx);
        }
    }

    // void pose_estimator::_precalc_cache(
    //     const frame_ptr& ref, size_t level
    // ) {
    //     const Eigen::Vector2d& focal_len = ref->camera->focal_len();
    //     const double scale = (1 << level);
    //     const cv::Mat& img_leveln = ref->pyramid[level];
    //     const int img_stride = img_leveln.step.p[0];
    //     constexpr int check_sz = patch_type::half_sz + 1;

    //     _patches_ref.resize(ref->n_features, cv::Mat(patch_type::size, patch_type::size, CV_32F));
    //     _jaccobians_ref.resize(ref->n_features);
    //     _visibles_ref.resize(ref->n_features, false);

    //     size_t idx = 0;

    //     for (const auto& each_feat : ref->features) {
    //         if (each_feat->describe_nothing()) { ++idx; continue; }
    //         Eigen::Vector2d uv_leveln = each_feat->uv / scale;
    //         if (!utils::in_image(
    //                 img_leveln, uv_leveln.x(), uv_leveln.y(), check_sz
    //             )
    //         ) { ++idx; continue; }

    //         _visibles_ref[idx] = true;
            
    //         Eigen::Vector3d xyz_ref = 
    //             ref->t_cw * each_feat->map_point_describing->position;
            
    //         auto dxy1deps = jaccobian_dxy1deps(xyz_ref);

    //         auto& patch = _patches_ref[idx];
    //         auto& jacc  = _jaccobians_ref[idx];

    //         int x = std::floor(uv_leveln.x());
    //         int y = std::floor(uv_leveln.y());
    //         double dx = uv_leveln.x() - x;
    //         double dy = uv_leveln.y() - y;

    //         double w00 = (1. - dx) * (1. - dy);
    //         double w01 =        dx * (1. - dy);
    //         double w10 = (1. - dx) * dy;
    //         double w11 =        dx * dy;

    //         //uint8_t*   patch_ptr = patch.data;
    //         uint8_t* img_ptr   = 
    //             img_leveln.data + (y - patch_type::half_sz) * img_stride + (x - patch_type::half_sz);
                
    //         size_t count_pixels = 0;

    //         for (int r = 0; r < patch_type::size; ++r) {
    //             for (int c = 0; c < patch_type::size; ++c) {
    //                 //*patch_ptr 
    //                 patch.at<float>(r, c)
    //                 = w00 * img_ptr[0] + 
    //                              w01 * img_ptr[1] + 
    //                              w10 * img_ptr[img_stride] + 
    //                              w11 * img_ptr[img_stride + 1];
    //                 double gx =        dy * (double(img_ptr[img_stride + 1]) - double(img_ptr[img_stride])) + 
    //                             (1. - dy) * (double(img_ptr[1]) - double(img_ptr[0]));
    //                 double gy =        dx * (double(img_ptr[img_stride + 1]) - double(img_ptr[1])) + 
    //                             (1. - dx) * (double(img_ptr[img_stride]) - double(img_ptr[0]));
    //                 // dI(n)deps = dI(n)dI(0) * dI(0)du * dudxy1 * dxy1deps
    //                 //              1/scale    (gx, gy)   [fx, 0]
    //                 //                                    [0, fy]  
    //                 jacc.row(count_pixels) = 
    //                     (gx * focal_len.x() * dxy1deps.row(0) + 
    //                      gy * focal_len.y() * dxy1deps.row(1)) / scale;
    //                 ++count_pixels;
    //                 //++patch_ptr;
    //                 ++img_ptr;
    //             }
    //             img_ptr += (img_stride - patch_type::size);
    //         }

    //         ++idx;
    //     }
    // }

    // void pose_estimator::_clear_cache() {
    //     _jaccobians_ref.clear();
    //     _patches_ref.clear();
    //     _visibles_ref.clear();
    // }

    // void pose_estimator::_init_graph_and_optimize(
    //     const frame_ptr& ref, 
    //     const frame_ptr& cur, 
    //     size_t           level
    // ) {
    //     //_optimizer.clear();
    //     //_optimizer.setAlgorithm(_algo);
    //     //_optimizer.setVerbose(true);

    //     // // create vertices
    //     // icia::_vertex_pose_only* v = new icia::_vertex_pose_only();
    //     // v->setId(0);
    //     // v->setEstimate(_t_cr);

    //     // // add the vertex to graph
    //     // _optimizer.addVertex(v);

    //     // create edges
    //     // size_t idx = 0;
    //     // size_t count_edges = 0;

    //     // for (const auto& each_feat : ref->features) {
    //     //     if (!_visibles_ref[idx]) { ++idx; continue; }

    //     //     Eigen::Vector3d xyz_ref = 
    //     //         ref->t_cw * each_feat->map_point_describing->position;
            
    //     //     auto& patch = _patches_ref[idx];
    //     //     auto& jacc  = _jaccobians_ref[idx];

    //     //     icia::_uedge_patch_photometric_err* e = 
    //     //         new icia::_uedge_patch_photometric_err(
    //     //             xyz_ref, patch, level, cur->camera, jacc
    //     //         );
    //     //     e->setId(count_edges);
    //     //     e->setVertex(0, v);
    //     //     //e->setRobustKernel(new g2o::RobustKernelHuber());
    //     //     e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
    //     //     e->setMeasurement(cur->pyramid[level]);

    //     //     // add the edge to graph
    //     //     _optimizer.addEdge(e);
            
    //     //     ++count_edges;
    //     //     ++idx;
    //     // }

    //     // _optimizer.initializeOptimization();
    //     // _optimizer.optimize(_n_iterations);
    //     // _t_cr = v->estimate();

    //     // std::cout << "r: \n" << _t_cr.rotationMatrix() << std::endl;
    //     // std::cout << "t: \n" << _t_cr.translation() << std::endl;
    //     // constexpr double check_sz = patch_type::half_sz + 1;
    //     // const double scale = (1 << level);
    //     // double last_chi2 = std::numeric_limits<double>::max();
    //     // Sophus::SE3d last_t_cr;
    //     // Sophus::Matrix6d hessian;
    //     // Sophus::Vector6d jres;

    //     // for (size_t i = 0; i < _n_iterations; ++i) {

    //     //     hessian.setZero();
    //     //     jres.setZero();

    //     //     double chi2 = 0.0;
    //     //     double err = 0.0;
    //     //     size_t idx = 0;

    //     //     for (const auto& each_feat : ref->features) {

    //     //         if (!_visibles_ref[idx]) { ++idx; continue; }

    //     //         Eigen::Vector3d xyz_ref = 
    //     //             ref->t_cw * each_feat->map_point_describing->position;

    //     //         auto& patch_ref = _patches_ref[idx];
    //     //         auto& jacc_ref  = _jaccobians_ref[idx];

    //     //         const cv::Mat& img_leveln_cur = cur->pyramid[level];

    //     //         Eigen::Vector3d xyz_cur   = _t_cr * xyz_ref;
    //     //         Eigen::Vector2d uv_cur    = cur->camera->cam2pixel(xyz_cur);
    //     //         Eigen::Vector2d uv_leveln = uv_cur / scale;

    //     //         if (!utils::in_image(
    //     //                 img_leveln_cur, uv_leveln.x(), uv_leveln.y(), check_sz
    //     //             )
    //     //         ) { ++idx; continue; }
            
    //     //         auto w = utils::bilinear_weights(uv_leveln);

    //     //         // const int    ref_stride = patch_type::size;
    //     //         // const uint8_t* ref_ptr    = patch_ref.data;
            
    //     //         const int      cur_stride = img_leveln_cur.step.p[0];
    //     //         const uint8_t* cur_ptr    = img_leveln_cur.data + 
    //     //                                     (int(uv_leveln.y()) - patch_type::half_sz) * cur_stride + 
    //     //                                     (int(uv_leveln.x()) - patch_type::half_sz);
    //     //         size_t count_pixels = 0;

    //     //         for (int r = 0; r < patch_type::size; ++r) {
    //     //             for (int c = 0; c < patch_type::size; ++c) {
    //     //                 float intensity_cur = w(0, 0) * cur_ptr[0] + 
    //     //                                       w(0, 1) * cur_ptr[1] + 
    //     //                                       w(1, 0) * cur_ptr[cur_stride] + 
    //     //                                       w(1, 1) * cur_ptr[cur_stride + 1];
    //     //                 err += intensity_cur - patch_ref.at<float>(r, c);//*ref_ptr;
    //     //                 chi2 += 0.5 * err * err;
    //     //                 Sophus::Vector6d j = jacc_ref.row(count_pixels++).transpose();
    //     //                 hessian += j * j.transpose();
    //     //                 jres += err * j;
    //     //                 //++ref_ptr; 
    //     //                 ++cur_ptr;
    //     //             }
    //     //             cur_ptr += (cur_stride - patch_type::size);
    //     //         }
    //     //         ++idx;
    //     //     }

    //     //     Sophus::Vector6d update = hessian.ldlt().solve(jres);
    //     //     if (update.hasNaN()) { assert(false); return; }

    //     //     if (1 < i && 1.2 * last_chi2 < chi2) { 
    //     //         std::cout << "loss inc." << std::endl;
    //     //         _t_cr = last_t_cr;
    //     //         break;
    //     //     }

    //     //     if (update.norm() < CONST_EPS) {
    //     //         std::cout << "chi2: " << chi2 << std::endl;
    //     //         std::cout << update.transpose() << std::endl;
    //     //         std::cout << "converged. " << i << std::endl;
    //     //         break;
    //     //     }

    //     //     last_t_cr = _t_cr;
    //     //     last_chi2 = chi2;
    //     //     _t_cr = _t_cr * Sophus::SE3d::exp(-update);
    //     //     std::cout << "_R:\n" << _t_cr.rotationMatrix() << std::endl;
    //     //     std::cout << "_t:\n" << _t_cr.translation() << std::endl;
    //     // }
    // }

} // namespace vslam

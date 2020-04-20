#include <vo/pose_estimator.hpp>
#include <vo/frame.hpp>
#include <vo/feature.hpp>
#include <vo/map_point.hpp>
#include <vo/camera.hpp>
#include <vo/jaccobian.hpp>

#include <utils/utils.hpp>

namespace icia {

    /**
     * @brief ICIA optical flow estimation
     * @cite 'Lucas-Kanade 20 Years On: A Unifying Framework'
     */ 
    struct _algo_impl : vslam::twoframe_estimator::algo_impl {
        
        static constexpr int half_sz  = vslam::twoframe_estimator::win_half_sz;
        static constexpr int sz       = vslam::twoframe_estimator::win_sz;
        static constexpr int area     = sz * sz;
        static constexpr int check_sz = half_sz + 1;

        static constexpr size_t min_visibles = std::numeric_limits<size_t>::max();

        using jacc_type = Eigen::Matrix<double, 6, area>;

        virtual ~_algo_impl() = default;

        size_t optimize_single_level(
            const vslam::frame_ptr&   ref,
            const vslam::frame_ptr&   cur,
            size_t                    level,
            size_t                    n_iterations,
            Sophus::SE3d&             t_cr
        ) override {
            size_t n_visibles = _compute_cache(ref, level);
            if (n_visibles < min_visibles) {
                // TODO too few features to track
            }

            // constant variables
            const double   scale = (1 << level);
            const cv::Mat& img   = cur->pyramid[level];

            double       last_chi2 = std::numeric_limits<double>::max();
            Sophus::SE3d last_t_cr = t_cr;

            Sophus::Matrix6d hessian;
            Sophus::Vector6d jres;

            size_t itr = 0;
            for (; itr < n_iterations; ++itr) {

                double chi2 = 0.0;
                hessian.setZero();
                jres.setZero();

                size_t feat_idx = 0;
                for (auto& each_feat : ref->features) {
                    if (!_visibles[feat_idx]) { ++feat_idx; continue; }

                    Eigen::Vector3d xyz = t_cr * _xyzs_ref[feat_idx];
                    Eigen::Vector2d uv  = cur->camera->cam2pixel(xyz) / scale; 
                    if (!utils::in_image(img, uv.x(), uv.y(), check_sz)) { 
                        ++feat_idx; continue;
                    }

                    int iu = uv.x(), iv = uv.y();
                    Eigen::Matrix2d w = utils::bilinear_weights(uv);

                    const auto& jacc  = _jaccobians[feat_idx];
                    const auto& templ = _templates[feat_idx]; 

                    const uint8_t* t_ptr = templ.data;

                    const int      i_stride = img.step.p[0];
                    const uint8_t* i_ptr    = img.data + (iv - half_sz) * i_stride + (iu - half_sz);

                    size_t count_pixels = 0;
                    for (int r = 0; r < sz; ++r) {
                        for (int c = 0; c < sz; ++c) {
                            double intensity_i = 
                                w(0, 0) * i_ptr[0]        + w(0, 1) * i_ptr[1] + 
                                w(1, 0) * i_ptr[i_stride] + w(1, 1) * i_ptr[i_stride + 1];
                            double err = intensity_i - *((const float*) t_ptr);
                            chi2 += 0.5 * err * err;
                            Sophus::Vector6d j = jacc.col(count_pixels++);
                            jres    += j * err;
                            hessian += j * j.transpose();

                            ++i_ptr;
                            t_ptr += 4;
                        }
                        i_ptr += (i_stride - sz);
                    }
                    ++feat_idx;
                }

                Sophus::Vector6d update = hessian.ldlt().solve(jres);
                if (update.hasNaN()) { assert(false); }

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

                t_cr = t_cr * Sophus::SE3d::exp(-update);
            }

            return itr;
        }

    protected:
        /**
         * @brief ICIA compute descent images grad(T)*dWdp
         * @cite 'Lucas-Kanade 20 Years On: A Unifying Framework'
         */ 
        size_t _compute_cache(const vslam::frame_ptr& ref, size_t level) {
            // constant variables
            const double          scale    = (1 << level);
            const cv::Mat&        img      = ref->pyramid[level];
            const Eigen::Matrix2d duv0dxy1 = ref->camera->focal_len().asDiagonal();
            const double          duvnduv0 = 1. / scale;

            _clear_cache();

            _visibles.resize(ref->n_features, false);
            _xyzs_ref.resize(ref->n_features);
            _templates.resize(ref->n_features, cv::Mat(sz, sz, CV_32FC1));
            _jaccobians.resize(ref->n_features);

            size_t count_visibles = 0;
            size_t feat_idx       = 0;

            for (auto& each_feat : ref->features) {
                if (//level != each_feat->level || 
                    each_feat->describe_nothing()) 
                { ++feat_idx; continue; }
                Eigen::Vector2d uv_n = each_feat->uv / scale;
                if (!utils::in_image(img, uv_n.x(), uv_n.y(), check_sz)) {
                    ++feat_idx; continue;
                }
                _visibles[feat_idx] = true;
                ++count_visibles;

                Eigen::Vector3d xyz = ref->t_cw * each_feat->map_point_describing->position;
                _xyzs_ref[feat_idx] = xyz;
                
                int    iu = uv_n.x()     , iv = uv_n.y();
		        double dx = uv_n.x() - iu, dy = uv_n.y() - iv;

		        double w00 = (1. - dx) * (1. - dy);
		        double w01 =        dx * (1. - dy);
		        double w10 = (1. - dx) * dy;
		        double w11 =        dx * dy;

                Eigen::Matrix26d duv0deps = duv0dxy1 * vslam::jaccobian_dxy1deps(xyz);

                cv::Mat&   templ = _templates[feat_idx];
                jacc_type& jacc  = _jaccobians[feat_idx];

                uint8_t* t_ptr = templ.data;

                const int      i_stride = img.step.p[0];
                const uint8_t* i_ptr    = img.data + (iv - half_sz) * i_stride + iu - half_sz;

                size_t count_pixels = 0;
                for (int r = 0; r < sz; ++r) {
                    for (int c = 0; c < sz; ++c) {
                        *((float*) t_ptr) = 
                            w00 * i_ptr[0]        + w01 * i_ptr[1] +
					        w10 * i_ptr[i_stride] + w11 * i_ptr[i_stride + 1];
                        
                        double gx =        dy * (double(i_ptr[i_stride + 1]) - double(i_ptr[i_stride])) +
					                (1. - dy) * (double(i_ptr[1]) - double(i_ptr[0]));
				        double gy =        dx * (double(i_ptr[i_stride + 1]) - double(i_ptr[1])) +
				                    (1. - dx) * (double(i_ptr[i_stride]) - double(i_ptr[0]));
                        Eigen::Matrix<double, 1, 2> dInduvn; dInduvn << gx, gy;
                        jacc.col(count_pixels++) = dInduvn * duvnduv0 * duv0deps;

                        ++i_ptr;
                        t_ptr += 4;
                    }
                    i_ptr += (i_stride - sz);
                }
                ++feat_idx;
            }
            return count_visibles;
        }

        void _clear_cache() {
            _visibles.clear();
            _xyzs_ref.clear();
            _templates.clear();
            _jaccobians.clear();
        }

        /**
         * @brief since the the jaccobians and the feature 
         *        patches of reference frame will be used 
         *        several times
         * @field caches 
         */
        std::vector<bool>            _visibles;
        std::vector<Eigen::Vector3d> _xyzs_ref;
        std::vector<cv::Mat>         _templates;
        std::vector<jacc_type>       _jaccobians;
    };

    /**
     * @brief g2o vertex
     */ 
    struct _vertex_pose_only : 
        g2o::BaseVertex<6, Sophus::SE3d> {

        _vertex_pose_only() = default;
        ~_vertex_pose_only() = default;
        
        bool read(std::istream& is) override { return false; }
        bool write(std::ostream& os) const override { return false; }

        void setToOriginImpl() override { _estimate = Sophus::SE3d(); }

        void oplusImpl(const double* u) override {
            Sophus::Vector6d update;
            update << u[0], u[1], u[2], u[3], u[4], u[5];
            _estimate = _estimate * Sophus::SE3d::exp(-update);
        }
    };

    /**
     * @brief g2o edge
     */ 
    struct _uedge_patch_photometric_err : 
        g2o::BaseUnaryEdge<1, cv::Mat, _vertex_pose_only> {

        _uedge_patch_photometric_err(
            const Eigen::Vector3d&          xyz_ref, 
            const cv::Mat&                  templ, 
            size_t                          level,
            const vslam::camera_ptr&        cam,
            const _algo_impl::jacc_type&    jacc
        ) : _visible(true), _xyz_ref(xyz_ref), _templ(templ),
            _level(level), _camera(cam), _jacc_caches(jacc) 
        { }

        bool read(std::istream& is) override { return false; }
        bool write(std::ostream& os) const override { return false; }

        void computeError() override {
            if (!_visible) { this->setLevel(1); return; }

            auto v = (const _vertex_pose_only*) _vertices[0];
            const auto& t_cr = v->estimate();
            const cv::Mat& img = _measurement;

            Eigen::Vector3d xyz  = t_cr * _xyz_ref;
            Eigen::Vector2d uv_0 = _camera->cam2pixel(xyz);
            Eigen::Vector2d uv_n = uv_0 / (1 << _level);

            _error.setZero();

            if (!utils::in_image(
                    img, uv_n.x(), uv_n.y(), _algo_impl::check_sz
                )
            ) { _visible = false; this->setLevel(1); return; }
            
            auto w = utils::bilinear_weights(uv_n);

            const uint8_t* t_ptr = _templ.data;
            
            const int      i_stride = img.step.p[0];
            const uint8_t* i_ptr    = img.data + 
                                        (int(uv_n.y()) - _algo_impl::half_sz) * i_stride + 
                                        (int(uv_n.x()) - _algo_impl::half_sz);

            for (int r = 0; r < _algo_impl::sz; ++r) {
                for (int c = 0; c < _algo_impl::sz; ++c) {
                    float intensity_i = w(0, 0) * i_ptr[0] + 
                                        w(0, 1) * i_ptr[1] + 
                                        w(1, 0) * i_ptr[i_stride] + 
                                        w(1, 1) * i_ptr[i_stride + 1];
                    _error(0, 0) += (intensity_i - *((float*) t_ptr));

                    t_ptr += 4; 
                    ++i_ptr;
                }
                i_ptr += (i_stride - _algo_impl::sz);
            }
        }

        void linearizeOplus() override {

            if (!_visible) { this->setLevel(1); return; }

            auto v = (const _vertex_pose_only*) _vertices[0];
            const auto& t_cr = v->estimate();
            const cv::Mat& img = _measurement;

            Eigen::Vector3d xyz  = t_cr * _xyz_ref;
            Eigen::Vector2d uv_0 = _camera->cam2pixel(xyz);
            Eigen::Vector2d uv_n = uv_0 / (1 << _level);

            if (!utils::in_image(
                    img, uv_n.x(), uv_n.y(), _algo_impl::check_sz
                )
            ) { _visible = false; this->setLevel(1); return; }

            for (int i = 0; i < _algo_impl::area; ++i) {
                _jacobianOplusXi -= _jacc_caches.col(i);
            }
        }

    private:
        bool                         _visible;
        const Eigen::Vector3d        _xyz_ref;
        const cv::Mat&               _templ;
        size_t                       _level;
        vslam::camera_ptr            _camera;
        const _algo_impl::jacc_type& _jacc_caches;
    };

    struct _g2o_impl : _algo_impl {

        using block_solver_t  = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
        using linear_solver_t = g2o::LinearSolverDense<typename block_solver_t::PoseMatrixType>;

        _g2o_impl() {
            auto algo = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<block_solver_t>(
                    g2o::make_unique<linear_solver_t>()
                )
            );
            _optimizer.setAlgorithm(algo);
        }

        size_t optimize_single_level(
            const vslam::frame_ptr&   ref,
            const vslam::frame_ptr&   cur,
            size_t                    level,
            size_t                    n_iterations,
            Sophus::SE3d&             t_cr
        ) override {
            _optimizer.clear();
            _optimizer.setVerbose(true);
            //_optimizer.setVerbose(false);

            // create vertex, only one vertex
            icia::_vertex_pose_only* v = new icia::_vertex_pose_only();
            v->setId(0);
            v->setEstimate(t_cr);

            // add the vertex to graph
            _optimizer.addVertex(v);

            //create edges
            size_t idx = 0;
            size_t count_edges = 0;

            for (const auto& each_feat : ref->features) {
                if (!_visibles[idx]) { ++idx; continue; }
                Eigen::Vector3d xyz_ref = 
                    ref->t_cw * each_feat->map_point_describing->position;
            
                auto& templ = _templates[idx];
                auto& jacc  = _jaccobians[idx];

                icia::_uedge_patch_photometric_err* e = 
                    new icia::_uedge_patch_photometric_err(
                        xyz_ref, templ, level, cur->camera, jacc
                    );
                e->setId(count_edges);
                e->setVertex(0, v);
                e->setRobustKernel(new g2o::RobustKernelHuber());
                e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
                e->setMeasurement(cur->pyramid[level]);

                // add the edge to graph
                _optimizer.addEdge(e);
            
                ++count_edges;
                ++idx;
            }

            _optimizer.initializeOptimization();
            int itr = _optimizer.optimize(n_iterations);
            t_cr = v->estimate();

            return itr;
        }

    private:
        /**
         * @field g2o optimizer
         */ 
        g2o::SparseOptimizer _optimizer;
    };
}

namespace fcfa {

    struct _algo_impl : vslam::twoframe_estimator::algo_impl {

        static constexpr int check_sz = vslam::twoframe_estimator::win_half_sz + 1;
        static constexpr int half_sz  = vslam::twoframe_estimator::win_half_sz;
        static constexpr int sz       = vslam::twoframe_estimator::win_sz;

        size_t optimize_single_level(
            const vslam::frame_ptr&   ref,
            const vslam::frame_ptr&   cur,
            size_t                    level,
            size_t                    n_iterations,
            Sophus::SE3d&             t_cr
        ) override {
            
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
                        //level != feat_ref->level || 
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
    };

} // namespace fcfa

namespace vslam {

    twoframe_estimator::twoframe_estimator(
        size_t    n_iterations,
        size_t    max_level,   
        size_t    min_level,  
        algorithm algo        
    ) : _n_iterations(n_iterations), 
        _min_level(min_level), 
        _max_level(max_level) 
    {
        assert(_min_level <= _max_level);
        switch (algo) {
            case LK_FCFA : {
                _algo_impl = utils::mk_vptr<fcfa::_algo_impl>();
                break;
            }
            case LK_ICIA : {
                _algo_impl = utils::mk_vptr<icia::_algo_impl>();
                break;
            }
            case LK_ICIA_G2O : {
                _algo_impl = utils::mk_vptr<icia::_g2o_impl>();
                break;
            }
            default : assert(false);
        }
    }

    void twoframe_estimator::estimate(
        const frame_ptr& ref, 
        const frame_ptr& cur, 
        Sophus::SE3d&    t_cr
    ) {
        auto l = _max_level + 1;
        while (_min_level < l) {
            _algo_impl->optimize_single_level(
                ref, cur, --l, _n_iterations, t_cr
            );
        }
    }

} // namespace vslam

namespace pnp {

    struct _ba_impl : 
        vslam::singleframe_estimator::algo_impl {

        size_t estimate(
            const vslam::frame_ptr& frame, 
            size_t                  n_iterations,
            Sophus::SE3d&           t_cw
        ) override {
            t_cw = frame->t_cw;

            size_t itr = 0;
            double last_chi2 = 0.0;
            Sophus::SE3d last_t(t_cw);

            Sophus::Matrix6d hessian;
            Sophus::Vector6d jres;

            for (; itr < n_iterations; ++itr) {

                double chi2 = 0.0;
                hessian.setZero();
                jres.setZero();

                for (auto& each : frame->features) {
                    if (each->describe_nothing()) { continue; }

                    Eigen::Vector3d xyz = t_cw * each->map_point_describing->position;
                    Eigen::Vector2d err = each->xy1.head<2>() - utils::project(xyz);
                    Eigen::Matrix26d jacc = vslam::jaccobian_dxy1deps(xyz) * (-1.);

                    chi2 += 0.5 * err.norm();
                    jres += jacc.transpose() * (-err);
                    hessian += jacc.transpose() * jacc;
                }

                Sophus::Vector6d delta = hessian.ldlt().solve(jres);
                if (delta.hasNaN()) { assert(false); }

                if (0 < itr && last_chi2 < chi2) { 
                    std::cout << "loss increasing at " << itr << std::endl;
                    t_cw = last_t;
                    chi2 = last_chi2;
                    break;
                }

                if (delta.norm() < CONST_EPS) { 
                    std::cout << "converged at " << itr << std::endl;
                    break;
                }

                last_t = t_cw;
                last_chi2 = chi2;

                t_cw = Sophus::SE3d::exp(delta) * t_cw;
            }

            return itr;
        }
    };

    struct _cv_impl : 
        vslam::singleframe_estimator::algo_impl {

        size_t estimate(
            const vslam::frame_ptr& frame, 
            size_t                  n_iterations,
            Sophus::SE3d&           t_cw
        ) override {
            _clear_cache();
            _load_cache(frame);

            cv::Mat rvec, tvec;
            _se3_to_cv(frame->t_cw, rvec, tvec);
            this->cv_api(frame->camera->cv_mat(), rvec, tvec);
            _se3_from_cv(rvec, tvec, t_cw);
            return 1;
        }

    protected:
        virtual void cv_api(
            const cv::Mat& cam_mat, 
            const cv::Mat& rvec, 
            const cv::Mat& tvec
        ) = 0;

        void _clear_cache() { _xyzs.clear(); _uvs.clear(); }

        void _load_cache(const vslam::frame_ptr& frame) {
            _xyzs.reserve(frame->n_features);
            _uvs.reserve(frame->n_features);
            for (auto& each : frame->features) {
                if (each->describe_nothing()) { continue; }
                const auto& uv  = each->uv;
                const auto& xyz = each->map_point_describing->position;
                _uvs.emplace_back(uv.x(), uv.y());
                _xyzs.emplace_back(xyz.x(), xyz.y(), xyz.z());
            }
        }

        static void _se3_to_cv(
            const Sophus::SE3d& t_cr, 
            cv::Mat&            rvec, 
            cv::Mat&            tvec
        ) { 
            Sophus::Vector6d se3 = t_cr.log();
            rvec = (cv::Mat_<double>(3, 1) << se3[3], se3[4], se3[5]);
            tvec = (cv::Mat_<double>(3, 1) << se3[0], se3[1], se3[2]);
        }

        static void _se3_from_cv(
            const cv::Mat& rvec, 
            const cv::Mat& tvec,
            Sophus::SE3d&  t_cr
        ) {
            Sophus::Vector6d se3;
            se3 << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2), 
                   rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2);
            t_cr = Sophus::SE3d::exp(se3);
        }

        /**
         * @field cache
         */
        std::vector<cv::Point3d> _xyzs;
        std::vector<cv::Point2d> _uvs;
    };

    struct _pnp_refine_cv_impl : _cv_impl {

        void cv_api(
            const cv::Mat& cam_mat, 
            const cv::Mat& rvec, 
            const cv::Mat& tvec
        ) override {
            cv::solvePnPRefineLM(_xyzs, _uvs, cam_mat, cv::Mat(), rvec, tvec);
        }
    };

    struct _epnp_cv_impl : _cv_impl {

        void cv_api(
            const cv::Mat& cam_mat, 
            const cv::Mat& rvec, 
            const cv::Mat& tvec
        ) override {
            cv::solvePnP(_xyzs, _uvs, cam_mat, cv::Mat(), rvec, tvec, false, cv::SOLVEPNP_EPNP);
        }
    };

    struct _pnp_dls_cv_impl : _cv_impl {

        void cv_api(
            const cv::Mat& cam_mat, 
            const cv::Mat& rvec, 
            const cv::Mat& tvec
        ) override {
            cv::solvePnP(_xyzs, _uvs, cam_mat, cv::Mat(), rvec, tvec, false, cv::SOLVEPNP_DLS);
        }
    };
    
} // namespace pnp

namespace vslam {

    singleframe_estimator::singleframe_estimator(
        size_t n_iterations, algorithm algo
    ) : _n_iterations(n_iterations) 
    {
        switch (algo) {
            case PNP_BA : {
                _algo_impl = utils::mk_vptr<pnp::_ba_impl>();
                break;
            }
            case PNP_G2O : { assert(false); }
            case PNP_CV : { 
                _algo_impl = utils::mk_vptr<pnp::_pnp_refine_cv_impl>();
                break;
            }
            case EPNP_CV : {
                _algo_impl = utils::mk_vptr<pnp::_epnp_cv_impl>();
                break;
            }
            case PNP_DLS_CV : {
                _algo_impl = utils::mk_vptr<pnp::_pnp_dls_cv_impl>();
                break;
            }
            default : assert(false);
        }
    }

    void singleframe_estimator::compute_inliers_and_reporj_err(
        const frame_ptr&          frame, 
        const Sophus::SE3d&       t_cw, 
        double                    reproj_thresh,
        std::vector<feature_ptr>& inliers,
        std::vector<feature_ptr>& outliers,
        double&                   err
    ) {
        err = 0;
        inliers.clear();  inliers.reserve(frame->n_features);
        outliers.clear(); outliers.reserve(frame->n_features);
        
        for (auto& each : frame->features) {
            if (each->describe_nothing()) { 
                outliers.emplace_back(each);
                continue; 
            }

            Eigen::Vector2d xy_unit_plane = 
                utils::project(t_cw * each->map_point_describing->position);
            double each_err = (xy_unit_plane - each->xy1.head<2>()).norm();
            err += 0.5* each_err;
            if (reproj_thresh < each_err) { 
                outliers.emplace_back(each);
                continue;  
            }
            inliers.emplace_back(each);
        }
    }
    
} // namespace vslam


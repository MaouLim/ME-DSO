#include <vo/pose_estimator.hpp>
#include <vo/frame.hpp>
#include <vo/feature.hpp>
#include <vo/map_point.hpp>
#include <vo/camera.hpp>
#include <vo/jaccobian.hpp>

#include <utils/utils.hpp>

namespace icia {

    struct _vertex_pose_only : 
        g2o::BaseVertex<6, Sophus::SE3d> {

        _vertex_pose_only() = default;
        
        bool read(std::istream& is) override { return false; }
        bool write(std::ostream& os) const override { return false; }

        void setToOriginImpl() override { _estimate = Sophus::SE3d(); }

        void oplusImpl(const double* u) override {
            Sophus::Vector6d update;
            update << u[0], u[1], u[2], u[3], u[4], u[5];
            _estimate = _estimate * Sophus::SE3d::exp(update).inverse();
        }
    };

    struct _uedge_patch_photometric_err : 
        g2o::BaseUnaryEdge<1, std::pair<uint8_t*, int>, _vertex_pose_only> {

        using patch_type = vslam::pose_estimator::patch_type;

        _uedge_patch_photometric_err() : _jacc_set(false) { }

        bool read(std::istream& is) override { return false; }
        bool write(std::ostream& os) const override { return false; }

        void computeError() override {
            auto v = (const _vertex_pose_only*) _vertices[0];
            auto t_cr = v->estimate();
            Eigen::Vector3d xyz_cur   = t_cr * _xyz_ref;
            Eigen::Vector2d uv_cur    = _camera->cam2pixel(xyz_cur);
            Eigen::Vector2d uv_leveln = uv_cur / (1 << _level);

            if (!utils::in_image(
                    _img_leveln_cur, uv_leveln.x(), uv_leveln.y(), _check_sz
                )
            ) { _error.setZero(); this->setLevel(1); return; }
            
            auto w = utils::bilinear_weights(uv_leveln);

            const float* ref_ptr    = _patch_ref.data;
            const int    ref_stride = patch_type::size;

            const uint8_t* cur_ptr    = _measurement.first;
            const int      cur_stride = _measurement.second;

            _error.setZero();

            size_t count_pixels = 0;
            
            for (int r = 0; r < patch_type::size; ++r) {
                for (int c = 0; c < patch_type::size; ++c) {
                    float intensity_cur = w(0, 0) * cur_ptr[0] + 
                                          w(0, 1) * cur_ptr[1] + 
                                          w(1, 0) * cur_ptr[cur_stride] + 
                                          w(1, 1) * cur_ptr[cur_stride + 1];
                    _error(0, 0) += intensity_cur - *ref_ptr;
                    ++ref_ptr; ++cur_ptr;

                    _jacc.noalias() += _error(0, 0) * _jacc_caches.row(count_pixels++);
                }
                cur_ptr += (cur_stride - patch_type::size);
            }
            _jacc_set = true;
        }

        void linearizeOplus() override {
            assert(_jacc_set);
            _jacobianOplusXi = _jacc;
        }

    private:
        static constexpr int _check_sz = patch_type::half_sz + 1;

        const Eigen::Vector3d&   _xyz_ref;
        const patch_type&        _patch_ref;
        const cv::Mat&           _img_leveln_cur;
        size_t                   _level;
        const vslam::camera_ptr& _camera;

        const Eigen::Matrix<double, patch_type::area, 6>& _jacc_caches;
        bool                                              _jacc_set;
        Eigen::Matrix<double, 1, 6>                       _jacc;
    };

} // namespace icia

namespace vslam {

    pose_estimator::pose_estimator() { 
        using block_solver_t  = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
        using linear_solver_t = g2o::LinearSolverDense<typename block_solver_t::PoseMatrixType>;
        _algo = new g2o::OptimizationAlgorithmGaussNewton(
            g2o::make_unique<block_solver_t>(
	            g2o::make_unique<linear_solver_t>()
	        )
        );
        _optimizer.setAlgorithm(_algo);
    }

    pose_estimator::~pose_estimator() { }

    size_t pose_estimator::estimate(
        const frame_ptr& ref, 
        const frame_ptr& cur, 
        Sophus::SE3d&    t_cr
    ) {
        _t_cr = t_cr;
        auto idx = _max_level + 1;
        while (_min_level < idx) {
            --idx;
            _clear_cache();
            _precalc_cache(ref, idx);
            _init_graph(cur, idx);
            _optimizer.optimize(_n_iterations);
        }
    }

    void pose_estimator::_precalc_cache(
        const frame_ptr& ref, size_t level
    ) {
        const Eigen::Vector2d& focal_len = ref->camera->focal_len();
        const double scale = (1 << level);
        const cv::Mat& img_leveln = ref->pyramid[level];
        const int img_stride = img_leveln.step.p[0];
        constexpr int check_sz = patch_type::half_sz + 1;

        _patches_ref.resize(ref->n_features);
        _jaccobians_ref.resize(ref->n_features);
        _visibles_ref.resize(ref->n_features, false);

        size_t idx = 0;

        for (const auto& each_feat : ref->features) {
            if (each_feat->describe_nothing()) { continue; }
            Eigen::Vector2d uv_leveln = each_feat->uv / scale;
            if (utils::in_image(
                    img_leveln, uv_leveln.x(), uv_leveln.y(), check_sz
                )
            ) { continue; }

            _visibles_ref[idx] = true;
            
            Eigen::Vector3d xyz_ref = 
                ref->t_cw * each_feat->map_point_describing->position;
            
            auto dxy1deps = jaccobian_dxy1deps(xyz_ref);

            auto& patch = _patches_ref[idx];
            auto& jacc  = _jaccobians_ref[idx];

            int x = std::floor(uv_leveln.x());
            int y = std::floor(uv_leveln.y());
            double dx = uv_leveln.x() - x;
            double dy = uv_leveln.y() - y;

            double w00 = (1. - dx) * (1. - dy);
            double w01 =        dx * (1. - dy);
            double w10 = (1. - dx) * dy;
            double w11 =        dx * dy;

            float*   patch_ptr = patch.start();
            uint8_t* img_ptr   = 
                img_leveln.data + (y - patch_type::half_sz) * img_stride + (x - patch_type::half_sz);
                
            size_t count_pixels = 0;

            for (int r = 0; r < patch_type::size; ++r) {
                for (int c = 0; c < patch_type::size; ++c) {
                    *patch_ptr = w00 * img_ptr[0] + 
                                 w01 * img_ptr[1] + 
                                 w10 * img_ptr[img_stride] + 
                                 w11 * img_ptr[img_stride + 1];
                    double gx =        dy * (img_ptr[img_stride + 1] - img_ptr[img_stride]) + 
                                (1. - dy) * (img_ptr[1] - img_ptr[0]);
                    double gy =        dx * (img_ptr[img_stride + 1] - img_ptr[1]) + 
                                (1. - dx) * (img_ptr[img_stride] - img_ptr[0]);
                    // dI(n)deps = dI(n)dI(0) * dI(0)du * dudxy1 * dxy1deps
                    //              1/scale    (gx, gy)   [fx, 0]
                    //                                    [0, fy]  
                    jacc.row(count_pixels) = 
                        (gx * focal_len.x() * dxy1deps.row(0) + 
                         gy * focal_len.y() * dxy1deps.row(1)) / scale;
                    ++count_pixels;
                }
            }

            ++idx;
        }
    }

    void pose_estimator::_clear_cache() {
        _jaccobians_ref.clear();
        _patches_ref.clear();
        _visibles_ref.clear();
    }

    void pose_estimator::_init_graph(const frame_ptr& cur, size_t level) {
        _optimizer.clear();

        // create vertex
        icia::_vertex_pose_only* v = new icia::_vertex_pose_only();
        v->setId(0);
        v->setEstimate(_t_cr);

        // add the vertex to graph
        _optimizer.addVertex(v);

    }
    
} // namespace vslam

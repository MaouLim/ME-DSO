#include <iostream>
#include <random>

#include <vo/matcher.hpp>

using namespace vslam;
using namespace Eigen;

std::mt19937_64 rand_engine(time(0));
std::normal_distribution<float> dist(1.0, 2.0);

bool align2D(
    const cv::Mat& cur_img,
    uint8_t* ref_patch_with_border,
    uint8_t* ref_patch,
    const int n_iter,
    Vector2d& cur_px_estimate,
    bool no_simd)
{

  const int halfpatch_size_ = 4;
  const int patch_size_ = 8;
  const int patch_area_ = 64;
  bool converged=false;

  // compute derivative of template and prepare inverse compositional
  //* 模板图像对像素 x,y 坐标进行求导
  float __attribute__((__aligned__(16))) ref_patch_dx[patch_area_];
  float __attribute__((__aligned__(16))) ref_patch_dy[patch_area_];
  Matrix3f H; H.setZero();

  // compute gradient and hessian
  const int ref_step = patch_size_+2;
  float* it_dx = ref_patch_dx;
  float* it_dy = ref_patch_dy;
  for(int y=0; y<patch_size_; ++y) // 行循环
  {
    uint8_t* it = ref_patch_with_border + (y+1)*ref_step + 1; // ref图像去掉border部分的行首指针
    for(int x=0; x<patch_size_; ++x, ++it, ++it_dx, ++it_dy)
    {
      Vector3f J;
      J[0] = 0.5 * (it[1] - it[-1]); // x方向导数
      J[1] = 0.5 * (it[ref_step] - it[-ref_step]); // y方向导数
      J[2] = 1; // 亮度误差导数
      *it_dx = J[0];
      *it_dy = J[1];
      H += J*J.transpose(); // Hessian矩阵求和
    }
  }
  Matrix3f Hinv = H.inverse();
  float mean_diff = 0;

  // Compute pixel location in new image:
  float u = cur_px_estimate.x();
  float v = cur_px_estimate.y();

  // termination condition
  const float min_update_squared = 0.03*0.03;
  const int cur_step = cur_img.step.p[0];
  float chi2 = 0;
  Vector3f update; update.setZero();
  for(int iter = 0; iter<n_iter; ++iter)
  {
    int u_r = floor(u);
    int v_r = floor(v);
    if(u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols-halfpatch_size_ || v_r >= cur_img.rows-halfpatch_size_)
      break;

    if(isnan(u) || isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
      return false;

    // compute interpolation weights
    float subpix_x = u-u_r;
    float subpix_y = v-v_r;
    float wTL = (1.0-subpix_x)*(1.0-subpix_y);
    float wTR = subpix_x * (1.0-subpix_y);
    float wBL = (1.0-subpix_x)*subpix_y;
    float wBR = subpix_x * subpix_y;

    // loop through search_patch, interpolate
    uint8_t* it_ref = ref_patch;
    float* it_ref_dx = ref_patch_dx;
    float* it_ref_dy = ref_patch_dy;
    float new_chi2 = 0.0;
    Vector3f Jres; Jres.setZero();
    for(int y=0; y<patch_size_; ++y)
    {
      uint8_t* it = (uint8_t*) cur_img.data + (v_r+y-halfpatch_size_)*cur_step + u_r-halfpatch_size_; // cur图像的patch行首指针
      for(int x=0; x<patch_size_; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy)
      {
        float search_pixel = wTL*it[0] + wTR*it[1] + wBL*it[cur_step] + wBR*it[cur_step+1]; // cur图像线性插值
        float res = search_pixel - *it_ref + mean_diff;
        Jres[0] -= res*(*it_ref_dx);
        Jres[1] -= res*(*it_ref_dy);
        Jres[2] -= res;
       new_chi2 += res*res;
      }
    }
    if (0 < iter && chi2 < new_chi2) {  
        std::cout << "LOSS INC AT" << iter << std::endl;
    }

    chi2 = new_chi2;

    update = Hinv * Jres;
    u += update[0];
    v += update[1];
    mean_diff += update[2];

    if(update[0]*update[0]+update[1]*update[1] < min_update_squared)
    {
#if SUBPIX_VERBOSE
      cout << "converged." << endl;
#endif
      converged=true;
      break;
    }
  }

  cur_px_estimate << u, v;
  return converged;
}

int main(int argc, char** argv) {

    cv::Mat img_ref = cv::imread("data/01.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img_cur = cv::imread("data/04.png", cv::IMREAD_GRAYSCALE);

    assert(img_ref.data && img_cur.data);

    cv::Ptr<cv::GFTTDetector> det = cv::GFTTDetector::create(1000, 0.01, 5);
    std::vector<cv::KeyPoint> kpts;
    det->detect(img_ref, kpts);

    std::vector<cv::Point2f> uvs_ref, uvs_cur;
    for (auto& each : kpts) { uvs_ref.push_back(each.pt); }
    uvs_cur = uvs_ref;
    std::vector<uchar> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(img_ref, img_cur, uvs_ref, uvs_cur, status, err);

    cv::Mat a0 = img_ref.clone(), a1 = img_cur.clone();
    for (auto i = 0; i < status.size(); ++i) {
        if (!status[i]) { continue; }
        if (!utils::in_image(img_ref, uvs_ref[i].x, uvs_ref[i].y, 10)) {
            status[i] = 0; continue;
        }
        if (!utils::in_image(img_cur, uvs_cur[i].x, uvs_cur[i].y, 10)) {
            status[i] = 0; continue;
        }
        cv::circle(a0, uvs_ref[i], 2, 255, 1);
        cv::circle(a1, uvs_cur[i], 2, 255, 1);
    }

    cv::imshow("a0", a0);
    cv::imshow("a1", a1);
    cv::waitKey();

    std::vector<cv::Point2f> uvs_cur_noisy;
    for (auto& each : uvs_cur) {
        cv::Point2f noise = { 0.5f + 2.f * std::sin(each.x), 0.5f + 2.f * std::cos(each.y) };
        uvs_cur_noisy.push_back(each + noise);
    }

    for (auto i = 0; i < status.size(); ++i) {
        if (!status[i]) { continue; }
        cv::circle(a1, uvs_cur_noisy[i], 2, 0, 1);
    }
    cv::imshow("a1", a1);
    cv::waitKey();

    const int half_sz = 5;

    for (auto i = 0; i < status.size(); ++i) {
        if (!status[i]) { continue; }

        patch_t patch_ref;
        uint8_t* ptr = patch_ref.data;

        double center_x = uvs_ref[i].x;
        double center_y = uvs_ref[i].y;

        for (auto r = 0; r < 10; ++r) {
            for (auto c = 0; c < 10; ++c) {
                double x = center_x - 5. + c;
                double y = center_y - 5. + r;
                double v = utils::bilinear_interoplate<uint8_t>(img_ref, x, y);
                assert(v < 255.);
                *ptr = (uint8_t) v;
                ++ptr;
            }
        }

        auto pt = uvs_cur_noisy[i];
        Eigen::Vector2d uv_cur = { pt.x, pt.y };
        alignment::align2d(img_cur, patch_ref, 10, uv_cur);
        uvs_cur_noisy[i] = cv::Point2f(uv_cur.x(), uv_cur.y());
    }

    cv::Mat a2;
    cv::cvtColor(img_cur, a2, cv::COLOR_GRAY2RGB);
    for (auto i = 0; i < status.size(); ++i) {
        if (!status[i]) { continue; }
        cv::circle(a2, uvs_cur_noisy[i], 2, { 0, 0, 255 }, 1);
    }
    cv::imshow("a2", a2);
    cv::waitKey();

    return 0;
}
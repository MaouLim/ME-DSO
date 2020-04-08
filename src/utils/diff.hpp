#ifndef _ME_VSLAM_DIFF_HPP_
#define _ME_VSLAM_DIFF_HPP_

namespace utils {

    /**
     * @brief to compute difference between two 2D-array
     */
    template <
        typename _Dtype, int _BlockSize = 8
    >
    struct diff_2d {

        static const int block_sz   = _BlockSize;
        static const int block_area = _BlockSize * _BlockSize;

        using ptr  = _Dtype*;
        using cptr = const _Dtype*;

        /**
         * @brief Sum of Absolute Difference of 2D area
         */ 
        static double sad(
            const _Dtype* a, int a_stride, 
            const _Dtype* b, int b_stride
        );

        /**
         * @brief Sum of Squared Distance
         */ 
        static double ssd(
            const _Dtype* a, int a_stride, 
            const _Dtype* b, int b_stride
        );

        /**
         * @brief Zero mean Sum of Squared Distance
         */ 
        static double zm_ssd(
            const _Dtype* a, int a_stride, 
            const _Dtype* b, int b_stride
        );

        /**
         * @brief Zero mean Normalized Cross Correlation
         */ 
        static double zm_ncc(
            const _Dtype* a, int a_stride, 
            const _Dtype* b, int b_stride
        );
    };

    template <
        typename _Dtype, int _BlockSize
    >
    inline double 
    diff_2d<_Dtype, _BlockSize>::sad(
        const _Dtype* a, int a_stride, 
        const _Dtype* b, int b_stride
    ) {
        double sum = 0.;
        for (size_t r = 0; r < block_sz; ++r) {

            cptr a_row = a + r * a_stride;
            cptr b_row = b + r * b_stride;

            for (size_t c = 0; c < block_sz; ++c) {
                sum += std::abs(double(a_row[c]) - double(b_row[c]));
            }
        }
        return sum;
    }

    template <
        typename _Dtype, int _BlockSize
    >
    inline double 
    diff_2d<_Dtype, _BlockSize>::ssd(
        const _Dtype* a, int a_stride, 
        const _Dtype* b, int b_stride
    ) {
        double sum_a2 = 0., sum_ab = 0., sum_b2 = 0.;

        for (size_t r = 0; r < block_sz; ++r) {

            cptr a_row = a + r * a_stride;
            cptr b_row = b + r * b_stride;

            for (size_t c = 0; c < block_sz; ++c) {
                double a_rc = a_row[c];
                double b_rc = b_row[c];
                sum_a2 += a_rc * a_rc;
                sum_b2 += b_rc * b_rc;
                sum_ab += a_rc * b_rc;
            }
        }
        return sum_a2 - sum_ab + sum_b2;
    }

    template <
        typename _Dtype, int _BlockSize
    >
    inline double 
    diff_2d<_Dtype, _BlockSize>::zm_ssd(
        const _Dtype* a, int a_stride, 
        const _Dtype* b, int b_stride
    ) {
        double sum_a2 = 0., sum_ab = 0., sum_b2 = 0.;
        double sum_a = 0., sum_b = 0.;

        for (size_t r = 0; r < block_sz; ++r) {

            cptr a_row = a + r * a_stride;
            cptr b_row = b + r * b_stride;

            for (size_t c = 0; c < block_sz; ++c) {

                double a_rc = a_row[c];
                double b_rc = b_row[c];
                sum_a2 += a_rc * a_rc;
                sum_b2 += b_rc * b_rc;
                sum_ab += a_rc * b_rc;

                sum_a += a_rc;
                sum_b += b_rc;
            }
        }
        return sum_a2 - sum_ab + sum_b2 - // ssd
               (sum_a * sum_a - sum_a *sum_b * 2. + sum_b * sum_b) / block_area; // mean difference
    }

    template <
        typename _Dtype, int _BlockSize
    >
    inline double 
    diff_2d<_Dtype, _BlockSize>::zm_ncc(
        const _Dtype* a, int a_stride, 
        const _Dtype* b, int b_stride
    ) {
        double mean_a = 0., mean_b = 0.;
        for (size_t r = 0; r < block_sz; ++r) {

            cptr a_row = a + r * a_stride;
            cptr b_row = b + r * b_stride;

            for (size_t c = 0; c < block_sz; ++c) {
                double a_rc = a_row[c];
                double b_rc = b_row[c];
                mean_a += a_rc;
                mean_b += b_rc;
            }
        }

        mean_a /= block_area;
        mean_b /= block_area;

        double sum_ab = 0., sum_a2 = 0., sum_b2 = 0.;
        for (size_t r = 0; r < block_sz; ++r) {

            cptr a_row = a + r * a_stride;
            cptr b_row = b + r * b_stride;

            for (size_t c = 0; c < block_sz; ++c) {
                double zm_a = a_row[c] - mean_a;
                double zm_b = b_row[c] - mean_b;
                sum_ab += zm_a * zm_b;
                sum_a2 += zm_a * zm_a;
                sum_b2 += zm_b * zm_b;
            }
        }

        return sum_ab / std::sqrt(sum_a2 * sum_b2);
    }
    
} // namespace utils


#endif
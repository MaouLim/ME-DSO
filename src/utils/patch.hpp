#ifndef _ME_VSLAM_PATCH_HPP_
#define _ME_VSLAM_PATCH_HPP_

namespace utils {

    template <
        typename _Dtype, 
        int      _HalfSize, 
        int      _BorderSize = 0
    >
    struct patch2d {

        using value_type      = _Dtype;
        using pointer         = _Dtype*;
        using const_pointer   = const _Dtype*;
        using reference       = _Dtype&;
        using const_reference = const _Dtype&;

        static constexpr int half_sz          = _HalfSize;
        static constexpr int border_sz        = _BorderSize;
        static constexpr int size             = half_sz * 2;
        static constexpr int area             = size * size;
        static constexpr int size_with_border = size + border_sz * 2;
        static constexpr int area_with_border = size_with_border * size_with_border;

        value_type __attribute__ ((aligned (16))) data[area_with_border];

        const_pointer start() const { return data + size_with_border + border_sz;  }
        pointer       start()       { return data + size_with_border + border_sz;  }

        const_pointer row(size_t idx) const { return data + (idx + 1) * size_with_border + border_sz; }
        pointer       row(size_t idx)       { return data + (idx + 1) * size_with_border + border_sz; }

        const_reference operator()(size_t r, size_t c) const { return row(r)[c]; }
        reference       operator()(size_t r, size_t c)       { return row(r)[c]; }

        static constexpr int stride() { return size_with_border; }
    };

} // namespace utils


#endif
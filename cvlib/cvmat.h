#pragma once

template <typename T>
inline T *align_ptr(T *ptr, uintptr_t n = (uintptr_t)sizeof(T)) {
    return (T *)(((uintptr_t)ptr + n - 1) & -n);
}

template <typename T>
inline T align_ptr(T ptr, uintptr_t n = (uintptr_t)sizeof(T)) {
    return (T)(((uintptr_t)ptr + n - 1) & -n);
}

inline size_t align_size(size_t size, size_t n) {
    return (size + n - 1) & -n;
}

inline void *memory_alloc(size_t size, size_t alignment) {
    void *ptr = nullptr;
    int32_t ret = posix_memalign(&ptr, alignment, size);
    if (ret != 0) {
        if (ret == ENOMEM) {
            throw std::runtime_error("posix_memalign no sufficient memory available");
        } else if (ret == EINVAL) {
            throw std::runtime_error("posix_memalign alignment not a power of 2 multiple of sizeof(void *)");
        }
    }
    return ptr;
}

inline void memory_dealloc(void *ptr) {
    free(ptr);
}

struct Corner {
    int32_t x;
    int32_t y;
    int32_t level;
    double score;
    double angle;
    Corner(int32_t x, int32_t y, int32_t level, double score, double angle) :
        x(x), y(y), level(level), score(score), angle(angle) {
    }
};

enum BorderType : int32_t {
    BORDER_ISOLATED = 0,
    BORDER_CONSTANT = 1,
    BORDER_REFLECT101 = 2
};

enum PixelType {
    u8 = 0,
    s8 = 1,
    u16 = 2,
    s16 = 3,
    u32 = 4,
    s32 = 5,
    f32 = 6,
    d64 = 7,
};

constexpr size_t PixelSize[] = {1, 1, 2, 2, 4, 4, 4, 8};

class CvMat {
public:
    ~CvMat();
    CvMat();
    CvMat(const CvMat &other);
    CvMat(CvMat &&other);
    CvMat &operator=(const CvMat &other);
    CvMat &operator=(CvMat &&other);

    CvMat(const uint8_t *buf, int32_t rows, int32_t cols, int32_t type = u8, int32_t channels = 1, bool clahe_tile = false, int32_t tile_size = 8);
    CvMat(int32_t rows, int32_t cols, int32_t type = u8, int32_t channels = 1, bool allocate_storage = true, bool clahe_tile = false, int32_t tile_size = 8);

    void create(const uint8_t *buf, int32_t t_rows, int32_t t_cols, int32_t t_type = u8, int32_t t_channels = 1, bool clahe_tile = false, int32_t tile_size = 8);
    void create(int32_t t_rows, int32_t t_cols, int32_t t_type = u8, int32_t t_channels = 1, bool allocate_storage = true, bool clahe_tile = false, int32_t tile_size = 8);

    void adjust_roi(int32_t dtop, int32_t dbottom, int32_t dleft, int32_t dright);

    void copy_to(CvMat &dst) const;
    void make_border(int32_t top, int32_t bottom, int32_t left, int32_t right, int32_t border_type = BORDER_ISOLATED);
    void copy_to_with_border(CvMat &dst, int32_t top, int32_t bottom, int32_t left, int32_t right, int32_t border_type = BORDER_ISOLATED) const;

    uint8_t *storage = nullptr;
    uint8_t *data = nullptr;

    int32_t rows = 0;
    int32_t cols = 0;
    int32_t type = u8;
    int32_t channels = 1;
    int32_t channel_size = PixelSize[u8];
    int32_t elem_size = PixelSize[u8] * 1;
    int32_t size = 0;

    int32_t whole_width = 0;
    int32_t whole_height = 0;
    int32_t row_stride = 0;
    int32_t whole_size = 0;
    bool is_submatrix = false;
    int32_t ofs_x = 0;
    int32_t ofs_y = 0;

    template <typename T>
    inline T *ptr(int32_t y = 0) {
        return (T *)(data + row_stride * y);
    }

    template <typename T>
    inline const T *ptr(int32_t y = 0) const {
        return (const T *)(data + row_stride * y);
    }

    template <typename T>
    inline T *ptr(int32_t y, int32_t x) {
        return (T *)(data + row_stride * y + elem_size * x);
    }

    template <typename T>
    inline const T *ptr(int32_t y, int32_t x) const {
        return (const T *)(data + row_stride * y + elem_size * x);
    }
};

#include <iostream>
#include <vector>
#include "cvmat.h"
#include "lkopticalflow.h"
#include "../mylog.h"

CvMat::CvMat() = default;

CvMat::~CvMat() {
    if (storage) {
        memory_dealloc(storage);
    }
}

CvMat::CvMat(const CvMat &other) :
    rows(other.rows), cols(other.cols), type(other.type), channels(other.channels),
    channel_size(other.channel_size), elem_size(other.elem_size), size(other.size),
    whole_width(other.whole_width), whole_height(other.whole_height),
    row_stride(other.row_stride), whole_size(other.whole_size),
    is_submatrix(other.is_submatrix), ofs_x(other.ofs_x), ofs_y(other.ofs_y) {
    storage = (uint8_t *)memory_alloc(whole_size, 16);
    std::memcpy(storage, other.storage, whole_size);
    data = storage + ofs_y * row_stride + ofs_x * elem_size;
}

CvMat::CvMat(CvMat &&other) :
    storage(other.storage), data(other.data),
    rows(other.rows), cols(other.cols), type(other.type), channels(other.channels),
    channel_size(other.channel_size), elem_size(other.elem_size), size(other.size),
    whole_width(other.whole_width), whole_height(other.whole_height), row_stride(other.row_stride),
    is_submatrix(other.is_submatrix), ofs_x(other.ofs_x), ofs_y(other.ofs_y) {
    other.storage = nullptr;
    other.data = nullptr;
}

CvMat &CvMat::operator=(const CvMat &other) {
    if (&other == this) {
        return *this;
    }
    rows = other.rows;
    cols = other.cols;
    type = other.type;
    channels = other.channels;
    channel_size = other.channel_size;
    elem_size = other.elem_size;
    size = other.size;
    whole_width = other.whole_width;
    whole_height = other.whole_height;
    row_stride = other.row_stride;
    whole_size = other.whole_size;
    is_submatrix = other.is_submatrix;
    ofs_x = other.ofs_x;
    ofs_y = other.ofs_y;

    if (storage) {
        memory_dealloc(storage);
    }
    storage = (uint8_t *)memory_alloc(whole_size, 16);
    std::memcpy(storage, other.storage, whole_size);
    data = storage + ofs_y * row_stride + ofs_x * elem_size;
    return *this;
}

CvMat &CvMat::operator=(CvMat &&other) {
    if (&other == this) {
        return *this;
    }
    rows = other.rows;
    cols = other.cols;
    type = other.type;
    channels = other.channels;
    channel_size = other.channel_size;
    elem_size = other.elem_size;
    size = other.size;
    whole_width = other.whole_width;
    whole_height = other.whole_height;
    row_stride = other.row_stride;
    whole_size = other.whole_size;
    is_submatrix = other.is_submatrix;
    ofs_x = other.ofs_x;
    ofs_y = other.ofs_y;

    if (storage) {
        memory_dealloc(storage);
    }
    storage = other.storage;
    other.storage = nullptr;
    data = other.data;
    other.data = nullptr;
    return *this;
}

CvMat::CvMat(const uint8_t *buf, int32_t rows, int32_t cols, int32_t type, int32_t channels,
               bool clahe_tile, int32_t tile_size) {
    this->create(buf, rows, cols, type, channels, clahe_tile, tile_size);
}

CvMat::CvMat(int32_t rows, int32_t cols, int32_t type, int32_t channels, bool allocate_storage,
               bool clahe_tile, int32_t tile_size) {
    this->create(rows, cols, type, channels, allocate_storage, clahe_tile, tile_size);
}

void CvMat::create(const uint8_t *buf, int32_t t_rows, int32_t t_cols, int32_t t_type, int32_t t_channels,
                    bool clahe_tile, int32_t tile_size) {
    rows = t_rows;
    cols = t_cols;
    type = t_type;
    channels = t_channels;
    channel_size = PixelSize[type];
    elem_size = channels * channel_size;
    size = rows * cols * elem_size;
    whole_height = rows;
    whole_width = cols;
    row_stride = whole_width * elem_size;
    whole_size = whole_height * row_stride;
    is_submatrix = false;
    ofs_x = 0;
    ofs_y = 0;

    if (storage) {
        memory_dealloc(storage);
    }

    if (clahe_tile) {
        int32_t delta_rows = (tile_size - rows % tile_size) % tile_size;
        int32_t delta_cols = (tile_size - cols % tile_size) % tile_size;
        if (delta_rows != 0 || delta_cols != 0) {
            whole_height += delta_rows;
            whole_width += delta_cols;
            row_stride = whole_width * elem_size;
            whole_size = whole_height * row_stride;
            is_submatrix = true;
        }
        storage = (uint8_t *)memory_alloc(whole_size, 16);
        // std::memset(storage, 0x00, whole_height * whole_width * elem_size);
        data = storage;
        if (delta_cols == 0) {
            std::memcpy(data, buf, whole_size);
        } else {
            for (int32_t i = 0; i < rows; ++i) {
                std::memcpy(data + i * row_stride, buf + i * cols * elem_size, cols * elem_size);
            }
        }
        make_border(0, delta_rows, 0, delta_cols, BORDER_REFLECT101);
    } else {
        storage = (uint8_t *)memory_alloc(size, 16);
        data = storage;
        std::memcpy(data, buf, size);
    }
}

void CvMat::create(int32_t t_rows, int32_t t_cols, int32_t t_type, int32_t t_channels, bool allocate_storage,
                    bool clahe_tile, int32_t tile_size) {
    rows = t_rows;
    cols = t_cols;
    type = t_type;
    channels = t_channels;
    channel_size = PixelSize[type];
    elem_size = channels * channel_size;
    size = rows * cols * elem_size;

    whole_height = rows;
    whole_width = cols;
    row_stride = whole_width * elem_size;
    whole_size = whole_height * row_stride;
    is_submatrix = false;
    ofs_x = 0;
    ofs_y = 0;

    if (storage) {
        memory_dealloc(storage);
    }

    if (allocate_storage) {
        if (clahe_tile) {
            int32_t delta_rows = tile_size - rows % tile_size;
            int32_t delta_cols = tile_size - cols % tile_size;
            if (delta_rows != 0 || delta_cols != 0) {
                whole_height += delta_rows;
                whole_width += delta_cols;
                row_stride = whole_width * elem_size;
                whole_size = whole_height * row_stride;
                is_submatrix = true;
            }
        }
        storage = (uint8_t *)memory_alloc(whole_size, 16);
        data = storage;
    } else {
        storage = nullptr;
        data = nullptr;
    }
}

void CvMat::adjust_roi(int32_t dtop, int32_t dbottom, int32_t dleft, int32_t dright) {
    int32_t row1 = std::min(std::max(0, ofs_y - dtop), whole_height);
    int32_t row2 = std::max(0, std::min(ofs_y + rows + dbottom, whole_height));
    int32_t col1 = std::min(std::max(0, ofs_x - dleft), whole_width);
    int32_t col2 = std::max(0, std::min(ofs_x + cols + dright, whole_width));
    if (row1 > row2) {
        std::swap(row1, row2);
    }
    if (col1 > col2) {
        std::swap(col1, col2);
    }

    data += (row1 - ofs_y) * row_stride + (col1 - ofs_x) * elem_size;
    rows = row2 - row1;
    cols = col2 - col1;
    size = rows * cols;
    ofs_x = col1;
    ofs_y = row1;

    if (ofs_x == 0 && ofs_y == 0 && cols == whole_width && rows == whole_height) {
        is_submatrix = false;
    } else {
        is_submatrix = true;
    }
}

void CvMat::copy_to(CvMat &dst) const {
    runtime_assert(rows == dst.rows && cols == dst.cols && type == dst.type && channels == dst.channels
                       && channel_size == dst.channel_size && elem_size == dst.elem_size && size == dst.size,
                   "copy_to size not match!");

    for (int32_t y = 0; y < rows; ++y) {
        const uint8_t *srow = ptr<uint8_t>(y);
        uint8_t *drow = dst.ptr<uint8_t>(y);
        std::memcpy(drow, srow, cols * elem_size);
    }
}

void CvMat::make_border(int32_t top, int32_t bottom, int32_t left, int32_t right, int32_t border_type) {
    runtime_assert(ofs_y >= top && ofs_y + rows + bottom <= whole_height
                       && ofs_x >= left && ofs_x + cols + right <= whole_width,
                   "make_border size wrong!");
    if (border_type == BORDER_ISOLATED) {
        return;
    } else if (border_type == BORDER_CONSTANT) { // now sets all pixels in the border to 0
        for (int32_t y = -top; y < 0; ++y) {
            std::memset(ptr<uint8_t>(y, -left), 0x00, (left + cols + right) * elem_size);
        }
        for (int32_t y = 0; y < rows; ++y) {
            std::memset(ptr<uint8_t>(y, -left), 0x00, left * elem_size);
            std::memset(ptr<uint8_t>(y, cols), 0x00, right * elem_size);
        }
        for (int32_t y = rows; y < rows + bottom; ++y) {
            std::memset(ptr<uint8_t>(y, -left), 0x00, (left + cols + right) * elem_size);
        }
    } else if (border_type == BORDER_REFLECT101) {
        std::vector<int32_t> tab_buf(left + right);
        int32_t *tab = tab_buf.data();
        int32_t x, y, y_interp;

        for (x = 0; x < left; ++x) {
            tab[x] = border_interpolate(x - left, cols, border_type);
        }
        for (x = 0; x < right; ++x) {
            tab[x + left] = border_interpolate(x + cols, cols, border_type);
        }

        for (y = 0; y < rows; ++y) {
            uint8_t *row = ptr<uint8_t>(y);
            for (x = 0; x < left; ++x) {
                row[x - left] = row[tab[x]];
            }
            for (x = 0; x < right; ++x) {
                row[x + cols] = row[tab[x + left]];
            }
        }
        for (y = -top; y < 0; ++y) {
            y_interp = border_interpolate(y, rows, border_type);
            std::memcpy(ptr<uint8_t>(y, -left), ptr<uint8_t>(y_interp, -left), (left + cols + right) * elem_size);
        }
        for (y = rows; y < rows + bottom; ++y) {
            y_interp = border_interpolate(y, rows, border_type);
            std::memcpy(ptr<uint8_t>(y, -left), ptr<uint8_t>(y_interp, -left), (left + cols + right) * elem_size);
        }
    }
}

void CvMat::copy_to_with_border(CvMat &dst, int32_t top, int32_t bottom, int32_t left, int32_t right, int32_t border_type) const {
    runtime_assert(rows == dst.rows && cols == dst.cols && type == dst.type && channels == dst.channels
                       && channel_size == dst.channel_size && elem_size == dst.elem_size && size == dst.size,
                   "copy_to_with_border size not match!");

    if (border_type == BORDER_ISOLATED) {
        copy_to(dst);
    } else if (border_type == BORDER_CONSTANT) { // now sets all pixels in the border to 0
        for (int32_t y = -top; y < 0; ++y) {
            std::memset(dst.ptr<uint8_t>(y, -left), 0x00, (left + cols + right) * elem_size);
        }
        for (int32_t y = 0; y < rows; ++y) {
            std::memset(dst.ptr<uint8_t>(y, -left), 0x00, left * elem_size);
            std::memcpy(dst.ptr<uint8_t>(y, 0), ptr<uint8_t>(y, 0), cols * elem_size);
            std::memset(dst.ptr<uint8_t>(y, cols), 0x00, right * elem_size);
        }
        for (int32_t y = rows; y < rows + bottom; ++y) {
            std::memset(dst.ptr<uint8_t>(y, -left), 0x00, (left + cols + right) * elem_size);
        }
    } else if (border_type == BORDER_REFLECT101) {
        std::vector<int32_t> tab_buf(left + right);
        int32_t *tab = tab_buf.data();
        int32_t x, y, y_interp;

        for (x = 0; x < left; ++x) {
            tab[x] = border_interpolate(x - left, cols, border_type);
        }
        for (x = 0; x < right; ++x) {
            tab[x + left] = border_interpolate(x + cols, cols, border_type);
        }

        for (y = 0; y < rows; ++y) {
            const uint8_t *srow = ptr<uint8_t>(y);
            uint8_t *drow = dst.ptr<uint8_t>(y);
            for (x = 0; x < left; ++x) {
                drow[x - left] = srow[tab[x]];
            }
            std::memcpy(drow, srow, cols * elem_size);
            for (x = 0; x < right; ++x) {
                drow[x + cols] = srow[tab[x + left]];
            }
        }
        for (y = -top; y < 0; ++y) {
            y_interp = border_interpolate(y, rows, border_type);
            std::memcpy(dst.ptr<uint8_t>(y, -left), dst.ptr<uint8_t>(y_interp, -left), (left + cols + right) * elem_size);
        }
        for (y = rows; y < rows + bottom; ++y) {
            y_interp = border_interpolate(y, rows, border_type);
            std::memcpy(dst.ptr<uint8_t>(y, -left), dst.ptr<uint8_t>(y_interp, -left), (left + cols + right) * elem_size);
        }
    }
}

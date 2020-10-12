#include <cmath>
#include "lkopticalflow.h"
#include "cvmat.h"
#include "../mylog.h"

#define DESCALE(x, n) (((x) + (1 << ((n)-1))) >> (n))

void calc_opticalflow_pyramid_lk(
    const std::vector<CvMat> &src_image_pyr, const std::vector<CvMat> &src_deriv_pyr,
    const std::vector<CvMat> &dst_image_pyr, const std::vector<CvMat> &dst_deriv_pyr,
    const std::vector<std::vector<double>> &src_kpts, std::vector<std::vector<double>> &dst_kpts, std::vector<uint8_t> &status,
    int32_t win_width, int32_t win_height, int32_t max_level) {
    runtime_assert(src_image_pyr.size() == dst_image_pyr.size(), "calc_opticalflow_pyramid_lk: pyramids size not valid!");
    runtime_assert(src_deriv_pyr.size() == dst_deriv_pyr.size(), "calc_opticalflow_pyramid_lk: pyramids size not valid!");

    if (status.size() != src_kpts.size()) {
        status = std::vector<uint8_t>(src_kpts.size(), 1);
    }

    if (dst_kpts.empty()) {
        dst_kpts = src_kpts;
    }
    runtime_assert(dst_kpts.size() == src_kpts.size(), "calc_opticalflow_pyramid_lk: keypoints size not match!");

    std::vector<double> half_win = {0.5f * (win_width - 1), 0.5f * (win_height - 1)};

    constexpr int32_t min_level = 1;
    for (int32_t level = max_level; level >= min_level; --level) {
        const CvMat &I = src_image_pyr[level];
        const CvMat &J = dst_image_pyr[level];
        const CvMat &deriv_I = src_deriv_pyr[level];

        // log_debug("level: %d. I: cols %d, rows %d. J: cols %d, rows %d. deriv_I: cols %d, rows %d",
        //           level, I.cols, I.rows, J.cols, J.rows, deriv_I.cols, deriv_I.rows);

        const int32_t stepI = I.row_stride / I.channel_size;
        const int32_t stepJ = J.row_stride / J.channel_size;
        const int32_t stepdI = deriv_I.row_stride / deriv_I.channel_size;

        CvMat I_window(win_height, win_width, s16, 1);
        CvMat deriv_I_window(win_height, win_width, s16, 2);

        for (int32_t ptidx = 0; ptidx < src_kpts.size(); ++ptidx) {
            if (status[ptidx] == 0) {
                continue;
            }

            std::vector<double> src_pt(2);
            src_pt[0] = src_kpts[ptidx][0] * (float)(1.0 / (1 << level));
            src_pt[1] = src_kpts[ptidx][1] * (float)(1.0 / (1 << level));

            std::vector<double> dst_pt(2);
            if (level == max_level) {
                dst_pt[0] = dst_kpts[ptidx][0] * (float)(1.0 / (1 << level));
                dst_pt[1] = dst_kpts[ptidx][1] * (float)(1.0 / (1 << level));
            } else {
                dst_pt[0] = dst_kpts[ptidx][0] * 2.f;
                dst_pt[1] = dst_kpts[ptidx][1] * 2.f;
            }
            dst_kpts[ptidx] = dst_pt;

            std::vector<int32_t> isrc_pt(2);
            std::vector<int32_t> idst_pt(2);
            src_pt[0] -= half_win[0];
            src_pt[1] -= half_win[1];

            isrc_pt[0] = std::floor(src_pt[0]);
            isrc_pt[1] = std::floor(src_pt[1]);

            if (isrc_pt[0] < -win_width || isrc_pt[0] >= deriv_I.cols
                || isrc_pt[1] < -win_height || isrc_pt[1] >= deriv_I.rows) {
                if (level == min_level) {
                    status[ptidx] = 0;
                }
                continue;
            }

            float a = src_pt[0] - isrc_pt[0];
            float b = src_pt[1] - isrc_pt[1];
            constexpr int32_t W_BITS = 14;
            constexpr float FLT_SCALE = 1.f / (1 << 20);
            int32_t iw00 = std::round((1.f - a) * (1.f - b) * (1 << W_BITS));
            int32_t iw01 = std::round(a * (1.f - b) * (1 << W_BITS));
            int32_t iw10 = std::round((1.f - a) * b * (1 << W_BITS));
            int32_t iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

            float iA11 = 0, iA12 = 0, iA22 = 0;
            float A11, A12, A22;

#if __HAS_SIMD__
            alignas(16) float nA11[4] = {0, 0, 0, 0};
            alignas(16) float nA12[4] = {0, 0, 0, 0};
            alignas(16) float nA22[4] = {0, 0, 0, 0};
            constexpr int32_t shift1 = -(W_BITS - 5);
            constexpr int32_t shift2 = -(W_BITS);

            const int16x4_t d26 = vdup_n_s16((int16_t)(iw00));
            const int16x4_t d27 = vdup_n_s16((int16_t)(iw01));
            const int16x4_t d28 = vdup_n_s16((int16_t)(iw10));
            const int16x4_t d29 = vdup_n_s16((int16_t)(iw11));

            const int32x4_t q11 = vdupq_n_s32((int32_t)(shift1));
            const int32x4_t q12 = vdupq_n_s32((int32_t)(shift2));
#endif

            int32_t x, y;
            for (y = 0; y < win_height; ++y) {
                const uint8_t *src = I.ptr<uint8_t>(y + isrc_pt[1], isrc_pt[0]);
                const int16_t *dsrc = deriv_I.ptr<int16_t>(y + isrc_pt[1], isrc_pt[0]);

                int16_t *Iptr = I_window.ptr<int16_t>(y);
                int16_t *dIptr = deriv_I_window.ptr<int16_t>(y);

                x = 0;

#if __HAS_SIMD__
                int32x4_t inA11_inner = vdupq_n_s32(0);
                int32x4_t inA12_inner = vdupq_n_s32(0);
                int32x4_t inA22_inner = vdupq_n_s32(0);

                for (; x <= win_width - 4; x += 4, dsrc += 8, dIptr += 8) {
                    uint8x8_t d0 = vld1_u8(src + x);
                    uint8x8_t d2 = vld1_u8(src + x + 1);
                    uint16x8_t q0 = vmovl_u8(d0);
                    uint16x8_t q1 = vmovl_u8(d2);

                    int32x4_t q5 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q0)), d26);
                    int32x4_t q6 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q1)), d27);

                    uint8x8_t d4 = vld1_u8(src + x + stepI);
                    uint8x8_t d6 = vld1_u8(src + x + stepI + 1);
                    uint16x8_t q2 = vmovl_u8(d4);
                    uint16x8_t q3 = vmovl_u8(d6);

                    int32x4_t q7 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q2)), d28);
                    int32x4_t q8 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q3)), d29);

                    q5 = vaddq_s32(q5, q6);
                    q7 = vaddq_s32(q7, q8);
                    q5 = vaddq_s32(q5, q7);

                    q5 = vqrshlq_s32(q5, q11);
                    int16x4_t nd0 = vmovn_s32(q5);
                    vst1_s16(Iptr + x, nd0);

                    int16x4x2_t d0d1 = vld2_s16(dsrc);
                    int16x4x2_t d2d3 = vld2_s16(dsrc + 2);

                    int32x4_t q4 = vmull_s16(d0d1.val[0], d26);
                    q6 = vmull_s16(d0d1.val[1], d26);
                    q7 = vmull_s16(d2d3.val[0], d27);
                    q8 = vmull_s16(d2d3.val[1], d27);

                    int16x4x2_t d4d5 = vld2_s16(dsrc + stepdI);
                    int16x4x2_t d6d7 = vld2_s16(dsrc + stepdI + 2);

                    q4 = vaddq_s32(q4, q7);
                    q6 = vaddq_s32(q6, q8);

                    q7 = vmull_s16(d4d5.val[0], d28);
                    int32x4_t q14 = vmull_s16(d4d5.val[1], d28);
                    q8 = vmull_s16(d6d7.val[0], d29);
                    int32x4_t q15 = vmull_s16(d6d7.val[1], d29);

                    q7 = vaddq_s32(q7, q8);
                    q14 = vaddq_s32(q14, q15);

                    q4 = vaddq_s32(q4, q7);
                    q6 = vaddq_s32(q6, q14);

                    q4 = vqrshlq_s32(q4, q12);
                    q6 = vqrshlq_s32(q6, q12);

                    q7 = vmulq_s32(q4, q4);
                    q8 = vmulq_s32(q4, q6);
                    q15 = vmulq_s32(q6, q6);

                    inA11_inner = vaddq_s32(inA11_inner, q7);
                    inA12_inner = vaddq_s32(inA12_inner, q8);
                    inA22_inner = vaddq_s32(inA22_inner, q15);

                    int16x4_t d8 = vmovn_s32(q4);
                    int16x4_t d12 = vmovn_s32(q6);

                    int16x4x2_t d8d12;
                    d8d12.val[0] = d8;
                    d8d12.val[1] = d12;
                    vst2_s16(dIptr, d8d12);
                }

                float32x4_t nq0 = vld1q_f32(nA11);
                float32x4_t nq1 = vld1q_f32(nA12);
                float32x4_t nq2 = vld1q_f32(nA22);

                nq0 = vaddq_f32(nq0, vcvtq_f32_s32(inA11_inner));
                nq1 = vaddq_f32(nq1, vcvtq_f32_s32(inA12_inner));
                nq2 = vaddq_f32(nq2, vcvtq_f32_s32(inA22_inner));

                vst1q_f32(nA11, nq0);
                vst1q_f32(nA12, nq1);
                vst1q_f32(nA22, nq2);
#endif

                for (; x < win_width; ++x, dsrc += 2, dIptr += 2) {
                    int32_t ival = DESCALE(src[x] * iw00 + src[x + 1] * iw01
                                               + src[x + stepI] * iw10 + src[x + stepI + 1] * iw11,
                                           W_BITS - 5);
                    int32_t ixval = DESCALE(dsrc[0] * iw00 + dsrc[2] * iw01
                                                + dsrc[stepdI] * iw10 + dsrc[stepdI + 2] * iw11,
                                            W_BITS);
                    int32_t iyval = DESCALE(dsrc[1] * iw00 + dsrc[3] * iw01
                                                + dsrc[stepdI + 1] * iw10 + dsrc[stepdI + 3] * iw11,
                                            W_BITS);

                    Iptr[x] = (int16_t)ival;
                    dIptr[0] = (int16_t)ixval;
                    dIptr[1] = (int16_t)iyval;

                    iA11 += (float)(ixval * ixval);
                    iA12 += (float)(ixval * iyval);
                    iA22 += (float)(iyval * iyval);
                }
            }

#if __HAS_SIMD__
            iA11 += nA11[0] + nA11[1] + nA11[2] + nA11[3];
            iA12 += nA12[0] + nA12[1] + nA12[2] + nA12[3];
            iA22 += nA22[0] + nA22[1] + nA22[2] + nA22[3];
#endif

            A11 = iA11 * FLT_SCALE;
            A12 = iA12 * FLT_SCALE;
            A22 = iA22 * FLT_SCALE;

            float D = A11 * A22 - A12 * A12;
            float min_eigenvalue = (A11 + A22 - std::sqrt((A11 - A22) * (A11 - A22) + 4.f * A12 * A12))
                                   / (2 * win_width * win_height);

            if (min_eigenvalue < 1e-4f || D < std::numeric_limits<float>::epsilon()) {
                if (level == min_level) {
                    status[ptidx] = 0;
                }
                continue;
            }

            D = 1.f / D;

            dst_pt[0] -= half_win[0];
            dst_pt[1] -= half_win[1];
            std::vector<double> prev_delta = {0, 0};

            for (int32_t j = 0; j < 30; ++j) {
                idst_pt[0] = std::floor(dst_pt[0]);
                idst_pt[1] = std::floor(dst_pt[1]);

                if (idst_pt[0] < -win_width || idst_pt[0] >= J.cols
                    || idst_pt[1] < -win_height || idst_pt[1] >= J.rows) {
                    if (level == min_level) {
                        status[ptidx] = 0;
                    }
                    continue;
                }

                a = dst_pt[0] - idst_pt[0];
                b = dst_pt[1] - idst_pt[1];

                iw00 = std::round((1.f - a) * (1.f - b) * (1 << W_BITS));
                iw01 = std::round(a * (1.f - b) * (1 << W_BITS));
                iw10 = std::round((1.f - a) * b * (1 << W_BITS));
                iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

                float ib1 = 0, ib2 = 0;
                float b1, b2;

#if __HAS_SIMD__
                alignas(16) float nB1[4] = {0, 0, 0, 0}, nB2[4] = {0, 0, 0, 0};

                const int16x4_t d26_2 = vdup_n_s16((int16_t)(iw00));
                const int16x4_t d27_2 = vdup_n_s16((int16_t)(iw01));
                const int16x4_t d28_2 = vdup_n_s16((int16_t)(iw10));
                const int16x4_t d29_2 = vdup_n_s16((int16_t)(iw11));
#endif

                for (y = 0; y < win_height; ++y) {
                    const uint8_t *Jptr = J.ptr<uint8_t>(y + idst_pt[1], idst_pt[0]);
                    const int16_t *Iptr = I_window.ptr<int16_t>(y);
                    const int16_t *dIptr = deriv_I_window.ptr<int16_t>(y);

                    x = 0;

#if __HAS_SIMD__

                    int32x4_t inB1_inner = vdupq_n_s32(0);
                    int32x4_t inB2_inner = vdupq_n_s32(0);

                    for (; x <= win_width - 8; x += 8, dIptr += 16) {
                        uint8x8_t d0 = vld1_u8(Jptr + x);
                        uint8x8_t d2 = vld1_u8(Jptr + x + 1);
                        uint8x8_t d4 = vld1_u8(Jptr + x + stepJ);
                        uint8x8_t d6 = vld1_u8(Jptr + x + stepJ + 1);

                        uint16x8_t q0 = vmovl_u8(d0);
                        uint16x8_t q1 = vmovl_u8(d2);
                        uint16x8_t q2 = vmovl_u8(d4);
                        uint16x8_t q3 = vmovl_u8(d6);

                        int32x4_t nq4 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q0)), d26_2);
                        int32x4_t nq5 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q0)), d26_2);

                        int32x4_t nq6 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q1)), d27_2);
                        int32x4_t nq7 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q1)), d27_2);

                        int32x4_t nq8 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q2)), d28_2);
                        int32x4_t nq9 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q2)), d28_2);

                        int32x4_t nq10 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q3)), d29_2);
                        int32x4_t nq11 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q3)), d29_2);

                        nq4 = vaddq_s32(nq4, nq6);
                        nq5 = vaddq_s32(nq5, nq7);
                        nq8 = vaddq_s32(nq8, nq10);
                        nq9 = vaddq_s32(nq9, nq11);

                        nq4 = vaddq_s32(nq4, nq8);
                        nq5 = vaddq_s32(nq5, nq9);

                        int16x8_t q6 = vld1q_s16(Iptr + x);
                        nq6 = vmovl_s16(vget_low_s16(q6));
                        nq8 = vmovl_s16(vget_high_s16(q6));

                        nq4 = vqrshlq_s32(nq4, q11);
                        nq5 = vqrshlq_s32(nq5, q11);

                        nq4 = vsubq_s32(nq4, nq6);
                        nq5 = vsubq_s32(nq5, nq8);

                        int16x8x2_t q0q1 = vld2q_s16(dIptr);
                        int32x4_t nq2 = vmovl_s16(vget_low_s16(q0q1.val[0]));
                        int32x4_t nq3 = vmovl_s16(vget_high_s16(q0q1.val[0]));

                        nq7 = vmovl_s16(vget_low_s16(q0q1.val[1]));
                        nq8 = vmovl_s16(vget_high_s16(q0q1.val[1]));

                        nq9 = vmulq_s32(nq4, nq2);
                        nq10 = vmulq_s32(nq5, nq3);

                        nq4 = vmulq_s32(nq4, nq7);
                        nq5 = vmulq_s32(nq5, nq8);

                        nq9 = vaddq_s32(nq9, nq10);
                        nq4 = vaddq_s32(nq4, nq5);

                        inB1_inner = vaddq_s32(inB1_inner, nq9);
                        inB2_inner = vaddq_s32(inB2_inner, nq4);
                    }

                    float32x4_t nB1v = vld1q_f32(nB1);
                    float32x4_t nB2v = vld1q_f32(nB2);

                    nB1v = vaddq_f32(nB1v, vcvtq_f32_s32(inB1_inner));
                    nB2v = vaddq_f32(nB2v, vcvtq_f32_s32(inB2_inner));

                    vst1q_f32(nB1, nB1v);
                    vst1q_f32(nB2, nB2v);
#endif

                    for (; x < win_width; ++x, dIptr += 2) {
                        int32_t diff = DESCALE(Jptr[x] * iw00 + Jptr[x + 1] * iw01
                                                   + Jptr[x + stepJ] * iw10 + Jptr[x + stepJ + 1] * iw11,
                                               W_BITS - 5)
                                       - Iptr[x];
                        ib1 += (float)(diff * dIptr[0]);
                        ib2 += (float)(diff * dIptr[1]);
                    }
                }

#if __HAS_SIMD__
                ib1 += (float)(nB1[0] + nB1[1] + nB1[2] + nB1[3]);
                ib2 += (float)(nB2[0] + nB2[1] + nB2[2] + nB2[3]);
#endif

                b1 = ib1 * FLT_SCALE;
                b2 = ib2 * FLT_SCALE;

                std::vector<double> delta = {(float)((A12 * b2 - A22 * b1) * D), (float)((A12 * b1 - A11 * b2) * D)};

                dst_pt[0] += delta[0];
                dst_pt[1] += delta[1];

                dst_kpts[ptidx][0] = dst_pt[0] + half_win[0];
                dst_kpts[ptidx][1] = dst_pt[1] + half_win[1];

                if ((delta[0] * delta[0] + delta[1] * delta[1]) <= 0.0001) {
                    break;
                }

                if (j > 0
                    && std::abs(delta[0] + prev_delta[0]) < 0.01f
                    && std::abs(delta[1] + prev_delta[1]) < 0.01f) {
                    dst_kpts[ptidx][0] -= delta[0] * 0.5f;
                    dst_kpts[ptidx][1] -= delta[1] * 0.5f;
                    break;
                }

                prev_delta = delta;
            }

            if (status[ptidx] && level == min_level) {
                std::vector<double> dst_pt = {dst_kpts[ptidx][0] - half_win[0], dst_kpts[ptidx][1] - half_win[1]};
                std::vector<int32_t> idst_pt(2);

                idst_pt[0] = std::floor(dst_pt[0]);
                idst_pt[1] = std::floor(dst_pt[1]);

                if (idst_pt[0] < -win_width || idst_pt[0] >= J.cols
                    || idst_pt[1] < -win_height || idst_pt[1] >= J.rows) {
                    status[ptidx] = 0;
                }
            }
        }
    }

    for (size_t i = 0; i < dst_kpts.size(); ++i) {
        dst_kpts[i][0] *= (1 << min_level);
        dst_kpts[i][1] *= (1 << min_level);
    }
}

int32_t build_opticalflow_pyramid(
    const CvMat &img, std::vector<CvMat> &image_levels, std::vector<CvMat> &deriv_levels,
    int32_t win_width, int32_t win_height, int32_t max_level) {
    image_levels.clear();
    image_levels.resize(max_level + 1);
    deriv_levels.clear();
    deriv_levels.resize(max_level + 1);

    // level 0 image
    image_levels[0].create(img.rows + win_height * 2, img.cols + win_width * 2, img.type, img.channels);
    image_levels[0].adjust_roi(-win_height, -win_height, -win_width, -win_width);
    img.copy_to_with_border(image_levels[0], win_height, win_height, win_width, win_width, BORDER_REFLECT101);

    int32_t width = img.cols;
    int32_t height = img.rows;

    for (int32_t level = 0; level <= max_level; ++level) {
        if (level != 0) {
            image_levels[level].create(height + win_height * 2, width + win_width * 2, img.type, img.channels);
            image_levels[level].adjust_roi(-win_height, -win_height, -win_width, -win_width);
            pyramid_downsample(image_levels[level - 1], image_levels[level], BORDER_REFLECT101);
            image_levels[level].make_border(win_height, win_height, win_width, win_width, BORDER_REFLECT101);
        }

        // compute derivatives
        deriv_levels[level].create(height + win_height * 2, width + win_width * 2, s16, 2);
        deriv_levels[level].adjust_roi(-win_height, -win_height, -win_width, -win_width);
        calc_scharr_deriv(image_levels[level], deriv_levels[level]);
        deriv_levels[level].make_border(win_height, win_height, win_width, win_width, BORDER_CONSTANT);

        width = (width + 1) / 2;
        height = (height + 1) / 2;

        if (width <= win_width || height <= win_height) {
            image_levels.resize(level + 1);
            deriv_levels.resize(level + 1);
            return level;
        }
    }

    return max_level;
}

void calc_scharr_deriv(const CvMat &src, CvMat &dst) {
    runtime_assert(src.rows == dst.rows && src.cols == dst.cols, "calc_scharr_deriv: src, dst not matched!");
    int32_t rows = src.rows;
    int32_t cols = src.cols;
    int32_t colsn = cols;
    // dst.create(rows, cols, DataType<int16_t>::pixel_type, 2);

    int32_t x, y;
    int32_t delta = align_size(cols + 2, 16);
    std::vector<int16_t> buffer(delta * 2 + 64);
    int16_t *trow0 = align_ptr(buffer.data() + 1, 16);
    int16_t *trow1 = align_ptr(trow0 + delta, 16);
    // printf("%p %p\n", trow0, trow1);

#if __HAS_SIMD__
    int16x8_t c3 = vdupq_n_s16(3);
    int16x8_t c10 = vdupq_n_s16(10);
#endif

    for (y = 0; y < rows; ++y) {
        const uint8_t *srow0 = src.ptr<uint8_t>(y > 0 ? y - 1 : rows > 1 ? 1 : 0);
        const uint8_t *srow1 = src.ptr<uint8_t>(y);
        const uint8_t *srow2 = src.ptr<uint8_t>(y < rows - 1 ? y + 1 : rows > 1 ? rows - 2 : 0);
        int16_t *drow = dst.ptr<int16_t>(y);

        // do vertical convolution
        x = 0;
#if __HAS_SIMD__
        for (; x <= cols - 8; x += 8) {
            int16x8_t s0 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(srow0 + x)));
            int16x8_t s1 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(srow1 + x)));
            int16x8_t s2 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(srow2 + x)));

            int16x8_t t1 = vqsubq_s16(s2, s0);
            int16x8_t t0 = vqaddq_s16(vmulq_s16(vqaddq_s16(s0, s2), c3), vmulq_s16(s1, c10));

            vst1q_s16(trow0 + x, t0);
            vst1q_s16(trow1 + x, t1);
        }
#endif
        for (; x < cols; ++x) {
            int16_t t0 = ((int16_t)srow0[x] + (int16_t)srow2[x]) * 3 + (int16_t)srow1[x] * 10;
            int16_t t1 = (int16_t)srow2[x] - (int16_t)srow0[x];
            trow0[x] = t0;
            trow1[x] = t1;
        }

        // make border
        int32_t x0 = cols > 1 ? 1 : 0;
        int32_t x1 = cols > 1 ? cols - 2 : 0;
        for (int32_t k = 0; k < 1; ++k) {
            trow0[-1 + k] = trow0[x0 + k];
            trow0[cols + k] = trow0[x1 + k];
            trow1[-1 + k] = trow1[x0 + k];
            trow1[cols + k] = trow1[x1 + k];
        }

        // do horizontal convolution, interleave the results and store them to dst
        x = 0;
#if __HAS_SIMD__
        for (; x <= cols - 8; x += 8) {
            int16x8_t s0 = vld1q_s16(trow0 + x - 1);
            int16x8_t s1 = vld1q_s16(trow0 + x + 1);
            int16x8_t s2 = vld1q_s16(trow1 + x - 1);
            int16x8_t s3 = vld1q_s16(trow1 + x);
            int16x8_t s4 = vld1q_s16(trow1 + x + 1);

            int16x8x2_t t;
            t.val[0] = vqsubq_s16(s1, s0);
            t.val[1] = vqaddq_s16(vmulq_s16(vqaddq_s16(s2, s4), c3), vmulq_s16(s3, c10));

            vst2q_s16(drow + x * 2, t);
        }
#endif
        for (; x < cols; ++x) {
            int16_t t0 = trow0[x + 1] - trow0[x - 1];
            int16_t t1 = (trow1[x + 1] + trow1[x - 1]) * 3 + trow1[x] * 10;
            drow[x * 2] = t0;
            drow[x * 2 + 1] = t1;
        }
    }
}

int32_t border_interpolate(int32_t p, int32_t len, int32_t border_type) {
    if (p >= 0 && p < len) {
        ;
    } else if (border_type == BORDER_REFLECT101) {
        if (len == 1) {
            return 0;
        }
        do {
            if (p < 0) {
                p = -p;
            } else {
                p = (len - 1) - (p - len) - 1;
            }
        } while (p < 0 || p >= len);
    } else if (border_type == BORDER_CONSTANT) {
        p = -1;
    }
    return p;
}

int32_t pyr_down_vec_hor(const uint8_t *src, int32_t *row, int32_t width) {
    int32_t x = 0;
    const uint8_t *src0 = src;
    const uint8_t *src2 = src + 2;
    const uint8_t *src4 = src + 3;
#if __HAS_SIMD__
    int16x8_t v_1_4 = vdupq_n_s32(0x00040001);
    int16x8_t v_6_4 = vdupq_n_s32(0x00040006);
    for (; x <= width - 4; x += 4, src0 += 8, src2 += 8, src4 += 8, row += 4) {
        int16x8_t v0 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src0)));
        int16x8_t v2 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src2)));
        int16x8_t v4 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src4)));

        int32x4_t v0c = vmull_s16(vget_low_s16(v0), vget_low_s16(v_1_4));
        int32x4_t v0d = vmull_s16(vget_high_s16(v0), vget_high_s16(v_1_4));
        int32x4x2_t v0cd = vuzpq_s32(v0c, v0d);

        int32x4_t v2c = vmull_s16(vget_low_s16(v2), vget_low_s16(v_6_4));
        int32x4_t v2d = vmull_s16(vget_high_s16(v2), vget_high_s16(v_6_4));
        int32x4x2_t v2cd = vuzpq_s32(v2c, v2d);

        vst1q_s32(
            row,
            vaddq_s32(
                vaddq_s32(vaddq_s32(v0cd.val[0], v0cd.val[1]), vaddq_s32(v2cd.val[0], v2cd.val[1])),
                vshlq_s32(vreinterpretq_s32_s16(v4), vdupq_n_s32((int32_t)(-16)))));
    }
#endif
    return x;
}

int32_t pyr_down_vec_vert(int32_t **src, uint8_t *dst, int32_t width) {
    int32_t x = 0;
    const int32_t *row0 = src[0];
    const int32_t *row1 = src[1];
    const int32_t *row2 = src[2];
    const int32_t *row3 = src[3];
    const int32_t *row4 = src[4];

#if __HAS_SIMD__
    for (; x <= width - 16; x += 16) {
        uint16x8_t r0, r1, r2, r3, r4, t0, t1;
        r0 = vcombine_s16(vqmovn_s32(vld1q_s32(row0 + x)), vqmovn_s32(vld1q_s32(row0 + x + 4)));
        r1 = vcombine_s16(vqmovn_s32(vld1q_s32(row1 + x)), vqmovn_s32(vld1q_s32(row1 + x + 4)));
        r2 = vcombine_s16(vqmovn_s32(vld1q_s32(row2 + x)), vqmovn_s32(vld1q_s32(row2 + x + 4)));
        r3 = vcombine_s16(vqmovn_s32(vld1q_s32(row3 + x)), vqmovn_s32(vld1q_s32(row3 + x + 4)));
        r4 = vcombine_s16(vqmovn_s32(vld1q_s32(row4 + x)), vqmovn_s32(vld1q_s32(row4 + x + 4)));
        t0 = vqaddq_u16(
            vqaddq_u16(vqaddq_u16(r0, r4), vqaddq_u16(r2, r2)),
            vshlq_n_u16(vqaddq_u16(vqaddq_u16(r1, r3), r2), 2));
        r0 = vcombine_s16(vqmovn_s32(vld1q_s32(row0 + x + 8)), vqmovn_s32(vld1q_s32(row0 + x + 12)));
        r1 = vcombine_s16(vqmovn_s32(vld1q_s32(row1 + x + 8)), vqmovn_s32(vld1q_s32(row1 + x + 12)));
        r2 = vcombine_s16(vqmovn_s32(vld1q_s32(row2 + x + 8)), vqmovn_s32(vld1q_s32(row2 + x + 12)));
        r3 = vcombine_s16(vqmovn_s32(vld1q_s32(row3 + x + 8)), vqmovn_s32(vld1q_s32(row3 + x + 12)));
        r4 = vcombine_s16(vqmovn_s32(vld1q_s32(row4 + x + 8)), vqmovn_s32(vld1q_s32(row4 + x + 12)));
        t1 = vqaddq_u16(
            vqaddq_u16(vqaddq_u16(r0, r4), vqaddq_u16(r2, r2)),
            vshlq_n_u16(vqaddq_u16(vqaddq_u16(r1, r3), r2), 2));
        // store 16 uint8_t numbers
        vst1q_u8(dst + x, vcombine_u8(vqrshrn_n_u16(t0, 8), vqrshrn_n_u16(t1, 8)));
    }
    if (x <= width - 8) {
        uint16x8_t r0, r1, r2, r3, r4, t0;
        r0 = vcombine_s16(vqmovn_s32(vld1q_s32(row0 + x)), vqmovn_s32(vld1q_s32(row0 + x + 4)));
        r1 = vcombine_s16(vqmovn_s32(vld1q_s32(row1 + x)), vqmovn_s32(vld1q_s32(row1 + x + 4)));
        r2 = vcombine_s16(vqmovn_s32(vld1q_s32(row2 + x)), vqmovn_s32(vld1q_s32(row2 + x + 4)));
        r3 = vcombine_s16(vqmovn_s32(vld1q_s32(row3 + x)), vqmovn_s32(vld1q_s32(row3 + x + 4)));
        r4 = vcombine_s16(vqmovn_s32(vld1q_s32(row4 + x)), vqmovn_s32(vld1q_s32(row4 + x + 4)));
        t0 = vqaddq_u16(
            vqaddq_u16(vqaddq_u16(r0, r4), vqaddq_u16(r2, r2)),
            vshlq_n_u16(vqaddq_u16(vqaddq_u16(r1, r3), r2), 2));
        // store 8 uint8_t numbers
        vst1_u8(dst + x, vqrshrn_n_u16(t0, 8));
        x += 8;
    }
    for (; x <= width - 4; x += 4) {
        int32x4_t r0, r1, r2, r3, r4, t0;
        r0 = vld1q_s32(row0 + x);
        r1 = vld1q_s32(row1 + x);
        r2 = vld1q_s32(row2 + x);
        r3 = vld1q_s32(row3 + x);
        r4 = vld1q_s32(row4 + x);
        t0 = vaddq_s32(
            vaddq_s32(vaddq_s32(r0, r4), vaddq_s32(r2, r2)),
            vshlq_n_s32(vaddq_s32(vaddq_s32(r1, r3), r2), 2));
        // store 4 uint8_t numbers
        int16x4_t p0 = vqrshrn_n_s32(t0, 8);
        uint16x8_t p1 = vreinterpretq_u16_s16(vcombine_s16(p0, p0));
        *((int32_t *)(dst + x)) = vget_lane_s32(vreinterpret_s32_u8(vmovn_u16(p1)), 0);
    }
#endif

    return x;
}

void pyramid_downsample(const CvMat &src, CvMat &dst, int32_t border_type) {
    constexpr int32_t sp_size = 5; // sample_size, 5x5 sample area
                                   //     |  1   4   6   4   1  |
                                   //  1  |  4  16  24  16   4  |
                                   // --- |  6  24  36  24   6  |
                                   // 256 |  4  16  24  16   4  |
                                   //     |  1   4   6   4   1  |
    int32_t swidth = src.cols;
    int32_t sheight = src.rows;
    int32_t dwidth = dst.cols;
    int32_t dheight = dst.rows;

    int32_t bufstep = align_size(dwidth, 16);
    std::vector<int32_t> buffer(bufstep * sp_size + 16);
    int32_t *buf = align_ptr(buffer.data(), 16);
    int32_t tab_l[sp_size + 2];
    int32_t tab_r[sp_size + 2];
    std::vector<int32_t> tab_m_buffer(dwidth);
    int32_t *tab_m = tab_m_buffer.data();
    int32_t *rows[sp_size];

    runtime_assert(swidth > 0 && sheight > 0
                       && swidth - dwidth * 2 <= 2 && sheight - dheight * 2 <= 2,
                   "pyramid_downsample: size not match");
    int32_t k, x, y;
    int32_t sy0 = -sp_size / 2; // source row starting point
    int32_t sy = sy0;           // source row index
    int32_t width0 = std::min(dwidth, (swidth - sp_size / 2 - 1) / 2 + 1);

    for (x = 0; x <= sp_size + 1; ++x) {
        tab_l[x] = border_interpolate(x - sp_size / 2, swidth, border_type);
        tab_r[x] = border_interpolate(x - sp_size / 2 + width0 * 2, swidth, border_type);
    }
    for (x = 0; x < dwidth; ++x) {
        tab_m[x] = x * 2;
    }

    for (y = 0; y < dheight; ++y) {
        uint8_t *drow = dst.ptr<uint8_t>(y); // dest row
        int32_t *row0, *row1, *row2, *row3, *row4;

        // fill the ring buffer (horizontal convolution and decimation)
        for (; sy <= y * 2 + 2; ++sy) {
            int32_t *buf_row = buf + ((sy - sy0) % sp_size) * bufstep; // buffer row
            int32_t sy_interp = border_interpolate(sy, sheight, border_type);
            const uint8_t *srow = src.ptr<uint8_t>(sy_interp); // source row
            int32_t limit = 1;
            const int32_t *tab = tab_l;

            for (x = 0;;) {
                // actually run twice for borders on both sides
                for (; x < limit; ++x) {
                    buf_row[x] = srow[tab[x + 2]] * 6
                                 + (srow[tab[x + 1]] + srow[tab[x + 3]]) * 4
                                 + srow[tab[x]] + srow[tab[x + 4]];
                }

                if (x == dwidth) {
                    break;
                }

                x += pyr_down_vec_hor(srow + x * 2 - 2, buf_row + x, width0 - x);
                for (; x < width0; ++x) {
                    buf_row[x] = srow[x * 2] * 6
                                 + (srow[x * 2 - 1] + srow[x * 2 + 1]) * 4
                                 + srow[x * 2 - 2] + srow[x * 2 + 2];
                }

                limit = dwidth;
                tab = tab_r - x;
            }
        }

        // do vertical convolution and decimation and write the result to the destination image
        for (k = 0; k < sp_size; ++k) {
            rows[k] = buf + ((y * 2 - sp_size / 2 + k - sy0) % sp_size) * bufstep;
        }
        row0 = rows[0];
        row1 = rows[1];
        row2 = rows[2];
        row3 = rows[3];
        row4 = rows[4];

        x = pyr_down_vec_vert(rows, drow, dwidth);
        for (; x < dwidth; ++x) {
            int32_t v = row2[x] * 6 + (row1[x] + row3[x]) * 4 + row0[x] + row4[x];
            drow[x] = (uint8_t)((v + (1 << (8 - 1))) >> 8);
        }
    }
}

namespace {
void make_offsets(int32_t pixel[], int32_t row_stride) {
    pixel[0] = 0 + row_stride * 3;
    pixel[1] = 1 + row_stride * 3;
    pixel[2] = 2 + row_stride * 2;
    pixel[3] = 3 + row_stride * 1;
    pixel[4] = 3 + row_stride * 0;
    pixel[5] = 3 + row_stride * -1;
    pixel[6] = 2 + row_stride * -2;
    pixel[7] = 1 + row_stride * -3;
    pixel[8] = 0 + row_stride * -3;
    pixel[9] = -1 + row_stride * -3;
    pixel[10] = -2 + row_stride * -2;
    pixel[11] = -3 + row_stride * -1;
    pixel[12] = -3 + row_stride * 0;
    pixel[13] = -3 + row_stride * 1;
    pixel[14] = -2 + row_stride * 2;
    pixel[15] = -1 + row_stride * 3;
}
} // namespace

uint8_t corner_score(const uint8_t *ptr, const int32_t pixel[], uint8_t threshold) {
    constexpr int32_t K = 8;
    constexpr int32_t N = K * 3 + 1;
    int32_t k;
    int32_t v = ptr[0];
    int16_t d[(N + 7) & ~7];
    for (k = 0; k < N; ++k) {
        d[k] = (int16_t)(v - ptr[pixel[k]]);
    }

#if __HAS_SIMD__
    int16x8_t q0 = vdupq_n_s16(-1000);
    int16x8_t q1 = vdupq_n_s16(1000);
    for (k = 0; k < 16; k += 8) {
        int16x8_t v0 = vld1q_s16((const int16_t *)(d + k + 1));
        int16x8_t v1 = vld1q_s16((const int16_t *)(d + k + 2));
        int16x8_t a = vminq_s16(v0, v1);
        int16x8_t b = vmaxq_s16(v0, v1);
        v0 = vld1q_s16((const int16_t *)(d + k + 3));
        a = vminq_s16(a, v0);
        b = vmaxq_s16(b, v0);
        v0 = vld1q_s16((const int16_t *)(d + k + 4));
        a = vminq_s16(a, v0);
        b = vmaxq_s16(b, v0);
        v0 = vld1q_s16((const int16_t *)(d + k + 5));
        a = vminq_s16(a, v0);
        b = vmaxq_s16(b, v0);
        v0 = vld1q_s16((const int16_t *)(d + k + 6));
        a = vminq_s16(a, v0);
        b = vmaxq_s16(b, v0);
        v0 = vld1q_s16((const int16_t *)(d + k + 7));
        a = vminq_s16(a, v0);
        b = vmaxq_s16(b, v0);
        v0 = vld1q_s16((const int16_t *)(d + k + 8));
        a = vminq_s16(a, v0);
        b = vmaxq_s16(b, v0);
        v0 = vld1q_s16((const int16_t *)(d + k));
        q0 = vmaxq_s16(q0, vminq_s16(a, v0));
        q1 = vminq_s16(q1, vmaxq_s16(b, v0));
        v0 = vld1q_s16((const int16_t *)(d + k + 9));
        q0 = vmaxq_s16(q0, vminq_s16(a, v0));
        q1 = vminq_s16(q1, vmaxq_s16(b, v0));
    }
    q0 = vmaxq_s16(q0, vsubq_s16(vdupq_n_s16(0), q1));

    q0 = vmaxq_s16(q0, vextq_s16(q0, q0, 4));
    q0 = vmaxq_s16(q0, vextq_s16(q0, q0, 2));
    q0 = vmaxq_s16(q0, vextq_s16(q0, q0, 1));
    int16_t result;
    vst1q_lane_s16((int16_t *)&result, q0, 0);
    threshold = result - 1;
#else
    int16_t a0 = threshold;
    for (k = 0; k < 16; k += 2) {
        int16_t a = std::min(d[k + 1], d[k + 2]);
        a = std::min(a, d[k + 3]);
        if (a <= a0) {
            continue;
        }
        a = std::min(a, d[k + 4]);
        a = std::min(a, d[k + 5]);
        a = std::min(a, d[k + 6]);
        a = std::min(a, d[k + 7]);
        a = std::min(a, d[k + 8]);
        a0 = std::max(a0, std::min(a, d[k]));
        a0 = std::max(a0, std::min(a, d[k + 9]));
    }

    int16_t b0 = -a0;
    for (k = 0; k < 16; k += 2) {
        int16_t b = std::max(d[k + 1], d[k + 2]);
        b = std::max(b, d[k + 3]);
        b = std::max(b, d[k + 4]);
        b = std::max(b, d[k + 5]);
        if (b >= b0) {
            continue;
        }
        b = std::max(b, d[k + 6]);
        b = std::max(b, d[k + 7]);
        b = std::max(b, d[k + 8]);

        b0 = std::min(b0, std::max(b, d[k]));
        b0 = std::min(b0, std::max(b, d[k + 9]));
    }

    threshold = -b0 - 1;
#endif

    return threshold;
}

void detect_fast_corners(const CvMat &img, double scale, int32_t level, std::vector<Corner> &grid_keypoints,
                         std::vector<bool> &grid_has_new_keypoint, size_t grid_size,
                         size_t num_grid_col, uint8_t threshold, bool nonmax_suppression) {
    int32_t width = img.cols;
    int32_t height = img.rows;

    constexpr int32_t K = 8;
    constexpr int32_t N = 16 + K + 1;

    int32_t i, j, k;
    int32_t pixel[N];
    make_offsets(pixel, width);
    for (k = 16; k < N; ++k) {
        pixel[k] = pixel[k - 16];
    }

    uint8_t threshold_tab[512];
    for (i = -255; i <= 255; ++i) {
        threshold_tab[i + 255] = (uint8_t)(i < -threshold ? 1 : i > threshold ? 2 : 0);
    }

#if __HAS_SIMD__
    uint8x16_t t = vdupq_n_u8(threshold);
    uint8x16_t K16 = vdupq_n_u8((uint8_t)K);
#endif

    std::vector<uint8_t> buffer((width + 16) * 3 * (sizeof(int32_t) + sizeof(uint8_t)) + 128);
    uint8_t *buf[3];
    buf[0] = buffer.data();
    buf[1] = buf[0] + width;
    buf[2] = buf[1] + width;
    memset(buf[0], 0, width * 3);
    int32_t *cpbuf[3];
    cpbuf[0] = (int32_t *)align_ptr(buf[2] + width, sizeof(int32_t)) + 1;
    cpbuf[1] = cpbuf[0] + width + 1;
    cpbuf[2] = cpbuf[1] + width + 1;

    for (i = 3; i < height - 2; ++i) {
        const uint8_t *ptr = (uint8_t *)img.data + i * img.row_stride + 3;
        uint8_t *curr = buf[(i - 3) % 3];
        memset(curr, 0, width);
        int32_t *corner_pos = cpbuf[(i - 3) % 3];
        int32_t corner_num = 0;

        if (i < height - 3) {
            j = 3;
#if __HAS_SIMD__
            for (; j < width - 16 - 3; j += 16, ptr += 16) {
                uint8x16_t v0 = vld1q_u8(ptr);
                uint8x16_t v1 = vqsubq_u8(v0, t);
                v0 = vqaddq_u8(v0, t);

                uint8x16_t x0 = vld1q_u8(ptr + pixel[0]);
                uint8x16_t x1 = vld1q_u8(ptr + pixel[4]);
                uint8x16_t x2 = vld1q_u8(ptr + pixel[8]);
                uint8x16_t x3 = vld1q_u8(ptr + pixel[12]);

                uint8x16_t m0 = vandq_u8(vcgtq_u8(x0, v0), vcgtq_u8(x1, v0));
                uint8x16_t m1 = vandq_u8(vcgtq_u8(v1, x0), vcgtq_u8(v1, x1));
                m0 = vorrq_u8(m0, vandq_u8(vcgtq_u8(x1, v0), vcgtq_u8(x2, v0)));
                m1 = vorrq_u8(m1, vandq_u8(vcgtq_u8(v1, x1), vcgtq_u8(v1, x2)));
                m0 = vorrq_u8(m0, vandq_u8(vcgtq_u8(x2, v0), vcgtq_u8(x3, v0)));
                m1 = vorrq_u8(m1, vandq_u8(vcgtq_u8(v1, x2), vcgtq_u8(v1, x3)));
                m0 = vorrq_u8(m0, vandq_u8(vcgtq_u8(x3, v0), vcgtq_u8(x0, v0)));
                m1 = vorrq_u8(m1, vandq_u8(vcgtq_u8(v1, x3), vcgtq_u8(v1, x0)));
                m0 = vorrq_u8(m1, m0);

                uint64_t mask[2];
                vst1q_u64(mask, vreinterpretq_u64_u8(m0));

                if (mask[0] == 0) {
                    if (mask[1] != 0) {
                        j -= 8;
                        ptr -= 8;
                    }
                    continue;
                }

                uint8x16_t c0 = vdupq_n_s16(0);
                uint8x16_t c1 = vdupq_n_s16(0);
                uint8x16_t max0 = vdupq_n_u16(0);
                uint8x16_t max1 = vdupq_n_u16(0);
                uint8x16_t x;
                for (k = 0; k < N; ++k) {
                    x = vld1q_u8(ptr + pixel[k]);
                    m0 = vcgtq_u8(x, v0);
                    m1 = vcgtq_u8(v1, x);

                    c0 = vandq_u8(vsubq_s8(c0, m0), m0);
                    c1 = vandq_u8(vsubq_s8(c1, m1), m1);

                    max0 = vmaxq_u8(max0, c0);
                    max1 = vmaxq_u8(max1, c1);
                }
                max0 = vmaxq_u8(max0, max1);

                uint8_t m_array[16];
                vst1q_u8(m_array, vcgtq_u8(max0, K16));
                for (k = 0; k < 16; ++k) {
                    if (m_array[k] != 0) {
                        corner_pos[corner_num++] = j + k;
                        if (nonmax_suppression) {
                            curr[j + k] = corner_score(ptr + k, pixel, threshold);
                        }
                    }
                }
            }
#endif

            for (; j < width - 3; ++j, ++ptr) {
                const int32_t x = j;
                const int32_t y = i - 1;
                const size_t idx = static_cast<size_t>(y / scale) / grid_size * num_grid_col
                                   + static_cast<size_t>(x / scale) / grid_size;
                if (grid_keypoints[idx].score > 1e30) {
                    continue;
                }

                int32_t v = ptr[0];
                const uint8_t *tab = &threshold_tab[0] - v + 255;
                int32_t d = tab[ptr[pixel[0]]] | tab[ptr[pixel[8]]];
                if (d == 0) {
                    continue;
                }
                d &= tab[ptr[pixel[2]]] | tab[ptr[pixel[10]]];
                d &= tab[ptr[pixel[4]]] | tab[ptr[pixel[12]]];
                d &= tab[ptr[pixel[6]]] | tab[ptr[pixel[14]]];
                if (d == 0) {
                    continue;
                }
                d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[9]]];
                d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[11]]];
                d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[13]]];
                d &= tab[ptr[pixel[7]]] | tab[ptr[pixel[15]]];
                if (d & 1) {
                    int32_t vt = v - threshold;
                    int32_t count = 0;
                    for (k = 0; k < N; ++k) {
                        int32_t x = ptr[pixel[k]];
                        if (x < vt) {
                            if (++count > K) {
                                corner_pos[corner_num++] = j;
                                if (nonmax_suppression) {
                                    curr[j] = corner_score(ptr, pixel, threshold);
                                }
                                break;
                            }
                        } else {
                            count = 0;
                        }
                    }
                }
                if (d & 2) {
                    int32_t vt = v + threshold;
                    int32_t count = 0;
                    for (k = 0; k < N; ++k) {
                        int32_t x = ptr[pixel[k]];
                        if (x > vt) {
                            if (++count > K) {
                                corner_pos[corner_num++] = j;
                                if (nonmax_suppression) {
                                    curr[j] = corner_score(ptr, pixel, threshold);
                                }
                                break;
                            }
                        } else {
                            count = 0;
                        }
                    }
                }
            }
        }

        corner_pos[-1] = corner_num;

        if (i == 3) {
            continue;
        }

        const uint8_t *prev = buf[(i - 4 + 3) % 3];
        const uint8_t *pprev = buf[(i - 5 + 3) % 3];
        corner_pos = cpbuf[(i - 4 + 3) % 3];
        corner_num = corner_pos[-1];

        for (k = 0; k < corner_num; ++k) {
            j = corner_pos[k];
            int32_t score = prev[j];
            if (!nonmax_suppression
                || (score > prev[j + 1] && score > prev[j - 1]
                    && score > pprev[j - 1] && score > pprev[j] && score > pprev[j + 1]
                    && score > curr[j - 1] && score > curr[j] && score > curr[j + 1])) {
                const int32_t x = j;
                const int32_t y = i - 1;
                const size_t idx = static_cast<size_t>(y / scale) / grid_size * num_grid_col
                                   + static_cast<size_t>(x / scale) / grid_size;

                if (score > grid_keypoints[idx].score) {
                    grid_keypoints[idx].x = x;
                    grid_keypoints[idx].y = y;
                    grid_keypoints[idx].level = level;
                    grid_keypoints[idx].score = score;
                    // grid_keypoints[idx].angle = -0.5;
                    grid_has_new_keypoint[idx] = true;
                }
            }
        }
    }
}

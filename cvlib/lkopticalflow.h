#pragma once

#include <iostream>
#include <vector>
#include "cvmat.h"

void calc_opticalflow_pyramid_lk(
    const std::vector<CvMat> &src_image_pyr, const std::vector<CvMat> &src_deriv_pyr,
    const std::vector<CvMat> &dst_image_pyr, const std::vector<CvMat> &dst_deriv_pyr,
    const std::vector<std::vector<double>> &src_pts, std::vector<std::vector<double>> &dst_pts, std::vector<uint8_t> &status,
    int32_t win_width, int32_t win_height, int32_t max_level);

int32_t build_opticalflow_pyramid(
    const CvMat &img, std::vector<CvMat> &image_levels, std::vector<CvMat> &deriv_levels,
    int32_t win_width, int32_t win_height, int32_t max_level);

void calc_scharr_deriv(const CvMat &src, CvMat &dst);

int32_t border_interpolate(int32_t p, int32_t len, int32_t border_type);

int32_t pyr_down_vec_hor(const uint8_t *src, int32_t *row, int32_t width);
int32_t pyr_down_vec_vert(int32_t **src, uint8_t *dst, int32_t width);

void pyramid_downsample(const CvMat &src, CvMat &dst, int32_t border_type);

void detect_fast_corners(const CvMat &img, double scale, int32_t level, std::vector<Corner> &grid_keypoints, std::vector<bool> &grid_has_new_keypoint,
                         size_t grid_size, size_t num_grid_col, uint8_t threshold = 10, bool nonmax_suppression = true);

uint8_t corner_score(const uint8_t *ptr, const int32_t pixel[], uint8_t threshold);

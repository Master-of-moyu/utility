#include <iostream>
#include <opencv2/opencv.hpp>
#include "cvmat.h"
#include "lkopticalflow.h"
#include "utility.h"
#include "../mylog.h"

const uint8_t grid_size = 30;
size_t num_grid_col;
size_t num_grid_row;
static cv::Ptr<cv::FastFeatureDetector> g_cvfast = cv::FastFeatureDetector::create();
std::vector<cv::KeyPoint> opencv_keypoints;
// 2d coordinate of the keypoint
std::vector<std::vector<double>> corners_current, corners_next;

void opencv_detect_kps(cv::Mat &img, std::vector<cv::KeyPoint> &kps) {
    // detect
    num_grid_row = std::ceil((double)img.rows / grid_size);
    num_grid_col = std::ceil((double)img.cols / grid_size);
    std::vector<cv::KeyPoint> grid_keypoints(num_grid_col * num_grid_row, cv::KeyPoint(0.0, 0.0, 0.0));

    g_cvfast->setThreshold(10);
    std::vector<cv::KeyPoint> cv_new_keypoints;
    g_cvfast->detect(img, cv_new_keypoints);

    for (size_t i = 0; i < cv_new_keypoints.size(); ++i) {
        size_t k = static_cast<size_t>(cv_new_keypoints[i].pt.y / 1) / grid_size * num_grid_col
                   + static_cast<size_t>(cv_new_keypoints[i].pt.x / 1) / grid_size;

        if (cv_new_keypoints[i].response > grid_keypoints[k].response) {
            grid_keypoints[k] = cv_new_keypoints[i];
            grid_keypoints[k].octave = 0;
        }
    }

    for (size_t i = 0; i < grid_keypoints.size(); ++i) {
        if (grid_keypoints[i].response > 5.0) {
            cv::KeyPoint kp;
            kp.pt.x = grid_keypoints[i].pt.x / 1;
            kp.pt.y = grid_keypoints[i].pt.y / 1;
            kps.push_back(kp);
        }
    }
}

static std::vector<cv::Point2f> to_opencv(const std::vector<cv::KeyPoint> &v) {
    std::vector<cv::Point2f> r(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        r[i].x = (float)v[i].pt.x;
        r[i].y = (float)v[i].pt.y;
    }
    return r;
}

int main(int argc, char **argv) {
    cv::Mat image2 = cv::imread("./images/1.png");
    cv::Mat image3 = cv::imread("./images/4.png", cv::IMREAD_GRAYSCALE);
    cv::Mat image4 = cv::imread("./images/6.png", cv::IMREAD_GRAYSCALE);
    const int of_size = 11;
    const int level_num = 3;

    // opencv detect.
    {
        opencv_detect_kps(image3, opencv_keypoints);
        cv::Mat opencv_kps_show = show_kps(image3, opencv_keypoints);
        cv::imshow("opencv_kps", opencv_kps_show);
        log_debug("opencv detect kps: %zu", opencv_keypoints.size());
    }
    // opencv optical flow.
    {
        std::vector<cv::Mat> cvimage_pyramid_current, cvimage_pyramid_next;
        cv::buildOpticalFlowPyramid(image3, cvimage_pyramid_current, cv::Size(of_size, of_size), level_num, true);
        cv::buildOpticalFlowPyramid(image4, cvimage_pyramid_next, cv::Size(of_size, of_size), level_num, true);

        std::vector<cv::Point2f> current_cvpoints = to_opencv(opencv_keypoints);
        std::vector<cv::Point2f> tracked_keypoints(opencv_keypoints.size());
        std::vector<uint8_t> status;
        std::vector<float> errs;

        cv::calcOpticalFlowPyrLK(cvimage_pyramid_current, cvimage_pyramid_next, current_cvpoints,
                                 tracked_keypoints, status, errs, cv::Size(of_size, of_size));

        size_t n = std::count_if(status.begin(), status.end(), [](const uint8_t &a) { return a != 0; });
        log_debug(" tracked keypoints: %zu", n);

        for (int i = 0; i < tracked_keypoints.size(); i++) {
            if (status[i] != 0) {
                cv::line(image2, cv::Point(current_cvpoints[i].x, current_cvpoints[i].y),
                         cv::Point(tracked_keypoints[i].x, tracked_keypoints[i].y), cv::Scalar(0, 255, 0), 2);
            }
        }
        cv::imshow("opencv_flow", image2);
    }
    // my detect
    {
        log_debug("");
        int img_w = image3.cols;
        int img_h = image3.rows;

        uint8_t *image_data1 = new uint8_t[img_w * img_h];
        uint8_t *image_data2 = new uint8_t[img_w * img_h];

        std::memset(image_data1, 0, img_w * img_h);
        std::memset(image_data2, 0, img_w * img_h);

        std::memcpy(image_data1, image3.data, img_w * img_h);
        std::memcpy(image_data2, image4.data, img_w * img_h);

        CvMat mymat1(image_data1, img_h, img_w);
        CvMat mymat2(image_data2, img_h, img_w);

        std::vector<bool> grid_has_new_keypoint(num_grid_col * num_grid_row, false);
        std::vector<Corner> grid_keypoints(num_grid_col * num_grid_row, Corner(0, 0, 0, 0.0, -1.1));

        detect_fast_corners(mymat1, 1.0, 0, grid_keypoints, grid_has_new_keypoint,
                            grid_size, num_grid_col, 10, true);

        for (size_t k = 0; k < grid_keypoints.size(); ++k) {
            if (grid_has_new_keypoint[k]) {
                const int32_t x = grid_keypoints[k].x;
                const int32_t y = grid_keypoints[k].y;
                const int32_t level = grid_keypoints[k].level;
                std::vector<double> new_keypoint = {x / 1.0, y / 1.0};
                corners_current.push_back(new_keypoint);
            }
        }
        log_debug("mycv detect kps: %zu", corners_current.size());
        cv::Mat mycv_kps_show = show_kps(image3, corners_current);
        cv::imshow("mycv_kps", mycv_kps_show);

        std::vector<CvMat> myimage_pyramid_current, myimage_pyramid_next;
        std::vector<CvMat> myderiv_levels_current, myderiv_levels_next;

        build_opticalflow_pyramid(mymat1, myimage_pyramid_current, myderiv_levels_current, of_size, of_size, level_num);
        build_opticalflow_pyramid(mymat2, myimage_pyramid_next, myderiv_levels_next, of_size, of_size, level_num);

        // my optical flow
        std::vector<uint8_t> status;
        calc_opticalflow_pyramid_lk(
            myimage_pyramid_current, myderiv_levels_current, myimage_pyramid_next, myderiv_levels_next,
            corners_current, corners_next, status, of_size, of_size, myderiv_levels_current.size() - 1);

        size_t n = std::count_if(status.begin(), status.end(), [](const uint8_t &a) { return a != 0; });
        log_debug(" tracked keypoints: %zu", n);

        cv::Mat image2_copy = image2;
        for (int i = 0; i < corners_next.size(); i++) {
            if (status[i] != 0) {
                cv::line(image2_copy, cv::Point(corners_current[i][0], corners_current[i][1]),
                         cv::Point(corners_next[i][0], corners_next[i][1]), cv::Scalar(0, 255, 0), 2);
            }
        }
        cv::imshow("mycv_flow", image2_copy);
    }

    cv::waitKey(0);

    return 0;
}

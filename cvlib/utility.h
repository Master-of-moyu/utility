#pragma once

#include <opencv2/opencv.hpp>
#include "cvmat.h"

cv::Mat show_kps(cv::Mat &I, std::vector<cv::KeyPoint> &kps) {
    cv::Mat temp_img = I.clone();
    cv::Mat im_show = cv::Mat(temp_img.rows, temp_img.cols, CV_8UC3);
    im_show = cv::Mat(temp_img.rows, temp_img.cols, CV_8UC3);
    cv::cvtColor(temp_img, im_show, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < kps.size(); ++i) {
        circle(im_show, cv::Point(cvRound(kps[i].pt.x), cvRound(kps[i].pt.y)), 4, cv::Scalar(0, 255, 0), 1, cv::LINE_8, 0);
    }
    return im_show;
}

cv::Mat show_kps(cv::Mat &I, std::vector<std::vector<double>> &kps) {
    cv::Mat temp_img = I.clone();
    cv::Mat im_show = cv::Mat(temp_img.rows, temp_img.cols, CV_8UC3);
    im_show = cv::Mat(temp_img.rows, temp_img.cols, CV_8UC3);
    cv::cvtColor(temp_img, im_show, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < kps.size(); ++i) {
        cv::circle(im_show, cv::Point(cvRound(kps[i][0]), cvRound(kps[i][1])), 4, cv::Scalar(0, 255, 0), 1, cv::LINE_8, 0);
    }
    return im_show;
}

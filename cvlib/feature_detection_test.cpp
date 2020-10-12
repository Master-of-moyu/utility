#include <iostream>
#include <opencv2/opencv.hpp>

#include "cvmat.h"
#include "lkopticalflow.h"
#include "utility.h"
#include "../mylog.h"

using std::cout;
using std::endl;
using namespace cv;

static Ptr<FastFeatureDetector> g_cvfast = FastFeatureDetector::create();

const uint8_t grid_size = 25;
std::vector<cv::Mat> image_levels;
std::vector<CvMat> myimage_levels;
std::vector<CvMat> myderiv_levels;
std::vector<double> scale_levels;
size_t num_grid_col;
size_t num_grid_row;
int feature_detection_levels = 0;
int level_num = 3;
std::vector<std::vector<double>> corners;
CvMat g_mymat;

struct RawDataReader {
    FILE *fp;

    RawDataReader(const std::string file_name) {
        std::string temp = (file_name).c_str();
        fp = fopen((file_name).c_str(), "rb");
        if (fp == NULL) {
            fprintf(stderr, "%s fopen error!\n", file_name.c_str());
        }
    }

    ~RawDataReader() {
        if (fp) {
            fclose(fp);
            fp = NULL;
        }
    }

    template <typename T>
    void Read(T *data, int size, const int N = 1) {
        fread(data, size, N, fp);
    }
};

void convert_mat2opencv(CvMat &img, cv::Mat &cvimg);
void cv_detect_kps(cv::Mat &img, std::vector<KeyPoint> &kps);
void preprocess(CvMat &img);
void detect_keypoints();

int main(int argc, char **argv) {
    RawDataReader reader(argv[1]);
    unsigned char type;
    double img_time, gtime, acc_time;
    unsigned char image_data[640 * 480] = {0};
    int img_h, img_w;
    double gravity[3];
    double att[4];

    while (1) {
        // read
        reader.Read<unsigned char>(&type, sizeof(unsigned char));
        if (type == 0) {
            log_debug("");
            reader.Read<double>(&img_time, sizeof(double));
            reader.Read<int>(&img_w, sizeof(int));
            reader.Read<int>(&img_h, sizeof(int));
            reader.Read<unsigned char>(image_data, sizeof(unsigned char), img_w * img_h);

            CvMat mymat(image_data, img_h, img_w);
            cv::Mat cvimg(mymat.rows, mymat.cols, CV_8UC1);
            g_mymat = mymat;
            convert_mat2opencv(mymat, cvimg);
            cv::imshow("cvimg", cvimg);

            preprocess(mymat);
            detect_keypoints();

            // 2. opencv feature detection & show
            std::vector<KeyPoint> cv_new_keypoints;
            cv_detect_kps(cvimg, cv_new_keypoints);
            log_debug("opencv detect kps: %zu", cv_new_keypoints.size());
            Mat cv_kps_show = show_kps(cvimg, cv_new_keypoints);
            cv::imshow("opencvkps", cv_kps_show);

            Mat my_kps_show = show_kps(cvimg, corners);
            cv::imshow("mykps", my_kps_show);

            cv::waitKey(0);
            type = -1;
            corners.clear();
        } else if (type == 18 || type == 2 || type == 1) {
            reader.Read<double>(&gtime, sizeof(double));
            reader.Read<double>(gravity, sizeof(double), 3);
            type = -2;
        } else if (type == 17) { // attitude
            reader.Read<double>(&gtime, sizeof(double));
            reader.Read<double>(att, sizeof(double), 4);
        } else {
            cout << "type error!" << endl;
            break;
        }
    }

    return 0;
}

void convert_mat2opencv(CvMat &img, cv::Mat &cvimg) {
    for (int y = 0; y < img.rows; y++) {
        uint8_t *cv_ptr = cvimg.ptr<uint8_t>(y);
        uint8_t *image_ptr = img.ptr<uint8_t>(y);
        for (int x = 0; x < img.cols; x++, image_ptr++, cv_ptr++) {
            *cv_ptr = *image_ptr;
        }
    }
}

void cv_detect_kps(cv::Mat &img, std::vector<KeyPoint> &kps) {
    // create image pyramid
    image_levels.resize(level_num + 1);
    scale_levels.resize(level_num + 1);
    image_levels[0] = img;
    scale_levels[0] = 1.0;
    for (size_t l = 1; l <= level_num; ++l) {
        image_levels[l] = cv::Mat(image_levels[l - 1].size() / 2, CV_8UC1);
        cv::resize(image_levels[l - 1], image_levels[l], image_levels[l].size());
        scale_levels[l] = scale_levels[l - 1] / 2.0;
    }

    // detect
    num_grid_row = std::ceil((double)img.rows / grid_size);
    num_grid_col = std::ceil((double)img.cols / grid_size);
    std::vector<cv::KeyPoint> grid_keypoints(num_grid_col * num_grid_row, cv::KeyPoint(0.0, 0.0, 0.0));

    g_cvfast->setThreshold(10);
    for (int level = 0; level <= feature_detection_levels; ++level) {
        std::vector<cv::KeyPoint> cv_new_keypoints;
        g_cvfast->detect(image_levels[level], cv_new_keypoints);

        for (size_t i = 0; i < cv_new_keypoints.size(); ++i) {
            size_t k = static_cast<size_t>(cv_new_keypoints[i].pt.y / scale_levels[level]) / grid_size * num_grid_col
                       + static_cast<size_t>(cv_new_keypoints[i].pt.x / scale_levels[level]) / grid_size;

            if (cv_new_keypoints[i].response > grid_keypoints[k].response) {
                grid_keypoints[k] = cv_new_keypoints[i];
                grid_keypoints[k].octave = level;
            }
        }
    }

    for (size_t i = 0; i < grid_keypoints.size(); ++i) {
        if (grid_keypoints[i].response > 5.0) {
            cv::KeyPoint kp;
            kp.pt.x = grid_keypoints[i].pt.x / scale_levels[grid_keypoints[i].octave];
            kp.pt.y = grid_keypoints[i].pt.y / scale_levels[grid_keypoints[i].octave];
            kps.push_back(kp);
        }
    }
}

void preprocess(CvMat &image) {
    int32_t detection_border = 5;
    int32_t of_win_size = 11;

    num_grid_col = std::ceil(double(image.cols) / grid_size);
    num_grid_row = std::ceil(double(image.rows) / grid_size);

    build_opticalflow_pyramid(image, myimage_levels, myderiv_levels, of_win_size, of_win_size, level_num);
    for (size_t l = 0; l <= level_num; ++l) {
        const double sx = image.cols / myimage_levels[l].cols;
        scale_levels.push_back(1.0 / sx);
    }
}

void detect_keypoints() {
    std::vector<bool> grid_has_new_keypoint(num_grid_col * num_grid_row, false);
    std::vector<Corner> grid_keypoints(num_grid_col * num_grid_row, Corner(0, 0, 0, 0.0, -1.1));

    // detect_fast_corners(myimage_levels[feature_detection_levels], scale_levels[feature_detection_levels],
    //                     feature_detection_levels, grid_keypoints, grid_has_new_keypoint, grid_size, num_grid_col, 10, true);
    detect_fast_corners(g_mymat, scale_levels[feature_detection_levels],
                        feature_detection_levels, grid_keypoints, grid_has_new_keypoint, grid_size, num_grid_col, 10, true);

    for (size_t k = 0; k < grid_keypoints.size(); ++k) {
        if (grid_has_new_keypoint[k]) {
            const int32_t x = grid_keypoints[k].x;
            const int32_t y = grid_keypoints[k].y;
            const int32_t level = grid_keypoints[k].level;
            std::vector<double> new_keypoint = {x / scale_levels[level], y / scale_levels[level]};
            corners.push_back(new_keypoint);
        }
    }
    log_debug("my kps size: %zu", corners.size());
}

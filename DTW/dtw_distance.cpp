#include <iostream>
#include <vector>
#include "mydtw.h"

int main() {
    double p = 2; // the p-norm to use; 2.0 = euclidean, 1.0 = manhattan
    std::vector<double> a = {10.6928, 10.5972, 10.417, 10.2129, 10.0833, 9.95799, 9.85148,
                             9.69707, 9.58058, 9.48399, 9.37746, 9.25864, 9.20625, 9.16172,
                             9.17338, 9.23253, 9.30864, 9.37809, 9.42583, 9.44704, 9.53012,
                             9.63139, 9.71748, 9.87756, 9.93723, 9.95668, 10.066, 10.2632,
                             10.4308, 10.4874, 10.6171, 10.7274, 10.7882};

    std::vector<double> b = {10.7882, 10.6241, 10.4801, 10.3578, 10.2488, 10.148, 10.0544,
                             9.97485, 9.87539, 9.71731, 9.50626, 9.33536, 9.2095, 9.13573,
                             9.07302, 9.10298, 9.13462, 9.21543, 9.28411, 9.43096, 9.57045,
                             9.68902, 9.82552, 10.0785, 10.3394, 10.5611, 10.6306, 10.6896,
                             10.7752, 11.028};

    std::vector<double> c = {10.2909, 10.2555, 10.0757, 9.95154, 9.78212, 9.6553, 9.4948,
                             9.40699, 9.4311, 9.48217, 9.49493, 9.60148, 9.77896, 9.82215,
                             10.0559, 10.1687, 10.3462};

    std::vector<double> d = {11.3217, 10.724, 10.2159, 10.0559, 10.7934, 11.4202, 11.3798,
                             11.3738, 11.3688, 12.065, 12.6568, 12.3226, 12.0174, 11.6166,
                             12.2844, 12.852, 12.933};

    std::vector<double> e = {10.6928, 10.5972, 10.417, 10.2129, 10.0833, 9.95799, 9.85148,
                             9.69707, 9.58058, 9.48399, 9.37746, 9.25864, 9.20625, 9.16172,
                             9.17338, 9.23253, 9.30864, 9.37809, 9.42583, 9.44704, 9.53012,
                             9.63139, 9.71748, 9.87756, 9.93723, 9.95668, 10.066, 10.2632,
                             10.4308, 10.4874, 10.6171, 10.7274, 10.7882};
    std::vector<double> f;

    double mean = 0.0;
    for (int i = 0; i < e.size(); i++)
        mean += e[i];
    mean /= e.size();

    for (int i = 0; i < e.size(); i++) {
        f.push_back(2 * (mean - e[i]) + e[i]);
    }

    std::cout << "DTW distance1: " << dtw_distance_only(a, b, 1) << std::endl;
    std::cout << "DTW distance2: " << dtw_distance_only(a, c, 1) << std::endl;
    std::cout << "DTW distance3: " << dtw_distance_only(a, d, 1) << std::endl;
    std::cout << "DTW distance4: " << dtw_distance_only(a, e, 1) << std::endl;
    std::cout << "DTW distance5: " << dtw_distance_only(a, f, 1) << std::endl;
    std::cout << "----------" << std::endl;
    std::cout << "DTW distance1: " << dtw_distance(a, b) << std::endl;
    std::cout << "DTW distance2: " << dtw_distance(a, c) << std::endl;
    std::cout << "DTW distance3: " << dtw_distance(a, d) << std::endl;
    std::cout << "DTW distance4: " << dtw_distance(a, e) << std::endl;
    std::cout << "DTW distance5: " << dtw_distance(a, f) << std::endl;
    return 0;
}

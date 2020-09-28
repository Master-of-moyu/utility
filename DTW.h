// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <vector>
#include <cmath>
#include <algorithm>

/**
   * Compute the p_norm between two 1D c++ vectors.
   *
   * The p_norm is sometimes referred to as the Minkowski norm. Common
   * p_norms include p=2.0 for the euclidean norm, or p=1.0 for the
   * manhattan distance. See also
   * https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm
   *
   * @a 1D vector of m size, where m is the number of dimensions.
   * @b 1D vector of m size (must be the same size as b).
   * @p value of norm to use.
   */
double p_norm(double a, double b, double p) {
    double d = 0;
    d += std::pow(std::abs(a - b), p);
    return std::pow(d, 1.0 / p);
};

/**
   * Compute the DTW distance between two 2D c++ vectors.
   *
   * The c++ vectors can have different number of data points, but must
   * have the same number of dimensions. This will raise
   * std::invalid_argument if the dimmensions of a and b are different.
   * Here the vectors should be formatted as
   * [number_of_data_points][number_of_dimensions]. The DTW distance can
   * be computed for any p_norm. See the wiki for more DTW info.
   * https://en.wikipedia.org/wiki/Dynamic_time_warping
   *
   * @a 2D vector of [number_of_data_points][number_of_dimensions].
   * @b 2D vector of [number_of_data_points][number_of_dimensions].
   * @p value of p_norm to use.
   */
double dtw_distance_only(std::vector<double> a, std::vector<double> b, double p) {
    int len_a = a.size();
    int len_b = b.size();

    std::vector<std::vector<double>> d(len_a, std::vector<double>(len_b, 0.0));
    d[0][0] = p_norm(a[0], b[0], p);

    for (int i = 1; i < len_a; i++) {
        d[i][0] = d[i - 1][0] + p_norm(a[i], b[0], p);
    }
    for (int i = 1; i < len_b; i++) {
        d[0][i] = d[0][i - 1] + p_norm(a[i], b[0], p);
    }

    for (int i = 1; i < len_a; i++) {
        for (int j = 1; j < len_b; j++) {
            d[i][j] = p_norm(a[i], b[j], p) + std::fmin(std::fmin(d[i - 1][j], d[i][j - 1]), d[i - 1][j - 1]);
        }
    }
    return d[len_a - 1][len_b - 1];
};

double dtw_distance(std::vector<double> a, std::vector<double> b) {
    int len_a = a.size();
    int len_b = b.size();

    std::vector<std::vector<double>> d(len_a, std::vector<double>(len_b, 0.0));
    d[0][0] = std::abs(a[0] - b[0]);

    for (int i = 1; i < len_a; i++) {
        d[i][0] = d[i - 1][0] + std::abs(a[i] - b[0]);
    }
    for (int i = 1; i < len_b; i++) {
        d[0][i] = d[0][i - 1] + std::abs(a[i] - b[0]);
    }

    for (int i = 1; i < len_a; i++) {
        for (int j = 1; j < len_b; j++) {
            d[i][j] = std::abs(a[i] - b[j]) + std::min(std::min(d[i - 1][j], d[i][j - 1]), d[i - 1][j - 1]);
        }
    }
    return d[len_a - 1][len_b - 1];
};

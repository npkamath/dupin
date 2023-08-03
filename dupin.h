#include <iostream>
#include <getopt.h>
#include <vector>
#include <limits>
#include <utility>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <array>
#include <unordered_map>
#pragma once

class dupinalgo {
public:
    int num_bkps;
    int num_parameters;
    int num_timesteps;
    int jump;
    int min_size;
    std::vector<std::vector<double>> datum;
    std::vector<std::vector<double>> cost_matrix;
 

    struct linear_fit_struct {
        std::vector<std::vector<double>> y;
        std::vector<double> x;
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_xx = 0.0;
        double sum_xy = 0.0;
    };


    void readinputpybind();
    std::unordered_map<int, std::unordered_map<int, double>> memo;

    void regression_setup(linear_fit_struct &lfit);


    std::vector<double> regressionline(int start, int end, int dim, linear_fit_struct& lfit);

    double l2_cost(std::vector<std::vector<double>>& predicted_y, int start, int end);

   // std::vector<double> computeCumulativeSum(std::vector<double>& arr);

    std::vector<std::vector<double>> predicted(int start, int end, linear_fit_struct& lfit);

    double cost_function(int start, int end);

    std::vector<int> admissible_bkps(int start, int end, int num_bkps);

    double seg(int start, int end, int num_bkps);

    std::vector<int> return_breakpoints();

    std::vector<int> bottomup_bkps();


    void read_input();
    std::vector<std::vector<double>> initialize_cost_matrix(std::vector<std::vector<double>>& datum);
    std::vector<int> getTopDownBreakpoints();
    std::vector<int> getBottomUpBreakpoints();
};
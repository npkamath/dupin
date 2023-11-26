#pragma once

#include <iostream>
#include <vector>
#include <unordered_map>
#include <limits>
#include <algorithm>




// dupinalgo class for dynamic programming based segmentation.
class dupinalgo {
private:
    // Struct for memoization key, combining start, end, and number of breakpoints.
    struct MemoKey {
        int start;
        int end;
        int num_bkps;

        bool operator==(const MemoKey &other) const {
            return start == other.start && end == other.end && num_bkps == other.num_bkps;
        }
    };

    // Custom XOR-bit hash function for MemoKey, avoids clustering of data in unordered map to improve efficiency
    struct MemoKeyHash {
        std::size_t operator()(const MemoKey& key) const {
            return ((std::hash<int>()(key.start) ^ (std::hash<int>()(key.end) << 1)) >> 1) ^ std::hash<int>()(key.num_bkps);
        }
    };

    // Memoization map to store the cost and partition for given parameters.
    std::unordered_map<MemoKey, std::pair<double, std::vector<int>>, MemoKeyHash> memo;

    int num_bkps; // Number of breakpoints.
    int num_parameters; // Number of parameters in the data.
    int num_timesteps; // Number of timesteps in the data.
    int jump; // Jump size for checking breakpoints.
    int min_size; // Minimum segment size.
    std::vector<std::vector<double>> datum; // Input data matrix.
    std::vector<std::vector<double>> cost_matrix; // Matrix storing cost values.

    // Structure for linear fit computation.
    struct linear_fit_struct {
        std::vector<std::vector<double>> y; // labels
        std::vector<double> x; // timesteps
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_xx = 0.0;
        double sum_xy = 0.0;
    };

public:
    dupinalgo(); // Default constructor.
    dupinalgo(int num_bkps_, int num_parameters_, int num_timesteps_, int jump_, int min_size_); // Parameterized constructor.

    // Function to normalize data by min-max normalization.
    void scale_datum();

    // Function to read input data(mainly used for testing).
    void read_input();

    // Function for reading input with pybind11 support(not required anymore).
    void readinputpybind();

    // Function to set up linear regression variables and normalize x-values
    void regression_setup(linear_fit_struct &lfit);

    // Function to compute the regression line for a given segment and feature using lfit structure
    std::vector<double> regressionline(int start, int end, int dim, linear_fit_struct& lfit);

    // Function to generate predicted data based on regression line
    std::vector<std::vector<double>> predicted(int start, int end, linear_fit_struct& lfit);

    // Function to calculate L2 cost between predicted data and actual labels in order to minimize the objective function
    double l2_cost(std::vector<std::vector<double>>& predicted_y, int start, int end);

    // Function to compute cost between two points.
    double cost_function(int start, int end);

    // Function to calculate all costs between all points and store in a cost matrix
    // only stores in upper triangular form to save memory
    // utilizes and combines all cost function components and scales data before computing
    std::vector<std::vector<double>> initialize_cost_matrix(std::vector<std::vector<double>>& datum);

    // Function to segment data using dynamic programming (top-down approach).
    std::pair<double, std::vector<int>> seg(int start, int end, int num_bkps);

    // Function to return optimal breakpoints after dynamic programming.
    std::vector<int> return_breakpoints();
    // Function to get top-down breakpoints.
    std::vector<int> getTopDownBreakpoints();

    // Getter functions for private class members.
    int get_num_timesteps();
    int get_num_parameters();
    int get_num_bkps();
    std::vector<std::vector<double>>& getDatum();
    std::vector<std::vector<double>>& getCostMatrix();
    
    // Setter functions for private class members.
    void set_num_timesteps(int value);
    void set_num_parameters(int value);
    void set_num_bkps(int value);
    void setDatum(const std::vector<std::vector<double>>& value);   
    void setCostMatrix(const std::vector<std::vector<double>>& value);
};

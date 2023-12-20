#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <unordered_map>
#include "dupin.h"
#include <iomanip>
#include <omp.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <tbb/global_control.h>



//#include <pybind11/numpy.h>
using namespace std;
using namespace Eigen;

//Constructors

dupinalgo::dupinalgo()
    : num_bkps(0), num_parameters(0), num_timesteps(0), jump(1), min_size(3) {
  
}

dupinalgo::dupinalgo(int num_bkps_, int num_parameters_, int num_timesteps_, int jump_, int min_size_)
    : num_bkps(num_bkps_), num_parameters(num_parameters_), 
      num_timesteps(num_timesteps_), jump(jump_), min_size(min_size_) {

}

void dupinalgo::read_input() {
    cin >> jump >> min_size >> num_bkps >> num_parameters >> num_timesteps;
    datum.resize(num_timesteps, num_parameters);
    
    for (int i = 0; i < num_timesteps; ++i) {
        for (int j = 0; j < num_parameters; ++j) {
            cin >> datum(i, j);
        }
    }
}


void dupinalgo::scale_datum() {
    VectorXd min_val = datum.colwise().minCoeff();
    VectorXd max_val = datum.colwise().maxCoeff();
    VectorXd range = max_val - min_val;

    for (int j = 0; j < num_parameters; ++j) {
        if (range(j) == 0.0) {
            datum.col(j).setZero();
        } else {
            datum.col(j) = (datum.col(j).array() - min_val(j)) / range(j);
        }
    }
}
void dupinalgo::regression_setup(linear_fit_struct &lfit) {
    lfit.x = VectorXd::LinSpaced(num_timesteps, 0, num_timesteps - 1) / (num_timesteps - 1);
    lfit.y = datum;
}

VectorXd dupinalgo::regressionline(int start, int end, int dim, linear_fit_struct &lfit) {
    int n = end - start;
    VectorXd x = lfit.x.segment(start, n);
    VectorXd y = lfit.y.col(dim).segment(start, n);

    double x_mean = x.mean();
    double y_mean = y.mean();

    // Convert the expressions to vectors before calculating the dot product
    VectorXd x_centered = x.array() - x_mean;
    VectorXd y_centered = y.array() - y_mean;

    double slope = x_centered.dot(y_centered) / x_centered.squaredNorm();
    double intercept = y_mean - slope * x_mean;

    // Return the fitted line
    return x.unaryExpr([slope, intercept](double xi) { return slope * xi + intercept; });
}


double dupinalgo::l2_cost(MatrixXd &predicted_y, int start, int end) {
    MatrixXd diff = predicted_y.block(start, 0, end - start, num_parameters) - datum.block(start, 0, end - start, num_parameters);
    return sqrt(diff.array().square().sum());
}



MatrixXd dupinalgo::predicted(int start, int end, linear_fit_struct &lfit) {
    MatrixXd predicted_y(num_timesteps, num_parameters);
    for (int i = 0; i < num_parameters; ++i) {
        predicted_y.block(start, i, end - start, 1) = regressionline(start, end, i, lfit);
    }
    return predicted_y;
}

double dupinalgo::cost_function(int start, int end) {
    linear_fit_struct lfit;
    regression_setup(lfit);
    MatrixXd predicted_y = predicted(start, end, lfit);
    return l2_cost(predicted_y, start, end);
}

void dupinalgo::initialize_cost_matrix_old() {
    tbb::global_control c(tbb::global_control::max_allowed_parallelism, 1);
    scale_datum();
    cost_matrix.resize(num_timesteps, num_timesteps);
    cost_matrix.setZero();
    tbb::parallel_for(tbb::blocked_range<int>(0, num_timesteps), [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i < r.end(); ++i) {
            for (int j = i + min_size; j < num_timesteps; ++j) {
                cost_matrix(i, j) = cost_function(i, j);
            }
        }
    });
}
void dupinalgo::initialize_cost_matrix() {
    tbb::global_control c(tbb::global_control::max_allowed_parallelism, 1);
    scale_datum();
    cost_matrix_new.initialize(num_timesteps);

    tbb::parallel_for(tbb::blocked_range<int>(0, num_timesteps), [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i < r.end(); ++i) {
            for (int j = i + min_size; j < num_timesteps; ++j) {
                cost_matrix_new(i, j) = cost_function(i, j);
            }
        }
    });
}

//DP Solution Part


pair<double, vector<int>> dupinalgo::segold(int start, int end, int num_bkps) {
        MemoKey key = {start, end, num_bkps};
        auto it = memo.find(key);
        if (it != memo.end()) {
            return it->second;
            cout <<"memo used!\n";
        }

        if (num_bkps == 0) {
            return {cost_matrix(start, end), {end}};
        }

        pair<double, vector<int>> best = {numeric_limits<double>::infinity(), {}};
        
        for (int bkp = start + min_size; bkp < end; bkp++) {
            if ((bkp - start) >= min_size && (end - bkp) >= min_size) {
                auto left = segold(start, bkp, num_bkps - 1);
                auto right = segold(bkp, end, 0);
                double cost = left.first + right.first;
                if (cost < best.first) {
                    best.first = cost;
                    best.second = left.second;
                    best.second.push_back(bkp);
                    best.second.insert(best.second.end(), right.second.begin(), right.second.end());
                }
            }
        }

        memo[key] = best;
        return best;
    }


    vector<int> dupinalgo::return_breakpoints_old() {
        auto result = segold(0, num_timesteps-1, num_bkps);
        vector<int> breakpoints = result.second;
        sort(breakpoints.begin(), breakpoints.end());
        breakpoints.erase(unique(breakpoints.begin(), breakpoints.end()), breakpoints.end());
 //       cout << "Memo count: "<< memo_count << "no memo count: "<< no_memo<< endl; 
        return breakpoints;
    }

//think about using 2d vector/array here//1d vector
//top down recursive implementation
// Recursive function to segment the data
pair<double, vector<int>> dupinalgo::seg(int start, int end, int num_bkps) {
        MemoKey key = {start, end, num_bkps};
        auto it = memo.find(key);
        if (it != memo.end()) {
//            memo_count++; 
            return it->second;
        }
//            no_memo++;
        if (num_bkps == 0) {
            return {cost_matrix_new(start, end), {end}};
        }

        pair<double, vector<int>> best = {numeric_limits<double>::infinity(), {}};
        
        for (int bkp = start + min_size; bkp < end; bkp++) {
            if ((bkp - start) >= min_size && (end - bkp) >= min_size) {
                auto left = seg(start, bkp, num_bkps - 1);
                auto right = seg(bkp, end, 0);
                double cost = left.first + right.first;
                if (cost < best.first) {
                    best.first = cost;
                    best.second = left.second;
                    best.second.push_back(bkp);
                    best.second.insert(best.second.end(), right.second.begin(), right.second.end());
                }
            }
        }

        memo[key] = best;
        return best;
    }

    vector<int> dupinalgo::return_breakpoints() {
        auto result = seg(0, num_timesteps-1, num_bkps);
        vector<int> breakpoints = result.second;
        sort(breakpoints.begin(), breakpoints.end());
        breakpoints.erase(unique(breakpoints.begin(), breakpoints.end()), breakpoints.end());
 //       cout << "Memo count: "<< memo_count << "no memo count: "<< no_memo<< endl; 
        return breakpoints;
    }


vector<int> dupinalgo::getTopDownBreakpoints(string type) {
    if(type == "old"){
        return dupinalgo::return_breakpoints_old();
    }
    else{
        return dupinalgo::return_breakpoints();
    }
	
}


int dupinalgo::get_num_timesteps() {
    return num_timesteps;
}

int dupinalgo::get_num_parameters() {
    return num_parameters;
}

int dupinalgo::get_num_bkps() {
    return num_bkps;
}

Eigen::MatrixXd& dupinalgo::getDatum() {
    return datum;
}

Eigen::MatrixXd& dupinalgo::getCostMatrix() {
    return cost_matrix;
}

void dupinalgo::set_num_timesteps(int value) {
    num_timesteps = value;
}

void dupinalgo::set_num_parameters(int value) {
    num_parameters = value;
}

void dupinalgo::set_num_bkps(int value) {
    num_bkps = value;
}

void dupinalgo::setDatum(const Eigen::MatrixXd& value) {
    datum = value;
}   

void dupinalgo::setCostMatrix(const Eigen::MatrixXd&value) {
    cost_matrix = value;
}

int main() {
    
	dupinalgo dupin;
    dupin.read_input();
    cout << "Validating input: \n";
    for (int i = 0; i < dupin.get_num_timesteps(); i++) {
        for (int j = 0; j < dupin.get_num_parameters(); j++) {
            cout << dupin.getDatum()(i, j) << " ";
        }
        cout << endl;
    }

    dupin.initialize_cost_matrix();
    cout << "Validating cost matrix: \n";
    for (int i = 1; i <= dupin.get_num_timesteps(); i++) {
        cout << setw(12) << i - 1 << " ";
    }
    cout << endl;
    for (int i = 0; i < dupin.get_num_timesteps(); i++) {
        cout << i  << " ";
        for (int j = 0; j < dupin.get_num_timesteps(); j++) {
            cout << setw(12) << setprecision(8) << dupin.getCostMatrix()(i, j) << "|";
        }
        cout << endl;
    }
//test top down
    string newp = "new";
	auto topbreakponts = dupin.getTopDownBreakpoints(newp);
	cout << "top down results: ";
	for (auto &i : topbreakponts) {
		cout << i << " "; 
	}
	cout << endl; 

}
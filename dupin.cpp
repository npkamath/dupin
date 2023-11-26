#include <iostream>
#include <vector>
#include <limits>
#include <utility>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <array>
#include <unordered_map>
#include "dupin.h"


//#include <pybind11/numpy.h>
using namespace std;

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
	datum.resize(num_timesteps, vector<double>(num_parameters));
	double temp;
	for (int i = 0; i < num_timesteps; i++) {
		for (int j = 0; j < num_parameters; j++) {
			cin >> temp;
			datum[i][j] = temp;

		}

	}
	
}


void dupinalgo::scale_datum() {
    vector<double> min_val(num_parameters, std::numeric_limits<double>::max());
    vector<double> max_val(num_parameters, std::numeric_limits<double>::lowest());

    // Find min and max values for each parameter
    for (const auto& row : datum) {
        for (int j = 0; j < num_parameters; ++j) {
            min_val[j] = min(min_val[j], row[j]);
            max_val[j] = max(max_val[j], row[j]);
        }
    }

    // Scale the datum using min-max scaling
    for (int i = 0; i < num_timesteps; ++i) {
        for (int j = 0; j < num_parameters; ++j) {
            double denominator = max_val[j] - min_val[j];
            if (denominator == 0) {
                datum[i][j] = 0; 
            }
			else{
				datum[i][j] = (datum[i][j] - min_val[j]) / denominator;
			}
            
        }
    }
}

void dupinalgo::regression_setup(linear_fit_struct &lfit) {
	lfit.x.resize(num_timesteps);
	//
	for (int i = 0; i < num_timesteps; ++i) {
		lfit.x[i] = static_cast<double>(i) / (static_cast<double>(num_timesteps) - 1);
	}
	lfit.y = datum;
}

vector<double> dupinalgo::regressionline(int start, int end, int dim, linear_fit_struct &lfit) {
    int n = end - start;
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
    vector<double> line;
    line.reserve(n); // Preallocate memory

    for (int i = start; i < end; ++i) {
        double current_x = lfit.x[i];
        double current_y = lfit.y[i][dim];
        sum_x += current_x;
        sum_y += current_y;
        sum_xy += current_x * current_y;
        sum_xx += current_x * current_x;
    }

    double denom = n * sum_xx - sum_x * sum_x;
    double m = 0.0, b = sum_y / n; // Default to horizontal line at the average y value

    if (denom != 0.0) { // Only calculate slope if denominator is not zero
        m = (n * sum_xy - sum_x * sum_y) / denom;
        b = (sum_y - m * sum_x) / n;
    }

    for (int i = start; i < end; i++) {
        double y = m * lfit.x[i] + b;
        line.push_back(y);
    }
    return line;
}

double dupinalgo::l2_cost(vector<vector<double>> &predicted_y, int start, int end) {
    double sum = 0.0;
    for (int i = start; i < end; i++) {
        for (int j = 0; j < num_parameters; j++) {
            double diff = predicted_y[i][j] - datum[i][j];
            sum += diff * diff;
        }
    }
    return sqrt(sum);
}




vector <vector <double>>  dupinalgo::predicted(int start, int end, linear_fit_struct &lfit) {
	vector<vector<double>> predicted_y(num_timesteps, vector<double>(num_parameters));
	
	for (int i = 0; i < num_parameters; i++) {
		vector<double> line = regressionline(start, end, i, lfit);
		for (int j = start; j < end; j++) {
			predicted_y[j][i] = line[j - start];
		}
	}
	return predicted_y;
}
double dupinalgo::cost_function(int start, int end) {
	
	linear_fit_struct lfit;
	regression_setup(lfit);

	// Compute predicted values
	vector<vector<double>> predicted_y = predicted(start, end, lfit);

	// Compute and return the L2 cost
	double final_cost = l2_cost(predicted_y, start, end);
	return final_cost;
}


vector<vector<double>> dupinalgo::initialize_cost_matrix(vector<vector<double>>& datum) {  //initialize and return the cost matrix
	
	scale_datum();
	cost_matrix.resize(num_timesteps, vector<double>(num_timesteps, 0.0));// only fill out half the matrix
	for (int i = 0; i < num_timesteps; i++) {
		for (int j = i + min_size; j < num_timesteps; j++) {
			cost_matrix[i][j] = cost_function(i, j); //fix to i and j
		}
	}
	return cost_matrix;

}

//DP Solution Part



int recursiv_count = 0;
//think about using 2d vector/array here//1d vector
//top down recursive implementation
// Recursive function to segment the data
 pair<double, vector<int>> dupinalgo::seg(int start, int end, int num_bkps) {
        MemoKey key = {start, end, num_bkps};
        auto it = memo.find(key);
        if (it != memo.end()) {
            return it->second;
        }

        if (num_bkps == 0) {
            return {cost_matrix[start][end], {end}};
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
        return breakpoints;
    }


vector<int> dupinalgo::getTopDownBreakpoints() {
	return dupinalgo::return_breakpoints();
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

vector<vector<double>>& dupinalgo::getDatum() {
    return datum;
}

vector<vector<double>>& dupinalgo::getCostMatrix() {
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

void dupinalgo::setDatum(const vector<vector<double>>& value) {
    datum = value;
}   

void dupinalgo::setCostMatrix(const vector<vector<double>>& value) {
    cost_matrix = value;
}

int main() {
/*
	dupinalgo dupin;
    dupin.read_input();
    cout << "Validating input: \n";
    for (int i = 0; i < dupin.get_num_timesteps(); i++) {
        for (int j = 0; j < dupin.get_num_parameters(); j++) {
            cout << dupin.getDatum()[i][j] << " ";
        }
        cout << endl;
    }

    dupin.initialize_cost_matrix(dupin.getDatum());
    cout << "Validating cost matrix: \n";
    for (int i = 1; i <= dupin.get_num_timesteps(); i++) {
        cout << setw(12) << i - 1 << " ";
    }
    cout << endl;
    for (int i = 0; i < dupin.get_num_timesteps(); i++) {
        cout << i  << " ";
        for (int j = 0; j < dupin.get_num_timesteps(); j++) {
            cout << setw(12) << setprecision(8) << dupin.getCostMatrix()[i][j] << "|";
        }
        cout << endl;
    }
//test top down
	auto topbreakponts = dupin.getTopDownBreakpoints();
	cout << "top down results: ";
	for (auto &i : topbreakponts) {
		cout << i << " "; 
	}
	cout << endl; 
	auto botbreakponts = dupin.bottomup_bkps();
	cout << "bot up results: "; 
	for (auto &i : botbreakponts) {
		cout << i << " "; 
	}
*/
}
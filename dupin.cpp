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
#include "dupin.h"
//#include <pybind11/numpy.h>
using namespace std;
/*
int num_bkps;
int num_parameters;
int num_timesteps;
int jump;
int min_size;
vector<vector<double>> datum;
vector <vector<double>> cost_matrix;
*/



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

void dupinalgo::readinputpybind() {

}




void dupinalgo::regression_setup(linear_fit_struct &lfit) {
	lfit.x.resize(num_timesteps);
	//normalize timestep vecotr
	for (int i = 0; i < num_timesteps; ++i) {
		lfit.x[i] = static_cast<double>(i) / (num_timesteps - 1) * 1.0;
	//	cout << lfit.x[i] << " timestep check ";
	}
	lfit.y = datum;
}

vector<double> dupinalgo::regressionline(int start, int end, int dim, linear_fit_struct &lfit) {
	vector<double> line;
	int n = end - start;
	// cout << "Dimension: " << dim << "\n";
	double current_x = 0.0;
	double current_y = 0.0;
	for (int i = start; i < end; ++i) {
		current_x = lfit.x[i];
		current_y = lfit.y[i][dim];

		lfit.sum_x += current_x;
		lfit.sum_y += current_y;
		lfit.sum_xy += current_x * current_y;
		lfit.sum_xx += current_x * current_x;
	}
	// cout <<"sum check: "<< " x: "<< lfit.sum_x << " y: " << lfit.sum_y
	// 	<< " xy: " << lfit.sum_xy << " xx: " << lfit.sum_xx << "\n";
	double denom = (n * lfit.sum_xx - (lfit.sum_x * lfit.sum_x)); 
	double m = 0.0;
	double b = 0.0;
	if (denom != 0.0) {
		m = (n * lfit.sum_xy - (lfit.sum_x * lfit.sum_y)) / (n * lfit.sum_xx - (lfit.sum_x * lfit.sum_x));
	}
	if (n != 0.0) {
		b = (lfit.sum_y - (m * lfit.sum_x)) / n;
	}

	for (int i = start; i < end; i++) {
		double x = lfit.x[i];
		double y = m * x + b;
	// 	cout <<"Line: " << y << " = " << m << " * " << x << " + " << b << endl;
		line.push_back(y);
	}
	return line;
	 
}

double dupinalgo::l2_cost(vector<vector<double>>& predicted_y, int start, int end) {
	double sum = 0;
	int n = end - start;
	double diff = 0;
	for (int i = start; i < end; i++) {
		for (int j = 0; j < num_parameters; j++) {
			diff = abs(predicted_y[i][j] - datum[i][j]);
		//	cout << "difference of " << i << " " << j << " is " << diff << endl;
			sum += diff * diff;
		}
	}
	return sqrt(sum);

}


/*vector<double> computeCumulativeSum(vector<double>& arr) {
	vector<double> cumsum(arr.size() + 1, 0.0);
	for (int i = 0; i < arr.size(); ++i) {
		cumsum[i + 1] = cumsum[i] + arr[i];
	}
	return cumsum;
}
*/

vector <vector <double>>  dupinalgo::predicted(int start, int end, linear_fit_struct &lfit) {
	vector<vector<double>> predicted_y(num_timesteps, vector<double>(num_parameters));
	
	for (int i = 0; i < num_parameters; i++) {
	//	cout << "Calling regressionline on paramater: " << i << "\n";
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
	int temp;
	bool swapped = false;
	if (start > end) {
		temp = end;
		end = start;
		start = temp;
		swapped = true;
	}

	vector<vector<double>> predicted_y = predicted(start, end, lfit);

	// Compute and return the L2 cost
	double final_cost = l2_cost(predicted_y, start, end);
	if (!swapped) {
		//cout << "Final Cost for " << start << " " << end << " is " << final_cost << endl << endl;
	}
	else {
		//cout << "Final Cost for " << end << " " << start << " is " << final_cost << endl << endl;
	}
	return final_cost;
}


vector<vector<double>> dupinalgo::initialize_cost_matrix(vector<vector<double>>& datum) {  //initialize and return the cost matrix
	

	cost_matrix.resize(num_timesteps, vector<double>(num_timesteps, 0.0));// only fill out half the matrix
	for (int i = 0; i < num_timesteps; i++) {
		for (int j = i + min_size; j < num_timesteps; j++) {

			//cout << "checking distance between " << i << " " << j<<endl; 
			cost_matrix[i][j] = cost_function(i, j); //fix to i and j
		}
	}
	return cost_matrix;

}
vector<int> dupinalgo::admissible_bkps(int start, int end, int num_bkps) {
	vector <int> list;
	//cout << "admissible bkps for " << start << " to " << end << endl;
	for (int i = start + min_size; i < end; i = i + 1) {
		int left_samples = i - start;
		int right_samples = end - i;
		if (left_samples >= min_size - 1 && right_samples >= min_size - 1 && i % jump == 0) {
			list.push_back(i);
			cout << i << " ";
		}
	}
	//cout << endl;
	return list;
}

//think about using 2d vector/array here//1d vector
//top down recursive implementation
double dupinalgo::seg(int start, int end, int num_bkps) {
	//recurrence to find the optimal partition
	double cost;
	vector<int> goodbkps;
	if (num_bkps == 0) {
		if (end - start > 2) {
			cost = cost_matrix[start][end];
			return cost;
		}
		else {
			return std::numeric_limits<double>::infinity();
		}
	}
	if (memo.count(start) && memo[start].count(end)) {
		return memo[start][end];
	}
	else {
		goodbkps = admissible_bkps(start, end, num_bkps);
		


			unordered_map<int, double> subproblems;

			for (auto& bkp : goodbkps) {
				double left_partition = seg(start, bkp, num_bkps - 1);
				double right_partition = seg(bkp, end, 0);
				double tmp_partition = left_partition + right_partition;
				subproblems[bkp] = tmp_partition;
			}
			double min_cost = std::numeric_limits<double>::infinity();

			for (auto it = subproblems.begin(); it != subproblems.end(); it++) {
				min_cost = min(min_cost, it->second);
			}
			memo[start][end] = min_cost;
			return min_cost;
		}

	
}

//Now.. return the breakpoints with a while loop !
vector<int> dupinalgo::return_breakpoints() {
	vector<int> opt_bkps;
	opt_bkps.reserve(num_bkps + 1);
	opt_bkps.push_back(num_timesteps - 1);

	if (num_bkps > cost_matrix.size()) {
	//	cout << "num_bkps is bigger than cost_matrix.size()" << endl;
		return opt_bkps; // or throw an exception
	}

	int little_k = num_bkps;
	int start = 0;
	int end = num_timesteps - 1;

	while (little_k > 0) {
	//	cout << " little k: " << little_k << "\n" << "_________________\n";
		int optimal_bkp = -1;
	

		if (end >= cost_matrix.size()) {
	//		cout << "end index is out of range: end=" << end << ", cost_matrix.size()=" << cost_matrix.size() << endl;
			return opt_bkps; // or throw an exception
		}


		vector<int> goodbkps = admissible_bkps(start, end, num_bkps);
		std::reverse(goodbkps.begin(), goodbkps.end()); 
		double min_cost = std::numeric_limits<double>::infinity();
		for (auto& bkp : goodbkps) {
			
			double total_cost = seg(start, bkp, little_k - 1) + cost_matrix[bkp][end];
	//		cout << "current min_cost is " << min_cost << " total cost for " <<bkp<< ": "<< total_cost<<endl;
			if (total_cost < min_cost) {
				min_cost = total_cost;
				optimal_bkp = bkp;
	//			cout << bkp << " is an optimal breakpoint with a cost of "<< total_cost << endl;
			}
			
			else {
	//			cout << bkp << " is not an optimal breakpoint with a cost of " << total_cost<<  endl; 
			}
		}

		if (optimal_bkp == -1) {
			little_k--;
			continue;
		}

		opt_bkps.push_back(optimal_bkp);
		end = end - min_size;
		little_k--;
	}
	std::reverse(opt_bkps.begin(), opt_bkps.end());
	return opt_bkps;
}



vector<int> dupinalgo::getTopDownBreakpoints() {
	return dupinalgo::return_breakpoints();
}
vector<int> dupinalgo::bottomup_bkps() {

	//bottom up implementation; avoid recursion so can avoid a 
//very large stack of memory needed for much larger structural changes
//same time complexity as top_down, but ruptures does not use iterative approach
//so unsure if algorithm is completely correct


	vector<vector<double>> tab(num_timesteps + 1, vector<double>(num_bkps + 1, numeric_limits<double>::infinity()));
	vector <vector<int>> bkps;
	bkps.resize(num_timesteps + 1, vector<int>(num_bkps));
	vector<int> optimal_bkps;

	tab[0][0] = 0;

	for (int i = min_size; i <= num_timesteps; ++i) {
		for (int j = 1; j <= num_bkps; ++j) {
			for (int k = min_size; k <= i; k += 1) {
				double current_cost = tab[k - 1][j - 1] + cost_matrix[k][i];
				if (current_cost < tab[i][j]) {
					tab[i][j] = current_cost;
					bkps[i][j] = k;
				}
			}
		}
	}
	int end = num_timesteps;
	int little_k = num_bkps;
	while (little_k > 0) {
		int start = bkps[end][little_k];
		optimal_bkps.push_back(start);
		end = start;
		little_k--;
		}
	std::reverse(optimal_bkps.begin(), optimal_bkps.end());                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
	return optimal_bkps;
}

int main() {
	dupinalgo dupin;
	dupin.read_input();
	//cout << "Validating input: \n";
	for (int i = 0; i < dupin.num_timesteps; i++) {
		for (int j = 0; j < dupin.num_parameters; j++) {
		///	cout << dupin.datum[i][j]<< " ";
		}
		//cout << endl;
	}
	//cout << "num_bkps: " << dupin.num_bkps << "\n";
	//cout << "num_parameters: " << dupin.num_parameters << "\n";
	//cout << "num_timesteps: " << dupin.num_timesteps << "\n";
	//cout << "jump: " << dupin.jump << "\n";
	//cout << "min_size: " << dupin.min_size << "\n";

	dupin.initialize_cost_matrix(dupin.datum);
	//cout << "Validating cost matrix: \n";
	for (int i = 1; i <= dupin.num_timesteps; i++) {
		//cout << setw(12) << i - 1 << " ";
	}
	//cout << endl;
	for (int i = 0; i < dupin.num_timesteps; i++) {
		//cout << i  << " ";
		for (int j = 0; j < dupin.num_timesteps; j++) {
			//cout << setw(12)<< setprecision(8)<< dupin.cost_matrix[i][j] << "|";
		}
	//	cout << endl;
	}
//test top down
	auto breakponts = dupin.getTopDownBreakpoints();
	//cout << "top down results: ";
	for (auto &i : breakponts) {
		cout << i << " "; 
	}
	
	//initialize_cost_matrix(datum);

//test bottom up
	/*auto bottomupbkp = bottomup_bkps();
	for (int i = 0; i < num_bkps; i++) {
		cout << bottomupbkp[i] << " "; */
	//}

}
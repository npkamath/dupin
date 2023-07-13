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

using namespace std;

int num_bkps;
int num_parameters;
int num_timesteps;
int jump;
int min_size;
vector<vector<double>> datum;
vector <vector<double>> cost_matrix;

void read_input() {
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



struct linear_fit_struct {
	vector<vector<double>> y;
	vector<double> x;
	double sum_x = 0.0;
	double sum_y = 0.0;
	double sum_xx = 0.0;
	double sum_xy = 0.0;
	
};


void regression_setup(linear_fit_struct &lfit) {
	lfit.x.resize(num_timesteps);
	//normalize timestep vecotr
	for (int i = 0; i < num_timesteps; ++i) {
		lfit.x[i] = static_cast<double>(i) / (num_timesteps - 1) * 1.0;
	}
	lfit.y = datum;
}

vector<double> regressionline(int start, int end, int dim, linear_fit_struct &lfit) {
	vector<double> line;
	int n = end - start;
	cout << "Dimension: " << dim << "\n";
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
	cout <<"sum check: "<< " x: "<< lfit.sum_x << " y: " << lfit.sum_y
		<< " xy: " << lfit.sum_xy << " xx: " << lfit.sum_xx << "\n";
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
		cout <<"Line: " << y << " = " << m << " * " << x << " + " << b << endl;
		line.push_back(y);
	}
	return line;
	 
}

double l2_cost(vector<vector<double>>& predicted_y, int start, int end) {
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


vector<double> computeCumulativeSum(vector<double>& arr) {
	vector<double> cumsum(arr.size() + 1, 0.0);
	for (int i = 0; i < arr.size(); ++i) {
		cumsum[i + 1] = cumsum[i] + arr[i];
	}
	return cumsum;
}


vector <vector <double>>  predicted(int start, int end, linear_fit_struct &lfit) {
	vector<vector<double>> predicted_y(num_timesteps, vector<double>(num_parameters));
	
	for (int i = 0; i < num_parameters; i++) {
		cout << "Calling regressionline on paramater: " << i << "\n";
		vector<double> line = regressionline(start, end, i, lfit);
		for (int j = start; j < end; j++) {
			predicted_y[j][i] = line[j - start];
		}
	}
	return predicted_y;
}
double cost_function(int start, int end) {
	
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
		cout << "Final Cost for " << start << " " << end << " is " << final_cost << endl << endl;
	}
	else {
		cout << "Final Cost for " << end << " " << start << " is " << final_cost << endl << endl;
	}
	return final_cost;
}


vector<vector<double>> initialize_cost_matrix(vector<vector<double>>& datum) {  //initialize and return the cost matrix
	

	cost_matrix.resize(num_timesteps, vector<double>(num_timesteps, 0.0));// only fill out half the matrix
	for (int i = 0; i < num_timesteps; i++) {
		for (int j = 0; j < num_timesteps; j++) {

			cout << "checking distance between " << i << " " << j<<endl; 
			cost_matrix[i][j] = cost_function(i, j); //fix to i and j
		}
	}
	return cost_matrix;

}
vector<int> admissible_bkps(int start, int end, int num_bkps) {
	vector <int> list;
	int num_samples;
	for (int i = start; i < end; i = i + 1) {
		num_samples = i - start;
		int diff = end - i;
		if (diff >= min_size){
			list.push_back(i);
		}
	}
	return list;

}

unordered_map<int, unordered_map<int, double>> memo; //think about using 2d vector/array here//1d vector
//top down recursive implementation
double seg(int start, int end, int num_bkps) {
	//recurrence to find the optimal partition
	double cost;
	vector<int> goodbkps;
	if (num_bkps == 0) {
		cost = cost_matrix[start][end];
		return cost;
	}
	if (memo.count(start) && memo[start].count(end)) {
		return memo[start][end];
	}
	else {
		vector<int> goodbkps = admissible_bkps(start, end, num_bkps);

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
vector<int> return_breakpoints() {
	vector<int> opt_bkps;
	opt_bkps.reserve(num_bkps);
	
	int little_k = num_bkps;
	
	int start = 0;
	while (little_k > 0) {
		int optimal_bkp = start;
		int end = num_timesteps - little_k;
		double min_cost = std::numeric_limits<double>::infinity();
		for (int i = start; i <= end; i += 1) {
			double left_partition = seg(start, i, num_bkps - 1);
			double right_partition = seg(i, end , 0);
			double total_partition = left_partition + right_partition;
			if (total_partition < min_cost) {
				min_cost = total_partition;
				optimal_bkp = i;
				cout << optimal_bkp << " i: " << i << " end: " << end << endl;
			}
		}
		opt_bkps.push_back(optimal_bkp);
		start = optimal_bkp;
		little_k--;
	}
	opt_bkps.push_back(num_timesteps);
//	std::reverse(opt_bkps.begin(), opt_bkps.end());
	return opt_bkps;
}
vector<int> bottomup_bkps() {

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
	read_input();
	cout << "Validating input: \n";
	for (int i = 0; i < num_timesteps; i++) {
		for (int j = 0; j < num_parameters; j++) {
			cout << datum[i][j]<< " ";
		}
		cout << endl;
	}
	initialize_cost_matrix(datum);
	cout << "Validating cost matrix: \n";
	for (int i = 1; i <= num_timesteps; i++) {
		cout << setw(12) << i << " ";
	}
	cout << endl;
	for (int i = 0; i < num_timesteps; i++) {
		cout << i + 1 << " ";
		for (int j = 0; j < num_timesteps; j++) {
			cout << setw(12)<< setprecision(8)<< cost_matrix[i][j] << "|";
		}
		cout << endl;
	}
//test top down
	auto breakponts = return_breakpoints();
	cout << "top down results: ";
	for (int i = 0; i <= num_bkps; i++) {
		cout << breakponts[i] << " "; 
	}
	
	//initialize_cost_matrix(datum);

//test bottom up
	/*auto bottomupbkp = bottomup_bkps();
	for (int i = 0; i < num_bkps; i++) {
		cout << bottomupbkp[i] << " "; */
	//}

}
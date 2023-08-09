import numpy as np
import _dupin
import dupin
from dupin.detect.costs import CostLinearFit


def compute_python_cost_matrix(signal: np.ndarray, min_size: int = 3):
    cost = CostLinearFit(metric = "l2")
    cost.fit(signal)

    n_samples = signal.shape[0]
    pycost_matrix = np.zeros((n_samples, n_samples))
    for start in range(n_samples):
        for end in range(start + min_size, n_samples):  
            pycost_matrix[start, end] = cost.error(start, end)  
    return pycost_matrix


def compute_cplus_cost_matrix(signal: np.ndarray):
    algo = _dupin.DupinAlgo()
    algo.num_bkps = 2; 
    algo.num_timesteps = signal.shape[0]
    algo.num_parameters = signal.shape[1]
    algo.data = signal.tolist()
    algo.initialize_cost_matrix(algo.data)
    return np.array(algo.cost_matrix)


def generate_data():
    return np.repeat([0, 4, 9, 20], 4).reshape((-1, 1))


def test_dupin():
    data = generate_data()

    cppcost_matrix = compute_cplus_cost_matrix(data)
    pycost_matrix = compute_python_cost_matrix(data)

    diff = np.abs(cppcost_matrix - pycost_matrix)
    print("Array difference\n", diff)
    printdiff = np.isclose(cppcost_matrix, pycost_matrix)
    print("Where is the array different\n", printdiff)
    total_diff = np.sum(diff)
    print(f"Total absolute difference between matrices: {total_diff}")
    if total_diff < 1e-1:  # 
        print("Matrices are almost identical!")
    else:
        print("Matrices differ!")
    

if __name__ == "__main__":
    test_dupin()

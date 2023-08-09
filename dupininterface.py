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
        for end in range(start + min_size, n_samples + 1):  
            pycost_matrix[start, end - 1] = cost.error(start, end)  
    return pycost_matrix

def compute_cplus_cost_matrix(signal: np.ndarray):
    algo = _dupin.DupinAlgo()
    algo.num_bkps = 3; 
    algo.num_timesteps = len(signal)
    algo.data = signal.reshape((-1,1)).tolist()
    algo.num_parameters = len(algo.data[0])
    algo.initialize_cost_matrix(algo.data)
    cppcost_matrix = np.array(algo.cost_matrix)
    return cppcost_matrix
def generate_data():
   
    segment1 = np.ones(4) * 1
    segment2 = np.ones(4) * 5
    segment3 = np.ones(4) * 10
    segment4 = np.ones(4) * 20
    
    data = np.concatenate([segment1, segment2, segment3, segment4])
    return data

def test_dupin():
    data = generate_data()

    cppcost_matrix = compute_cplus_cost_matrix(data)
    pycost_matrix = compute_python_cost_matrix(data)
    print("C++ Cost Matrix:")
    print(cppcost_matrix)
    print("\nPython Cost Matrix:")
    print(pycost_matrix)

    diff = np.abs(cppcost_matrix - pycost_matrix)
    printdiff = np.isclose(cppcost_matrix, pycost_matrix)
    print(printdiff)
    total_diff = np.sum(diff)
    print(f"Total absolute difference between matrices: {total_diff}")

    if total_diff < 1e-16:  # 
        print("Matrices are almost identical!")
    else:
        print("Matrices differ!")
    
if __name__ == "__main__":
    test_dupin()
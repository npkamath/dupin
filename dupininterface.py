import numpy as np
import _dupin
import dupin as du
import ruptures as rpt
import time
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
    algo.num_bkps = 1; 
    algo.num_timesteps = signal.shape[0]
    algo.num_parameters = signal.shape[1]
    algo.datum = signal.tolist()
    start1 = time.time()
    algo.initialize_cost_matrix(algo.datum)
 
    
    
    topcppbkps = algo.getTopDownBreakpoints()
    end1 = time.time()
    
    cpptime = end1-start1
    print("cpptime: ", cpptime)
    return np.array(algo.cost_matrix), topcppbkps



def generate_data():
    data = (np.repeat([0, 8, 16, 24, 100], 240) + np.random.random(1200)).reshape((-1, 3))
    return data



def test_dupin():
    data = generate_data()
    print(f"{data=}") 
    
    cppcost_matrix, topcppbkps = compute_cplus_cost_matrix(data)
    
    
    pycost_matrix = compute_python_cost_matrix(data)
    print ("diff: ", pycost_matrix - cppcost_matrix)
    

    printdiff = np.allclose(cppcost_matrix, pycost_matrix, atol=1e-1, rtol= 1e-2)

    print("Are the arrays the same?: ", printdiff) 
    
    start2 = time.time()
    dynp = rpt.Dynp(custom_cost = du.detect.CostLinearFit("l2"), jump=1)
    sweepsweep = du.detect.SweepDetector(dynp, 10)
    pybkps = sweepsweep.fit(data)
    end2 = time.time()
    pythontime= end2-start2
    print("pythontime: ", pythontime)
    print(f"python breakpoints: {sweepsweep.change_points_} opt change points: {sweepsweep.opt_change_points_}")
    print("Top Down C++ breakpoints:", topcppbkps)

if __name__ == "__main__":
    test_dupin()

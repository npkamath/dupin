import numpy as np
import _dupin
import random
import dupin as du
import ruptures as rpt
import time
from dupin.detect.costs import CostLinearFit
from guppy import hpy




random.seed(445)
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
    algo.num_bkps = 4; 
    algo.num_timesteps = signal.shape[0]
    algo.num_parameters = signal.shape[1]
    algo.datum = np.asfortranarray(signal) 
    start1 = time.time()
    algo.initialize_cost_matrix()
    
    

    topcppbkps = algo.getTopDownBreakpoints()
    end1 = time.time()
  
    
    cpptime = end1-start1
    print("cpptime: ", cpptime)
    return np.array(algo.cost_matrix), topcppbkps, cpptime



def generate_data():
    soize = 1080
    data = (np.repeat([0, -200, -160, -201, -152, 200], soize) + np.random.random(soize*6)).reshape((-1, 6))
    newdata = (np.repeat([0, 200, 0, 0], 180) + np.random.random(180*4)).reshape((-1,6))
    data = np.concatenate((data, newdata), axis=0)
    return data

def compute_cpp_operations(data):
    cppcost_matrix, topcppbkps, cpptime = compute_cplus_cost_matrix(data)
    return cppcost_matrix, topcppbkps, cpptime

def compute_python_operations(data):
    start_time = time.time()
    dynp = rpt.Dynp(custom_cost=du.detect.CostLinearFit("l2"), jump=1)
    sweepsweep = du.detect.SweepDetector(dynp, 10)
    sweepsweep.fit(data)
    end_time = time.time()
    pythontime = end_time - start_time
    return sweepsweep.change_points_, sweepsweep.opt_change_points_, pythontime

def test_dupin():
    data = generate_data()

    # C++ Operations
    cppcost_matrix, topcppbkps, cpptime = compute_cpp_operations(data)

    # Python Operations
    python_bkps, opt_python_bkps, pythontime = compute_python_operations(data)

    # Comparing and printing results
    multiplier = pythontime / cpptime
    print("C++ is", multiplier, "times faster!")
    print(f"Python breakpoints: {python_bkps} opt change points: {opt_python_bkps}")
    print("Top Down C++ breakpoints:", topcppbkps)

if __name__ == "__main__":
    test_dupin()

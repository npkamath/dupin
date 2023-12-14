import numpy as np
import _dupin
import random
import dupin as du
import ruptures as rpt
import time
from dupin.detect.costs import CostLinearFit
from guppy import hpy

h = hpy()


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
    algo.num_bkps = 5; 
    algo.num_timesteps = signal.shape[0]
    algo.num_parameters = signal.shape[1]
    algo.datum = np.asfortranarray(signal) 
    start1 = time.time()
    algo.initialize_cost_matrix(algo.datum)
    
    

    topcppbkps = algo.getTopDownBreakpoints()
    end1 = time.time()
  
    
    cpptime = end1-start1
    print("cpptime: ", cpptime)
    return np.array(algo.cost_matrix), topcppbkps, cpptime



def generate_data():
    soize = 200
    data = (np.repeat([0, -200, -160, -201, -152, 200], soize) + np.random.random(soize*6)).reshape((-1, 3))
    newdata = (np.repeat([0, 200, 0, 0], 180) + np.random.random(180*4)).reshape((-1,3))
    data = np.concatenate((data, newdata), axis=0)
    return data



def test_dupin():
    data = generate_data()
    print(f"{data=}") 
    
    cppcost_matrix, topcppbkps, cpptime = compute_cplus_cost_matrix(data)
    
    
    pycost_matrix = compute_python_cost_matrix(data)
    print ("diff: ", pycost_matrix - cppcost_matrix)
    

    printdiff = np.allclose(cppcost_matrix, pycost_matrix, atol=1e-1, rtol= 1e-2)

    print("Are the arrays the same?: ", printdiff) 
    
    start2 = time.time()
    dynp = rpt.Dynp(custom_cost = du.detect.CostLinearFit("l2"), jump=1)
    sweepsweep = du.detect.SweepDetector(dynp, 10)
    sweepsweep.fit(data)
    end2 = time.time()
    pythontime= end2-start2
    print("pythontime: ", pythontime)
    multiplier = pythontime/cpptime
    print("C++ is ", multiplier, " times faster!")
    print(f"python breakpoints: {sweepsweep.change_points_} opt change points: {sweepsweep.opt_change_points_}")
    print("Top Down C++ breakpoints:", topcppbkps)
    print (h.heap())

if __name__ == "__main__":
    test_dupin()

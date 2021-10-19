import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

#%% logistic model
def logistic(x, start, K, x_peak, r):
    """
    Logistic model
    
    This function runs a logistic model.
    
    Args:
        x (array_like): The control variable as a sequence of numeric values \
        in a list or a numpy array.
        start (float): The initial value of the return variable.
        K (float): The carrying capacity.
        x_peak (float): The x-value with the steepest growth.
        r (float): The growth rate.
        
    Returns:
        array_like: A numpy array or a single floating-point number with \
        the return variable.
    """
    
    if isinstance(x, list):
        x = np.array(x)
    return start + K / (1 + np.exp(r * (x_peak-x)))

def calibration(x, y):
    """
    Calibration
    
    This function calibrates a logistic model.
    The logistic model can have a positive or negative growth.
    
    Args:
        x (array_like): The explanatory variable as a sequence of numeric values \
        in a list or a numpy array.
        y (array_like): The response variable as a sequence of numeric values \
        in a list or a numpy array.
        
    Returns:
        tuple: A tuple including four values: 1) the initial value (start), \
        2) the carrying capacity (K), 3) the x-value with the steepest growth \
        (x_peak), and 4) the growth rate (r).
    """
    if isinstance(x, pd.Series): x = x.to_numpy(dtype='int')
    if isinstance(y, pd.Series): y = y.to_numpy(dtype='float')
    
    if len(np.unique(y)) == 1:
        return y[0], 0, 2000.0, 0
    
    # initial parameter guesses
    slope = [None] * (len(x) - 1)
    for i in range(len(slope)):
        slope[i] = (y[i+1] - y[i]) / (x[i+1] - x[i])
        slope[i] = abs(slope[i])
    x_peak = x[slope.index(max(slope))] + 0.5
    
    if y[0] < y[-1]: # positive growth
        start = min(y)
        K = 2 * (sum([y[slope.index(max(slope))], \
                        y[slope.index(max(slope))+1]])/2 - start)
    else: # negative growth
        K = 2 * (max(y) - sum([y[slope.index(max(slope))], \
                        y[slope.index(max(slope))+1]])/2)
        start = max(y) - K
        if start < 0 :
            start = 0
        else: 
            pass
        # implement a check to make sure that the start value is non-negative
    # curve fitting
    popt, _ = curve_fit(logistic, x, y, p0 = [start, K, x_peak, 0], maxfev = 100000,
                        bounds = ([0.5*start, 0.5*K, 1995, -10],
                                  [2*(start+0.001), 2*K, 2030, 10]))
    # +0.001 so that upper bound always larger than lower bound even if start=0
    return popt

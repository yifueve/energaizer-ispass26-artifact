import os
import sys

import numpy as np
import pandas as pd

import re
import math
import copy
import yaml

import warnings
warnings.filterwarnings('ignore')

import cvxpy

def optimize(z, y, const=True):
    N, k = z.shape
    assert y.shape == (N,), "y must have shape (N,)"

    # Normalize for numerical stability issues
    y_max = float(y.max())
    z = z.astype(np.float64) / y_max
    y = y.astype(np.float64) / y_max
    
    # Define the optimization variables
    # beta has k elements (one for each feature)
    beta = cvxpy.Variable(k)
    # beta0 is the constant term (intercept)
    if const:
        beta0 = cvxpy.Variable()
    
    # Formulate the objective function
    # We want to minimize ||z @ beta + beta0 - y||^2
    # print(z.shape, beta.shape)
    if const:
        residuals = z @ beta + beta0 * np.ones(N) - y
    else:
        residuals = z @ beta - y
    objective = cvxpy.Minimize(cvxpy.sum_squares(residuals))
    
    # Add non-negativity constraints for all coefficients
    if const:
        constraints = [
            beta >= 0,     # All feature coefficients must be non-negative
            beta0 >= 0     # Constant term must be non-negative
        ]
    else:
        constraints = [
            beta >= 0     # All feature coefficients must be non-negative
        ]

    problem = cvxpy.Problem(objective, constraints)
    # result = problem.solve(verbose=False)
    
    try:
        problem = cvxpy.Problem(objective, constraints)
        result = problem.solve(verbose=False)
    except Exception as e:
        problem = cvxpy.Problem(objective)
        problem.solve()
    
    return np.concatenate([beta.value, [beta0.value * y_max]]) if const else beta.value
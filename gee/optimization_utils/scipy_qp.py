import numpy as np
from scipy.optimize import minimize

def optimize(z, y, const=True):
    N, k = z.shape
    assert y.shape == (N,), "y must have shape (N,)"
    
    # Initial guess for coefficients (all zeros)
    if const:
        x0 = np.zeros(k + 1)  # k features + intercept
    else:
        x0 = np.zeros(k)  # k features only
    
    # Define the objective function in QP form
    # min 0.5 * x^T H x + f^T x
    def objective(x):
        if const:
            beta = x[:k]
            beta0 = x[k]
            residuals = z @ beta + beta0 * np.ones(N) - y
        else:
            beta = x
            residuals = z @ beta - y
        return np.sum(residuals**2)
    
    # We can also compute the gradient for better performance
    def gradient(x):
        if const:
            beta = x[:k]
            beta0 = x[k]
            residuals = z @ beta + beta0 - y
            grad_beta = 2 * z.T @ residuals
            grad_beta0 = 2 * np.sum(residuals)
            return np.concatenate([grad_beta, [grad_beta0]])
        else:
            residuals = z @ x - y
            return 2 * z.T @ residuals
    
    # Define constraints for non-negativity
    bounds = [(0, None) for _ in range(len(x0))]  # All variables >= 0
    
    # Solve the optimization problem using SLSQP which handles QP well
    # try:
    result = minimize(
        objective,
        x0,
        method='SLSQP',  # Better for QP problems
        jac=gradient,    # Provide gradient for better performance
        bounds=bounds,
        options={'disp': False}
    )
    
    # if result.success:
    return result.x
            
    # except Exception as e:
    #     # If there's an exception, try unconstrained optimization
    #     result = minimize(objective, x0, method='SLSQP', jac=gradient, options={'disp': False})
    #     return result.x
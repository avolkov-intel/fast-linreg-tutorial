import numpy as np
import scipy
from time import time
from sklearn.linear_model import LinearRegression


def generate_data(n, p, seed=None):
    """
    Generate random data of shape (n, p).
    """
    if (seed):
        np.random.seed(seed)
    X = np.random.normal(0, 5, (n, p)).astype(np.float64)
    w = np.random.normal(0, 2, (p,)).astype(np.float64)
    y = X @ w + np.random.normal(0, 1, (n,)).astype(np.float64)
    return X, y

def benchmark(func, args, n_iter=10):
    """
    Benchmarking custom function
    """
    times = []
    for i in range(n_iter):
        start = time()
        res = func(*args)
        end = time()
        times.append(end - start)
    return round(sum(times) / n_iter, 6), res

def compute_xtx_xty_baseline(X, y):
    """
    Implementation of XtX, Xty kernels with nested loops, attendees will be asked to implement it.
    """
    A = np.zeros((X.shape[1], X.shape[1]), dtype=X.dtype)
    b = np.zeros((X.shape[1],), dtype=X.dtype)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            b[j] += X[i, j] * y[i]
            for k in range(X.shape[1]):
                A[j, k] += X[i, j] * X[i, k]
    return A, b

def compute_xtx_xty_numpy(X, y):
    """
    Implementation of XtX, Xty kernels using numpy, attendees will be asked to implement it.
    """
    A = X.T @ X
    b = X.T @ y
    return A, b

def linear_regression_gth(X, y):
    """
    Getting predictions using Linear Regression from scikit-learn
    """
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    return model.coef_

def linear_regression(compute_xtx_xty, X, y, use_openblas=None):
    """
    Training the own implementation of Linear Regression using XtX, Xty kernels and solver from scipy.
    Attendees will be asked to implement it.
    """
    if (use_openblas is None):
        A, b = compute_xtx_xty(X, y)
    else:
        A, b = compute_xtx_xty(X, y, use_openblas)
    coefs = scipy.linalg.lstsq(A, b)[0]
    return coefs
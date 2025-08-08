from utils_cython import compute_xtx_xty as compute_xtx_xty_cython
from utils_pybind import compute_xtx_xty as compute_xtx_xty_pybind
from utils import compute_xtx_xty_baseline, compute_xtx_xty_numpy, generate_data, benchmark, linear_regression, linear_regression_gth
from time import time
import numpy as np

n_samples = 10 ** 5
n_features = 100

print("Generating data...")

X, y = generate_data(n_samples, n_features, seed=42)


"""
Runtime comparison of different approaches
Note that baseline can be really slow, it is recommended to decrease input dimensions before running it
"""
print("Running XtX and Xty benchmarks...")

#print(f"Baseline: \n{benchmark_syrk_gemv(compute_xtx_xty_baseline, X, y)} \n")
print(f"Numpy: \n{benchmark(compute_xtx_xty_numpy, [X, y])[0]} \n")
print(f"Cython: \n{benchmark(compute_xtx_xty_cython, [X, y])[0]} \n")
print(f"Pybind: \n{benchmark(compute_xtx_xty_pybind, [X, y, False])[0]} \n")
print(f"Cython with OpenBLAS: \n{benchmark(compute_xtx_xty_cython, [X, y, True])[0]} \n")
print(f"Pybind with OpenBLAS: \n{benchmark(compute_xtx_xty_pybind, [X, y, True])[0]} \n")

print("Running Linear regression benchmarks...")
sklearn_time, sklearn_result = benchmark(linear_regression_gth, [X, y])
print(f"Baseline: \n{sklearn_time}\n")
print("Ground truth coefficients:")
print(sklearn_result[:10])
pybind_time, pybind_result = benchmark(linear_regression, [compute_xtx_xty_pybind, X, y, True])
print(f"Pybind with OpenBLAS: \n{pybind_time}\n")
print("Coefficients:")
print(pybind_result[:10])
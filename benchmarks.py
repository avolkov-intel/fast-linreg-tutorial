from utils_cython import compute_xtx_xty as compute_xtx_xty_cython
from utils_pybind import compute_xtx_xty as compute_xtx_xty_pybind
from utils import compute_xtx_xty_baseline, compute_xtx_xty_numpy, generate_data, benchmark, linear_regression, linear_regression_gth
from time import time
import numpy as np
from joblib import cpu_count

n_samples = 10 ** 5
n_features = 100
num_physical_cores = cpu_count(only_physical_cores=True)

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
print(f"Cython with BLAS: \n{benchmark(compute_xtx_xty_cython, [X, y, True])[0]} \n")
print(f"Pybind: \n{benchmark(compute_xtx_xty_pybind, [X, y, False, False, 1])[0]} \n")
print(f"Pybind with BLAS: \n{benchmark(compute_xtx_xty_pybind, [X, y, True, False, 1])[0]} \n")
print(f"Cython with BLAS (blocked): \n{benchmark(compute_xtx_xty_cython, [X, y, True, True, num_physical_cores])[0]} \n")
print(f"Pybind with BLAS (blocked): \n{benchmark(compute_xtx_xty_pybind, [X, y, True, True, num_physical_cores])[0]} \n")


print("Running Linear regression benchmarks...")
sklearn_time, sklearn_result = benchmark(linear_regression_gth, [X, y])
print(f"Baseline: \n{sklearn_time}\n")
print("Ground truth coefficients:")
print(sklearn_result[:10])
pybind_blocked_time, pybind_blocked_result = benchmark(linear_regression, [compute_xtx_xty_pybind, X, y, True, True, num_physical_cores])
print(f"Pybind with BLAS (blocked): \n{pybind_blocked_time}\n")
print("Coefficients:")
print(pybind_blocked_result[:10])

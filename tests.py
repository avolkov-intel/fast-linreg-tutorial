from utils_cython import compute_xtx_xty as compute_xtx_xty_cython
from utils_pybind import compute_xtx_xty as compute_xtx_xty_pybind
from utils import compute_xtx_xty_baseline, compute_xtx_xty_numpy, generate_data, benchmark, linear_regression, linear_regression_gth
from time import time
import numpy as np

def check_all(X, y):
    # Compute ground truth using numpy
    xtx_gth, xty_gth = compute_xtx_xty_numpy(X, y)

    # Compute results using different implementations
    #xtx_baseline, xty_baseline = compute_xtx_xty_baseline(X, y)
    xtx_numpy, xty_numpy = compute_xtx_xty_numpy(X, y)
    xtx_cython, xty_cython = compute_xtx_xty_cython(X, y)
    xtx_pybind, xty_pybind = compute_xtx_xty_pybind(X, y, False)
    xtx_pybind_openbals, xty_pybind_openblas = compute_xtx_xty_pybind(X, y, True)

    # Compare results with ground truth
    #assert np.allclose(xtx_gth, xtx_baseline), "Baseline xtx mismatch"
    #assert np.allclose(xty_gth, xty_baseline), "Baseline xty mismatch"
    assert np.allclose(xtx_gth, xtx_numpy), "Numpy xtx mismatch"
    assert np.allclose(xty_gth, xty_numpy), "Numpy xty mismatch"
    assert np.allclose(xtx_gth, xtx_cython), "Cython xtx mismatch"
    assert np.allclose(xty_gth, xty_cython), "Cython xty mismatch"
    assert np.allclose(xtx_gth, xtx_pybind), "Pybind xtx mismatch"
    assert np.allclose(xty_gth, xty_pybind), "Pybind xty mismatch"
    assert np.allclose(xtx_gth, xtx_pybind_openbals), "Pybind OpenBLAS xtx mismatch"
    assert np.allclose(xty_gth, xty_pybind_openblas), "Pybind OpenBLAS xty mismatch"

    print("All implementations match the ground truth.")

if __name__ == "__main__":
    # Define dimensions for testing
    dimensions = [(100, 10), (1000, 50), (10000, 100)]

    for n_samples, n_features in dimensions:
        print(f"Testing with {n_samples} samples and {n_features} features...")
        X, y = generate_data(n_samples, n_features)
        check_all(X, y)
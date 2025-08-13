from utils_cython import compute_xtx_xty as compute_xtx_xty_cython
from utils_pybind import compute_xtx_xty as compute_xtx_xty_pybind
from utils import generate_data, compute_xtx_xty_numpy
import numpy as np
from joblib import cpu_count

num_physical_cores = cpu_count(only_physical_cores=True)


def test_one_func(func, args, A_gth, b_gth, name):
    """
    Testing custom function
    """
    A, b = func(*args)
    assert np.allclose(A_gth, A), f"{name} xtx mismatch"
    assert np.allclose(b_gth, b), f"{name} xty mismatch"


def check_all(X, y):
    # Compute ground truth using numpy
    A_gth, b_gth = compute_xtx_xty_numpy(X, y)

    test_one_func(compute_xtx_xty_cython, [X, y], A_gth, b_gth, "Cython")
    test_one_func(compute_xtx_xty_pybind, [X, y, False, False, 1], A_gth, b_gth, "Pybind")
    test_one_func(compute_xtx_xty_cython, [X, y, True], A_gth, b_gth, "Cython with BLAS")
    test_one_func(compute_xtx_xty_pybind, [X, y, True, False, 1], A_gth, b_gth, "Pybind with BLAS")
    test_one_func(compute_xtx_xty_cython, [X, y, True, True, num_physical_cores], A_gth, b_gth, "Cython with BLAS (blocked)")
    test_one_func(compute_xtx_xty_pybind, [X, y, True, True, num_physical_cores], A_gth, b_gth, "Pybind with BLAS (blocked)")

    print("All implementations match the ground truth.")


if __name__ == "__main__":
    # Define dimensions for testing
    dimensions = [(100, 10), (1000, 50), (10000, 100)]

    for n_samples, n_features in dimensions:
        print(f"Testing with {n_samples} samples and {n_features} features...")
        X, y = generate_data(n_samples, n_features)
        check_all(X, y)

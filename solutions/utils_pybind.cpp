#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "solutions/utils_pybind_blas.hpp"
#include "solutions/utils_pybind_blas_blocked.hpp"

#include <vector>
#include <memory>
#include <algorithm>
typedef pybind11::ssize_t ssize_t;

namespace py = pybind11;
using namespace pybind11::literals;


void compute_xtx_xty_naive(const double* X, 
                           const double* y, 
                           const int n, 
                           const int p,
                           double* A, 
                           double* b) {
    // Compute X^T X and X^T y
    for (ssize_t i = 0; i < n; ++i) {
        for (ssize_t j = 0; j < p; ++j) {
            b[j] += X[i * p + j] * y[i];
            for (ssize_t k = 0; k < p; ++k) {
                A[j * p + k] += X[i * p + j] * X[i * p + k];
            }
        }
    }
}

py::tuple compute_xtx_xty(
    py::array_t<double> X,
    py::array_t<double> y,
    bool use_blas,
    bool blocked,
    int n_threads_blocked
) {

    // Request buffers
    py::buffer_info X_buf = X.request();
    py::buffer_info y_buf = y.request();

    // Check dimensions
    if (X_buf.ndim != 2 || y_buf.ndim != 1)
        throw std::runtime_error("X must be 2D and y must be 1D");

    ssize_t n_samples = X_buf.shape[0];   // rows
    ssize_t n_features = X_buf.shape[1];  // cols

    if (y_buf.shape[0] != n_samples)
        throw std::runtime_error("Mismatch between X rows and y length");

    auto A = py::array_t<double>({n_features, n_features});
    auto b = py::array_t<double>({n_features});

    auto X_ptr = static_cast<const double*>(X_buf.ptr);
    auto y_ptr = static_cast<const double*>(y_buf.ptr);
    auto A_ptr = static_cast<double*>(A.request().ptr);
    auto b_ptr = static_cast<double*>(b.request().ptr);

    // Zero-initialize
    std::fill(A_ptr, A_ptr + n_features * n_features, 0.0);
    std::fill(b_ptr, b_ptr + n_features, 0.0);

    if (!use_blas) {
        compute_xtx_xty_naive(
            X_ptr,
            y_ptr,
            n_samples, n_features,
            A_ptr,
            b_ptr
        );
    } else if (!blocked) {
        compute_xtx_xty_blas(
            X_ptr,
            y_ptr,
            n_samples, n_features,
            A_ptr,
            b_ptr
        );
    } else {
        compute_xtx_xty_blocked(
            X_ptr,
            y_ptr,
            n_samples, n_features,
            A_ptr,
            b_ptr,
            n_threads_blocked
        );
    }

    return py::make_tuple(A, b);
}

PYBIND11_MODULE(utils_pybind, m) {
    m.def("compute_xtx_xty", &compute_xtx_xty, "Compute X^T X and X^T y using nested loops or OpenBLAS");
}

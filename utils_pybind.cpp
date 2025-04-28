#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cblas.h>

namespace py = pybind11;

/*
 Here are two implementation of XtX and Xty kernels:
    - in first result is computed using nested loops
    - in second result is computed using OpenBLAS library
 Attendees wouldn't be required to fill in much gaps here, code
 is provided to demonstrate what can be possibly done, not to learn
 how to code in C++.
*/

py::tuple compute_xtx_xty(py::array_t<double> X, py::array_t<double> y, bool use_openblas) {

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

    auto X_ptr = static_cast<double*>(X_buf.ptr);
    auto y_ptr = static_cast<double*>(y_buf.ptr);
    auto A_ptr = static_cast<double*>(A.request().ptr);
    auto b_ptr = static_cast<double*>(b.request().ptr);

    // Zero-initialize
    std::fill(A_ptr, A_ptr + n_features * n_features, 0.0);
    std::fill(b_ptr, b_ptr + n_features, 0.0);

    if (!use_openblas) {

        // Compute X^T X and X^T y
        for (ssize_t i = 0; i < n_samples; ++i) {
            for (ssize_t j = 0; j < n_features; ++j) {
                b_ptr[j] += X_ptr[i * n_features + j] * y_ptr[i];
                for (ssize_t k = 0; k < n_features; ++k) {
                    A_ptr[j * n_features + k] += X_ptr[i * n_features + j] * X_ptr[i * n_features + k];
                }
                
            }
        }

    } else {

        // Compute X^T X using cblas_dsyrk (symmetric rank-k update)
        cblas_dsyrk(
            CblasRowMajor,    // Row major
            CblasUpper,       // We'll fill upper triangle
            CblasTrans,       // Need X^T * X, so transpose X
            n_features,       // Size of output (n_features x n_features)
            n_samples,        // k
            1.0,              // alpha
            X_ptr,            // input matrix (X)
            n_features,       // leading dimension of inpuit matrix (lda)
            0.0,              // beta
            A_ptr,            // output matrix (XtX)
            n_features        // leading dimension of output matrix (ldc)
        );

        // Since only the upper part is filled by dsyrk, we copy it to the lower part manually
        for (ssize_t i = 0; i < n_features; ++i) {
            for (ssize_t j = i+1; j < n_features; ++j) {
                A_ptr[j * n_features + i] = A_ptr[i * n_features + j];
            }
        }

        // Compute X^T y using cblas_dgemv (matrix-vector multiplication)
        cblas_dgemv(
            CblasRowMajor,
            CblasTrans,       // transpose X
            n_samples,        // input matrix dimensions
            n_features,       // input matrix dimensions
            1.0,              // alpha
            X_ptr,            // input matrix (X)
            n_features,       // leading dimension of inpuit matrix (lda)
            y_ptr,            // input vector (y)
            1,                // leading dimension of input vector
            0.0,              // beta 
            b_ptr,            // output vector (b)
            1                 // leading dimension of output vector
        );

    }

    return py::make_tuple(A, b);
}

PYBIND11_MODULE(utils_pybind, m) {
    m.def("compute_xtx_xty", &compute_xtx_xty, "Compute X^T X and X^T y using nested loops or OpenBLAS");
}
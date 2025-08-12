#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cblas.h>
#include <vector>
#include <memory>
#include <algorithm>
#ifdef _OPENMP
#   include <omp.h>
#endif
#include "blas_helpers.h"
typedef pybind11::ssize_t ssize_t;

namespace py = pybind11;

/*
 Here are two implementation of XtX and Xty kernels:
    - in first result is computed using nested loops
    - in second result is computed using OpenBLAS library
 Attendees wouldn't be required to fill in much gaps here, code
 is provided to demonstrate what can be possibly done, not to learn
 how to code in C++.
*/

void compute_xtx_xty_blocked(
    const double *X,
    const double *y,
    const int n, const int p,
    int n_threads,
    double *A,
    double *b
);

py::tuple compute_xtx_xty(
    py::array_t<double> X,
    py::array_t<double> y,
    bool use_openblas,
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

#ifndef _OPENMP
    blocked = false; /* requires OpenMP */
#endif

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

    } else if (!blocked) {

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

#ifdef _OPENMP
    else {
        compute_xtx_xty_blocked(
            X_ptr,
            y_ptr,
            n_samples, n_features,
            n_threads_blocked,
            A_ptr,
            b_ptr
        );
    }
#endif

    return py::make_tuple(A, b);
}

#ifdef _OPENMP
void compute_xtx_xty_blocked(
    const double *X,
    const double *y,
    const int n, const int p,
    int n_threads,
    double *A,
    double *b
)
{
    const int block_size = 256;
    const int num_blocks = n / block_size;
    const int size_remainder = n - num_blocks * block_size;
    n_threads = std::min(n_threads, num_blocks);

    const int dim_A = p*p;
    const int dim_b = p;
    
    std::vector< std::unique_ptr<double[]> > A_thread_memory(n_threads);
    std::vector< std::unique_ptr<double[]> > b_thread_memory(n_threads);

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int block_id = 0; block_id < num_blocks; block_id++)
    {
        const int thread_id = omp_get_thread_num();
        if (!A_thread_memory[thread_id]) A_thread_memory[thread_id] = std::unique_ptr<double[]>(new double[dim_A]());
        if (!b_thread_memory[thread_id]) b_thread_memory[thread_id] = std::unique_ptr<double[]>(new double[dim_b]());

        double *A_thread = A_thread_memory[thread_id].get();
        double *b_thread = b_thread_memory[thread_id].get();

        const double *X_block = X + block_id*block_size*p;
        const double *y_block = y + block_id*block_size;

        cblas_dsyrk(
            CblasRowMajor, CblasUpper, CblasTrans,
            p, block_size,
            1.0, X_block, p,
            1.0, A_thread, p
        );
        cblas_dgemv(
            CblasRowMajor, CblasTrans,
            block_size, p,
            1.0, X_block, p,
            y_block, 1,
            1.0, b_thread, 1
        );
    }

    if (!size_remainder) {
        std::copy(
            A_thread_memory[0].get(),
            A_thread_memory[0].get() + dim_A,
            A
        );
        std::copy(
            b_thread_memory[0].get(),
            b_thread_memory[0].get() + dim_b,
            b
        );
    }

    else {
        cblas_dsyrk(
            CblasRowMajor, CblasUpper, CblasTrans,
            p, size_remainder,
            1.0, X + num_blocks*block_size*p, p,
            0.0, A, p
        );
        cblas_dgemv(
            CblasRowMajor, CblasTrans,
            size_remainder, p,
            1.0, X + num_blocks*block_size*p, p,
            y + num_blocks*block_size, 1,
            0.0, b, 1
        );
    }

    for (int thread_id = size_remainder? 0 : 1; thread_id < n_threads; thread_id++) {
        cblas_daxpy(dim_A, 1.0, A_thread_memory[thread_id].get(), 1, A, 1);
        cblas_daxpy(dim_b, 1.0, b_thread_memory[thread_id].get(), 1, b, 1);
    }

    for (int row = 1; row < p; row++) {
        for (int col = 0; col < row; col++) {
            A[col + row*p] = A[row + col*p];
        }
    }
}
#endif

PYBIND11_MODULE(utils_pybind, m) {
    m.def("compute_xtx_xty", &compute_xtx_xty, "Compute X^T X and X^T y using nested loops or OpenBLAS");
}

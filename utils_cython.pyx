# distutils: language=c++
import numpy as np
cimport numpy as np
from cython import wraparound
from libcpp cimport bool

# We need this include to build the kernel with solution for comparison
include "solutions/utils_cython_naive.pxi"

# TODO PRACTICE 3: Include the header
# Function declaration provided for reference
#cdef void compute_xtx_xty_blas(
#    double *X,
#    double *y,
#    int n,
#    int p,
#    double *A,
#    double *b,
#) noexcept nogil
# include "solutions/utils_cython_blas.pxi"

# TODO PRACTICE 4: Include the header
# Function declaration provided for reference
#cdef void compute_xtx_xty_blas_blocked(
#    double *X,
#    double *y,
#    int n,
#    int p,
#    double *A,
#    double *b,
#    int n_threads_blocked
#) noexcept nogil
# include "solutions/utils_cython_blas_blocked.pxi"


cdef void compute_xtx_xty_naive(
    const double[:, :] X,
    const double[:] y,
    const int n,
    const int p,
    double[:, :] A,
    double[:] b
):
    # TODO Practice 1: Fill A and b
    pass


def compute_xtx_xty(np.ndarray[np.float64_t, ndim=2] X,
                    np.ndarray[np.float64_t, ndim=1] y,
                    bool use_blas=False,
                    bool blocked=False,
                    int n_threads_blocked=1):
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]

    # Zero-initialize
    cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros((p, p), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] b = np.zeros(p, dtype=np.float64)

    # Note: data 'X' and 'y' are not modified, but Cython BLAS does not
    # declare these Fortran-type functions to have 'const' qualifier
    cdef double* X_ptr = &X[0, 0]
    cdef double* y_ptr = &y[0]

    cdef double* A_ptr = &A[0, 0]
    cdef double* b_ptr = &b[0]

    if not use_blas:
        compute_xtx_xty_naive(X, y, n, p, A, b)
    else:
        if not blocked:
            # TODO PRACTICE 3: call compute_xtx_xty_blas function
            pass
        else:
            # TODO PRACTICE 4: call compute_xtx_xty_blas_blocked function
            pass
    return A, b

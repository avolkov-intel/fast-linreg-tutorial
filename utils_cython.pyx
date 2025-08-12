import numpy as np
cimport numpy as np
from cython import boundscheck
from libcpp cimport bool
from scipy.linalg.cython_blas cimport dsyrk, dgemv

"""
 Here is the implementation of XtX and Xty kernels with nested loops
 Attendes will be required to fill some gaps in this implementation.
"""

cdef void compute_xtx_xty_naive(
    const double *X,
    const double *y,
    const int n,
    const int p,
    double *A,
    double *b
) noexcept nogil:
    cdef int i, j, k
    cdef double x_ij
    cdef const double *X_row_i
    cdef double *A_row_j

    # Compute X^T X and X^T y
    # Note: the auto-generated C++ file from Cython, when put under common compilers,
    # is not able to perform all the obvious loop optimizations as the manual C++ code
    # in the PyBind11 code. It needs to manually assign temporary pointers for each
    # row being reused, while similar code written in C/C++ directly would generate
    # something that compilers are better able to optimize.
    for i in range(n):
        X_row_i = X + i * p
        for j in range(p):
            b[j] += X_row_i[j] * y[i]
            x_ij = X_row_i[j]
            A_row_j = A + j * p
            for k in range(p):
                A_row_j[k] += x_ij * X_row_i[k]


cdef void compute_xtx_xty_blas(
    double *X,
    double *y,
    int n,
    int p,
    double *A,
    double *b
) noexcept nogil:
    # Compute X^T X using cblas_dsyrk (symmetric rank-k update)
    cdef double one = 1.0
    cdef double zero = 0.0
    cdef char L = 76 # ASCII for letter 'L'
    cdef char N = 78 # ASCII for letter 'N'
    dsyrk(
        &L, &N,
        &p, &n,
        &one, X, &p,
        &zero, A, &p
    )

    # Since only the upper part is filled by dsyrk, we copy it to the lower part manually
    cdef int i, j
    for i in range(p):
        for j in range(i + 1, p):
            A[i + j * p] = A[j + i * p]

    # Compute X^T y using cblas_dgemv (matrix-vector multiplication)
    cdef int one_int = 1
    dgemv(
        "N", &p, &n,
        &one, X, &p,
        y, &one_int,
        &zero, b, &one_int
    )


@boundscheck(False)
def compute_xtx_xty(np.ndarray[np.float64_t, ndim=2] X,
                    np.ndarray[np.float64_t, ndim=1] y,
                    bool use_blas=False):
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]

    # Zero-initialize
    cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros((p, p), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] b = np.zeros(p, dtype=np.float64)

    with nogil:
        if not use_blas:
            compute_xtx_xty_naive(
                &X[0, 0],
                &y[0],
                n,
                p,
                &A[0, 0],
                &b[0],
            )
        else:
            compute_xtx_xty_blas(
                &X[0, 0],
                &y[0],
                n,
                p,
                &A[0, 0],
                &b[0],
            )

    return A, b

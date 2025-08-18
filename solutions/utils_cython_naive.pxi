# distutils: language=c++
import numpy as np
cimport numpy as np
from cython import boundscheck


"""
 Here is the implementation of XtX and Xty kernels with nested loops
 Attendes will be required to fill some gaps in this implementation.
"""

@boundscheck(False)
cdef void compute_xtx_xty_naive_solution_impl(
    const double [:, :] X,
    const double [:] y,
    const int n,
    const int p,
    double[:, :] A,
    double[:] b
):

    cdef int i, j, k
    cdef double x_ij
    cdef const double *X_row_i
    cdef double *A_row_j

    cdef const double* X_ptr = &X[0, 0]
    cdef const double* y_ptr = &y[0]
    
    cdef double* A_ptr = &A[0, 0]
    cdef double* b_ptr = &b[0]

    # Compute X^T X and X^T y
    # Note: the auto-generated C++ file from Cython, when put under common compilers,
    # is not able to perform all the obvious loop optimizations as the manual C++ code
    # in the PyBind11 code. It needs to manually assign temporary pointers for each
    # row being reused, while similar code written in C/C++ directly would generate
    # something that compilers are better able to optimize.
    for i in range(n):
        X_row_i = X_ptr + i * p
        for j in range(p):
            b_ptr[j] += X_row_i[j] * y_ptr[i]
            x_ij = X_row_i[j]
            A_row_j = A_ptr + j * p
            for k in range(p):
                A_row_j[k] += x_ij * X_row_i[k]


def compute_xtx_xty_naive_solution(np.ndarray[np.float64_t, ndim=2] X,
                    np.ndarray[np.float64_t, ndim=1] y):
    
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    
    # Zero-initialize
    cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros((p, p), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] b = np.zeros(p, dtype=np.float64)

    compute_xtx_xty_naive_solution_impl(X, y, n, p, A, b)
    return A, b

# distutils: language=c++
import numpy as np
cimport numpy as np
from cython import boundscheck
include "solutions/utils_cython_naive.pxi"
include "solutions/utils_cython_blas.pxi"
include "solutions/utils_cython_blas_blocked.pxi"

"""
 Here is the implementation of XtX and Xty kernels with nested loops
 Attendes will be required to fill some gaps in this implementation.
"""


@boundscheck(False)
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
    
    if not use_blas:
        compute_xtx_xty_naive_solution_impl(X, y, n, p, A, b)
    else:
        with nogil:
            if not blocked:
                compute_xtx_xty_blas(
                    &X[0, 0],
                    &y[0],
                    n,
                    p,
                    &A[0, 0],
                    &b[0],
                )
            else:
                compute_xtx_xty_blas_blocked(
                    &X[0, 0],
                    &y[0],
                    n,
                    p,
                    &A[0, 0],
                    &b[0],
                    n_threads_blocked,
                )

    return A, b
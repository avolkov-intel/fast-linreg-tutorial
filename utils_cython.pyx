import numpy as np
cimport numpy as np

"""
 Here is the implementation of XtX and Xty kernels with nested loops
 Attendes will be required to fill some gaps in this implementation.
"""

def compute_xtx_xty(np.ndarray[np.float64_t, ndim=2] X,
                    np.ndarray[np.float64_t, ndim=1] y):
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]

    # Zero-initialize
    cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros((p, p), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] b = np.zeros(p, dtype=np.float64)

    cdef int i, j, k

    # Compute X^T X and X^T y
    for i in range(n):
        for j in range(p):
            b[j] += X[i, j] * y[i]
            for k in range(p):
                A[j, k] += X[i, j] * X[i, k]

    return A, b
# distutils: language=c++
import numpy as np
cimport numpy as np
from cython import boundscheck
from scipy.linalg.cython_blas cimport dsyrk, dgemv, daxpy

"""
Implementation using single BLAS call
"""

cdef void compute_xtx_xty_blas(
    double *X,
    double *y,
    int n,
    int p,
    double *A,
    double *b
) noexcept nogil:
    # Compute X^T X using dsyrk (symmetric rank-k update)
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
    cdef int row, col
    for row in range(1, p):
        for col in range(row):
            A[col + row*p] = A[row + col*p]

    # Compute X^T y using dgemv (matrix-vector multiplication)
    cdef int one_int = 1
    dgemv(
        "N", &p, &n,
        &one, X, &p,
        y, &one_int,
        &zero, b, &one_int
    )

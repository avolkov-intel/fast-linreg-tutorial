import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound

"""
 Here is the implementation of XtX and Xty kernels with nested loops
 Attendes will be required to fill some gaps in this implementation.
"""

@boundscheck(False) # Do not perform bound checks
@wraparound(False) # Do not allow negative indices e.g. arr[-5]
def compute_xtx_xty(double[:,:] X,
                    double[:] y):
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]

    # Zero-initialize
    cdef double[:,:] A = np.zeros((p, p), dtype=np.float64)
    cdef double[:] b = np.zeros(p, dtype=np.float64)

    cdef int i, j, k

    # TODO: Declare the pointers for matrices
    # Compute X^T X and X^T y
    for i in range(n):
        for j in range(p):
            b[j] += X[i, j] * y[i]
            for k in range(p):
                A[j, k] += X[i, j] * X[i, k]

    return A, b


from scipy.linalg.cython_blas cimport dsyrk, dgemm
@boundscheck(False) # Do not perform bound checks
@wraparound(False) # Do not allow negative indices e.g. arr[-5]
def compute_xtx_xty_cython_scipy(np.ndarray[np.float64_t, ndim=2] X,
                    np.ndarray[np.float64_t, ndim=1] y):
    """
    Compute A = XᵀX and b = Xᵀy using BLAS routines (dsyrk and dgemm).
    """
    cdef int n = X.shape[0]  # number of rows
    cdef int p = X.shape[1]  # number of columns

    # Ensure Fortran-order for BLAS compatibility
    cdef np.ndarray[np.float64_t, ndim=2, mode='fortran'] Xf = np.asfortranarray(X)
    cdef np.ndarray[np.float64_t, ndim=2, mode='fortran'] y_mat = np.asfortranarray(y.reshape((n, 1)))

    # Allocate output arrays
    cdef np.ndarray[np.float64_t, ndim=2, mode='fortran'] A = np.zeros((p, p), dtype=np.float64, order='F')
    cdef np.ndarray[np.float64_t, ndim=2, mode='fortran'] b_mat = np.zeros((p, 1), dtype=np.float64, order='F')

    # BLAS parameters
    cdef int one = 1
    cdef char uplo = 'U'
    cdef char trans = 'T'
    cdef char transa = 'T'
    cdef char transb = 'N'
    cdef double alpha = 1.0
    cdef double beta = 0.0

    # Compute A = XᵀX using dsyrk (only upper triangle filled)
    dsyrk(&uplo, &trans, &p, &n,
          &alpha, &Xf[0, 0], &n,
          &beta, &A[0, 0], &p)

    # Fill lower triangle to make A fully symmetric
    cdef int i, j
    for i in range(p):
        for j in range(i + 1, p):
            A[j, i] = A[i, j]

    # Compute b = Xᵀy using dgemm
    dgemm(&transa, &transb,
          &p, &one, &n,
          &alpha, &Xf[0, 0], &n,
                   &y_mat[0, 0], &n,
          &beta,  &b_mat[0, 0], &p)

    # Return as 1D vector
    return A, b_mat[:, 0]
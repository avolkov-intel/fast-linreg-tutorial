# distutils: language=c++
import numpy as np
cimport numpy as np
from cython import boundscheck
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.algorithm cimport copy
from cython.parallel cimport threadid, prange
from scipy.linalg.cython_blas cimport dsyrk, dgemv, daxpy
from threadpoolctl import threadpool_limits
from contextlib import nullcontext

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
    cdef int i, j
    for i in range(p):
        for j in range(i + 1, p):
            A[i + j * p] = A[j + i * p]

    # Compute X^T y using dgemv (matrix-vector multiplication)
    cdef int one_int = 1
    dgemv(
        "N", &p, &n,
        &one, X, &p,
        y, &one_int,
        &zero, b, &one_int
    )

cdef extern from *:
    """
    bool check_has_openmp()
    {
#ifdef _OPENMP
        return true;
#else
        return false;
#endif
    }
    """
    bool check_has_openmp() noexcept nogil
cdef bool printed_no_omp_msg = False

cdef void compute_xtx_xty_blas_blocked(
    double *X,
    double *y,
    int n,
    int p,
    double *A,
    double *b,
    int n_threads
) noexcept nogil:
    global printed_no_omp_msg
    cdef double one = 1.0
    cdef double zero = 0.0
    cdef char L = 76 # ASCII for letter 'L'
    cdef char N = 78 # ASCII for letter 'N'
    cdef int one_int = 1

    cdef int block_size = 256
    cdef int num_blocks = <int>(n / block_size)
    cdef int size_remainder = n - num_blocks * block_size
    if num_blocks < n_threads:
        n_threads = num_blocks
    if not check_has_openmp():
        n_threads = 1
        if not printed_no_omp_msg:
            printed_no_omp_msg = True
            with gil:
                print("\tNote: blocked version running single-threaded (no OpenMP)")

    cdef int dim_A = p*p
    cdef int dim_b = p

    # Note: the vectors inside allocated inside the parallel loop by
    # the thread that uses them, in order to preserve numa locality.
    cdef vector[ vector[double] ] A_thread_memory
    cdef vector[ vector[double] ] b_thread_memory
    A_thread_memory.resize(n_threads)
    b_thread_memory.resize(n_threads)

    cdef int block_id

    with gil:
        blas_lim_ctx = (
            threadpool_limits(limits="sequential_blas_under_openmp")
            if n_threads > 1 else
            nullcontext()
        )
        with blas_lim_ctx:
            for block_id in prange(num_blocks, nogil=True, schedule="static", num_threads=n_threads):

                if A_thread_memory[threadid()].empty():
                    A_thread_memory[threadid()].resize(dim_A)
                if b_thread_memory[threadid()].empty():
                    b_thread_memory[threadid()].resize(dim_b)

                dsyrk(
                    &L, &N,
                    &p, &block_size,
                    &one, X + block_id*block_size*p, &p,
                    &one, A_thread_memory[threadid()].data(), &p
                )
                dgemv(
                    &N, &p, &block_size,
                    &one, X + block_id*block_size*p, &p,
                    y + block_id*block_size, &one_int,
                    &one, b_thread_memory[threadid()].data(), &one_int
                )

    if not size_remainder:
        copy(
            A_thread_memory[0].data(),
            A_thread_memory[0].data() + dim_A,
            A
        )
        copy(
            b_thread_memory[0].data(),
            b_thread_memory[0].data() + dim_b,
            b
        )
    else:
        dsyrk(
            &L, &N,
            &p, &size_remainder,
            &one, X + num_blocks*block_size*p, &p,
            &zero, A, &p
        )
        dgemv(
            &N, &p, &size_remainder,
            &one, X + num_blocks*block_size*p, &p,
            y + num_blocks*block_size, &one_int,
            &zero, b, &one_int
        )

    cdef int thread_id
    cdef int thread_id_start = 0 if size_remainder else 1
    for thread_id in range(thread_id_start, n_threads):
        daxpy(&dim_A, &one, A_thread_memory[thread_id].data(), &one_int, A, &one_int)
        daxpy(&dim_b, &one, b_thread_memory[thread_id].data(), &one_int, b, &one_int)

    cdef int row, col
    for row in range(1, p):
        for col in range(row):
            A[col + row*p] = A[row + col*p]


@boundscheck(False)
def compute_xtx_xty(np.ndarray[np.float64_t, ndim=2] X,
                    np.ndarray[np.float64_t, ndim=1] y,
                    bool use_blas=False,
                    bool blocked=False,
                    int n_threads=1):
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
                    n_threads,
                )

    return A, b

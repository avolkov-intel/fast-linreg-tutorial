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

# Helper to symmetrize an array representing a square symmetric
# matrix when only the upper triangular part has been filled,
# as done by many BLAS functions
cdef void symmetrize_matrix(double *A, const int n) noexcept nogil:
    cdef int row, col
    for row in range(1, n):
        for col in range(row):
            A[col + row*n] = A[row + col*n]

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


"""
Multithreaded implementation with multiple calls for BLAS
"""

cdef int compute_xtx_xty_blas_blocked(
    double *X,
    double *y,
    int n,
    int p,
    double *A,
    double *b,
    int n_threads_blocked
) except + nogil:
    global printed_no_omp_msg
    cdef double one = 1.0
    cdef double zero = 0.0
    cdef char L = 76 # ASCII for letter 'L'
    cdef char N = 78 # ASCII for letter 'N'
    cdef int one_int = 1

    cdef int block_size = 256
    cdef int num_blocks = <int>(n / block_size)
    cdef int size_remainder = n - num_blocks * block_size
    if num_blocks < n_threads_blocked:
        n_threads_blocked = num_blocks
    if not check_has_openmp():
        n_threads_blocked = 1
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
    A_thread_memory.resize(n_threads_blocked)
    b_thread_memory.resize(n_threads_blocked)

    cdef int block_id

    with gil:
        blas_lim_ctx = (
            threadpool_limits(limits="sequential_blas_under_openmp")
            if n_threads_blocked > 1 else
            nullcontext()
        )
        with blas_lim_ctx:
            for block_id in prange(num_blocks, nogil=True, schedule="static", num_threads=n_threads_blocked):

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
    for thread_id in range(thread_id_start, n_threads_blocked):
        daxpy(&dim_A, &one, A_thread_memory[thread_id].data(), &one_int, A, &one_int)
        daxpy(&dim_b, &one, b_thread_memory[thread_id].data(), &one_int, b, &one_int)

    symmetrize_matrix(A, p)

    return 0

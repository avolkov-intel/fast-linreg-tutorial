#include "solutions/utils_pybind_blas.hpp"
#ifdef _OPENMP
#   include <omp.h>
#else
#   define omp_get_thread_num() 0
#endif

namespace py = pybind11;
using namespace pybind11::literals;


/*
 Here is the implementation of XtX and Xty kernels 
 which are computed using OpenBLAS and OpenMP.
*/

bool printed_no_omp_msg = false;
void compute_xtx_xty_blocked(
    const double *X,
    const double *y,
    const int n, const int p,
    double *A,
    double *b,
    int n_threads_blocked);

void compute_xtx_xty_blocked(
    const double *X,
    const double *y,
    const int n, const int p,
    double *A,
    double *b,
    int n_threads_blocked)
{
#ifndef _OPENMP
    n_threads_blocked = 1;
    if (!printed_no_omp_msg) {
        PyGILState_STATE gil_state = PyGILState_Ensure();
        PySys_WriteStdout("\tNote: blocked version running single-threaded (no OpenMP)\n");
        PyGILState_Release(gil_state);
        printed_no_omp_msg = true;
    }
#endif

    const int block_size = 256;
    const int num_blocks = n / block_size;
    const int size_remainder = n - num_blocks * block_size;
    n_threads_blocked = std::min(n_threads_blocked, num_blocks);

    const int dim_A = p*p;
    const int dim_b = p;
    
    std::vector< std::unique_ptr<double[]> > A_thread_memory(n_threads_blocked);
    std::vector< std::unique_ptr<double[]> > b_thread_memory(n_threads_blocked);

    const double one = 1.0;
    const double zero = 0.0;
    const int one_int = 1;

    py::object thread_limit_ctx = (n_threads_blocked > 1)?
        py::module_::import("threadpoolctl").attr("threadpool_limits")("limits"_a="sequential_blas_under_openmp")
        :
        py::module_::import("contextlib").attr("nullcontext")()
    ;
    thread_limit_ctx.attr("__enter__")();

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_blocked)
#endif
    for (int block_id = 0; block_id < num_blocks; block_id++)
    {
        const int thread_id = omp_get_thread_num();
        if (!A_thread_memory[thread_id]) A_thread_memory[thread_id] = std::unique_ptr<double[]>(new double[dim_A]());
        if (!b_thread_memory[thread_id]) b_thread_memory[thread_id] = std::unique_ptr<double[]>(new double[dim_b]());

        double *A_thread = A_thread_memory[thread_id].get();
        double *b_thread = b_thread_memory[thread_id].get();

        const double *X_block = X + block_id*block_size*p;
        const double *y_block = y + block_id*block_size;

        dsyrk_(
            "L", "N",
            &p, &block_size,
            &one, X_block, &p,
            &one, A_thread, &p
        );
        dgemv_(
            "N", &p, &block_size,
            &one, X_block, &p,
            y_block, &one_int,
            &one, b_thread, &one_int
        );
    }

    thread_limit_ctx.attr("__exit__")(py::none(), py::none(), py::none());

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
        dsyrk_(
            "L", "N",
            &p, &size_remainder,
            &one, X + num_blocks*block_size*p, &p,
            &zero, A, &p
        );
        dgemv_(
            "N", &p, &size_remainder,
            &one, X + num_blocks*block_size*p, &p,
            y + num_blocks*block_size, &one_int,
            &zero, b, &one_int
        );
    }

    for (int thread_id = size_remainder? 0 : 1; thread_id < n_threads_blocked; thread_id++) {
        daxpy_(&dim_A, &one, A_thread_memory[thread_id].get(), &one_int, A, &one_int);
        daxpy_(&dim_b, &one, b_thread_memory[thread_id].get(), &one_int, b, &one_int);
    }

    symmetrize_matrix(A, p);
}

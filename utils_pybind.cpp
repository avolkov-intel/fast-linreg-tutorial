#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <memory>
#include <algorithm>
#ifdef _OPENMP
#   include <omp.h>
#else
#   define omp_get_thread_num() 0
#endif
typedef pybind11::ssize_t ssize_t;

namespace py = pybind11;
using namespace pybind11::literals;

/*
 Workaround for loading required BLAS functions from SciPy.
 This is not required if linking directly to a BLAS library.
*/

extern "C" {
typedef void (*dsyrk_t)(const char*, const char*, const int*, const int*, const double*, const double*, const int*, const double*, double*, const int*);
typedef void (*dgemv_t)(const char*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*);
typedef void (*daxpy_t)(const int*, const double*, const double*, const int*, double*, const int*);
dsyrk_t dsyrk_;
dgemv_t dgemv_;
daxpy_t daxpy_;
} /* extern "C" */

int load_blas_funs()
{
    PyGILState_STATE gil_state = PyGILState_Ensure();
    PyObject *cython_blas_module = PyImport_ImportModule("scipy.linalg.cython_blas");
    PyObject *pyx_capi_obj = PyObject_GetAttrString(cython_blas_module, "__pyx_capi__");
    PyObject *cobj_dsyrk = PyDict_GetItemString(pyx_capi_obj, "dsyrk");
    PyObject *cobj_dgemv = PyDict_GetItemString(pyx_capi_obj, "dgemv");
    PyObject *cobj_daxpy = PyDict_GetItemString(pyx_capi_obj, "daxpy");
    dsyrk_ = reinterpret_cast<dsyrk_t>(PyCapsule_GetPointer(cobj_dsyrk, PyCapsule_GetName(cobj_dsyrk)));
    dgemv_ = reinterpret_cast<dgemv_t>(PyCapsule_GetPointer(cobj_dgemv, PyCapsule_GetName(cobj_dgemv)));
    daxpy_ = reinterpret_cast<daxpy_t>(PyCapsule_GetPointer(cobj_daxpy, PyCapsule_GetName(cobj_daxpy)));
    Py_DECREF(cython_blas_module);
    PyGILState_Release(gil_state);
    return 0;
}

const int dummy = load_blas_funs();

/*
 Here are two implementation of XtX and Xty kernels:
    - in first result is computed using nested loops
    - in second result is computed using OpenBLAS library
 Attendees wouldn't be required to fill in much gaps here, code
 is provided to demonstrate what can be possibly done, not to learn
 how to code in C++.
*/

bool printed_no_omp_msg = false;
void compute_xtx_xty_blocked(
    const double *X,
    const double *y,
    const int n, const int p,
    int n_threads,
    double *A,
    double *b
);

py::tuple compute_xtx_xty(
    py::array_t<double> X,
    py::array_t<double> y,
    bool use_blas,
    bool blocked,
    int n_threads_blocked
) {

    // Request buffers
    py::buffer_info X_buf = X.request();
    py::buffer_info y_buf = y.request();

    // Check dimensions
    if (X_buf.ndim != 2 || y_buf.ndim != 1)
        throw std::runtime_error("X must be 2D and y must be 1D");

    ssize_t n_samples = X_buf.shape[0];   // rows
    ssize_t n_features = X_buf.shape[1];  // cols

    if (y_buf.shape[0] != n_samples)
        throw std::runtime_error("Mismatch between X rows and y length");

    auto A = py::array_t<double>({n_features, n_features});
    auto b = py::array_t<double>({n_features});

    auto X_ptr = static_cast<const double*>(X_buf.ptr);
    auto y_ptr = static_cast<const double*>(y_buf.ptr);
    auto A_ptr = static_cast<double*>(A.request().ptr);
    auto b_ptr = static_cast<double*>(b.request().ptr);

    // Zero-initialize
    std::fill(A_ptr, A_ptr + n_features * n_features, 0.0);
    std::fill(b_ptr, b_ptr + n_features, 0.0);

    if (!use_blas) {

        // Compute X^T X and X^T y
        for (ssize_t i = 0; i < n_samples; ++i) {
            for (ssize_t j = 0; j < n_features; ++j) {
                b_ptr[j] += X_ptr[i * n_features + j] * y_ptr[i];
                for (ssize_t k = 0; k < n_features; ++k) {
                    A_ptr[j * n_features + k] += X_ptr[i * n_features + j] * X_ptr[i * n_features + k];
                }
                
            }
        }

    } else if (!blocked) {

        const int p_ = n_features;
        const int n_ = n_samples;
        const double one = 1.0;
        const double zero = 0.0;
        dsyrk_(
            "L", "N",
            &p_, &n_,
            &one, X_ptr, &p_,
            &zero, A_ptr, &p_
        );
        const int one_int = 1;
        dgemv_(
            "N", &p_, &n_,
            &one, X_ptr, &p_,
            y_ptr, &one_int,
            &zero, b_ptr, &one_int
        );

        // Since only the upper part is filled by dsyrk, we copy it to the lower part manually
        for (ssize_t i = 0; i < n_features; ++i) {
            for (ssize_t j = i+1; j < n_features; ++j) {
                A_ptr[j * n_features + i] = A_ptr[i * n_features + j];
            }
        }

    }

    else {

        compute_xtx_xty_blocked(
            X_ptr,
            y_ptr,
            n_samples, n_features,
            n_threads_blocked,
            A_ptr,
            b_ptr
        );
    }

    return py::make_tuple(A, b);
}

void compute_xtx_xty_blocked(
    const double *X,
    const double *y,
    const int n, const int p,
    int n_threads,
    double *A,
    double *b
)
{
#ifndef _OPENMP
    n_threads = 1;
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
    n_threads = std::min(n_threads, num_blocks);

    const int dim_A = p*p;
    const int dim_b = p;
    
    std::vector< std::unique_ptr<double[]> > A_thread_memory(n_threads);
    std::vector< std::unique_ptr<double[]> > b_thread_memory(n_threads);

    const double one = 1.0;
    const double zero = 0.0;
    const int one_int = 1;

    py::object thread_limit_ctx = (n_threads > 1)?
        py::module_::import("threadpoolctl").attr("threadpool_limits")("limits"_a="sequential_blas_under_openmp")
        :
        py::module_::import("contextlib").attr("nullcontext")()
    ;
    thread_limit_ctx.attr("__enter__")();

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads)
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

    for (int thread_id = size_remainder? 0 : 1; thread_id < n_threads; thread_id++) {
        daxpy_(&dim_A, &one, A_thread_memory[thread_id].get(), &one_int, A, &one_int);
        daxpy_(&dim_b, &one, b_thread_memory[thread_id].get(), &one_int, b, &one_int);
    }

    for (int row = 1; row < p; row++) {
        for (int col = 0; col < row; col++) {
            A[col + row*p] = A[row + col*p];
        }
    }
}

PYBIND11_MODULE(utils_pybind, m) {
    m.def("compute_xtx_xty", &compute_xtx_xty, "Compute X^T X and X^T y using nested loops or OpenBLAS");
}

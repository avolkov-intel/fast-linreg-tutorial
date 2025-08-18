#ifndef pybind_blas_header
#define pybind_blas_header
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <memory>
#include <algorithm>

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
 Helper to symmetrize an array representing a square symmetric
 matrix when only the upper triangular part has been filled,
 as done by many BLAS functions
*/
void symmetrize_matrix(double *A, const int n)
{
    for (int row = 1; row < n; row++) {
        for (int col = 0; col < row; col++) {
            A[col + row*n] = A[row + col*n];
        }
    }
}

void compute_xtx_xty_blas(
    const double *X,
    const double *y,
    const int n, const int p,
    double *A,
    double *b
) {
    const double one = 1.0;
    const double zero = 0.0;
    dsyrk_(
        "L", "N",
        &p, &n,
        &one, X, &p,
        &zero, A, &p
    );
    const int one_int = 1;
    dgemv_(
        "N", &p, &n,
        &one, X, &p,
        y, &one_int,
        &zero, b, &one_int
    );

    // Since only the upper part is filled by dsyrk, we copy it to the lower part manually
    symmetrize_matrix(A, p);
}
#endif

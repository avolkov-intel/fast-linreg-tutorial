/*
This is a helper to bring the required CBLAS functions into a PyBind11 module,
done by fetching function pointers of the Fortran-style BLAS interface from SciPy
and re-creating the CBLAS versions by calling the Fortran functions behind the
scenes. This is not necessary if linking directly to a BLAS library, but doing
it this way allows getting BLAS out of a Python-level install of just SciPy.
*/

extern "C" {

typedef void (*dsyrk_t)(const char*, const char*, const int*, const int*, const double*, const double*, const int*, const double*, double*, const int*);
typedef void (*dgemv_t)(const char*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*);
dsyrk_t dsyrk_;
dgemv_t dgemv_;

typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef CBLAS_ORDER CBLAS_LAYOUT;
void cblas_dsyrk(
    const CBLAS_ORDER Order,
    const CBLAS_UPLO Uplo,
    const CBLAS_TRANSPOSE Trans,
    const int N,
    const int K,
    const double alpha,
    const double *A,
    const int lda,
    const double beta,
    double *C,
    const int ldc
);
void cblas_dgemv(
    const CBLAS_ORDER order,
    const CBLAS_TRANSPOSE TransA,
    const int m,
    const int n,
    const double alpha,
    const double  *a,
    const int lda,
    const double *x,
    const int incx,
    const double beta,
    double  *y,
    const int incy
);

} /* extern "C" */

int load_blas_funs()
{
    PyGILState_STATE gil_state = PyGILState_Ensure();
    PyObject *cython_blas_module = PyImport_ImportModule("scipy.linalg.cython_blas");
    PyObject *pyx_capi_obj = PyObject_GetAttrString(cython_blas_module, "__pyx_capi__");
    PyObject *cobj_dsyrk = PyDict_GetItemString(pyx_capi_obj, "dsyrk");
    PyObject *cobj_dgemv = PyDict_GetItemString(pyx_capi_obj, "dgemv");
    dsyrk_ = reinterpret_cast<dsyrk_t>(PyCapsule_GetPointer(cobj_dsyrk, PyCapsule_GetName(cobj_dsyrk)));
    dgemv_ = reinterpret_cast<dgemv_t>(PyCapsule_GetPointer(cobj_dgemv, PyCapsule_GetName(cobj_dgemv)));
    Py_DECREF(cython_blas_module);
    PyGILState_Release(gil_state);
    return 0;
}

const int dummy = load_blas_funs();

void cblas_dsyrk
(
    const CBLAS_ORDER Order,
    const CBLAS_UPLO Uplo,
    const CBLAS_TRANSPOSE Trans,
    const int N,
    const int K,
    const double alpha,
    const double *A,
    const int lda,
    const double beta,
    double *C,
    const int ldc
)
{
    char uplo = '\0';
    char trans = '\0';
    if (Order == CblasColMajor)
    {
        if (Uplo == CblasUpper)
            uplo = 'U';
        else
            uplo = 'L';

        if (Trans == CblasTrans)
            trans = 'T';
        else if (Trans == CblasConjTrans)
            trans = 'C';
        else
            trans = 'N';
    }

    else
    {
        if (Uplo == CblasUpper)
            uplo = 'L';
        else
            uplo = 'U';

        if (Trans == CblasTrans)
            trans = 'N';
        else if (Trans == CblasConjTrans)
            trans = 'N';
        else
            trans = 'T';
    }

    dsyrk_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc);
}

void cblas_dgemv
(
    const CBLAS_ORDER order,
    const CBLAS_TRANSPOSE TransA,
    const int m,
    const int n,
    const double alpha,
    const double  *a,
    const int lda,
    const double *x,
    const int incx,
    const double beta,
    double  *y,
    const int incy
)
{
    char trans = '\0';
    if (order == CblasColMajor)
    {
        if (TransA == CblasNoTrans)
            trans = 'N';
        else if (TransA == CblasTrans)
            trans = 'T';
        else
            trans = 'C';

        dgemv_(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }

    else
    {
        if (TransA == CblasNoTrans)
            trans = 'T';
        else if (TransA == CblasTrans)
            trans = 'N';
        else
            trans = 'N';

        dgemv_(&trans, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }
}

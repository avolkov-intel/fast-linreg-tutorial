from scipy.linalg.cython_blas cimport dsyrk, dgemv

# Helpers for getting BLAS functions
ctypedef enum CBLAS_ORDER:
    CblasRowMajor = 101
    CblasColMajor = 102

ctypedef enum CBLAS_TRANSPOSE:
    CblasNoTrans=111
    CblasTrans=112
    CblasConjTrans=113
    CblasConjNoTrans=114

ctypedef enum CBLAS_UPLO:
    CblasUpper=121
    CblasLower=122

# Workaround for non-const types in SciPy's .pxd files
ctypedef void (*dsyrk_t)(const char*, const char*, const int*, const int*, const double*, const double*, const int*, const double*, double*, const int*) noexcept nogil
ctypedef void (*dgemv_t)(const char*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*) noexcept nogil

cdef void cblas_dsyrk(
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
) noexcept nogil:
    cdef char uplo = 0
    cdef char trans = 0
    if (Order == CblasColMajor):
        if (Uplo == CblasUpper):
            uplo = 85 #'U'
        else:
            uplo = 76 #'L'

        if (Trans == CblasTrans):
            trans = 84 #'T'
        elif (Trans == CblasConjTrans):
            trans = 67 #'C'
        else:
            trans = 78 #'N'

    else:
        if (Uplo == CblasUpper):
            uplo = 76 #'L'
        else:
            uplo = 85 #'U'

        if (Trans == CblasTrans):
            trans = 78 #'N'
        elif (Trans == CblasConjTrans):
            trans = 78 #'N'
        else:
            trans = 84 #'T'

    (<dsyrk_t>dsyrk)(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc)

cdef void cblas_dgemv(
    const CBLAS_ORDER order,
    const CBLAS_TRANSPOSE TransA,
    const int m,
    const int n,
    const double alpha,
    const double *a,
    const int lda,
    const double *x,
    const int incx,
    const double beta,
    double *y,
    const int incy
) noexcept nogil:
    cdef char trans = 0
    if (order == CblasColMajor):
        if (TransA == CblasNoTrans):
            trans = 78 #'N'
        elif (TransA == CblasTrans):
            trans = 84 #'T'
        else:
            trans = 67 #'C'

        (<dgemv_t>dgemv)(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);

    else:
        if (TransA == CblasNoTrans):
            trans = 84 #'T'
        elif (TransA == CblasTrans):
            trans = 78 #'N'
        else:
            trans = 78 #'N'

        (<dgemv_t>dgemv)(&trans, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy)

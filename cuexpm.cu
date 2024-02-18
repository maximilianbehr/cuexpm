/* MIT License
 *
 * Copyright (c) 2024 Maximilian Behr
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <math.h>
#include <stdint.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include "cuexpm.h"

#define CHECK_CUEXPM(err)                                                                                  \
    do {                                                                                                   \
        int error_code = (err);                                                                            \
        if (error_code) {                                                                                  \
            fprintf(stderr, "cuExpm Error %d. In file '%s' on line %d\n", error_code, __FILE__, __LINE__); \
            fflush(stderr);                                                                                \
            return -1;                                                                                     \
        }                                                                                                  \
    } while (false)

#define CHECK_CUDA(err)                                                                                                                      \
    do {                                                                                                                                     \
        cudaError_t error_code = (err);                                                                                                      \
        if (error_code != cudaSuccess) {                                                                                                     \
            fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), __FILE__, __LINE__); \
            fflush(stderr);                                                                                                                  \
            return -1;                                                                                                                       \
        }                                                                                                                                    \
    } while (false)

#define CHECK_CUBLAS(err)                                                                                                                          \
    do {                                                                                                                                           \
        cublasStatus_t error_code = (err);                                                                                                         \
        if (error_code != CUBLAS_STATUS_SUCCESS) {                                                                                                 \
            fprintf(stderr, "CUBLAS Error %d - %s. In file '%s' on line %d\n", error_code, cublasGetStatusString(error_code), __FILE__, __LINE__); \
            fflush(stderr);                                                                                                                        \
            return -2;                                                                                                                             \
        }                                                                                                                                          \
    } while (false)

static inline const char *cuexpm_cusolverGetErrorEnum(cusolverStatus_t error) {
    switch (error) {
        case CUSOLVER_STATUS_SUCCESS:
            return "CUSOLVER_STATUS_SUCCESS";

        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "CUSOLVER_STATUS_ALLOC_FAILED";

        case CUSOLVER_STATUS_INVALID_VALUE:
            return "CUSOLVER_STATUS_INVALID_VALUE";

        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "CUSOLVER_STATUS_ARCH_MISMATCH";

        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "CUSOLVER_STATUS_EXECUTION_FAILED";

        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";
        default:
            return "unknown";
    }
}

#define CHECK_CUSOLVER(err)                                                                                                                                \
    do {                                                                                                                                                   \
        cusolverStatus_t error_code = (err);                                                                                                               \
        if (error_code != CUSOLVER_STATUS_SUCCESS) {                                                                                                       \
            fprintf(stderr, "CUSOLVER Error %d - %s. In file '%s' on line %d\n", error_code, cuexpm_cusolverGetErrorEnum(error_code), __FILE__, __LINE__); \
            fflush(stderr);                                                                                                                                \
            return -3;                                                                                                                                     \
        }                                                                                                                                                  \
    } while (false)

template <typename T>
struct cuexpm_traits;

template <>
struct cuexpm_traits<double> {
    typedef double S;
    /*-----------------------------------------------------------------------------
     * constants
     *-----------------------------------------------------------------------------*/
    static constexpr double one = 1.;
    static constexpr double mone = -1.;
    static constexpr double zero = 0.;

    /*-----------------------------------------------------------------------------
     * Pade coefficients
     *-----------------------------------------------------------------------------*/
    static constexpr double Pade3[] = {120., 60., 12., 1.};
    static constexpr double Pade5[] = {30240., 15120., 3360., 420., 30., 1.};
    static constexpr double Pade7[] = {17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.};
    static constexpr double Pade9[] = {17643225600., 8821612800., 2075673600., 302702400., 30270240., 2162160., 110880., 3960., 90., 1.};
    static constexpr double Pade13[] = {64764752532480000., 32382376266240000., 7771770303897600., 1187353796428800., 129060195264000., 10559470521600., 670442572800., 33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.};

    /*-----------------------------------------------------------------------------
     * absolute value, used for computing the 1-norm of a matrix
     *-----------------------------------------------------------------------------*/
    __device__ inline static double abs(const double x) {
        return fabs(x);
    }

    /*-----------------------------------------------------------------------------
     * scaling of a matrix, matrix addition and mulitplication using cuBLAS
     *-----------------------------------------------------------------------------*/
    inline static cublasStatus_t cublasXdscal(cublasHandle_t handle, int n, const double *alpha, double *x, int incx) {
        return cublasDscal(handle, n, alpha, x, incx);
    }

    inline static cublasStatus_t cublasXgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double *alpha, const double *A, int lda, const double *beta, const double *B, int ldb, double *C, int ldc) {
        return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    }

    inline static cublasStatus_t cublasXgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
        return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    /*-----------------------------------------------------------------------------
     * computeType and dataType for cuSolver
     *-----------------------------------------------------------------------------*/
    static constexpr cudaDataType dataType = CUDA_R_64F;
    static constexpr cudaDataType computeType = CUDA_R_64F;
};

template <>
struct cuexpm_traits<float> {
    typedef float S;
    /*-----------------------------------------------------------------------------
     * constants
     *-----------------------------------------------------------------------------*/
    static constexpr float one = 1.f;
    static constexpr float mone = -1.f;
    static constexpr float zero = 0.f;

    /*-----------------------------------------------------------------------------
     * Pade coefficients
     *-----------------------------------------------------------------------------*/
    static constexpr float Pade3[] = {120.f, 60.f, 12.f, 1.f};
    static constexpr float Pade5[] = {30240.f, 15120.f, 3360.f, 420.f, 30.f, 1.f};
    static constexpr float Pade7[] = {17297280.f, 8648640.f, 1995840.f, 277200.f, 25200.f, 1512.f, 56.f, 1.f};
    static constexpr float Pade9[] = {17643225600.f, 8821612800.f, 2075673600.f, 302702400.f, 30270240.f, 2162160.f, 110880.f, 3960.f, 90.f, 1.f};
    static constexpr float Pade13[] = {64764752532480000.f, 32382376266240000.f, 7771770303897600.f, 1187353796428800.f, 129060195264000.f, 10559470521600.f, 670442572800.f, 33522128640.f, 1323241920.f, 40840800.f, 960960.f, 16380.f, 182.f, 1.f};

    /*-----------------------------------------------------------------------------
     * absolute value, used for computing the 1-norm of a matrix
     *-----------------------------------------------------------------------------*/
    __device__ inline static double abs(const double x) {
        return fabsf(x);
    }

    /*-----------------------------------------------------------------------------
     * scaling of a matrix, matrix addition and mulitplication using cuBLAS
     *-----------------------------------------------------------------------------*/
    inline static cublasStatus_t cublasXdscal(cublasHandle_t handle, int n, const float *alpha, float *x, int incx) {
        return cublasSscal(handle, n, alpha, x, incx);
    }

    inline static cublasStatus_t cublasXgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float *alpha, const float *A, int lda, const float *beta, const float *B, int ldb, float *C, int ldc) {
        return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    }

    inline static cublasStatus_t cublasXgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
        return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    /*-----------------------------------------------------------------------------
     * computeType and dataType for cuSolver
     *-----------------------------------------------------------------------------*/
    static constexpr cudaDataType dataType = CUDA_R_32F;
    static constexpr cudaDataType computeType = CUDA_R_32F;
};

template <>
struct cuexpm_traits<cuDoubleComplex> {
    typedef double S;
    /*-----------------------------------------------------------------------------
     * constants
     *-----------------------------------------------------------------------------*/
    static constexpr cuDoubleComplex one = {1., 0.};
    static constexpr cuDoubleComplex mone = {-1., 0.};
    static constexpr cuDoubleComplex zero = {0., 0.};

    /*-----------------------------------------------------------------------------
     * Pade coefficients
     *-----------------------------------------------------------------------------*/
    static constexpr cuDoubleComplex Pade3[] = {{120., 0.}, {60., 0.}, {12., 0.}, {1., 0.}};
    static constexpr cuDoubleComplex Pade5[] = {{30240., 0.}, {15120., 0.}, {3360., 0.}, {420., 0.}, {30., 0.}, {1., 0.}};
    static constexpr cuDoubleComplex Pade7[] = {{17297280., 0.}, {8648640., 0.}, {1995840., 0.}, {277200., 0.}, {25200., 0.}, {1512., 0.}, {56., 0.}, {1., 0.}};
    static constexpr cuDoubleComplex Pade9[] = {{17643225600., 0.}, {8821612800., 0.}, {2075673600., 0.}, {302702400., 0.}, {30270240., 0.}, {2162160., 0.}, {110880., 0.}, {3960., 0.}, {90., 0.}, {1., 0.}};
    static constexpr cuDoubleComplex Pade13[] = {{64764752532480000., 0.}, {32382376266240000., 0.}, {7771770303897600., 0.}, {1187353796428800., 0.}, {129060195264000., 0.}, {10559470521600., 0.}, {670442572800., 0.}, {33522128640., 0.}, {1323241920., 0.}, {40840800., 0.}, {960960., 0.}, {16380., 0.}, {182., 0.}, {1., 0.}};

    /*-----------------------------------------------------------------------------
     * absolute value, used for computing the 1-norm of a matrix
     *-----------------------------------------------------------------------------*/
    __device__ inline static double abs(const cuDoubleComplex x) {
        return cuCabs(x);
    }

    /*-----------------------------------------------------------------------------
     * scaling of a matrix, matrix addition and mulitplication using cuBLAS
     *-----------------------------------------------------------------------------*/
    inline static cublasStatus_t cublasXdscal(cublasHandle_t handle, int n, const double *alpha, cuDoubleComplex *x, int incx) {
        return cublasZdscal(handle, n, alpha, x, incx);
    }

    inline static cublasStatus_t cublasXgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *beta, const cuDoubleComplex *B, int ldb, cuDoubleComplex *C, int ldc) {
        return cublasZgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    }

    inline static cublasStatus_t cublasXgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
        return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    /*-----------------------------------------------------------------------------
     * computeType and dataType for cuSolver
     *-----------------------------------------------------------------------------*/
    static constexpr cudaDataType dataType = CUDA_C_64F;
    static constexpr cudaDataType computeType = CUDA_C_64F;
};

template <>
struct cuexpm_traits<cuComplex> {
    typedef float S;
    /*-----------------------------------------------------------------------------
     * constants
     *-----------------------------------------------------------------------------*/
    static constexpr cuComplex one = {1.f, 0.f};
    static constexpr cuComplex mone = {-1.f, 0.f};
    static constexpr cuComplex zero = {0.f, 0.f};

    /*-----------------------------------------------------------------------------
     * Pade coefficients
     *-----------------------------------------------------------------------------*/
    static constexpr cuComplex Pade3[] = {{120.f, 0.f}, {60.f, 0.f}, {12.f, 0.f}, {1.f, 0.f}};
    static constexpr cuComplex Pade5[] = {{30240.f, 0.f}, {15120.f, 0.f}, {3360.f, 0.f}, {420.f, 0.f}, {30.f, 0.f}, {1.f, 0.f}};
    static constexpr cuComplex Pade7[] = {{17297280.f, 0.f}, {8648640.f, 0.f}, {1995840.f, 0.f}, {277200.f, 0.f}, {25200.f, 0.f}, {1512.f, 0.f}, {56.f, 0.f}, {1.f, 0.f}};
    static constexpr cuComplex Pade9[] = {{17643225600.f, 0.f}, {8821612800.f, 0.f}, {2075673600.f, 0.f}, {302702400.f, 0.f}, {30270240.f, 0.f}, {2162160.f, 0.f}, {110880.f, 0.f}, {3960.f, 0.f}, {90.f, 0.f}, {1.f, 0.f}};
    static constexpr cuComplex Pade13[] = {{64764752532480000.f, 0.f}, {32382376266240000.f, 0.f}, {7771770303897600.f, 0.f}, {1187353796428800.f, 0.f}, {129060195264000.f, 0.f}, {10559470521600.f, 0.f}, {670442572800.f, 0.f}, {33522128640.f, 0.f}, {1323241920.f, 0.f}, {40840800.f, 0.f}, {960960.f, 0.f}, {16380.f, 0.f}, {182.f, 0.f}, {1.f, 0.f}};

    /*-----------------------------------------------------------------------------
     * absolute value, used for computing the 1-norm of a matrix
     *-----------------------------------------------------------------------------*/
    __device__ inline static float abs(const cuComplex x) {
        return cuCabsf(x);
    }

    /*-----------------------------------------------------------------------------
     * scaling of a matrix, matrix addition and mulitplication using cuBLAS
     *-----------------------------------------------------------------------------*/
    inline static cublasStatus_t cublasXdscal(cublasHandle_t handle, int n, const float *alpha, cuComplex *x, int incx) {
        return cublasCsscal(handle, n, alpha, x, incx);
    }

    inline static cublasStatus_t cublasXgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *beta, const cuComplex *B, int ldb, cuComplex *C, int ldc) {
        return cublasCgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    }

    inline static cublasStatus_t cublasXgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
        return cublasCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    /*-----------------------------------------------------------------------------
     * computeType and dataType for cuSolver
     *-----------------------------------------------------------------------------*/
    static constexpr cudaDataType dataType = CUDA_C_32F;
    static constexpr cudaDataType computeType = CUDA_C_32F;
};

template <typename T>
__global__ static void cuexpm_absrowsums(const T *__restrict__ d_A, const int n, double *__restrict__ buffer) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (int j = i; j < n; j += blockDim.x * gridDim.x) {
        double tmp = 0.;
        for (int k = 0; k < n; ++k) {
            tmp += cuexpm_traits<T>::abs(d_A[k + j * n]);
        }
        buffer[j] = tmp;
    }
}

template <typename T>
static int cuexpm_matrix1norm(const T *__restrict__ d_A, const int n, void *d_buffer, double *__restrict__ d_nrmA1) {
    *d_nrmA1 = 0.;
    double *buffer = reinterpret_cast<double *>(d_buffer);
    cuexpm_absrowsums<<<(n + 255) / 256, 256>>>(d_A, n, buffer);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    *d_nrmA1 = *(thrust::max_element(thrust::device_pointer_cast(buffer), thrust::device_pointer_cast(buffer + n)));
    return 0;
}

template <typename T>
static int cuexpm_parameters(const T *__restrict__ d_A, const int n, void *d_buffer, int *m, int *s) {
    double eta1 = 0.;
    CHECK_CUEXPM(cuexpm_matrix1norm(d_A, n, d_buffer, &eta1));
    if constexpr (std::is_same<T, double>::value || std::is_same<T, cuDoubleComplex>::value) {
        const double theta[] = {1.495585217958292e-002, 2.539398330063230e-001, 9.504178996162932e-001, 2.97847961257068e+000, 5.371920351148152e+000};
        *s = 0;
        if (eta1 <= theta[0]) {
            *m = 3;
            return 0;
        }
        if (eta1 <= theta[1]) {
            *m = 5;
            return 0;
        }
        if (eta1 <= theta[2]) {
            *m = 7;
            return 0;
        }
        if (eta1 <= theta[3]) {
            *m = 9;
            return 0;
        }
        *s = ceil(log2(eta1 / theta[4]));
        if (*s < 0) {
            *s = 0;
        }
        *m = 13;
    } else {
        const double theta[] = {4.258730016922831e-001, 1.880152677804762e+000, 3.925724783138660e+000};
        *s = 0;
        if (eta1 <= theta[0]) {
            *m = 3;
            return 0;
        }
        if (eta1 <= theta[1]) {
            *m = 5;
            return 0;
        }
        *s = ceil(log2(eta1 / theta[2]));
        if (*s < 0) {
            *s = 0;
        }
        *m = 7;
    }
    return 0;
}

template <typename T>
__global__ static void setDiag(T *d_A, const int n, const T alpha) {
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int j0 = threadIdx.y + blockIdx.y * blockDim.y;
    for (int i = i0; i < n; i += blockDim.x * gridDim.x) {
        for (int j = j0; j < n; j += blockDim.y * gridDim.y) {
            if (i == j) {
                d_A[i + j * n] = alpha;
            } else {
                d_A[i + j * n] = cuexpm_traits<T>::zero;
            }
        }
    }
}

__device__ static cuComplex &operator+=(cuComplex &a, const cuComplex &b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

__device__ static cuDoubleComplex &operator+=(cuDoubleComplex &a, const cuDoubleComplex &b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

template <typename T>
__global__ static void addDiag(T *d_A, const int n, const T alpha) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (int j = i; j < n; j += blockDim.x * gridDim.x) {
        d_A[j + j * n] += alpha;
    }
}

const static cusolverAlgMode_t CUSOLVER_ALG = CUSOLVER_ALG_0;

template <typename T>
static int cuexpm_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    /*-----------------------------------------------------------------------------
     * initialize with zero
     *-----------------------------------------------------------------------------*/
    *d_bufferSize = 0;
    *h_bufferSize = 0;

    /*-----------------------------------------------------------------------------
     * get device and host workspace size for LU factorization
     *-----------------------------------------------------------------------------*/
    // create cusolver handle
    cusolverDnHandle_t cusolverH;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

    // create cusolver params
    cusolverDnParams_t params;
    CHECK_CUSOLVER(cusolverDnCreateParams(&params));
    CHECK_CUSOLVER(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG));

    // compute workspace size
    CHECK_CUSOLVER(cusolverDnXgetrf_bufferSize(cusolverH, params, n, n, cuexpm_traits<T>::dataType, nullptr, n, cuexpm_traits<T>::computeType, d_bufferSize, h_bufferSize));

    // free workspace
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));
    CHECK_CUSOLVER(cusolverDnDestroyParams(params));

    /*-----------------------------------------------------------------------------
     * compute final workspace size
     * matrix T1, T2, T4, T6, T8, U, V -> n * n * 5 +  n * n * 2
     * int64 array ipiv -> n * sizeof(int64_t)
     * int info -> sizeof(int)
     *-----------------------------------------------------------------------------*/
    *d_bufferSize = std::max(*d_bufferSize, sizeof(T) * n * n * 5) + sizeof(T) * n * n * 2 + sizeof(int64_t) * n + sizeof(int);

    return 0;
}

int cuexpmd_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    return cuexpm_bufferSize<double>(n, d_bufferSize, h_bufferSize);
}

int cuexpms_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    return cuexpm_bufferSize<float>(n, d_bufferSize, h_bufferSize);
}

int cuexpmc_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    return cuexpm_bufferSize<cuComplex>(n, d_bufferSize, h_bufferSize);
}

int cuexpmz_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    return cuexpm_bufferSize<cuDoubleComplex>(n, d_bufferSize, h_bufferSize);
}

template <typename T>
static int cuexpm(const T *d_A, const int n, void *d_buffer, void *h_buffer, T *d_F) {
    /*-----------------------------------------------------------------------------
     * kernel launch parameters
     *-----------------------------------------------------------------------------*/
    const size_t threadsPerBlock = 256;                                        // addDiag
    const size_t blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;  // addDiag
    dim3 grid((n + 15) / 16, (n + 15) / 16);                                   // setDiag
    dim3 block(16, 16);                                                        // setDiag

    /*-----------------------------------------------------------------------------
     * compute the scaling parameter and Pade approximant degree
     *-----------------------------------------------------------------------------*/
    int m, s;
    CHECK_CUEXPM(cuexpm_parameters(d_A, n, d_buffer, &m, &s));
    // printf("m = %d, s = %d\n", m, s);

    /*-----------------------------------------------------------------------------
     * split memory buffer
     * memory layout: |U, V, T1, T2, T4, T6, T8| from 0 to (n * n * 7) -1
     *-----------------------------------------------------------------------------*/
    T *T1, *T2, *T4, *T6, *T8, *U, *V;
    U = (T *)d_buffer;
    V = U + n * n * 1;
    T1 = U + n * n * 2;
    T2 = U + n * n * 3;
    T4 = U + n * n * 4;
    T6 = U + n * n * 5;
    T8 = U + n * n * 6;

    /*-----------------------------------------------------------------------------
     * create cuBlas handle
     *-----------------------------------------------------------------------------*/
    cublasHandle_t cublasH;
    CHECK_CUBLAS(cublasCreate(&cublasH));

    /*-----------------------------------------------------------------------------
     * rescale T, T = T / 2^s
     *-----------------------------------------------------------------------------*/
    CHECK_CUDA(cudaMemcpy(T1, d_A, sizeof(*d_A) * n * n, cudaMemcpyDeviceToDevice));

    typename cuexpm_traits<T>::S alpha = 1. / (1 << s);
    if (s != 0) {
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXdscal(cublasH, n * n, &alpha, T1, 1));
    }

    /*-----------------------------------------------------------------------------
     * compute powers of T if needed
     *-----------------------------------------------------------------------------*/
    CHECK_CUBLAS(cuexpm_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &cuexpm_traits<T>::one, T1, n, T1, n, &cuexpm_traits<T>::zero, T2, n));
    if (m >= 5) {
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &cuexpm_traits<T>::one, T2, n, T2, n, &cuexpm_traits<T>::zero, T4, n));
    }
    if (m >= 7) {
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &cuexpm_traits<T>::one, T2, n, T4, n, &cuexpm_traits<T>::zero, T6, n));
    }
    if (m == 9) {
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &cuexpm_traits<T>::one, T4, n, T4, n, &cuexpm_traits<T>::zero, T8, n));
    }

    /*-----------------------------------------------------------------------------
     * compute U and V for the Pade approximant independently on different streams
     *-----------------------------------------------------------------------------*/
    cudaStream_t streamU, streamV;
    CHECK_CUDA(cudaStreamCreate(&streamU));
    CHECK_CUDA(cudaStreamCreate(&streamV));
    if (m == 3) {
        // U = U + c(3)*T2 + c(1)*I
        setDiag<<<grid, block, 0, streamU>>>(U, n, cuexpm_traits<T>::Pade3[1]);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUBLAS(cublasSetStream(cublasH, streamU));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade3[3], T2, n, U, n));

        // V = V + c(2)*T2 + c(0)*I
        setDiag<<<grid, block, 0, streamV>>>(V, n, cuexpm_traits<T>::Pade3[0]);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUBLAS(cublasSetStream(cublasH, streamV));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade3[2], T2, n, V, n));
    } else if (m == 5) {
        // U = U + c(5)*T4 + c(3)*T2 + c(1)*I
        setDiag<<<grid, block, 0, streamU>>>(U, n, cuexpm_traits<T>::Pade5[1]);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUBLAS(cublasSetStream(cublasH, streamU));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade5[3], T2, n, U, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade5[5], T4, n, U, n));

        // V = V + c(4)*T4 + c(2)*T2 + c(0)*I
        setDiag<<<grid, block, 0, streamV>>>(V, n, cuexpm_traits<T>::Pade5[0]);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUBLAS(cublasSetStream(cublasH, streamV));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade5[2], T2, n, V, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade5[4], T4, n, V, n));
    } else if (m == 7) {
        // U = U + c(7)*T6 + c(5)*T4 + c(3)*T2 + c(1)*I
        setDiag<<<grid, block, 0, streamU>>>(U, n, cuexpm_traits<T>::Pade7[1]);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUBLAS(cublasSetStream(cublasH, streamU));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade7[3], T2, n, U, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade7[5], T4, n, U, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade7[7], T6, n, U, n));

        // V = V + c(6)*T6 + c(4)*T4 + c(2)*T2 + c(0)*I
        setDiag<<<grid, block, 0, streamV>>>(V, n, cuexpm_traits<T>::Pade7[0]);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUBLAS(cublasSetStream(cublasH, streamV));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade7[2], T2, n, V, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade7[4], T4, n, V, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade7[6], T6, n, V, n));
    } else if (m == 9) {
        // U = U + c(9)*T8 + c(7)*T6 + c(5)*T4 + c(3)*T2 + c(1)*I
        setDiag<<<grid, block, 0, streamU>>>(U, n, cuexpm_traits<T>::Pade9[1]);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUBLAS(cublasSetStream(cublasH, streamU));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade9[3], T2, n, U, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade9[5], T4, n, U, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade9[7], T6, n, U, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade9[9], T8, n, U, n));

        // V = V + c(6)*T6 + c(4)*T4 + c(2)*T2 + c(0)*I
        setDiag<<<grid, block, 0, streamV>>>(V, n, cuexpm_traits<T>::Pade9[0]);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUBLAS(cublasSetStream(cublasH, streamV));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade9[2], T2, n, V, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade9[4], T4, n, V, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade9[6], T6, n, V, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade9[8], T8, n, V, n));
    } else if (m == 13) {
        //  U = T6*(c(13)*T6 + c(11)*T4 + c(9)*T2) + c(7)*T6 + c(5)*T4 + c(3)*T2 + c(1)*I;
        setDiag<<<grid, block, 0, streamU>>>(U, n, cuexpm_traits<T>::Pade13[1]);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUBLAS(cublasSetStream(cublasH, streamU));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade13[3], T2, n, U, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade13[5], T4, n, U, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, U, n, &cuexpm_traits<T>::Pade13[7], T6, n, U, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::Pade13[9], T2, n, &cuexpm_traits<T>::Pade13[11], T4, n, T8, n));  // overwrite of T8
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::Pade13[13], T6, n, &cuexpm_traits<T>::one, T8, n, T8, n));        // overwrite of T8
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &cuexpm_traits<T>::one, T6, n, T8, n, &cuexpm_traits<T>::one, U, n));

        // V = T6*(c(12)*T6 + c(10)*T4 + c(8)*T2) + c(6)*T6 + c(4)*T4 + c(2)*T2 + c(0)*I;
        setDiag<<<grid, block, 0, streamV>>>(V, n, cuexpm_traits<T>::Pade13[0]);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUBLAS(cublasSetStream(cublasH, streamV));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade13[2], T2, n, V, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade13[4], T4, n, V, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::one, V, n, &cuexpm_traits<T>::Pade13[6], T6, n, V, n));
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::Pade13[8], T2, n, &cuexpm_traits<T>::Pade13[10], T4, n, T8, n));  // overwrite of T8
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::Pade13[12], T6, n, &cuexpm_traits<T>::one, T8, n, T8, n));        // overwrite of T8
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &cuexpm_traits<T>::one, T6, n, T8, n, &cuexpm_traits<T>::one, V, n));
    } else {
        fprintf(stderr, "m must be 3, 5, 7, 9, or 13\n");
        fflush(stderr);
        return -4;
    }
    CHECK_CUDA(cudaStreamSynchronize(streamU));
    CHECK_CUDA(cudaStreamSynchronize(streamV));
    CHECK_CUDA(cudaStreamDestroy(streamU));
    CHECK_CUDA(cudaStreamDestroy(streamV));
    CHECK_CUBLAS(cublasSetStream(cublasH, 0));

    // U = T*U
    CHECK_CUBLAS(cuexpm_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &cuexpm_traits<T>::one, T1, n, U, n, &cuexpm_traits<T>::zero, T8, n));
    std::swap(U, T8);

    /*-----------------------------------------------------------------------------
     *  compute F = (V-U)\(U+V) = (V-U)\2*U + I
     *-----------------------------------------------------------------------------*/
    // prepare right-hand side
    CHECK_CUBLAS(cuexpm_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &cuexpm_traits<T>::mone, U, n, &cuexpm_traits<T>::one, V, n, V, n));

    typename cuexpm_traits<T>::S two = 2.;
    CHECK_CUBLAS(cuexpm_traits<T>::cublasXdscal(cublasH, n * n, &two, U, 1));

    // create cusolver handle
    cusolverDnHandle_t cusolverH;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

    // create cusolver params
    cusolverDnParams_t params;
    CHECK_CUSOLVER(cusolverDnCreateParams(&params));
    CHECK_CUSOLVER(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG));

    // split the memory buffer
    // memory layout: |U, V, T1, T2, T4, T6, T8| from 0 to (n * n * 7) -1
    //                |x, x, ipiv, info, dwork|  from n * n * 2         to n * n * 2 + n -1 (ipiv) overwrites T1
    //                                           from n * n * 2 + n     to n * n * 2 + n    (info) overwrites T1, T2
    //                                           from n * n * 2 + n + 1 to XXXXXXXXXXXXX    (dwork) overwrites T1, T2, T4, T6, T8, ...
    int64_t *d_ipiv = (int64_t *)T1;    // use T1 as ipiv
    int *d_info = (int *)(d_ipiv + n);  // put d_info after d_ipiv
    void *d_work = d_info + 1;          // put d_work after d_info
    void *h_work = h_buffer;            // use h_buffer as workspace

    // compute LU factorization
    size_t lworkdevice = 0, lworkhost = 0;
    CHECK_CUSOLVER(cusolverDnXgetrf_bufferSize(cusolverH, params, n, n, cuexpm_traits<T>::dataType, V, n, cuexpm_traits<T>::computeType, &lworkdevice, &lworkhost));
    CHECK_CUSOLVER(cusolverDnXgetrf(cusolverH, params, n, n, cuexpm_traits<T>::dataType, V, n, d_ipiv, cuexpm_traits<T>::computeType, d_work, lworkdevice, h_work, lworkhost, d_info));

    // solve linear system
    CHECK_CUSOLVER(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, n, n, cuexpm_traits<T>::dataType, V, n, d_ipiv, cuexpm_traits<T>::computeType, U, n, d_info));

    // free workspace
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));
    CHECK_CUSOLVER(cusolverDnDestroyParams(params));

    // add identity
    addDiag<<<blocksPerGrid, threadsPerBlock>>>(U, n, cuexpm_traits<T>::one);
    CHECK_CUDA(cudaPeekAtLastError());

    /*-----------------------------------------------------------------------------
     * squaring phase
     *-----------------------------------------------------------------------------*/
    if (((s % 2) == 0) && s > 0) {
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &cuexpm_traits<T>::one, U, n, U, n, &cuexpm_traits<T>::zero, V, n));
        std::swap(U, V);
        s--;
    }

    for (int k = 0; k < s; ++k) {
        CHECK_CUBLAS(cuexpm_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &cuexpm_traits<T>::one, U, n, U, n, &cuexpm_traits<T>::zero, d_F, n));
        CHECK_CUDA(cudaDeviceSynchronize());
        std::swap(U, d_F);
    }

    /*-----------------------------------------------------------------------------
     * free memory and destroy cuBlas handle
     *-----------------------------------------------------------------------------*/
    CHECK_CUBLAS(cublasDestroy(cublasH));
    return 0;
}

int cuexpms(const float *d_A, const int n, void *d_buffer, void *h_buffer, float *d_expmA) {
    return cuexpm(d_A, n, d_buffer, h_buffer, d_expmA);
}

int cuexpmd(const double *d_A, const int n, void *d_buffer, void *h_buffer, double *d_expmA) {
    return cuexpm(d_A, n, d_buffer, h_buffer, d_expmA);
}

int cuexpmc(const cuComplex *d_A, const int n, void *d_buffer, void *h_buffer, cuComplex *d_expmA) {
    return cuexpm(d_A, n, d_buffer, h_buffer, d_expmA);
}

int cuexpmz(const cuDoubleComplex *d_A, const int n, void *d_buffer, void *h_buffer, cuDoubleComplex *d_expmA) {
    return cuexpm(d_A, n, d_buffer, h_buffer, d_expmA);
}

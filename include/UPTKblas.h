#if !defined(UPTKBLAS_API_H_)
#define UPTKBLAS_API_H_

#define UPTKBLASAPI

#include <stdint.h>

#include "cuComplex.h" /* import complex data type */

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <UPTK_runtime_api.h>

#include <UPTK_library_types.h>

#if defined(__cplusplus)
extern "C"
{
#endif /* __cplusplus */

#define UPTKBLAS_VER_MAJOR 1
#define UPTKBLAS_VER_MINOR 0
#define UPTKBLAS_VER_PATCH 0
#define UPTKBLAS_VER_BUILD 0
#define UPTKBLAS_VERSION (UPTKBLAS_VER_MAJOR * 10000 + UPTKBLAS_VER_MINOR * 100 + UPTKBLAS_VER_PATCH)

  /* CUBLAS status type returns */
  typedef enum
  {
    UPTKBLAS_STATUS_SUCCESS = 0,
    UPTKBLAS_STATUS_NOT_INITIALIZED = 1,
    UPTKBLAS_STATUS_ALLOC_FAILED = 3,
    UPTKBLAS_STATUS_INVALID_VALUE = 7,
    UPTKBLAS_STATUS_ARCH_MISMATCH = 8,
    UPTKBLAS_STATUS_MAPPING_ERROR = 11,
    UPTKBLAS_STATUS_EXECUTION_FAILED = 13,
    UPTKBLAS_STATUS_INTERNAL_ERROR = 14,
    UPTKBLAS_STATUS_NOT_SUPPORTED = 15,
    UPTKBLAS_STATUS_LICENSE_ERROR = 16
  } UPTKblasStatus_t;

  typedef enum
  {
    UPTKBLAS_FILL_MODE_LOWER = 0,
    UPTKBLAS_FILL_MODE_UPPER = 1,
    UPTKBLAS_FILL_MODE_FULL = 2
  } UPTKblasFillMode_t;

  typedef enum
  {
    UPTKBLAS_DIAG_NON_UNIT = 0,
    UPTKBLAS_DIAG_UNIT = 1
  } UPTKblasDiagType_t;

  typedef enum
  {
    UPTKBLAS_SIDE_LEFT = 0,
    UPTKBLAS_SIDE_RIGHT = 1
  } UPTKblasSideMode_t;

  typedef enum
  {
    UPTKBLAS_OP_N = 0,
    UPTKBLAS_OP_T = 1,
    UPTKBLAS_OP_C = 2,
    UPTKBLAS_OP_HERMITAN = 2, /* synonym if UPTKBLAS_OP_C */
    UPTKBLAS_OP_CONJG = 3     /* conjugate, placeholder - not supported in the current release */
  } UPTKblasOperation_t;

  typedef enum
  {
    UPTKBLAS_POINTER_MODE_HOST = 0,
    UPTKBLAS_POINTER_MODE_DEVICE = 1
  } UPTKblasPointerMode_t;

  typedef enum
  {
    UPTKBLAS_ATOMICS_NOT_ALLOWED = 0,
    UPTKBLAS_ATOMICS_ALLOWED = 1
  } UPTKblasAtomicsMode_t;

  /*For different GEMM algorithm */
  typedef enum
  {
    UPTKBLAS_GEMM_DFALT = -1,
    UPTKBLAS_GEMM_DEFAULT = -1,
    UPTKBLAS_GEMM_ALGO0 = 0,
    UPTKBLAS_GEMM_ALGO1 = 1,
    UPTKBLAS_GEMM_ALGO2 = 2,
    UPTKBLAS_GEMM_ALGO3 = 3,
    UPTKBLAS_GEMM_ALGO4 = 4,
    UPTKBLAS_GEMM_ALGO5 = 5,
    UPTKBLAS_GEMM_ALGO6 = 6,
    UPTKBLAS_GEMM_ALGO7 = 7,
    UPTKBLAS_GEMM_ALGO8 = 8,
    UPTKBLAS_GEMM_ALGO9 = 9,
    UPTKBLAS_GEMM_ALGO10 = 10,
    UPTKBLAS_GEMM_ALGO11 = 11,
    UPTKBLAS_GEMM_ALGO12 = 12,
    UPTKBLAS_GEMM_ALGO13 = 13,
    UPTKBLAS_GEMM_ALGO14 = 14,
    UPTKBLAS_GEMM_ALGO15 = 15,
    UPTKBLAS_GEMM_ALGO16 = 16,
    UPTKBLAS_GEMM_ALGO17 = 17,
    UPTKBLAS_GEMM_ALGO18 = 18, // sliced 32x32
    UPTKBLAS_GEMM_ALGO19 = 19, // sliced 64x32
    UPTKBLAS_GEMM_ALGO20 = 20, // sliced 128x32
    UPTKBLAS_GEMM_ALGO21 = 21, // sliced 32x32  -splitK
    UPTKBLAS_GEMM_ALGO22 = 22, // sliced 64x32  -splitK
    UPTKBLAS_GEMM_ALGO23 = 23, // sliced 128x32 -splitK
    UPTKBLAS_GEMM_DEFAULT_TENSOR_OP = 99,
    UPTKBLAS_GEMM_DFALT_TENSOR_OP = 99,
    UPTKBLAS_GEMM_ALGO0_TENSOR_OP = 100,
    UPTKBLAS_GEMM_ALGO1_TENSOR_OP = 101,
    UPTKBLAS_GEMM_ALGO2_TENSOR_OP = 102,
    UPTKBLAS_GEMM_ALGO3_TENSOR_OP = 103,
    UPTKBLAS_GEMM_ALGO4_TENSOR_OP = 104,
    UPTKBLAS_GEMM_ALGO5_TENSOR_OP = 105,
    UPTKBLAS_GEMM_ALGO6_TENSOR_OP = 106,
    UPTKBLAS_GEMM_ALGO7_TENSOR_OP = 107,
    UPTKBLAS_GEMM_ALGO8_TENSOR_OP = 108,
    UPTKBLAS_GEMM_ALGO9_TENSOR_OP = 109,
    UPTKBLAS_GEMM_ALGO10_TENSOR_OP = 110,
    UPTKBLAS_GEMM_ALGO11_TENSOR_OP = 111,
    UPTKBLAS_GEMM_ALGO12_TENSOR_OP = 112,
    UPTKBLAS_GEMM_ALGO13_TENSOR_OP = 113,
    UPTKBLAS_GEMM_ALGO14_TENSOR_OP = 114,
    UPTKBLAS_GEMM_ALGO15_TENSOR_OP = 115
  } UPTKblasGemmAlgo_t;

  /*Enum for default math mode/tensor operation*/
  typedef enum
  {
    UPTKBLAS_DEFAULT_MATH = 0,

    /* deprecated, same effect as using UPTKBLAS_COMPUTE_32F_FAST_16F, will be removed in a future release */
    UPTKBLAS_TENSOR_OP_MATH = 1,

    /* same as using matching _PEDANTIC compute type when using UPTKblas<T>routine calls or UPTKblasEx() calls with
       UPTKDataType as compute type */
    UPTKBLAS_PEDANTIC_MATH = 2,

    /* allow accelerating single precision routines using TF32 tensor cores */
    UPTKBLAS_TF32_TENSOR_OP_MATH = 3,

    /* flag to force any reductons to use the accumulator type and not output type in case of mixed precision routines
       with lower size output type */
    UPTKBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16,
  } UPTKblasMath_t;

  /* For backward compatibility purposes */
  typedef UPTKDataType UPTKblasDataType_t;

  /* Enum for compute type
   *
   * - default types provide best available performance using all available hardware features
   *   and guarantee internal storage precision with at least the same precision and range;
   * - _PEDANTIC types ensure standard arithmetic and exact specified internal storage format;
   * - _FAST types allow for some loss of precision to enable higher throughput arithmetic.
   */
  typedef enum
  {
    UPTKBLAS_COMPUTE_16F = 64,           /* half - default */
    UPTKBLAS_COMPUTE_16F_PEDANTIC = 65,  /* half - pedantic */
    UPTKBLAS_COMPUTE_32F = 68,           /* float - default */
    UPTKBLAS_COMPUTE_32F_PEDANTIC = 69,  /* float - pedantic */
    UPTKBLAS_COMPUTE_32F_FAST_16F = 74,  /* float - fast, allows down-converting inputs to half or TF32 */
    UPTKBLAS_COMPUTE_32F_FAST_16BF = 75, /* float - fast, allows down-converting inputs to bfloat16 or TF32 */
    UPTKBLAS_COMPUTE_32F_FAST_TF32 = 77, /* float - fast, allows down-converting inputs to TF32 */
    UPTKBLAS_COMPUTE_64F = 70,           /* double - default */
    UPTKBLAS_COMPUTE_64F_PEDANTIC = 71,  /* double - pedantic */
    UPTKBLAS_COMPUTE_32I = 72,           /* signed 32-bit int - default */
    UPTKBLAS_COMPUTE_32I_PEDANTIC = 73,  /* signed 32-bit int - pedantic */
  } UPTKblasComputeType_t;

  /* Opaque structure holding CUBLAS library context */
  struct UPTKblasContext;
  typedef struct UPTKblasContext *UPTKblasHandle_t;

  /* Cublas logging */
  typedef void (*UPTKblasLogCallback)(const char *msg);

  /* cuBLAS Exported API {{{ */

  /* --------------- CUBLAS Helper Functions  ---------------- */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCreate(UPTKblasHandle_t *handle);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDestroy(UPTKblasHandle_t handle);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasGetVersion(UPTKblasHandle_t handle, int *version);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasGetProperty(libraryPropertyType type, int *value);

  UPTKBLASAPI size_t UPTKblasGetCudartVersion(void);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSetWorkspace(UPTKblasHandle_t handle,
                                        void *workspace,
                                        size_t workspaceSizeInBytes);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSetStream(UPTKblasHandle_t handle, UPTKStream_t streamId);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasGetStream(UPTKblasHandle_t handle, UPTKStream_t *streamId);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasGetPointerMode(UPTKblasHandle_t handle, UPTKblasPointerMode_t *mode);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSetPointerMode(UPTKblasHandle_t handle, UPTKblasPointerMode_t mode);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasGetAtomicsMode(UPTKblasHandle_t handle, UPTKblasAtomicsMode_t *mode);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSetAtomicsMode(UPTKblasHandle_t handle, UPTKblasAtomicsMode_t mode);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasGetMathMode(UPTKblasHandle_t handle, UPTKblasMath_t *mode);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSetMathMode(UPTKblasHandle_t handle, UPTKblasMath_t mode);

  UPTKBLASAPI const char *UPTKblasGetStatusName(UPTKblasStatus_t status);

  UPTKBLASAPI const char *UPTKblasGetStatusString(UPTKblasStatus_t status);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSetVector(int n, int elemSize, const void *x, int incx, void *devicePtr, int incy);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasSetVector_64(int64_t n, int64_t elemSize, const void *x, int64_t incx, void *devicePtr, int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasGetVector_64(int64_t n, int64_t elemSize, const void *x, int64_t incx, void *y, int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasSetMatrix_64(int64_t rows, int64_t cols, int64_t elemSize, const void *A, int64_t lda, void *B, int64_t ldb);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasGetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasGetMatrix_64(int64_t rows, int64_t cols, int64_t elemSize, const void *A, int64_t lda, void *B, int64_t ldb);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSetVectorAsync(
      int n, int elemSize, const void *hostPtr, int incx, void *devicePtr, int incy, UPTKStream_t stream);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSetVectorAsync_64(
      int64_t n, int64_t elemSize, const void *hostPtr, int64_t incx, void *devicePtr, int64_t incy, UPTKStream_t stream);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasGetVectorAsync(
      int n, int elemSize, const void *devicePtr, int incx, void *hostPtr, int incy, UPTKStream_t stream);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasGetVectorAsync_64(
      int64_t n, int64_t elemSize, const void *devicePtr, int64_t incx, void *hostPtr, int64_t incy, UPTKStream_t stream);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasSetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, UPTKStream_t stream);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSetMatrixAsync_64(int64_t rows,
                                             int64_t cols,
                                             int64_t elemSize,
                                             const void *A,
                                             int64_t lda,
                                             void *B,
                                             int64_t ldb,
                                             UPTKStream_t stream);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasGetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, UPTKStream_t stream);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasGetMatrixAsync_64(int64_t rows,
                                             int64_t cols,
                                             int64_t elemSize,
                                             const void *A,
                                             int64_t lda,
                                             void *B,
                                             int64_t ldb,
                                             UPTKStream_t stream);

  /* --------------- CUBLAS BLAS1 Functions  ---------------- */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasNrm2Ex(UPTKblasHandle_t handle,
                                  int n,
                                  const void *x,
                                  UPTKDataType xType,
                                  int incx,
                                  void *result,
                                  UPTKDataType resultType,
                                  UPTKDataType executionType);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasNrm2Ex_64(UPTKblasHandle_t handle,
                                     int64_t n,
                                     const void *x,
                                     UPTKDataType xType,
                                     int64_t incx,
                                     void *result,
                                     UPTKDataType resultType,
                                     UPTKDataType executionType);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasSnrm2(UPTKblasHandle_t handle, int n, const float *x, int incx, float *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasSnrm2_64(UPTKblasHandle_t handle, int64_t n, const float *x, int64_t incx, float *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDnrm2(UPTKblasHandle_t handle, int n, const double *x, int incx, double *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDnrm2_64(UPTKblasHandle_t handle, int64_t n, const double *x, int64_t incx, double *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasScnrm2(UPTKblasHandle_t handle, int n, const cuComplex *x, int incx, float *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasScnrm2_64(UPTKblasHandle_t handle, int64_t n, const cuComplex *x, int64_t incx, float *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDznrm2(UPTKblasHandle_t handle, int n, const cuDoubleComplex *x, int incx, double *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDznrm2_64(UPTKblasHandle_t handle, int64_t n, const cuDoubleComplex *x, int64_t incx, double *result);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDotEx(UPTKblasHandle_t handle,
                                 int n,
                                 const void *x,
                                 UPTKDataType xType,
                                 int incx,
                                 const void *y,
                                 UPTKDataType yType,
                                 int incy,
                                 void *result,
                                 UPTKDataType resultType,
                                 UPTKDataType executionType);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDotEx_64(UPTKblasHandle_t handle,
                                    int64_t n,
                                    const void *x,
                                    UPTKDataType xType,
                                    int64_t incx,
                                    const void *y,
                                    UPTKDataType yType,
                                    int64_t incy,
                                    void *result,
                                    UPTKDataType resultType,
                                    UPTKDataType executionType);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDotcEx(UPTKblasHandle_t handle,
                                  int n,
                                  const void *x,
                                  UPTKDataType xType,
                                  int incx,
                                  const void *y,
                                  UPTKDataType yType,
                                  int incy,
                                  void *result,
                                  UPTKDataType resultType,
                                  UPTKDataType executionType);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDotcEx_64(UPTKblasHandle_t handle,
                                     int64_t n,
                                     const void *x,
                                     UPTKDataType xType,
                                     int64_t incx,
                                     const void *y,
                                     UPTKDataType yType,
                                     int64_t incy,
                                     void *result,
                                     UPTKDataType resultType,
                                     UPTKDataType executionType);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasSdot(UPTKblasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSdot_64(
      UPTKblasHandle_t handle, int64_t n, const float *x, int64_t incx, const float *y, int64_t incy, float *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDdot(UPTKblasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDdot_64(
      UPTKblasHandle_t handle, int64_t n, const double *x, int64_t incx, const double *y, int64_t incy, double *result);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCdotu(
      UPTKblasHandle_t handle, int n, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *result);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCdotu_64(UPTKblasHandle_t handle,
                                    int64_t n,
                                    const cuComplex *x,
                                    int64_t incx,
                                    const cuComplex *y,
                                    int64_t incy,
                                    cuComplex *result);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCdotc(
      UPTKblasHandle_t handle, int n, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *result);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCdotc_64(UPTKblasHandle_t handle,
                                    int64_t n,
                                    const cuComplex *x,
                                    int64_t incx,
                                    const cuComplex *y,
                                    int64_t incy,
                                    cuComplex *result);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZdotu(UPTKblasHandle_t handle,
                                 int n,
                                 const cuDoubleComplex *x,
                                 int incx,
                                 const cuDoubleComplex *y,
                                 int incy,
                                 cuDoubleComplex *result);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZdotu_64(UPTKblasHandle_t handle,
                                    int64_t n,
                                    const cuDoubleComplex *x,
                                    int64_t incx,
                                    const cuDoubleComplex *y,
                                    int64_t incy,
                                    cuDoubleComplex *result);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZdotc(UPTKblasHandle_t handle,
                                 int n,
                                 const cuDoubleComplex *x,
                                 int incx,
                                 const cuDoubleComplex *y,
                                 int incy,
                                 cuDoubleComplex *result);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZdotc_64(UPTKblasHandle_t handle,
                                    int64_t n,
                                    const cuDoubleComplex *x,
                                    int64_t incx,
                                    const cuDoubleComplex *y,
                                    int64_t incy,
                                    cuDoubleComplex *result);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasScalEx(UPTKblasHandle_t handle,
                                  int n,
                                  const void *alpha,
                                  UPTKDataType alphaType,
                                  void *x,
                                  UPTKDataType xType,
                                  int incx,
                                  UPTKDataType executionType);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasScalEx_64(UPTKblasHandle_t handle,
                                     int64_t n,
                                     const void *alpha,
                                     UPTKDataType alphaType,
                                     void *x,
                                     UPTKDataType xType,
                                     int64_t incx,
                                     UPTKDataType executionType);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasSscal(UPTKblasHandle_t handle, int n, const float *alpha, float *x, int incx);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasSscal_64(UPTKblasHandle_t handle, int64_t n, const float *alpha, float *x, int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDscal(UPTKblasHandle_t handle, int n, const double *alpha, double *x, int incx);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDscal_64(UPTKblasHandle_t handle, int64_t n, const double *alpha, double *x, int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasCscal(UPTKblasHandle_t handle, int n, const cuComplex *alpha, cuComplex *x, int incx);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasCscal_64(UPTKblasHandle_t handle, int64_t n, const cuComplex *alpha, cuComplex *x, int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasCsscal(UPTKblasHandle_t handle, int n, const float *alpha, cuComplex *x, int incx);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasCsscal_64(UPTKblasHandle_t handle, int64_t n, const float *alpha, cuComplex *x, int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasZscal(UPTKblasHandle_t handle, int n, const cuDoubleComplex *alpha, cuDoubleComplex *x, int incx);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasZscal_64(UPTKblasHandle_t handle, int64_t n, const cuDoubleComplex *alpha, cuDoubleComplex *x, int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasZdscal(UPTKblasHandle_t handle, int n, const double *alpha, cuDoubleComplex *x, int incx);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasZdscal_64(UPTKblasHandle_t handle, int64_t n, const double *alpha, cuDoubleComplex *x, int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasAxpyEx(UPTKblasHandle_t handle,
                                  int n,
                                  const void *alpha,
                                  UPTKDataType alphaType,
                                  const void *x,
                                  UPTKDataType xType,
                                  int incx,
                                  void *y,
                                  UPTKDataType yType,
                                  int incy,
                                  UPTKDataType executiontype);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasAxpyEx_64(UPTKblasHandle_t handle,
                                     int64_t n,
                                     const void *alpha,
                                     UPTKDataType alphaType,
                                     const void *x,
                                     UPTKDataType xType,
                                     int64_t incx,
                                     void *y,
                                     UPTKDataType yType,
                                     int64_t incy,
                                     UPTKDataType executiontype);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasSaxpy(UPTKblasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSaxpy_64(
      UPTKblasHandle_t handle, int64_t n, const float *alpha, const float *x, int64_t incx, float *y, int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDaxpy(UPTKblasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDaxpy_64(
      UPTKblasHandle_t handle, int64_t n, const double *alpha, const double *x, int64_t incx, double *y, int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCaxpy(
      UPTKblasHandle_t handle, int n, const cuComplex *alpha, const cuComplex *x, int incx, cuComplex *y, int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCaxpy_64(UPTKblasHandle_t handle,
                                    int64_t n,
                                    const cuComplex *alpha,
                                    const cuComplex *x,
                                    int64_t incx,
                                    cuComplex *y,
                                    int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZaxpy(UPTKblasHandle_t handle,
                                 int n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *x,
                                 int incx,
                                 cuDoubleComplex *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZaxpy_64(UPTKblasHandle_t handle,
                                    int64_t n,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *x,
                                    int64_t incx,
                                    cuDoubleComplex *y,
                                    int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasScopy(UPTKblasHandle_t handle, int n, const float *x, int incx, float *y, int incy);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasScopy_64(UPTKblasHandle_t handle, int64_t n, const float *x, int64_t incx, float *y, int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDcopy(UPTKblasHandle_t handle, int n, const double *x, int incx, double *y, int incy);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDcopy_64(UPTKblasHandle_t handle, int64_t n, const double *x, int64_t incx, double *y, int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasCcopy(UPTKblasHandle_t handle, int n, const cuComplex *x, int incx, cuComplex *y, int incy);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasCcopy_64(UPTKblasHandle_t handle, int64_t n, const cuComplex *x, int64_t incx, cuComplex *y, int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasZcopy(UPTKblasHandle_t handle, int n, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZcopy_64(
      UPTKblasHandle_t handle, int64_t n, const cuDoubleComplex *x, int64_t incx, cuDoubleComplex *y, int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasSswap(UPTKblasHandle_t handle, int n, float *x, int incx, float *y, int incy);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasSswap_64(UPTKblasHandle_t handle, int64_t n, float *x, int64_t incx, float *y, int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDswap(UPTKblasHandle_t handle, int n, double *x, int incx, double *y, int incy);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDswap_64(UPTKblasHandle_t handle, int64_t n, double *x, int64_t incx, double *y, int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasCswap(UPTKblasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasCswap_64(UPTKblasHandle_t handle, int64_t n, cuComplex *x, int64_t incx, cuComplex *y, int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasZswap(UPTKblasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasZswap_64(UPTKblasHandle_t handle, int64_t n, cuDoubleComplex *x, int64_t incx, cuDoubleComplex *y, int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasIsamax(UPTKblasHandle_t handle, int n, const float *x, int incx, int *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasIsamax_64(UPTKblasHandle_t handle, int64_t n, const float *x, int64_t incx, int64_t *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasIdamax(UPTKblasHandle_t handle, int n, const double *x, int incx, int *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasIdamax_64(UPTKblasHandle_t handle, int64_t n, const double *x, int64_t incx, int64_t *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasIcamax(UPTKblasHandle_t handle, int n, const cuComplex *x, int incx, int *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasIcamax_64(UPTKblasHandle_t handle, int64_t n, const cuComplex *x, int64_t incx, int64_t *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasIzamax(UPTKblasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasIzamax_64(UPTKblasHandle_t handle, int64_t n, const cuDoubleComplex *x, int64_t incx, int64_t *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasIsamin(UPTKblasHandle_t handle, int n, const float *x, int incx, int *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasIsamin_64(UPTKblasHandle_t handle, int64_t n, const float *x, int64_t incx, int64_t *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasIdamin(UPTKblasHandle_t handle, int n, const double *x, int incx, int *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasIdamin_64(UPTKblasHandle_t handle, int64_t n, const double *x, int64_t incx, int64_t *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasIcamin(UPTKblasHandle_t handle, int n, const cuComplex *x, int incx, int *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasIcamin_64(UPTKblasHandle_t handle, int64_t n, const cuComplex *x, int64_t incx, int64_t *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasIzamin(UPTKblasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasIzamin_64(UPTKblasHandle_t handle, int64_t n, const cuDoubleComplex *x, int64_t incx, int64_t *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasSasum(UPTKblasHandle_t handle, int n, const float *x, int incx, float *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasSasum_64(UPTKblasHandle_t handle, int64_t n, const float *x, int64_t incx, float *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDasum(UPTKblasHandle_t handle, int n, const double *x, int incx, double *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDasum_64(UPTKblasHandle_t handle, int64_t n, const double *x, int64_t incx, double *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasScasum(UPTKblasHandle_t handle, int n, const cuComplex *x, int incx, float *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasScasum_64(UPTKblasHandle_t handle, int64_t n, const cuComplex *x, int64_t incx, float *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDzasum(UPTKblasHandle_t handle, int n, const cuDoubleComplex *x, int incx, double *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDzasum_64(UPTKblasHandle_t handle, int64_t n, const cuDoubleComplex *x, int64_t incx, double *result);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasSrot(UPTKblasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float *c, const float *s);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSrot_64(
      UPTKblasHandle_t handle, int64_t n, float *x, int64_t incx, float *y, int64_t incy, const float *c, const float *s);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDrot(UPTKblasHandle_t handle, int n, double *x, int incx, double *y, int incy, const double *c, const double *s);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDrot_64(UPTKblasHandle_t handle,
                                   int64_t n,
                                   double *x,
                                   int64_t incx,
                                   double *y,
                                   int64_t incy,
                                   const double *c,
                                   const double *s);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCrot(
      UPTKblasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy, const float *c, const cuComplex *s);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCrot_64(UPTKblasHandle_t handle,
                                   int64_t n,
                                   cuComplex *x,
                                   int64_t incx,
                                   cuComplex *y,
                                   int64_t incy,
                                   const float *c,
                                   const cuComplex *s);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCsrot(
      UPTKblasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy, const float *c, const float *s);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCsrot_64(UPTKblasHandle_t handle,
                                    int64_t n,
                                    cuComplex *x,
                                    int64_t incx,
                                    cuComplex *y,
                                    int64_t incy,
                                    const float *c,
                                    const float *s);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZrot(UPTKblasHandle_t handle,
                                int n,
                                cuDoubleComplex *x,
                                int incx,
                                cuDoubleComplex *y,
                                int incy,
                                const double *c,
                                const cuDoubleComplex *s);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZrot_64(UPTKblasHandle_t handle,
                                   int64_t n,
                                   cuDoubleComplex *x,
                                   int64_t incx,
                                   cuDoubleComplex *y,
                                   int64_t incy,
                                   const double *c,
                                   const cuDoubleComplex *s);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZdrot(UPTKblasHandle_t handle,
                                 int n,
                                 cuDoubleComplex *x,
                                 int incx,
                                 cuDoubleComplex *y,
                                 int incy,
                                 const double *c,
                                 const double *s);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZdrot_64(UPTKblasHandle_t handle,
                                    int64_t n,
                                    cuDoubleComplex *x,
                                    int64_t incx,
                                    cuDoubleComplex *y,
                                    int64_t incy,
                                    const double *c,
                                    const double *s);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasRotEx(UPTKblasHandle_t handle,
                                 int n,
                                 void *x,
                                 UPTKDataType xType,
                                 int incx,
                                 void *y,
                                 UPTKDataType yType,
                                 int incy,
                                 const void *c,
                                 const void *s,
                                 UPTKDataType csType,
                                 UPTKDataType executiontype);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasRotEx_64(UPTKblasHandle_t handle,
                                    int64_t n,
                                    void *x,
                                    UPTKDataType xType,
                                    int64_t incx,
                                    void *y,
                                    UPTKDataType yType,
                                    int64_t incy,
                                    const void *c,
                                    const void *s,
                                    UPTKDataType csType,
                                    UPTKDataType executiontype);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSrotg(UPTKblasHandle_t handle, float *a, float *b, float *c, float *s);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDrotg(UPTKblasHandle_t handle, double *a, double *b, double *c, double *s);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasCrotg(UPTKblasHandle_t handle, cuComplex *a, cuComplex *b, float *c, cuComplex *s);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasZrotg(UPTKblasHandle_t handle, cuDoubleComplex *a, cuDoubleComplex *b, double *c, cuDoubleComplex *s);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasSrotm(UPTKblasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float *param);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasSrotm_64(UPTKblasHandle_t handle, int64_t n, float *x, int64_t incx, float *y, int64_t incy, const float *param);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDrotm(UPTKblasHandle_t handle, int n, double *x, int incx, double *y, int incy, const double *param);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDrotm_64(
      UPTKblasHandle_t handle, int64_t n, double *x, int64_t incx, double *y, int64_t incy, const double *param);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasSrotmg(UPTKblasHandle_t handle, float *d1, float *d2, float *x1, const float *y1, float *param);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDrotmg(UPTKblasHandle_t handle, double *d1, double *d2, double *x1, const double *y1, double *param);

  /* --------------- CUBLAS BLAS2 Functions  ---------------- */

  /* GEMV */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemv(UPTKblasHandle_t handle,
                                 UPTKblasOperation_t trans,
                                 int m,
                                 int n,
                                 const float *alpha,
                                 const float *A,
                                 int lda,
                                 const float *x,
                                 int incx,
                                 const float *beta,
                                 float *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemv_64(UPTKblasHandle_t handle,
                                    UPTKblasOperation_t trans,
                                    int64_t m,
                                    int64_t n,
                                    const float *alpha,
                                    const float *A,
                                    int64_t lda,
                                    const float *x,
                                    int64_t incx,
                                    const float *beta,
                                    float *y,
                                    int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemv(UPTKblasHandle_t handle,
                                 UPTKblasOperation_t trans,
                                 int m,
                                 int n,
                                 const double *alpha,
                                 const double *A,
                                 int lda,
                                 const double *x,
                                 int incx,
                                 const double *beta,
                                 double *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemv_64(UPTKblasHandle_t handle,
                                    UPTKblasOperation_t trans,
                                    int64_t m,
                                    int64_t n,
                                    const double *alpha,
                                    const double *A,
                                    int64_t lda,
                                    const double *x,
                                    int64_t incx,
                                    const double *beta,
                                    double *y,
                                    int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemv(UPTKblasHandle_t handle,
                                 UPTKblasOperation_t trans,
                                 int m,
                                 int n,
                                 const cuComplex *alpha,
                                 const cuComplex *A,
                                 int lda,
                                 const cuComplex *x,
                                 int incx,
                                 const cuComplex *beta,
                                 cuComplex *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemv_64(UPTKblasHandle_t handle,
                                    UPTKblasOperation_t trans,
                                    int64_t m,
                                    int64_t n,
                                    const cuComplex *alpha,
                                    const cuComplex *A,
                                    int64_t lda,
                                    const cuComplex *x,
                                    int64_t incx,
                                    const cuComplex *beta,
                                    cuComplex *y,
                                    int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemv(UPTKblasHandle_t handle,
                                 UPTKblasOperation_t trans,
                                 int m,
                                 int n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A,
                                 int lda,
                                 const cuDoubleComplex *x,
                                 int incx,
                                 const cuDoubleComplex *beta,
                                 cuDoubleComplex *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemv_64(UPTKblasHandle_t handle,
                                    UPTKblasOperation_t trans,
                                    int64_t m,
                                    int64_t n,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *A,
                                    int64_t lda,
                                    const cuDoubleComplex *x,
                                    int64_t incx,
                                    const cuDoubleComplex *beta,
                                    cuDoubleComplex *y,
                                    int64_t incy);

  /* GBMV */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgbmv(UPTKblasHandle_t handle,
                                 UPTKblasOperation_t trans,
                                 int m,
                                 int n,
                                 int kl,
                                 int ku,
                                 const float *alpha,
                                 const float *A,
                                 int lda,
                                 const float *x,
                                 int incx,
                                 const float *beta,
                                 float *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgbmv_64(UPTKblasHandle_t handle,
                                    UPTKblasOperation_t trans,
                                    int64_t m,
                                    int64_t n,
                                    int64_t kl,
                                    int64_t ku,
                                    const float *alpha,
                                    const float *A,
                                    int64_t lda,
                                    const float *x,
                                    int64_t incx,
                                    const float *beta,
                                    float *y,
                                    int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgbmv(UPTKblasHandle_t handle,
                                 UPTKblasOperation_t trans,
                                 int m,
                                 int n,
                                 int kl,
                                 int ku,
                                 const double *alpha,
                                 const double *A,
                                 int lda,
                                 const double *x,
                                 int incx,
                                 const double *beta,
                                 double *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgbmv_64(UPTKblasHandle_t handle,
                                    UPTKblasOperation_t trans,
                                    int64_t m,
                                    int64_t n,
                                    int64_t kl,
                                    int64_t ku,
                                    const double *alpha,
                                    const double *A,
                                    int64_t lda,
                                    const double *x,
                                    int64_t incx,
                                    const double *beta,
                                    double *y,
                                    int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgbmv(UPTKblasHandle_t handle,
                                 UPTKblasOperation_t trans,
                                 int m,
                                 int n,
                                 int kl,
                                 int ku,
                                 const cuComplex *alpha,
                                 const cuComplex *A,
                                 int lda,
                                 const cuComplex *x,
                                 int incx,
                                 const cuComplex *beta,
                                 cuComplex *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgbmv_64(UPTKblasHandle_t handle,
                                    UPTKblasOperation_t trans,
                                    int64_t m,
                                    int64_t n,
                                    int64_t kl,
                                    int64_t ku,
                                    const cuComplex *alpha,
                                    const cuComplex *A,
                                    int64_t lda,
                                    const cuComplex *x,
                                    int64_t incx,
                                    const cuComplex *beta,
                                    cuComplex *y,
                                    int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgbmv(UPTKblasHandle_t handle,
                                 UPTKblasOperation_t trans,
                                 int m,
                                 int n,
                                 int kl,
                                 int ku,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A,
                                 int lda,
                                 const cuDoubleComplex *x,
                                 int incx,
                                 const cuDoubleComplex *beta,
                                 cuDoubleComplex *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgbmv_64(UPTKblasHandle_t handle,
                                    UPTKblasOperation_t trans,
                                    int64_t m,
                                    int64_t n,
                                    int64_t kl,
                                    int64_t ku,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *A,
                                    int64_t lda,
                                    const cuDoubleComplex *x,
                                    int64_t incx,
                                    const cuDoubleComplex *beta,
                                    cuDoubleComplex *y,
                                    int64_t incy);

  /* TRMV */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasStrmv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 const float *A,
                                 int lda,
                                 float *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasStrmv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    const float *A,
                                    int64_t lda,
                                    float *x,
                                    int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDtrmv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 const double *A,
                                 int lda,
                                 double *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDtrmv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    const double *A,
                                    int64_t lda,
                                    double *x,
                                    int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCtrmv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 const cuComplex *A,
                                 int lda,
                                 cuComplex *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCtrmv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    const cuComplex *A,
                                    int64_t lda,
                                    cuComplex *x,
                                    int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrmv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 const cuDoubleComplex *A,
                                 int lda,
                                 cuDoubleComplex *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrmv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    const cuDoubleComplex *A,
                                    int64_t lda,
                                    cuDoubleComplex *x,
                                    int64_t incx);

  /* TBMV */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasStbmv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 int k,
                                 const float *A,
                                 int lda,
                                 float *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasStbmv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    int64_t k,
                                    const float *A,
                                    int64_t lda,
                                    float *x,
                                    int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDtbmv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 int k,
                                 const double *A,
                                 int lda,
                                 double *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDtbmv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    int64_t k,
                                    const double *A,
                                    int64_t lda,
                                    double *x,
                                    int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCtbmv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 int k,
                                 const cuComplex *A,
                                 int lda,
                                 cuComplex *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCtbmv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    int64_t k,
                                    const cuComplex *A,
                                    int64_t lda,
                                    cuComplex *x,
                                    int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZtbmv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 int k,
                                 const cuDoubleComplex *A,
                                 int lda,
                                 cuDoubleComplex *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZtbmv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    int64_t k,
                                    const cuDoubleComplex *A,
                                    int64_t lda,
                                    cuDoubleComplex *x,
                                    int64_t incx);

  /* TPMV */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasStpmv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 const float *AP,
                                 float *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasStpmv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    const float *AP,
                                    float *x,
                                    int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDtpmv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 const double *AP,
                                 double *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDtpmv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    const double *AP,
                                    double *x,
                                    int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCtpmv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 const cuComplex *AP,
                                 cuComplex *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCtpmv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    const cuComplex *AP,
                                    cuComplex *x,
                                    int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZtpmv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 const cuDoubleComplex *AP,
                                 cuDoubleComplex *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZtpmv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    const cuDoubleComplex *AP,
                                    cuDoubleComplex *x,
                                    int64_t incx);

  /* TRSV */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasStrsv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 const float *A,
                                 int lda,
                                 float *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasStrsv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    const float *A,
                                    int64_t lda,
                                    float *x,
                                    int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDtrsv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 const double *A,
                                 int lda,
                                 double *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDtrsv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    const double *A,
                                    int64_t lda,
                                    double *x,
                                    int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCtrsv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 const cuComplex *A,
                                 int lda,
                                 cuComplex *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCtrsv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    const cuComplex *A,
                                    int64_t lda,
                                    cuComplex *x,
                                    int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrsv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 const cuDoubleComplex *A,
                                 int lda,
                                 cuDoubleComplex *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrsv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    const cuDoubleComplex *A,
                                    int64_t lda,
                                    cuDoubleComplex *x,
                                    int64_t incx);

  /* TPSV */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasStpsv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 const float *AP,
                                 float *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasStpsv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    const float *AP,
                                    float *x,
                                    int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDtpsv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 const double *AP,
                                 double *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDtpsv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    const double *AP,
                                    double *x,
                                    int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCtpsv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 const cuComplex *AP,
                                 cuComplex *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCtpsv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    const cuComplex *AP,
                                    cuComplex *x,
                                    int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZtpsv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 const cuDoubleComplex *AP,
                                 cuDoubleComplex *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZtpsv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    const cuDoubleComplex *AP,
                                    cuDoubleComplex *x,
                                    int64_t incx);

  /* TBSV */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasStbsv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 int k,
                                 const float *A,
                                 int lda,
                                 float *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasStbsv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    int64_t k,
                                    const float *A,
                                    int64_t lda,
                                    float *x,
                                    int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDtbsv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 int k,
                                 const double *A,
                                 int lda,
                                 double *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDtbsv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    int64_t k,
                                    const double *A,
                                    int64_t lda,
                                    double *x,
                                    int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCtbsv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 int k,
                                 const cuComplex *A,
                                 int lda,
                                 cuComplex *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCtbsv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    int64_t k,
                                    const cuComplex *A,
                                    int64_t lda,
                                    cuComplex *x,
                                    int64_t incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZtbsv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int n,
                                 int k,
                                 const cuDoubleComplex *A,
                                 int lda,
                                 cuDoubleComplex *x,
                                 int incx);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZtbsv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t n,
                                    int64_t k,
                                    const cuDoubleComplex *A,
                                    int64_t lda,
                                    cuDoubleComplex *x,
                                    int64_t incx);

  /* SYMV/HEMV */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSsymv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 const float *alpha,
                                 const float *A,
                                 int lda,
                                 const float *x,
                                 int incx,
                                 const float *beta,
                                 float *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSsymv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    const float *alpha,
                                    const float *A,
                                    int64_t lda,
                                    const float *x,
                                    int64_t incx,
                                    const float *beta,
                                    float *y,
                                    int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDsymv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 const double *alpha,
                                 const double *A,
                                 int lda,
                                 const double *x,
                                 int incx,
                                 const double *beta,
                                 double *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDsymv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    const double *alpha,
                                    const double *A,
                                    int64_t lda,
                                    const double *x,
                                    int64_t incx,
                                    const double *beta,
                                    double *y,
                                    int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCsymv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 const cuComplex *alpha,
                                 const cuComplex *A,
                                 int lda,
                                 const cuComplex *x,
                                 int incx,
                                 const cuComplex *beta,
                                 cuComplex *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCsymv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    const cuComplex *alpha,
                                    const cuComplex *A,
                                    int64_t lda,
                                    const cuComplex *x,
                                    int64_t incx,
                                    const cuComplex *beta,
                                    cuComplex *y,
                                    int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZsymv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A,
                                 int lda,
                                 const cuDoubleComplex *x,
                                 int incx,
                                 const cuDoubleComplex *beta,
                                 cuDoubleComplex *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZsymv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *A,
                                    int64_t lda,
                                    const cuDoubleComplex *x,
                                    int64_t incx,
                                    const cuDoubleComplex *beta,
                                    cuDoubleComplex *y,
                                    int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasChemv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 const cuComplex *alpha,
                                 const cuComplex *A,
                                 int lda,
                                 const cuComplex *x,
                                 int incx,
                                 const cuComplex *beta,
                                 cuComplex *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasChemv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    const cuComplex *alpha,
                                    const cuComplex *A,
                                    int64_t lda,
                                    const cuComplex *x,
                                    int64_t incx,
                                    const cuComplex *beta,
                                    cuComplex *y,
                                    int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZhemv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A,
                                 int lda,
                                 const cuDoubleComplex *x,
                                 int incx,
                                 const cuDoubleComplex *beta,
                                 cuDoubleComplex *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZhemv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *A,
                                    int64_t lda,
                                    const cuDoubleComplex *x,
                                    int64_t incx,
                                    const cuDoubleComplex *beta,
                                    cuDoubleComplex *y,
                                    int64_t incy);

  /* SBMV/HBMV */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSsbmv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 int k,
                                 const float *alpha,
                                 const float *A,
                                 int lda,
                                 const float *x,
                                 int incx,
                                 const float *beta,
                                 float *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSsbmv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    int64_t k,
                                    const float *alpha,
                                    const float *A,
                                    int64_t lda,
                                    const float *x,
                                    int64_t incx,
                                    const float *beta,
                                    float *y,
                                    int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDsbmv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 int k,
                                 const double *alpha,
                                 const double *A,
                                 int lda,
                                 const double *x,
                                 int incx,
                                 const double *beta,
                                 double *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDsbmv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    int64_t k,
                                    const double *alpha,
                                    const double *A,
                                    int64_t lda,
                                    const double *x,
                                    int64_t incx,
                                    const double *beta,
                                    double *y,
                                    int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasChbmv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 int k,
                                 const cuComplex *alpha,
                                 const cuComplex *A,
                                 int lda,
                                 const cuComplex *x,
                                 int incx,
                                 const cuComplex *beta,
                                 cuComplex *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasChbmv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    int64_t k,
                                    const cuComplex *alpha,
                                    const cuComplex *A,
                                    int64_t lda,
                                    const cuComplex *x,
                                    int64_t incx,
                                    const cuComplex *beta,
                                    cuComplex *y,
                                    int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZhbmv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 int k,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A,
                                 int lda,
                                 const cuDoubleComplex *x,
                                 int incx,
                                 const cuDoubleComplex *beta,
                                 cuDoubleComplex *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZhbmv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    int64_t k,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *A,
                                    int64_t lda,
                                    const cuDoubleComplex *x,
                                    int64_t incx,
                                    const cuDoubleComplex *beta,
                                    cuDoubleComplex *y,
                                    int64_t incy);

  /* SPMV/HPMV */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSspmv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 const float *alpha,
                                 const float *AP,
                                 const float *x,
                                 int incx,
                                 const float *beta,
                                 float *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSspmv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    const float *alpha,
                                    const float *AP,
                                    const float *x,
                                    int64_t incx,
                                    const float *beta,
                                    float *y,
                                    int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDspmv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 const double *alpha,
                                 const double *AP,
                                 const double *x,
                                 int incx,
                                 const double *beta,
                                 double *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDspmv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    const double *alpha,
                                    const double *AP,
                                    const double *x,
                                    int64_t incx,
                                    const double *beta,
                                    double *y,
                                    int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasChpmv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 const cuComplex *alpha,
                                 const cuComplex *AP,
                                 const cuComplex *x,
                                 int incx,
                                 const cuComplex *beta,
                                 cuComplex *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasChpmv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    const cuComplex *alpha,
                                    const cuComplex *AP,
                                    const cuComplex *x,
                                    int64_t incx,
                                    const cuComplex *beta,
                                    cuComplex *y,
                                    int64_t incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZhpmv(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *AP,
                                 const cuDoubleComplex *x,
                                 int incx,
                                 const cuDoubleComplex *beta,
                                 cuDoubleComplex *y,
                                 int incy);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZhpmv_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *AP,
                                    const cuDoubleComplex *x,
                                    int64_t incx,
                                    const cuDoubleComplex *beta,
                                    cuDoubleComplex *y,
                                    int64_t incy);

  /* GER */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSger(UPTKblasHandle_t handle,
                                int m,
                                int n,
                                const float *alpha,
                                const float *x,
                                int incx,
                                const float *y,
                                int incy,
                                float *A,
                                int lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSger_64(UPTKblasHandle_t handle,
                                   int64_t m,
                                   int64_t n,
                                   const float *alpha,
                                   const float *x,
                                   int64_t incx,
                                   const float *y,
                                   int64_t incy,
                                   float *A,
                                   int64_t lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDger(UPTKblasHandle_t handle,
                                int m,
                                int n,
                                const double *alpha,
                                const double *x,
                                int incx,
                                const double *y,
                                int incy,
                                double *A,
                                int lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDger_64(UPTKblasHandle_t handle,
                                   int64_t m,
                                   int64_t n,
                                   const double *alpha,
                                   const double *x,
                                   int64_t incx,
                                   const double *y,
                                   int64_t incy,
                                   double *A,
                                   int64_t lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgeru(UPTKblasHandle_t handle,
                                 int m,
                                 int n,
                                 const cuComplex *alpha,
                                 const cuComplex *x,
                                 int incx,
                                 const cuComplex *y,
                                 int incy,
                                 cuComplex *A,
                                 int lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgeru_64(UPTKblasHandle_t handle,
                                    int64_t m,
                                    int64_t n,
                                    const cuComplex *alpha,
                                    const cuComplex *x,
                                    int64_t incx,
                                    const cuComplex *y,
                                    int64_t incy,
                                    cuComplex *A,
                                    int64_t lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgerc(UPTKblasHandle_t handle,
                                 int m,
                                 int n,
                                 const cuComplex *alpha,
                                 const cuComplex *x,
                                 int incx,
                                 const cuComplex *y,
                                 int incy,
                                 cuComplex *A,
                                 int lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgerc_64(UPTKblasHandle_t handle,
                                    int64_t m,
                                    int64_t n,
                                    const cuComplex *alpha,
                                    const cuComplex *x,
                                    int64_t incx,
                                    const cuComplex *y,
                                    int64_t incy,
                                    cuComplex *A,
                                    int64_t lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgeru(UPTKblasHandle_t handle,
                                 int m,
                                 int n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *x,
                                 int incx,
                                 const cuDoubleComplex *y,
                                 int incy,
                                 cuDoubleComplex *A,
                                 int lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgeru_64(UPTKblasHandle_t handle,
                                    int64_t m,
                                    int64_t n,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *x,
                                    int64_t incx,
                                    const cuDoubleComplex *y,
                                    int64_t incy,
                                    cuDoubleComplex *A,
                                    int64_t lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgerc(UPTKblasHandle_t handle,
                                 int m,
                                 int n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *x,
                                 int incx,
                                 const cuDoubleComplex *y,
                                 int incy,
                                 cuDoubleComplex *A,
                                 int lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgerc_64(UPTKblasHandle_t handle,
                                    int64_t m,
                                    int64_t n,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *x,
                                    int64_t incx,
                                    const cuDoubleComplex *y,
                                    int64_t incy,
                                    cuDoubleComplex *A,
                                    int64_t lda);

  /* SYR/HER */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSsyr(UPTKblasHandle_t handle,
                                UPTKblasFillMode_t uplo,
                                int n,
                                const float *alpha,
                                const float *x,
                                int incx,
                                float *A,
                                int lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSsyr_64(UPTKblasHandle_t handle,
                                   UPTKblasFillMode_t uplo,
                                   int64_t n,
                                   const float *alpha,
                                   const float *x,
                                   int64_t incx,
                                   float *A,
                                   int64_t lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDsyr(UPTKblasHandle_t handle,
                                UPTKblasFillMode_t uplo,
                                int n,
                                const double *alpha,
                                const double *x,
                                int incx,
                                double *A,
                                int lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDsyr_64(UPTKblasHandle_t handle,
                                   UPTKblasFillMode_t uplo,
                                   int64_t n,
                                   const double *alpha,
                                   const double *x,
                                   int64_t incx,
                                   double *A,
                                   int64_t lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyr(UPTKblasHandle_t handle,
                                UPTKblasFillMode_t uplo,
                                int n,
                                const cuComplex *alpha,
                                const cuComplex *x,
                                int incx,
                                cuComplex *A,
                                int lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyr_64(UPTKblasHandle_t handle,
                                   UPTKblasFillMode_t uplo,
                                   int64_t n,
                                   const cuComplex *alpha,
                                   const cuComplex *x,
                                   int64_t incx,
                                   cuComplex *A,
                                   int64_t lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZsyr(UPTKblasHandle_t handle,
                                UPTKblasFillMode_t uplo,
                                int n,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *x,
                                int incx,
                                cuDoubleComplex *A,
                                int lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZsyr_64(UPTKblasHandle_t handle,
                                   UPTKblasFillMode_t uplo,
                                   int64_t n,
                                   const cuDoubleComplex *alpha,
                                   const cuDoubleComplex *x,
                                   int64_t incx,
                                   cuDoubleComplex *A,
                                   int64_t lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCher(UPTKblasHandle_t handle,
                                UPTKblasFillMode_t uplo,
                                int n,
                                const float *alpha,
                                const cuComplex *x,
                                int incx,
                                cuComplex *A,
                                int lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCher_64(UPTKblasHandle_t handle,
                                   UPTKblasFillMode_t uplo,
                                   int64_t n,
                                   const float *alpha,
                                   const cuComplex *x,
                                   int64_t incx,
                                   cuComplex *A,
                                   int64_t lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZher(UPTKblasHandle_t handle,
                                UPTKblasFillMode_t uplo,
                                int n,
                                const double *alpha,
                                const cuDoubleComplex *x,
                                int incx,
                                cuDoubleComplex *A,
                                int lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZher_64(UPTKblasHandle_t handle,
                                   UPTKblasFillMode_t uplo,
                                   int64_t n,
                                   const double *alpha,
                                   const cuDoubleComplex *x,
                                   int64_t incx,
                                   cuDoubleComplex *A,
                                   int64_t lda);

  /* SPR/HPR */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSspr(
      UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, float *AP);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSspr_64(UPTKblasHandle_t handle,
                                   UPTKblasFillMode_t uplo,
                                   int64_t n,
                                   const float *alpha,
                                   const float *x,
                                   int64_t incx,
                                   float *AP);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDspr(
      UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, double *AP);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDspr_64(UPTKblasHandle_t handle,
                                   UPTKblasFillMode_t uplo,
                                   int64_t n,
                                   const double *alpha,
                                   const double *x,
                                   int64_t incx,
                                   double *AP);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasChpr(UPTKblasHandle_t handle,
                                UPTKblasFillMode_t uplo,
                                int n,
                                const float *alpha,
                                const cuComplex *x,
                                int incx,
                                cuComplex *AP);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasChpr_64(UPTKblasHandle_t handle,
                                   UPTKblasFillMode_t uplo,
                                   int64_t n,
                                   const float *alpha,
                                   const cuComplex *x,
                                   int64_t incx,
                                   cuComplex *AP);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZhpr(UPTKblasHandle_t handle,
                                UPTKblasFillMode_t uplo,
                                int n,
                                const double *alpha,
                                const cuDoubleComplex *x,
                                int incx,
                                cuDoubleComplex *AP);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZhpr_64(UPTKblasHandle_t handle,
                                   UPTKblasFillMode_t uplo,
                                   int64_t n,
                                   const double *alpha,
                                   const cuDoubleComplex *x,
                                   int64_t incx,
                                   cuDoubleComplex *AP);

  /* SYR2/HER2 */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSsyr2(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 const float *alpha,
                                 const float *x,
                                 int incx,
                                 const float *y,
                                 int incy,
                                 float *A,
                                 int lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSsyr2_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    const float *alpha,
                                    const float *x,
                                    int64_t incx,
                                    const float *y,
                                    int64_t incy,
                                    float *A,
                                    int64_t lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDsyr2(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 const double *alpha,
                                 const double *x,
                                 int incx,
                                 const double *y,
                                 int incy,
                                 double *A,
                                 int lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDsyr2_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    const double *alpha,
                                    const double *x,
                                    int64_t incx,
                                    const double *y,
                                    int64_t incy,
                                    double *A,
                                    int64_t lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyr2(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 const cuComplex *alpha,
                                 const cuComplex *x,
                                 int incx,
                                 const cuComplex *y,
                                 int incy,
                                 cuComplex *A,
                                 int lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyr2_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    const cuComplex *alpha,
                                    const cuComplex *x,
                                    int64_t incx,
                                    const cuComplex *y,
                                    int64_t incy,
                                    cuComplex *A,
                                    int64_t lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZsyr2(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *x,
                                 int incx,
                                 const cuDoubleComplex *y,
                                 int incy,
                                 cuDoubleComplex *A,
                                 int lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZsyr2_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *x,
                                    int64_t incx,
                                    const cuDoubleComplex *y,
                                    int64_t incy,
                                    cuDoubleComplex *A,
                                    int64_t lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCher2(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 const cuComplex *alpha,
                                 const cuComplex *x,
                                 int incx,
                                 const cuComplex *y,
                                 int incy,
                                 cuComplex *A,
                                 int lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCher2_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    const cuComplex *alpha,
                                    const cuComplex *x,
                                    int64_t incx,
                                    const cuComplex *y,
                                    int64_t incy,
                                    cuComplex *A,
                                    int64_t lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZher2(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *x,
                                 int incx,
                                 const cuDoubleComplex *y,
                                 int incy,
                                 cuDoubleComplex *A,
                                 int lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZher2_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *x,
                                    int64_t incx,
                                    const cuDoubleComplex *y,
                                    int64_t incy,
                                    cuDoubleComplex *A,
                                    int64_t lda);

  /* SPR2/HPR2 */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSspr2(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 const float *alpha,
                                 const float *x,
                                 int incx,
                                 const float *y,
                                 int incy,
                                 float *AP);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSspr2_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    const float *alpha,
                                    const float *x,
                                    int64_t incx,
                                    const float *y,
                                    int64_t incy,
                                    float *AP);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDspr2(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 const double *alpha,
                                 const double *x,
                                 int incx,
                                 const double *y,
                                 int incy,
                                 double *AP);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDspr2_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    const double *alpha,
                                    const double *x,
                                    int64_t incx,
                                    const double *y,
                                    int64_t incy,
                                    double *AP);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasChpr2(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 const cuComplex *alpha,
                                 const cuComplex *x,
                                 int incx,
                                 const cuComplex *y,
                                 int incy,
                                 cuComplex *AP);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasChpr2_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    const cuComplex *alpha,
                                    const cuComplex *x,
                                    int64_t incx,
                                    const cuComplex *y,
                                    int64_t incy,
                                    cuComplex *AP);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZhpr2(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 int n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *x,
                                 int incx,
                                 const cuDoubleComplex *y,
                                 int incy,
                                 cuDoubleComplex *AP);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZhpr2_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    int64_t n,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *x,
                                    int64_t incx,
                                    const cuDoubleComplex *y,
                                    int64_t incy,
                                    cuDoubleComplex *AP);

  /* BATCH GEMV */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemvBatched(UPTKblasHandle_t handle,
                                        UPTKblasOperation_t trans,
                                        int m,
                                        int n,
                                        const float *alpha,
                                        const float *const Aarray[],
                                        int lda,
                                        const float *const xarray[],
                                        int incx,
                                        const float *beta,
                                        float *const yarray[],
                                        int incy,
                                        int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemvBatched_64(UPTKblasHandle_t handle,
                                           UPTKblasOperation_t trans,
                                           int64_t m,
                                           int64_t n,
                                           const float *alpha,
                                           const float *const Aarray[],
                                           int64_t lda,
                                           const float *const xarray[],
                                           int64_t incx,
                                           const float *beta,
                                           float *const yarray[],
                                           int64_t incy,
                                           int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemvBatched(UPTKblasHandle_t handle,
                                        UPTKblasOperation_t trans,
                                        int m,
                                        int n,
                                        const double *alpha,
                                        const double *const Aarray[],
                                        int lda,
                                        const double *const xarray[],
                                        int incx,
                                        const double *beta,
                                        double *const yarray[],
                                        int incy,
                                        int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemvBatched_64(UPTKblasHandle_t handle,
                                           UPTKblasOperation_t trans,
                                           int64_t m,
                                           int64_t n,
                                           const double *alpha,
                                           const double *const Aarray[],
                                           int64_t lda,
                                           const double *const xarray[],
                                           int64_t incx,
                                           const double *beta,
                                           double *const yarray[],
                                           int64_t incy,
                                           int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemvBatched(UPTKblasHandle_t handle,
                                        UPTKblasOperation_t trans,
                                        int m,
                                        int n,
                                        const cuComplex *alpha,
                                        const cuComplex *const Aarray[],
                                        int lda,
                                        const cuComplex *const xarray[],
                                        int incx,
                                        const cuComplex *beta,
                                        cuComplex *const yarray[],
                                        int incy,
                                        int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemvBatched_64(UPTKblasHandle_t handle,
                                           UPTKblasOperation_t trans,
                                           int64_t m,
                                           int64_t n,
                                           const cuComplex *alpha,
                                           const cuComplex *const Aarray[],
                                           int64_t lda,
                                           const cuComplex *const xarray[],
                                           int64_t incx,
                                           const cuComplex *beta,
                                           cuComplex *const yarray[],
                                           int64_t incy,
                                           int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemvBatched(UPTKblasHandle_t handle,
                                        UPTKblasOperation_t trans,
                                        int m,
                                        int n,
                                        const cuDoubleComplex *alpha,
                                        const cuDoubleComplex *const Aarray[],
                                        int lda,
                                        const cuDoubleComplex *const xarray[],
                                        int incx,
                                        const cuDoubleComplex *beta,
                                        cuDoubleComplex *const yarray[],
                                        int incy,
                                        int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemvBatched_64(UPTKblasHandle_t handle,
                                           UPTKblasOperation_t trans,
                                           int64_t m,
                                           int64_t n,
                                           const cuDoubleComplex *alpha,
                                           const cuDoubleComplex *const Aarray[],
                                           int64_t lda,
                                           const cuDoubleComplex *const xarray[],
                                           int64_t incx,
                                           const cuDoubleComplex *beta,
                                           cuDoubleComplex *const yarray[],
                                           int64_t incy,
                                           int64_t batchCount);

#if defined(__cplusplus)

  UPTKBLASAPI UPTKblasStatus_t UPTKblasHSHgemvBatched(UPTKblasHandle_t handle,
                                          UPTKblasOperation_t trans,
                                          int m,
                                          int n,
                                          const float *alpha,
                                          const __half *const Aarray[],
                                          int lda,
                                          const __half *const xarray[],
                                          int incx,
                                          const float *beta,
                                          __half *const yarray[],
                                          int incy,
                                          int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasHSHgemvBatched_64(UPTKblasHandle_t handle,
                                             UPTKblasOperation_t trans,
                                             int64_t m,
                                             int64_t n,
                                             const float *alpha,
                                             const __half *const Aarray[],
                                             int64_t lda,
                                             const __half *const xarray[],
                                             int64_t incx,
                                             const float *beta,
                                             __half *const yarray[],
                                             int64_t incy,
                                             int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasHSSgemvBatched(UPTKblasHandle_t handle,
                                          UPTKblasOperation_t trans,
                                          int m,
                                          int n,
                                          const float *alpha,
                                          const __half *const Aarray[],
                                          int lda,
                                          const __half *const xarray[],
                                          int incx,
                                          const float *beta,
                                          float *const yarray[],
                                          int incy,
                                          int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasHSSgemvBatched_64(UPTKblasHandle_t handle,
                                             UPTKblasOperation_t trans,
                                             int64_t m,
                                             int64_t n,
                                             const float *alpha,
                                             const __half *const Aarray[],
                                             int64_t lda,
                                             const __half *const xarray[],
                                             int64_t incx,
                                             const float *beta,
                                             float *const yarray[],
                                             int64_t incy,
                                             int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasTSTgemvBatched(UPTKblasHandle_t handle,
                                          UPTKblasOperation_t trans,
                                          int m,
                                          int n,
                                          const float *alpha,
                                          const __nv_bfloat16 *const Aarray[],
                                          int lda,
                                          const __nv_bfloat16 *const xarray[],
                                          int incx,
                                          const float *beta,
                                          __nv_bfloat16 *const yarray[],
                                          int incy,
                                          int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasTSTgemvBatched_64(UPTKblasHandle_t handle,
                                             UPTKblasOperation_t trans,
                                             int64_t m,
                                             int64_t n,
                                             const float *alpha,
                                             const __nv_bfloat16 *const Aarray[],
                                             int64_t lda,
                                             const __nv_bfloat16 *const xarray[],
                                             int64_t incx,
                                             const float *beta,
                                             __nv_bfloat16 *const yarray[],
                                             int64_t incy,
                                             int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasTSSgemvBatched(UPTKblasHandle_t handle,
                                          UPTKblasOperation_t trans,
                                          int m,
                                          int n,
                                          const float *alpha,
                                          const __nv_bfloat16 *const Aarray[],
                                          int lda,
                                          const __nv_bfloat16 *const xarray[],
                                          int incx,
                                          const float *beta,
                                          float *const yarray[],
                                          int incy,
                                          int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasTSSgemvBatched_64(UPTKblasHandle_t handle,
                                             UPTKblasOperation_t trans,
                                             int64_t m,
                                             int64_t n,
                                             const float *alpha,
                                             const __nv_bfloat16 *const Aarray[],
                                             int64_t lda,
                                             const __nv_bfloat16 *const xarray[],
                                             int64_t incx,
                                             const float *beta,
                                             float *const yarray[],
                                             int64_t incy,
                                             int64_t batchCount);

#endif

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemvStridedBatched(UPTKblasHandle_t handle,
                                               UPTKblasOperation_t trans,
                                               int m,
                                               int n,
                                               const float *alpha,
                                               const float *A,
                                               int lda,
                                               long long int strideA,
                                               const float *x,
                                               int incx,
                                               long long int stridex,
                                               const float *beta,
                                               float *y,
                                               int incy,
                                               long long int stridey,
                                               int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemvStridedBatched_64(UPTKblasHandle_t handle,
                                                  UPTKblasOperation_t trans,
                                                  int64_t m,
                                                  int64_t n,
                                                  const float *alpha,
                                                  const float *A,
                                                  int64_t lda,
                                                  long long int strideA,
                                                  const float *x,
                                                  int64_t incx,
                                                  long long int stridex,
                                                  const float *beta,
                                                  float *y,
                                                  int64_t incy,
                                                  long long int stridey,
                                                  int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemvStridedBatched(UPTKblasHandle_t handle,
                                               UPTKblasOperation_t trans,
                                               int m,
                                               int n,
                                               const double *alpha,
                                               const double *A,
                                               int lda,
                                               long long int strideA,
                                               const double *x,
                                               int incx,
                                               long long int stridex,
                                               const double *beta,
                                               double *y,
                                               int incy,
                                               long long int stridey,
                                               int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemvStridedBatched_64(UPTKblasHandle_t handle,
                                                  UPTKblasOperation_t trans,
                                                  int64_t m,
                                                  int64_t n,
                                                  const double *alpha,
                                                  const double *A,
                                                  int64_t lda,
                                                  long long int strideA,
                                                  const double *x,
                                                  int64_t incx,
                                                  long long int stridex,
                                                  const double *beta,
                                                  double *y,
                                                  int64_t incy,
                                                  long long int stridey,
                                                  int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemvStridedBatched(UPTKblasHandle_t handle,
                                               UPTKblasOperation_t trans,
                                               int m,
                                               int n,
                                               const cuComplex *alpha,
                                               const cuComplex *A,
                                               int lda,
                                               long long int strideA,
                                               const cuComplex *x,
                                               int incx,
                                               long long int stridex,
                                               const cuComplex *beta,
                                               cuComplex *y,
                                               int incy,
                                               long long int stridey,
                                               int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemvStridedBatched_64(UPTKblasHandle_t handle,
                                                  UPTKblasOperation_t trans,
                                                  int64_t m,
                                                  int64_t n,
                                                  const cuComplex *alpha,
                                                  const cuComplex *A,
                                                  int64_t lda,
                                                  long long int strideA,
                                                  const cuComplex *x,
                                                  int64_t incx,
                                                  long long int stridex,
                                                  const cuComplex *beta,
                                                  cuComplex *y,
                                                  int64_t incy,
                                                  long long int stridey,
                                                  int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemvStridedBatched(UPTKblasHandle_t handle,
                                               UPTKblasOperation_t trans,
                                               int m,
                                               int n,
                                               const cuDoubleComplex *alpha,
                                               const cuDoubleComplex *A,
                                               int lda,
                                               long long int strideA,
                                               const cuDoubleComplex *x,
                                               int incx,
                                               long long int stridex,
                                               const cuDoubleComplex *beta,
                                               cuDoubleComplex *y,
                                               int incy,
                                               long long int stridey,
                                               int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemvStridedBatched_64(UPTKblasHandle_t handle,
                                                  UPTKblasOperation_t trans,
                                                  int64_t m,
                                                  int64_t n,
                                                  const cuDoubleComplex *alpha,
                                                  const cuDoubleComplex *A,
                                                  int64_t lda,
                                                  long long int strideA,
                                                  const cuDoubleComplex *x,
                                                  int64_t incx,
                                                  long long int stridex,
                                                  const cuDoubleComplex *beta,
                                                  cuDoubleComplex *y,
                                                  int64_t incy,
                                                  long long int stridey,
                                                  int64_t batchCount);

#if defined(__cplusplus)

  UPTKBLASAPI UPTKblasStatus_t UPTKblasHSHgemvStridedBatched(UPTKblasHandle_t handle,
                                                 UPTKblasOperation_t trans,
                                                 int m,
                                                 int n,
                                                 const float *alpha,
                                                 const __half *A,
                                                 int lda,
                                                 long long int strideA,
                                                 const __half *x,
                                                 int incx,
                                                 long long int stridex,
                                                 const float *beta,
                                                 __half *y,
                                                 int incy,
                                                 long long int stridey,
                                                 int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasHSHgemvStridedBatched_64(UPTKblasHandle_t handle,
                                                    UPTKblasOperation_t trans,
                                                    int64_t m,
                                                    int64_t n,
                                                    const float *alpha,
                                                    const __half *A,
                                                    int64_t lda,
                                                    long long int strideA,
                                                    const __half *x,
                                                    int64_t incx,
                                                    long long int stridex,
                                                    const float *beta,
                                                    __half *y,
                                                    int64_t incy,
                                                    long long int stridey,
                                                    int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasHSSgemvStridedBatched(UPTKblasHandle_t handle,
                                                 UPTKblasOperation_t trans,
                                                 int m,
                                                 int n,
                                                 const float *alpha,
                                                 const __half *A,
                                                 int lda,
                                                 long long int strideA,
                                                 const __half *x,
                                                 int incx,
                                                 long long int stridex,
                                                 const float *beta,
                                                 float *y,
                                                 int incy,
                                                 long long int stridey,
                                                 int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasHSSgemvStridedBatched_64(UPTKblasHandle_t handle,
                                                    UPTKblasOperation_t trans,
                                                    int64_t m,
                                                    int64_t n,
                                                    const float *alpha,
                                                    const __half *A,
                                                    int64_t lda,
                                                    long long int strideA,
                                                    const __half *x,
                                                    int64_t incx,
                                                    long long int stridex,
                                                    const float *beta,
                                                    float *y,
                                                    int64_t incy,
                                                    long long int stridey,
                                                    int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasTSTgemvStridedBatched(UPTKblasHandle_t handle,
                                                 UPTKblasOperation_t trans,
                                                 int m,
                                                 int n,
                                                 const float *alpha,
                                                 const __nv_bfloat16 *A,
                                                 int lda,
                                                 long long int strideA,
                                                 const __nv_bfloat16 *x,
                                                 int incx,
                                                 long long int stridex,
                                                 const float *beta,
                                                 __nv_bfloat16 *y,
                                                 int incy,
                                                 long long int stridey,
                                                 int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasTSTgemvStridedBatched_64(UPTKblasHandle_t handle,
                                                    UPTKblasOperation_t trans,
                                                    int64_t m,
                                                    int64_t n,
                                                    const float *alpha,
                                                    const __nv_bfloat16 *A,
                                                    int64_t lda,
                                                    long long int strideA,
                                                    const __nv_bfloat16 *x,
                                                    int64_t incx,
                                                    long long int stridex,
                                                    const float *beta,
                                                    __nv_bfloat16 *y,
                                                    int64_t incy,
                                                    long long int stridey,
                                                    int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasTSSgemvStridedBatched(UPTKblasHandle_t handle,
                                                 UPTKblasOperation_t trans,
                                                 int m,
                                                 int n,
                                                 const float *alpha,
                                                 const __nv_bfloat16 *A,
                                                 int lda,
                                                 long long int strideA,
                                                 const __nv_bfloat16 *x,
                                                 int incx,
                                                 long long int stridex,
                                                 const float *beta,
                                                 float *y,
                                                 int incy,
                                                 long long int stridey,
                                                 int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasTSSgemvStridedBatched_64(UPTKblasHandle_t handle,
                                                    UPTKblasOperation_t trans,
                                                    int64_t m,
                                                    int64_t n,
                                                    const float *alpha,
                                                    const __nv_bfloat16 *A,
                                                    int64_t lda,
                                                    long long int strideA,
                                                    const __nv_bfloat16 *x,
                                                    int64_t incx,
                                                    long long int stridex,
                                                    const float *beta,
                                                    float *y,
                                                    int64_t incy,
                                                    long long int stridey,
                                                    int64_t batchCount);

#endif

  /* ---------------- CUBLAS BLAS3 Functions ---------------- */

  /* GEMM */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemm(UPTKblasHandle_t handle,
                                 UPTKblasOperation_t transa,
                                 UPTKblasOperation_t transb,
                                 int m,
                                 int n,
                                 int k,
                                 const float *alpha,
                                 const float *A,
                                 int lda,
                                 const float *B,
                                 int ldb,
                                 const float *beta,
                                 float *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemm_64(UPTKblasHandle_t handle,
                                    UPTKblasOperation_t transa,
                                    UPTKblasOperation_t transb,
                                    int64_t m,
                                    int64_t n,
                                    int64_t k,
                                    const float *alpha,
                                    const float *A,
                                    int64_t lda,
                                    const float *B,
                                    int64_t ldb,
                                    const float *beta,
                                    float *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemm(UPTKblasHandle_t handle,
                                 UPTKblasOperation_t transa,
                                 UPTKblasOperation_t transb,
                                 int m,
                                 int n,
                                 int k,
                                 const double *alpha,
                                 const double *A,
                                 int lda,
                                 const double *B,
                                 int ldb,
                                 const double *beta,
                                 double *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemm_64(UPTKblasHandle_t handle,
                                    UPTKblasOperation_t transa,
                                    UPTKblasOperation_t transb,
                                    int64_t m,
                                    int64_t n,
                                    int64_t k,
                                    const double *alpha,
                                    const double *A,
                                    int64_t lda,
                                    const double *B,
                                    int64_t ldb,
                                    const double *beta,
                                    double *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemm(UPTKblasHandle_t handle,
                                 UPTKblasOperation_t transa,
                                 UPTKblasOperation_t transb,
                                 int m,
                                 int n,
                                 int k,
                                 const cuComplex *alpha,
                                 const cuComplex *A,
                                 int lda,
                                 const cuComplex *B,
                                 int ldb,
                                 const cuComplex *beta,
                                 cuComplex *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemm_64(UPTKblasHandle_t handle,
                                    UPTKblasOperation_t transa,
                                    UPTKblasOperation_t transb,
                                    int64_t m,
                                    int64_t n,
                                    int64_t k,
                                    const cuComplex *alpha,
                                    const cuComplex *A,
                                    int64_t lda,
                                    const cuComplex *B,
                                    int64_t ldb,
                                    const cuComplex *beta,
                                    cuComplex *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemm3m(UPTKblasHandle_t handle,
                                   UPTKblasOperation_t transa,
                                   UPTKblasOperation_t transb,
                                   int m,
                                   int n,
                                   int k,
                                   const cuComplex *alpha,
                                   const cuComplex *A,
                                   int lda,
                                   const cuComplex *B,
                                   int ldb,
                                   const cuComplex *beta,
                                   cuComplex *C,
                                   int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemm3m_64(UPTKblasHandle_t handle,
                                      UPTKblasOperation_t transa,
                                      UPTKblasOperation_t transb,
                                      int64_t m,
                                      int64_t n,
                                      int64_t k,
                                      const cuComplex *alpha,
                                      const cuComplex *A,
                                      int64_t lda,
                                      const cuComplex *B,
                                      int64_t ldb,
                                      const cuComplex *beta,
                                      cuComplex *C,
                                      int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemm(UPTKblasHandle_t handle,
                                 UPTKblasOperation_t transa,
                                 UPTKblasOperation_t transb,
                                 int m,
                                 int n,
                                 int k,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A,
                                 int lda,
                                 const cuDoubleComplex *B,
                                 int ldb,
                                 const cuDoubleComplex *beta,
                                 cuDoubleComplex *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemm_64(UPTKblasHandle_t handle,
                                    UPTKblasOperation_t transa,
                                    UPTKblasOperation_t transb,
                                    int64_t m,
                                    int64_t n,
                                    int64_t k,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *A,
                                    int64_t lda,
                                    const cuDoubleComplex *B,
                                    int64_t ldb,
                                    const cuDoubleComplex *beta,
                                    cuDoubleComplex *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemm3m(UPTKblasHandle_t handle,
                                   UPTKblasOperation_t transa,
                                   UPTKblasOperation_t transb,
                                   int m,
                                   int n,
                                   int k,
                                   const cuDoubleComplex *alpha,
                                   const cuDoubleComplex *A,
                                   int lda,
                                   const cuDoubleComplex *B,
                                   int ldb,
                                   const cuDoubleComplex *beta,
                                   cuDoubleComplex *C,
                                   int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemm3m_64(UPTKblasHandle_t handle,
                                      UPTKblasOperation_t transa,
                                      UPTKblasOperation_t transb,
                                      int64_t m,
                                      int64_t n,
                                      int64_t k,
                                      const cuDoubleComplex *alpha,
                                      const cuDoubleComplex *A,
                                      int64_t lda,
                                      const cuDoubleComplex *B,
                                      int64_t ldb,
                                      const cuDoubleComplex *beta,
                                      cuDoubleComplex *C,
                                      int64_t ldc);

#if defined(__cplusplus)

  UPTKBLASAPI UPTKblasStatus_t UPTKblasHgemm(UPTKblasHandle_t handle,
                                 UPTKblasOperation_t transa,
                                 UPTKblasOperation_t transb,
                                 int m,
                                 int n,
                                 int k,
                                 const __half *alpha,
                                 const __half *A,
                                 int lda,
                                 const __half *B,
                                 int ldb,
                                 const __half *beta,
                                 __half *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasHgemm_64(UPTKblasHandle_t handle,
                                    UPTKblasOperation_t transa,
                                    UPTKblasOperation_t transb,
                                    int64_t m,
                                    int64_t n,
                                    int64_t k,
                                    const __half *alpha,
                                    const __half *A,
                                    int64_t lda,
                                    const __half *B,
                                    int64_t ldb,
                                    const __half *beta,
                                    __half *C,
                                    int64_t ldc);

#endif

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemmEx(UPTKblasHandle_t handle,
                                   UPTKblasOperation_t transa,
                                   UPTKblasOperation_t transb,
                                   int m,
                                   int n,
                                   int k,
                                   const float *alpha,
                                   const void *A,
                                   UPTKDataType Atype,
                                   int lda,
                                   const void *B,
                                   UPTKDataType Btype,
                                   int ldb,
                                   const float *beta,
                                   void *C,
                                   UPTKDataType Ctype,
                                   int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemmEx_64(UPTKblasHandle_t handle,
                                      UPTKblasOperation_t transa,
                                      UPTKblasOperation_t transb,
                                      int64_t m,
                                      int64_t n,
                                      int64_t k,
                                      const float *alpha,
                                      const void *A,
                                      UPTKDataType Atype,
                                      int64_t lda,
                                      const void *B,
                                      UPTKDataType Btype,
                                      int64_t ldb,
                                      const float *beta,
                                      void *C,
                                      UPTKDataType Ctype,
                                      int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasGemmEx(UPTKblasHandle_t handle,
                                  UPTKblasOperation_t transa,
                                  UPTKblasOperation_t transb,
                                  int m,
                                  int n,
                                  int k,
                                  const void *alpha,
                                  const void *A,
                                  UPTKDataType Atype,
                                  int lda,
                                  const void *B,
                                  UPTKDataType Btype,
                                  int ldb,
                                  const void *beta,
                                  void *C,
                                  UPTKDataType Ctype,
                                  int ldc,
                                  UPTKblasComputeType_t computeType,
                                  UPTKblasGemmAlgo_t algo);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasGemmEx_64(UPTKblasHandle_t handle,
                                     UPTKblasOperation_t transa,
                                     UPTKblasOperation_t transb,
                                     int64_t m,
                                     int64_t n,
                                     int64_t k,
                                     const void *alpha,
                                     const void *A,
                                     UPTKDataType Atype,
                                     int64_t lda,
                                     const void *B,
                                     UPTKDataType Btype,
                                     int64_t ldb,
                                     const void *beta,
                                     void *C,
                                     UPTKDataType Ctype,
                                     int64_t ldc,
                                     UPTKblasComputeType_t computeType,
                                     UPTKblasGemmAlgo_t algo);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemmEx(UPTKblasHandle_t handle,
                                   UPTKblasOperation_t transa,
                                   UPTKblasOperation_t transb,
                                   int m,
                                   int n,
                                   int k,
                                   const cuComplex *alpha,
                                   const void *A,
                                   UPTKDataType Atype,
                                   int lda,
                                   const void *B,
                                   UPTKDataType Btype,
                                   int ldb,
                                   const cuComplex *beta,
                                   void *C,
                                   UPTKDataType Ctype,
                                   int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemmEx_64(UPTKblasHandle_t handle,
                                      UPTKblasOperation_t transa,
                                      UPTKblasOperation_t transb,
                                      int64_t m,
                                      int64_t n,
                                      int64_t k,
                                      const cuComplex *alpha,
                                      const void *A,
                                      UPTKDataType Atype,
                                      int64_t lda,
                                      const void *B,
                                      UPTKDataType Btype,
                                      int64_t ldb,
                                      const cuComplex *beta,
                                      void *C,
                                      UPTKDataType Ctype,
                                      int64_t ldc);

  /* SYRK */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSsyrk(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 int n,
                                 int k,
                                 const float *alpha,
                                 const float *A,
                                 int lda,
                                 const float *beta,
                                 float *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSsyrk_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    int64_t n,
                                    int64_t k,
                                    const float *alpha,
                                    const float *A,
                                    int64_t lda,
                                    const float *beta,
                                    float *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDsyrk(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 int n,
                                 int k,
                                 const double *alpha,
                                 const double *A,
                                 int lda,
                                 const double *beta,
                                 double *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDsyrk_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    int64_t n,
                                    int64_t k,
                                    const double *alpha,
                                    const double *A,
                                    int64_t lda,
                                    const double *beta,
                                    double *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyrk(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 int n,
                                 int k,
                                 const cuComplex *alpha,
                                 const cuComplex *A,
                                 int lda,
                                 const cuComplex *beta,
                                 cuComplex *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyrk_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    int64_t n,
                                    int64_t k,
                                    const cuComplex *alpha,
                                    const cuComplex *A,
                                    int64_t lda,
                                    const cuComplex *beta,
                                    cuComplex *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZsyrk(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 int n,
                                 int k,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A,
                                 int lda,
                                 const cuDoubleComplex *beta,
                                 cuDoubleComplex *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZsyrk_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    int64_t n,
                                    int64_t k,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *A,
                                    int64_t lda,
                                    const cuDoubleComplex *beta,
                                    cuDoubleComplex *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyrkEx(UPTKblasHandle_t handle,
                                   UPTKblasFillMode_t uplo,
                                   UPTKblasOperation_t trans,
                                   int n,
                                   int k,
                                   const cuComplex *alpha,
                                   const void *A,
                                   UPTKDataType Atype,
                                   int lda,
                                   const cuComplex *beta,
                                   void *C,
                                   UPTKDataType Ctype,
                                   int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyrkEx_64(UPTKblasHandle_t handle,
                                      UPTKblasFillMode_t uplo,
                                      UPTKblasOperation_t trans,
                                      int64_t n,
                                      int64_t k,
                                      const cuComplex *alpha,
                                      const void *A,
                                      UPTKDataType Atype,
                                      int64_t lda,
                                      const cuComplex *beta,
                                      void *C,
                                      UPTKDataType Ctype,
                                      int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyrk3mEx(UPTKblasHandle_t handle,
                                     UPTKblasFillMode_t uplo,
                                     UPTKblasOperation_t trans,
                                     int n,
                                     int k,
                                     const cuComplex *alpha,
                                     const void *A,
                                     UPTKDataType Atype,
                                     int lda,
                                     const cuComplex *beta,
                                     void *C,
                                     UPTKDataType Ctype,
                                     int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyrk3mEx_64(UPTKblasHandle_t handle,
                                        UPTKblasFillMode_t uplo,
                                        UPTKblasOperation_t trans,
                                        int64_t n,
                                        int64_t k,
                                        const cuComplex *alpha,
                                        const void *A,
                                        UPTKDataType Atype,
                                        int64_t lda,
                                        const cuComplex *beta,
                                        void *C,
                                        UPTKDataType Ctype,
                                        int64_t ldc);

  /* HERK */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCherk(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 int n,
                                 int k,
                                 const float *alpha,
                                 const cuComplex *A,
                                 int lda,
                                 const float *beta,
                                 cuComplex *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCherk_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    int64_t n,
                                    int64_t k,
                                    const float *alpha,
                                    const cuComplex *A,
                                    int64_t lda,
                                    const float *beta,
                                    cuComplex *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZherk(UPTKblasHandle_t handle,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 int n,
                                 int k,
                                 const double *alpha,
                                 const cuDoubleComplex *A,
                                 int lda,
                                 const double *beta,
                                 cuDoubleComplex *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZherk_64(UPTKblasHandle_t handle,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    int64_t n,
                                    int64_t k,
                                    const double *alpha,
                                    const cuDoubleComplex *A,
                                    int64_t lda,
                                    const double *beta,
                                    cuDoubleComplex *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCherkEx(UPTKblasHandle_t handle,
                                   UPTKblasFillMode_t uplo,
                                   UPTKblasOperation_t trans,
                                   int n,
                                   int k,
                                   const float *alpha,
                                   const void *A,
                                   UPTKDataType Atype,
                                   int lda,
                                   const float *beta,
                                   void *C,
                                   UPTKDataType Ctype,
                                   int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCherkEx_64(UPTKblasHandle_t handle,
                                      UPTKblasFillMode_t uplo,
                                      UPTKblasOperation_t trans,
                                      int64_t n,
                                      int64_t k,
                                      const float *alpha,
                                      const void *A,
                                      UPTKDataType Atype,
                                      int64_t lda,
                                      const float *beta,
                                      void *C,
                                      UPTKDataType Ctype,
                                      int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCherk3mEx(UPTKblasHandle_t handle,
                                     UPTKblasFillMode_t uplo,
                                     UPTKblasOperation_t trans,
                                     int n,
                                     int k,
                                     const float *alpha,
                                     const void *A,
                                     UPTKDataType Atype,
                                     int lda,
                                     const float *beta,
                                     void *C,
                                     UPTKDataType Ctype,
                                     int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCherk3mEx_64(UPTKblasHandle_t handle,
                                        UPTKblasFillMode_t uplo,
                                        UPTKblasOperation_t trans,
                                        int64_t n,
                                        int64_t k,
                                        const float *alpha,
                                        const void *A,
                                        UPTKDataType Atype,
                                        int64_t lda,
                                        const float *beta,
                                        void *C,
                                        UPTKDataType Ctype,
                                        int64_t ldc);

  /* SYR2K / HER2K */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSsyr2k(UPTKblasHandle_t handle,
                                  UPTKblasFillMode_t uplo,
                                  UPTKblasOperation_t trans,
                                  int n,
                                  int k,
                                  const float *alpha,
                                  const float *A,
                                  int lda,
                                  const float *B,
                                  int ldb,
                                  const float *beta,
                                  float *C,
                                  int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSsyr2k_64(UPTKblasHandle_t handle,
                                     UPTKblasFillMode_t uplo,
                                     UPTKblasOperation_t trans,
                                     int64_t n,
                                     int64_t k,
                                     const float *alpha,
                                     const float *A,
                                     int64_t lda,
                                     const float *B,
                                     int64_t ldb,
                                     const float *beta,
                                     float *C,
                                     int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDsyr2k(UPTKblasHandle_t handle,
                                  UPTKblasFillMode_t uplo,
                                  UPTKblasOperation_t trans,
                                  int n,
                                  int k,
                                  const double *alpha,
                                  const double *A,
                                  int lda,
                                  const double *B,
                                  int ldb,
                                  const double *beta,
                                  double *C,
                                  int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDsyr2k_64(UPTKblasHandle_t handle,
                                     UPTKblasFillMode_t uplo,
                                     UPTKblasOperation_t trans,
                                     int64_t n,
                                     int64_t k,
                                     const double *alpha,
                                     const double *A,
                                     int64_t lda,
                                     const double *B,
                                     int64_t ldb,
                                     const double *beta,
                                     double *C,
                                     int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyr2k(UPTKblasHandle_t handle,
                                  UPTKblasFillMode_t uplo,
                                  UPTKblasOperation_t trans,
                                  int n,
                                  int k,
                                  const cuComplex *alpha,
                                  const cuComplex *A,
                                  int lda,
                                  const cuComplex *B,
                                  int ldb,
                                  const cuComplex *beta,
                                  cuComplex *C,
                                  int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyr2k_64(UPTKblasHandle_t handle,
                                     UPTKblasFillMode_t uplo,
                                     UPTKblasOperation_t trans,
                                     int64_t n,
                                     int64_t k,
                                     const cuComplex *alpha,
                                     const cuComplex *A,
                                     int64_t lda,
                                     const cuComplex *B,
                                     int64_t ldb,
                                     const cuComplex *beta,
                                     cuComplex *C,
                                     int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZsyr2k(UPTKblasHandle_t handle,
                                  UPTKblasFillMode_t uplo,
                                  UPTKblasOperation_t trans,
                                  int n,
                                  int k,
                                  const cuDoubleComplex *alpha,
                                  const cuDoubleComplex *A,
                                  int lda,
                                  const cuDoubleComplex *B,
                                  int ldb,
                                  const cuDoubleComplex *beta,
                                  cuDoubleComplex *C,
                                  int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZsyr2k_64(UPTKblasHandle_t handle,
                                     UPTKblasFillMode_t uplo,
                                     UPTKblasOperation_t trans,
                                     int64_t n,
                                     int64_t k,
                                     const cuDoubleComplex *alpha,
                                     const cuDoubleComplex *A,
                                     int64_t lda,
                                     const cuDoubleComplex *B,
                                     int64_t ldb,
                                     const cuDoubleComplex *beta,
                                     cuDoubleComplex *C,
                                     int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCher2k(UPTKblasHandle_t handle,
                                  UPTKblasFillMode_t uplo,
                                  UPTKblasOperation_t trans,
                                  int n,
                                  int k,
                                  const cuComplex *alpha,
                                  const cuComplex *A,
                                  int lda,
                                  const cuComplex *B,
                                  int ldb,
                                  const float *beta,
                                  cuComplex *C,
                                  int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCher2k_64(UPTKblasHandle_t handle,
                                     UPTKblasFillMode_t uplo,
                                     UPTKblasOperation_t trans,
                                     int64_t n,
                                     int64_t k,
                                     const cuComplex *alpha,
                                     const cuComplex *A,
                                     int64_t lda,
                                     const cuComplex *B,
                                     int64_t ldb,
                                     const float *beta,
                                     cuComplex *C,
                                     int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZher2k(UPTKblasHandle_t handle,
                                  UPTKblasFillMode_t uplo,
                                  UPTKblasOperation_t trans,
                                  int n,
                                  int k,
                                  const cuDoubleComplex *alpha,
                                  const cuDoubleComplex *A,
                                  int lda,
                                  const cuDoubleComplex *B,
                                  int ldb,
                                  const double *beta,
                                  cuDoubleComplex *C,
                                  int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZher2k_64(UPTKblasHandle_t handle,
                                     UPTKblasFillMode_t uplo,
                                     UPTKblasOperation_t trans,
                                     int64_t n,
                                     int64_t k,
                                     const cuDoubleComplex *alpha,
                                     const cuDoubleComplex *A,
                                     int64_t lda,
                                     const cuDoubleComplex *B,
                                     int64_t ldb,
                                     const double *beta,
                                     cuDoubleComplex *C,
                                     int64_t ldc);

  /* SYRKX / HERKX */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSsyrkx(UPTKblasHandle_t handle,
                                  UPTKblasFillMode_t uplo,
                                  UPTKblasOperation_t trans,
                                  int n,
                                  int k,
                                  const float *alpha,
                                  const float *A,
                                  int lda,
                                  const float *B,
                                  int ldb,
                                  const float *beta,
                                  float *C,
                                  int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSsyrkx_64(UPTKblasHandle_t handle,
                                     UPTKblasFillMode_t uplo,
                                     UPTKblasOperation_t trans,
                                     int64_t n,
                                     int64_t k,
                                     const float *alpha,
                                     const float *A,
                                     int64_t lda,
                                     const float *B,
                                     int64_t ldb,
                                     const float *beta,
                                     float *C,
                                     int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDsyrkx(UPTKblasHandle_t handle,
                                  UPTKblasFillMode_t uplo,
                                  UPTKblasOperation_t trans,
                                  int n,
                                  int k,
                                  const double *alpha,
                                  const double *A,
                                  int lda,
                                  const double *B,
                                  int ldb,
                                  const double *beta,
                                  double *C,
                                  int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDsyrkx_64(UPTKblasHandle_t handle,
                                     UPTKblasFillMode_t uplo,
                                     UPTKblasOperation_t trans,
                                     int64_t n,
                                     int64_t k,
                                     const double *alpha,
                                     const double *A,
                                     int64_t lda,
                                     const double *B,
                                     int64_t ldb,
                                     const double *beta,
                                     double *C,
                                     int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyrkx(UPTKblasHandle_t handle,
                                  UPTKblasFillMode_t uplo,
                                  UPTKblasOperation_t trans,
                                  int n,
                                  int k,
                                  const cuComplex *alpha,
                                  const cuComplex *A,
                                  int lda,
                                  const cuComplex *B,
                                  int ldb,
                                  const cuComplex *beta,
                                  cuComplex *C,
                                  int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyrkx_64(UPTKblasHandle_t handle,
                                     UPTKblasFillMode_t uplo,
                                     UPTKblasOperation_t trans,
                                     int64_t n,
                                     int64_t k,
                                     const cuComplex *alpha,
                                     const cuComplex *A,
                                     int64_t lda,
                                     const cuComplex *B,
                                     int64_t ldb,
                                     const cuComplex *beta,
                                     cuComplex *C,
                                     int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZsyrkx(UPTKblasHandle_t handle,
                                  UPTKblasFillMode_t uplo,
                                  UPTKblasOperation_t trans,
                                  int n,
                                  int k,
                                  const cuDoubleComplex *alpha,
                                  const cuDoubleComplex *A,
                                  int lda,
                                  const cuDoubleComplex *B,
                                  int ldb,
                                  const cuDoubleComplex *beta,
                                  cuDoubleComplex *C,
                                  int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZsyrkx_64(UPTKblasHandle_t handle,
                                     UPTKblasFillMode_t uplo,
                                     UPTKblasOperation_t trans,
                                     int64_t n,
                                     int64_t k,
                                     const cuDoubleComplex *alpha,
                                     const cuDoubleComplex *A,
                                     int64_t lda,
                                     const cuDoubleComplex *B,
                                     int64_t ldb,
                                     const cuDoubleComplex *beta,
                                     cuDoubleComplex *C,
                                     int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCherkx(UPTKblasHandle_t handle,
                                  UPTKblasFillMode_t uplo,
                                  UPTKblasOperation_t trans,
                                  int n,
                                  int k,
                                  const cuComplex *alpha,
                                  const cuComplex *A,
                                  int lda,
                                  const cuComplex *B,
                                  int ldb,
                                  const float *beta,
                                  cuComplex *C,
                                  int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCherkx_64(UPTKblasHandle_t handle,
                                     UPTKblasFillMode_t uplo,
                                     UPTKblasOperation_t trans,
                                     int64_t n,
                                     int64_t k,
                                     const cuComplex *alpha,
                                     const cuComplex *A,
                                     int64_t lda,
                                     const cuComplex *B,
                                     int64_t ldb,
                                     const float *beta,
                                     cuComplex *C,
                                     int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZherkx(UPTKblasHandle_t handle,
                                  UPTKblasFillMode_t uplo,
                                  UPTKblasOperation_t trans,
                                  int n,
                                  int k,
                                  const cuDoubleComplex *alpha,
                                  const cuDoubleComplex *A,
                                  int lda,
                                  const cuDoubleComplex *B,
                                  int ldb,
                                  const double *beta,
                                  cuDoubleComplex *C,
                                  int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZherkx_64(UPTKblasHandle_t handle,
                                     UPTKblasFillMode_t uplo,
                                     UPTKblasOperation_t trans,
                                     int64_t n,
                                     int64_t k,
                                     const cuDoubleComplex *alpha,
                                     const cuDoubleComplex *A,
                                     int64_t lda,
                                     const cuDoubleComplex *B,
                                     int64_t ldb,
                                     const double *beta,
                                     cuDoubleComplex *C,
                                     int64_t ldc);

  /* SYMM */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSsymm(UPTKblasHandle_t handle,
                                 UPTKblasSideMode_t side,
                                 UPTKblasFillMode_t uplo,
                                 int m,
                                 int n,
                                 const float *alpha,
                                 const float *A,
                                 int lda,
                                 const float *B,
                                 int ldb,
                                 const float *beta,
                                 float *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSsymm_64(UPTKblasHandle_t handle,
                                    UPTKblasSideMode_t side,
                                    UPTKblasFillMode_t uplo,
                                    int64_t m,
                                    int64_t n,
                                    const float *alpha,
                                    const float *A,
                                    int64_t lda,
                                    const float *B,
                                    int64_t ldb,
                                    const float *beta,
                                    float *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDsymm(UPTKblasHandle_t handle,
                                 UPTKblasSideMode_t side,
                                 UPTKblasFillMode_t uplo,
                                 int m,
                                 int n,
                                 const double *alpha,
                                 const double *A,
                                 int lda,
                                 const double *B,
                                 int ldb,
                                 const double *beta,
                                 double *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDsymm_64(UPTKblasHandle_t handle,
                                    UPTKblasSideMode_t side,
                                    UPTKblasFillMode_t uplo,
                                    int64_t m,
                                    int64_t n,
                                    const double *alpha,
                                    const double *A,
                                    int64_t lda,
                                    const double *B,
                                    int64_t ldb,
                                    const double *beta,
                                    double *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCsymm(UPTKblasHandle_t handle,
                                 UPTKblasSideMode_t side,
                                 UPTKblasFillMode_t uplo,
                                 int m,
                                 int n,
                                 const cuComplex *alpha,
                                 const cuComplex *A,
                                 int lda,
                                 const cuComplex *B,
                                 int ldb,
                                 const cuComplex *beta,
                                 cuComplex *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCsymm_64(UPTKblasHandle_t handle,
                                    UPTKblasSideMode_t side,
                                    UPTKblasFillMode_t uplo,
                                    int64_t m,
                                    int64_t n,
                                    const cuComplex *alpha,
                                    const cuComplex *A,
                                    int64_t lda,
                                    const cuComplex *B,
                                    int64_t ldb,
                                    const cuComplex *beta,
                                    cuComplex *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZsymm(UPTKblasHandle_t handle,
                                 UPTKblasSideMode_t side,
                                 UPTKblasFillMode_t uplo,
                                 int m,
                                 int n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A,
                                 int lda,
                                 const cuDoubleComplex *B,
                                 int ldb,
                                 const cuDoubleComplex *beta,
                                 cuDoubleComplex *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZsymm_64(UPTKblasHandle_t handle,
                                    UPTKblasSideMode_t side,
                                    UPTKblasFillMode_t uplo,
                                    int64_t m,
                                    int64_t n,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *A,
                                    int64_t lda,
                                    const cuDoubleComplex *B,
                                    int64_t ldb,
                                    const cuDoubleComplex *beta,
                                    cuDoubleComplex *C,
                                    int64_t ldc);

  /* HEMM */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasChemm(UPTKblasHandle_t handle,
                                 UPTKblasSideMode_t side,
                                 UPTKblasFillMode_t uplo,
                                 int m,
                                 int n,
                                 const cuComplex *alpha,
                                 const cuComplex *A,
                                 int lda,
                                 const cuComplex *B,
                                 int ldb,
                                 const cuComplex *beta,
                                 cuComplex *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasChemm_64(UPTKblasHandle_t handle,
                                    UPTKblasSideMode_t side,
                                    UPTKblasFillMode_t uplo,
                                    int64_t m,
                                    int64_t n,
                                    const cuComplex *alpha,
                                    const cuComplex *A,
                                    int64_t lda,
                                    const cuComplex *B,
                                    int64_t ldb,
                                    const cuComplex *beta,
                                    cuComplex *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZhemm(UPTKblasHandle_t handle,
                                 UPTKblasSideMode_t side,
                                 UPTKblasFillMode_t uplo,
                                 int m,
                                 int n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A,
                                 int lda,
                                 const cuDoubleComplex *B,
                                 int ldb,
                                 const cuDoubleComplex *beta,
                                 cuDoubleComplex *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZhemm_64(UPTKblasHandle_t handle,
                                    UPTKblasSideMode_t side,
                                    UPTKblasFillMode_t uplo,
                                    int64_t m,
                                    int64_t n,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *A,
                                    int64_t lda,
                                    const cuDoubleComplex *B,
                                    int64_t ldb,
                                    const cuDoubleComplex *beta,
                                    cuDoubleComplex *C,
                                    int64_t ldc);

  /* TRSM */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasStrsm(UPTKblasHandle_t handle,
                                 UPTKblasSideMode_t side,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int m,
                                 int n,
                                 const float *alpha,
                                 const float *A,
                                 int lda,
                                 float *B,
                                 int ldb);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasStrsm_64(UPTKblasHandle_t handle,
                                    UPTKblasSideMode_t side,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t m,
                                    int64_t n,
                                    const float *alpha,
                                    const float *A,
                                    int64_t lda,
                                    float *B,
                                    int64_t ldb);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDtrsm(UPTKblasHandle_t handle,
                                 UPTKblasSideMode_t side,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int m,
                                 int n,
                                 const double *alpha,
                                 const double *A,
                                 int lda,
                                 double *B,
                                 int ldb);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDtrsm_64(UPTKblasHandle_t handle,
                                    UPTKblasSideMode_t side,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t m,
                                    int64_t n,
                                    const double *alpha,
                                    const double *A,
                                    int64_t lda,
                                    double *B,
                                    int64_t ldb);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCtrsm(UPTKblasHandle_t handle,
                                 UPTKblasSideMode_t side,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int m,
                                 int n,
                                 const cuComplex *alpha,
                                 const cuComplex *A,
                                 int lda,
                                 cuComplex *B,
                                 int ldb);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCtrsm_64(UPTKblasHandle_t handle,
                                    UPTKblasSideMode_t side,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t m,
                                    int64_t n,
                                    const cuComplex *alpha,
                                    const cuComplex *A,
                                    int64_t lda,
                                    cuComplex *B,
                                    int64_t ldb);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrsm(UPTKblasHandle_t handle,
                                 UPTKblasSideMode_t side,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int m,
                                 int n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A,
                                 int lda,
                                 cuDoubleComplex *B,
                                 int ldb);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrsm_64(UPTKblasHandle_t handle,
                                    UPTKblasSideMode_t side,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t m,
                                    int64_t n,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *A,
                                    int64_t lda,
                                    cuDoubleComplex *B,
                                    int64_t ldb);

  /* TRMM */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasStrmm(UPTKblasHandle_t handle,
                                 UPTKblasSideMode_t side,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int m,
                                 int n,
                                 const float *alpha,
                                 const float *A,
                                 int lda,
                                 const float *B,
                                 int ldb,
                                 float *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasStrmm_64(UPTKblasHandle_t handle,
                                    UPTKblasSideMode_t side,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t m,
                                    int64_t n,
                                    const float *alpha,
                                    const float *A,
                                    int64_t lda,
                                    const float *B,
                                    int64_t ldb,
                                    float *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDtrmm(UPTKblasHandle_t handle,
                                 UPTKblasSideMode_t side,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int m,
                                 int n,
                                 const double *alpha,
                                 const double *A,
                                 int lda,
                                 const double *B,
                                 int ldb,
                                 double *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDtrmm_64(UPTKblasHandle_t handle,
                                    UPTKblasSideMode_t side,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t m,
                                    int64_t n,
                                    const double *alpha,
                                    const double *A,
                                    int64_t lda,
                                    const double *B,
                                    int64_t ldb,
                                    double *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCtrmm(UPTKblasHandle_t handle,
                                 UPTKblasSideMode_t side,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int m,
                                 int n,
                                 const cuComplex *alpha,
                                 const cuComplex *A,
                                 int lda,
                                 const cuComplex *B,
                                 int ldb,
                                 cuComplex *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCtrmm_64(UPTKblasHandle_t handle,
                                    UPTKblasSideMode_t side,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t m,
                                    int64_t n,
                                    const cuComplex *alpha,
                                    const cuComplex *A,
                                    int64_t lda,
                                    const cuComplex *B,
                                    int64_t ldb,
                                    cuComplex *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrmm(UPTKblasHandle_t handle,
                                 UPTKblasSideMode_t side,
                                 UPTKblasFillMode_t uplo,
                                 UPTKblasOperation_t trans,
                                 UPTKblasDiagType_t diag,
                                 int m,
                                 int n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A,
                                 int lda,
                                 const cuDoubleComplex *B,
                                 int ldb,
                                 cuDoubleComplex *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrmm_64(UPTKblasHandle_t handle,
                                    UPTKblasSideMode_t side,
                                    UPTKblasFillMode_t uplo,
                                    UPTKblasOperation_t trans,
                                    UPTKblasDiagType_t diag,
                                    int64_t m,
                                    int64_t n,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *A,
                                    int64_t lda,
                                    const cuDoubleComplex *B,
                                    int64_t ldb,
                                    cuDoubleComplex *C,
                                    int64_t ldc);

  /* BATCH GEMM */

#if defined(__cplusplus)

  UPTKBLASAPI UPTKblasStatus_t UPTKblasHgemmBatched(UPTKblasHandle_t handle,
                                        UPTKblasOperation_t transa,
                                        UPTKblasOperation_t transb,
                                        int m,
                                        int n,
                                        int k,
                                        const __half *alpha,
                                        const __half *const Aarray[],
                                        int lda,
                                        const __half *const Barray[],
                                        int ldb,
                                        const __half *beta,
                                        __half *const Carray[],
                                        int ldc,
                                        int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasHgemmBatched_64(UPTKblasHandle_t handle,
                                           UPTKblasOperation_t transa,
                                           UPTKblasOperation_t transb,
                                           int64_t m,
                                           int64_t n,
                                           int64_t k,
                                           const __half *alpha,
                                           const __half *const Aarray[],
                                           int64_t lda,
                                           const __half *const Barray[],
                                           int64_t ldb,
                                           const __half *beta,
                                           __half *const Carray[],
                                           int64_t ldc,
                                           int64_t batchCount);

#endif

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemmBatched(UPTKblasHandle_t handle,
                                        UPTKblasOperation_t transa,
                                        UPTKblasOperation_t transb,
                                        int m,
                                        int n,
                                        int k,
                                        const float *alpha,
                                        const float *const Aarray[],
                                        int lda,
                                        const float *const Barray[],
                                        int ldb,
                                        const float *beta,
                                        float *const Carray[],
                                        int ldc,
                                        int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemmBatched_64(UPTKblasHandle_t handle,
                                           UPTKblasOperation_t transa,
                                           UPTKblasOperation_t transb,
                                           int64_t m,
                                           int64_t n,
                                           int64_t k,
                                           const float *alpha,
                                           const float *const Aarray[],
                                           int64_t lda,
                                           const float *const Barray[],
                                           int64_t ldb,
                                           const float *beta,
                                           float *const Carray[],
                                           int64_t ldc,
                                           int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemmBatched(UPTKblasHandle_t handle,
                                        UPTKblasOperation_t transa,
                                        UPTKblasOperation_t transb,
                                        int m,
                                        int n,
                                        int k,
                                        const double *alpha,
                                        const double *const Aarray[],
                                        int lda,
                                        const double *const Barray[],
                                        int ldb,
                                        const double *beta,
                                        double *const Carray[],
                                        int ldc,
                                        int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemmBatched_64(UPTKblasHandle_t handle,
                                           UPTKblasOperation_t transa,
                                           UPTKblasOperation_t transb,
                                           int64_t m,
                                           int64_t n,
                                           int64_t k,
                                           const double *alpha,
                                           const double *const Aarray[],
                                           int64_t lda,
                                           const double *const Barray[],
                                           int64_t ldb,
                                           const double *beta,
                                           double *const Carray[],
                                           int64_t ldc,
                                           int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemmBatched(UPTKblasHandle_t handle,
                                        UPTKblasOperation_t transa,
                                        UPTKblasOperation_t transb,
                                        int m,
                                        int n,
                                        int k,
                                        const cuComplex *alpha,
                                        const cuComplex *const Aarray[],
                                        int lda,
                                        const cuComplex *const Barray[],
                                        int ldb,
                                        const cuComplex *beta,
                                        cuComplex *const Carray[],
                                        int ldc,
                                        int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemmBatched_64(UPTKblasHandle_t handle,
                                           UPTKblasOperation_t transa,
                                           UPTKblasOperation_t transb,
                                           int64_t m,
                                           int64_t n,
                                           int64_t k,
                                           const cuComplex *alpha,
                                           const cuComplex *const Aarray[],
                                           int64_t lda,
                                           const cuComplex *const Barray[],
                                           int64_t ldb,
                                           const cuComplex *beta,
                                           cuComplex *const Carray[],
                                           int64_t ldc,
                                           int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemmBatched(UPTKblasHandle_t handle,
                                        UPTKblasOperation_t transa,
                                        UPTKblasOperation_t transb,
                                        int m,
                                        int n,
                                        int k,
                                        const cuDoubleComplex *alpha,
                                        const cuDoubleComplex *const Aarray[],
                                        int lda,
                                        const cuDoubleComplex *const Barray[],
                                        int ldb,
                                        const cuDoubleComplex *beta,
                                        cuDoubleComplex *const Carray[],
                                        int ldc,
                                        int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemmBatched_64(UPTKblasHandle_t handle,
                                           UPTKblasOperation_t transa,
                                           UPTKblasOperation_t transb,
                                           int64_t m,
                                           int64_t n,
                                           int64_t k,
                                           const cuDoubleComplex *alpha,
                                           const cuDoubleComplex *const Aarray[],
                                           int64_t lda,
                                           const cuDoubleComplex *const Barray[],
                                           int64_t ldb,
                                           const cuDoubleComplex *beta,
                                           cuDoubleComplex *const Carray[],
                                           int64_t ldc,
                                           int64_t batchCount);

#if defined(__cplusplus)

  UPTKBLASAPI UPTKblasStatus_t UPTKblasHgemmStridedBatched(UPTKblasHandle_t handle,
                                               UPTKblasOperation_t transa,
                                               UPTKblasOperation_t transb,
                                               int m,
                                               int n,
                                               int k,
                                               const __half *alpha,
                                               const __half *A,
                                               int lda,
                                               long long int strideA,
                                               const __half *B,
                                               int ldb,
                                               long long int strideB,
                                               const __half *beta,
                                               __half *C,
                                               int ldc,
                                               long long int strideC,
                                               int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasHgemmStridedBatched_64(UPTKblasHandle_t handle,
                                                  UPTKblasOperation_t transa,
                                                  UPTKblasOperation_t transb,
                                                  int64_t m,
                                                  int64_t n,
                                                  int64_t k,
                                                  const __half *alpha,
                                                  const __half *A,
                                                  int64_t lda,
                                                  long long int strideA,
                                                  const __half *B,
                                                  int64_t ldb,
                                                  long long int strideB,
                                                  const __half *beta,
                                                  __half *C,
                                                  int64_t ldc,
                                                  long long int strideC,
                                                  int64_t batchCount);

#endif

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemmStridedBatched(UPTKblasHandle_t handle,
                                               UPTKblasOperation_t transa,
                                               UPTKblasOperation_t transb,
                                               int m,
                                               int n,
                                               int k,
                                               const float *alpha,
                                               const float *A,
                                               int lda,
                                               long long int strideA,
                                               const float *B,
                                               int ldb,
                                               long long int strideB,
                                               const float *beta,
                                               float *C,
                                               int ldc,
                                               long long int strideC,
                                               int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemmStridedBatched_64(UPTKblasHandle_t handle,
                                                  UPTKblasOperation_t transa,
                                                  UPTKblasOperation_t transb,
                                                  int64_t m,
                                                  int64_t n,
                                                  int64_t k,
                                                  const float *alpha,
                                                  const float *A,
                                                  int64_t lda,
                                                  long long int strideA,
                                                  const float *B,
                                                  int64_t ldb,
                                                  long long int strideB,
                                                  const float *beta,
                                                  float *C,
                                                  int64_t ldc,
                                                  long long int strideC,
                                                  int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemmStridedBatched(UPTKblasHandle_t handle,
                                               UPTKblasOperation_t transa,
                                               UPTKblasOperation_t transb,
                                               int m,
                                               int n,
                                               int k,
                                               const double *alpha,
                                               const double *A,
                                               int lda,
                                               long long int strideA,
                                               const double *B,
                                               int ldb,
                                               long long int strideB,
                                               const double *beta,
                                               double *C,
                                               int ldc,
                                               long long int strideC,
                                               int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemmStridedBatched_64(UPTKblasHandle_t handle,
                                                  UPTKblasOperation_t transa,
                                                  UPTKblasOperation_t transb,
                                                  int64_t m,
                                                  int64_t n,
                                                  int64_t k,
                                                  const double *alpha,
                                                  const double *A,
                                                  int64_t lda,
                                                  long long int strideA,
                                                  const double *B,
                                                  int64_t ldb,
                                                  long long int strideB,
                                                  const double *beta,
                                                  double *C,
                                                  int64_t ldc,
                                                  long long int strideC,
                                                  int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemmStridedBatched(UPTKblasHandle_t handle,
                                               UPTKblasOperation_t transa,
                                               UPTKblasOperation_t transb,
                                               int m,
                                               int n,
                                               int k,
                                               const cuComplex *alpha,
                                               const cuComplex *A,
                                               int lda,
                                               long long int strideA,
                                               const cuComplex *B,
                                               int ldb,
                                               long long int strideB,
                                               const cuComplex *beta,
                                               cuComplex *C,
                                               int ldc,
                                               long long int strideC,
                                               int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemmStridedBatched_64(UPTKblasHandle_t handle,
                                                  UPTKblasOperation_t transa,
                                                  UPTKblasOperation_t transb,
                                                  int64_t m,
                                                  int64_t n,
                                                  int64_t k,
                                                  const cuComplex *alpha,
                                                  const cuComplex *A,
                                                  int64_t lda,
                                                  long long int strideA,
                                                  const cuComplex *B,
                                                  int64_t ldb,
                                                  long long int strideB,
                                                  const cuComplex *beta,
                                                  cuComplex *C,
                                                  int64_t ldc,
                                                  long long int strideC,
                                                  int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemm3mStridedBatched(UPTKblasHandle_t handle,
                                                 UPTKblasOperation_t transa,
                                                 UPTKblasOperation_t transb,
                                                 int m,
                                                 int n,
                                                 int k,
                                                 const cuComplex *alpha,
                                                 const cuComplex *A,
                                                 int lda,
                                                 long long int strideA,
                                                 const cuComplex *B,
                                                 int ldb,
                                                 long long int strideB,
                                                 const cuComplex *beta,
                                                 cuComplex *C,
                                                 int ldc,
                                                 long long int strideC,
                                                 int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemm3mStridedBatched_64(UPTKblasHandle_t handle,
                                                    UPTKblasOperation_t transa,
                                                    UPTKblasOperation_t transb,
                                                    int64_t m,
                                                    int64_t n,
                                                    int64_t k,
                                                    const cuComplex *alpha,
                                                    const cuComplex *A,
                                                    int64_t lda,
                                                    long long int strideA,
                                                    const cuComplex *B,
                                                    int64_t ldb,
                                                    long long int strideB,
                                                    const cuComplex *beta,
                                                    cuComplex *C,
                                                    int64_t ldc,
                                                    long long int strideC,
                                                    int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemmStridedBatched(UPTKblasHandle_t handle,
                                               UPTKblasOperation_t transa,
                                               UPTKblasOperation_t transb,
                                               int m,
                                               int n,
                                               int k,
                                               const cuDoubleComplex *alpha,
                                               const cuDoubleComplex *A,
                                               int lda,
                                               long long int strideA,
                                               const cuDoubleComplex *B,
                                               int ldb,
                                               long long int strideB,
                                               const cuDoubleComplex *beta,
                                               cuDoubleComplex *C,
                                               int ldc,
                                               long long int strideC,
                                               int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemmStridedBatched_64(UPTKblasHandle_t handle,
                                                  UPTKblasOperation_t transa,
                                                  UPTKblasOperation_t transb,
                                                  int64_t m,
                                                  int64_t n,
                                                  int64_t k,
                                                  const cuDoubleComplex *alpha,
                                                  const cuDoubleComplex *A,
                                                  int64_t lda,
                                                  long long int strideA,
                                                  const cuDoubleComplex *B,
                                                  int64_t ldb,
                                                  long long int strideB,
                                                  const cuDoubleComplex *beta,
                                                  cuDoubleComplex *C,
                                                  int64_t ldc,
                                                  long long int strideC,
                                                  int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasGemmBatchedEx(UPTKblasHandle_t handle,
                                         UPTKblasOperation_t transa,
                                         UPTKblasOperation_t transb,
                                         int m,
                                         int n,
                                         int k,
                                         const void *alpha,
                                         const void *const Aarray[],
                                         UPTKDataType Atype,
                                         int lda,
                                         const void *const Barray[],
                                         UPTKDataType Btype,
                                         int ldb,
                                         const void *beta,
                                         void *const Carray[],
                                         UPTKDataType Ctype,
                                         int ldc,
                                         int batchCount,
                                         UPTKblasComputeType_t computeType,
                                         UPTKblasGemmAlgo_t algo);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasGemmBatchedEx_64(UPTKblasHandle_t handle,
                                            UPTKblasOperation_t transa,
                                            UPTKblasOperation_t transb,
                                            int64_t m,
                                            int64_t n,
                                            int64_t k,
                                            const void *alpha,
                                            const void *const Aarray[],
                                            UPTKDataType Atype,
                                            int64_t lda,
                                            const void *const Barray[],
                                            UPTKDataType Btype,
                                            int64_t ldb,
                                            const void *beta,
                                            void *const Carray[],
                                            UPTKDataType Ctype,
                                            int64_t ldc,
                                            int64_t batchCount,
                                            UPTKblasComputeType_t computeType,
                                            UPTKblasGemmAlgo_t algo);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasGemmStridedBatchedEx(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t transa,
                                                UPTKblasOperation_t transb,
                                                int m,
                                                int n,
                                                int k,
                                                const void *alpha,
                                                const void *A,
                                                UPTKDataType Atype,
                                                int lda,
                                                long long int strideA,
                                                const void *B,
                                                UPTKDataType Btype,
                                                int ldb,
                                                long long int strideB,
                                                const void *beta,
                                                void *C,
                                                UPTKDataType Ctype,
                                                int ldc,
                                                long long int strideC,
                                                int batchCount,
                                                UPTKblasComputeType_t computeType,
                                                UPTKblasGemmAlgo_t algo);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasGemmStridedBatchedEx_64(UPTKblasHandle_t handle,
                                                   UPTKblasOperation_t transa,
                                                   UPTKblasOperation_t transb,
                                                   int64_t m,
                                                   int64_t n,
                                                   int64_t k,
                                                   const void *alpha,
                                                   const void *A,
                                                   UPTKDataType Atype,
                                                   int64_t lda,
                                                   long long int strideA,
                                                   const void *B,
                                                   UPTKDataType Btype,
                                                   int64_t ldb,
                                                   long long int strideB,
                                                   const void *beta,
                                                   void *C,
                                                   UPTKDataType Ctype,
                                                   int64_t ldc,
                                                   long long int strideC,
                                                   int64_t batchCount,
                                                   UPTKblasComputeType_t computeType,
                                                   UPTKblasGemmAlgo_t algo);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemmGroupedBatched(UPTKblasHandle_t handle,
                                               const UPTKblasOperation_t transa_array[],
                                               const UPTKblasOperation_t transb_array[],
                                               const int m_array[],
                                               const int n_array[],
                                               const int k_array[],
                                               const float alpha_array[],
                                               const float *const Aarray[],
                                               const int lda_array[],
                                               const float *const Barray[],
                                               const int ldb_array[],
                                               const float beta_array[],
                                               float *const Carray[],
                                               const int ldc_array[],
                                               int group_count,
                                               const int group_size[]);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemmGroupedBatched_64(UPTKblasHandle_t handle,
                                                  const UPTKblasOperation_t transa_array[],
                                                  const UPTKblasOperation_t transb_array[],
                                                  const int64_t m_array[],
                                                  const int64_t n_array[],
                                                  const int64_t k_array[],
                                                  const float alpha_array[],
                                                  const float *const Aarray[],
                                                  const int64_t lda_array[],
                                                  const float *const Barray[],
                                                  const int64_t ldb_array[],
                                                  const float beta_array[],
                                                  float *const Carray[],
                                                  const int64_t ldc_array[],
                                                  int64_t group_count,
                                                  const int64_t group_size[]);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemmGroupedBatched(UPTKblasHandle_t handle,
                                               const UPTKblasOperation_t transa_array[],
                                               const UPTKblasOperation_t transb_array[],
                                               const int m_array[],
                                               const int n_array[],
                                               const int k_array[],
                                               const double alpha_array[],
                                               const double *const Aarray[],
                                               const int lda_array[],
                                               const double *const Barray[],
                                               const int ldb_array[],
                                               const double beta_array[],
                                               double *const Carray[],
                                               const int ldc_array[],
                                               int group_count,
                                               const int group_size[]);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemmGroupedBatched_64(UPTKblasHandle_t handle,
                                                  const UPTKblasOperation_t transa_array[],
                                                  const UPTKblasOperation_t transb_array[],
                                                  const int64_t m_array[],
                                                  const int64_t n_array[],
                                                  const int64_t k_array[],
                                                  const double alpha_array[],
                                                  const double *const Aarray[],
                                                  const int64_t lda_array[],
                                                  const double *const Barray[],
                                                  const int64_t ldb_array[],
                                                  const double beta_array[],
                                                  double *const Carray[],
                                                  const int64_t ldc_array[],
                                                  int64_t group_count,
                                                  const int64_t group_size[]);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasGemmGroupedBatchedEx(UPTKblasHandle_t handle,
                                                const UPTKblasOperation_t transa_array[],
                                                const UPTKblasOperation_t transb_array[],
                                                const int m_array[],
                                                const int n_array[],
                                                const int k_array[],
                                                const void *alpha_array,
                                                const void *const Aarray[],
                                                UPTKDataType_t Atype,
                                                const int lda_array[],
                                                const void *const Barray[],
                                                UPTKDataType_t Btype,
                                                const int ldb_array[],
                                                const void *beta_array,
                                                void *const Carray[],
                                                UPTKDataType_t Ctype,
                                                const int ldc_array[],
                                                int group_count,
                                                const int group_size[],
                                                UPTKblasComputeType_t computeType);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasGemmGroupedBatchedEx_64(UPTKblasHandle_t handle,
                                                   const UPTKblasOperation_t transa_array[],
                                                   const UPTKblasOperation_t transb_array[],
                                                   const int64_t m_array[],
                                                   const int64_t n_array[],
                                                   const int64_t k_array[],
                                                   const void *alpha_array,
                                                   const void *const Aarray[],
                                                   UPTKDataType_t Atype,
                                                   const int64_t lda_array[],
                                                   const void *const Barray[],
                                                   UPTKDataType_t Btype,
                                                   const int64_t ldb_array[],
                                                   const void *beta_array,
                                                   void *const Carray[],
                                                   UPTKDataType_t Ctype,
                                                   const int64_t ldc_array[],
                                                   int64_t group_count,
                                                   const int64_t group_size[],
                                                   UPTKblasComputeType_t computeType);

  /* ---------------- CUBLAS BLAS-like Extension ---------------- */

  /* GEAM */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgeam(UPTKblasHandle_t handle,
                                 UPTKblasOperation_t transa,
                                 UPTKblasOperation_t transb,
                                 int m,
                                 int n,
                                 const float *alpha,
                                 const float *A,
                                 int lda,
                                 const float *beta,
                                 const float *B,
                                 int ldb,
                                 float *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgeam_64(UPTKblasHandle_t handle,
                                    UPTKblasOperation_t transa,
                                    UPTKblasOperation_t transb,
                                    int64_t m,
                                    int64_t n,
                                    const float *alpha,
                                    const float *A,
                                    int64_t lda,
                                    const float *beta,
                                    const float *B,
                                    int64_t ldb,
                                    float *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgeam(UPTKblasHandle_t handle,
                                 UPTKblasOperation_t transa,
                                 UPTKblasOperation_t transb,
                                 int m,
                                 int n,
                                 const double *alpha,
                                 const double *A,
                                 int lda,
                                 const double *beta,
                                 const double *B,
                                 int ldb,
                                 double *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgeam_64(UPTKblasHandle_t handle,
                                    UPTKblasOperation_t transa,
                                    UPTKblasOperation_t transb,
                                    int64_t m,
                                    int64_t n,
                                    const double *alpha,
                                    const double *A,
                                    int64_t lda,
                                    const double *beta,
                                    const double *B,
                                    int64_t ldb,
                                    double *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgeam(UPTKblasHandle_t handle,
                                 UPTKblasOperation_t transa,
                                 UPTKblasOperation_t transb,
                                 int m,
                                 int n,
                                 const cuComplex *alpha,
                                 const cuComplex *A,
                                 int lda,
                                 const cuComplex *beta,
                                 const cuComplex *B,
                                 int ldb,
                                 cuComplex *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgeam_64(UPTKblasHandle_t handle,
                                    UPTKblasOperation_t transa,
                                    UPTKblasOperation_t transb,
                                    int64_t m,
                                    int64_t n,
                                    const cuComplex *alpha,
                                    const cuComplex *A,
                                    int64_t lda,
                                    const cuComplex *beta,
                                    const cuComplex *B,
                                    int64_t ldb,
                                    cuComplex *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgeam(UPTKblasHandle_t handle,
                                 UPTKblasOperation_t transa,
                                 UPTKblasOperation_t transb,
                                 int m,
                                 int n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A,
                                 int lda,
                                 const cuDoubleComplex *beta,
                                 const cuDoubleComplex *B,
                                 int ldb,
                                 cuDoubleComplex *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgeam_64(UPTKblasHandle_t handle,
                                    UPTKblasOperation_t transa,
                                    UPTKblasOperation_t transb,
                                    int64_t m,
                                    int64_t n,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *A,
                                    int64_t lda,
                                    const cuDoubleComplex *beta,
                                    const cuDoubleComplex *B,
                                    int64_t ldb,
                                    cuDoubleComplex *C,
                                    int64_t ldc);

  /* TRSM - Batched Triangular Solver */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasStrsmBatched(UPTKblasHandle_t handle,
                                        UPTKblasSideMode_t side,
                                        UPTKblasFillMode_t uplo,
                                        UPTKblasOperation_t trans,
                                        UPTKblasDiagType_t diag,
                                        int m,
                                        int n,
                                        const float *alpha,
                                        const float *const A[],
                                        int lda,
                                        float *const B[],
                                        int ldb,
                                        int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasStrsmBatched_64(UPTKblasHandle_t handle,
                                           UPTKblasSideMode_t side,
                                           UPTKblasFillMode_t uplo,
                                           UPTKblasOperation_t trans,
                                           UPTKblasDiagType_t diag,
                                           int64_t m,
                                           int64_t n,
                                           const float *alpha,
                                           const float *const A[],
                                           int64_t lda,
                                           float *const B[],
                                           int64_t ldb,
                                           int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDtrsmBatched(UPTKblasHandle_t handle,
                                        UPTKblasSideMode_t side,
                                        UPTKblasFillMode_t uplo,
                                        UPTKblasOperation_t trans,
                                        UPTKblasDiagType_t diag,
                                        int m,
                                        int n,
                                        const double *alpha,
                                        const double *const A[],
                                        int lda,
                                        double *const B[],
                                        int ldb,
                                        int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDtrsmBatched_64(UPTKblasHandle_t handle,
                                           UPTKblasSideMode_t side,
                                           UPTKblasFillMode_t uplo,
                                           UPTKblasOperation_t trans,
                                           UPTKblasDiagType_t diag,
                                           int64_t m,
                                           int64_t n,
                                           const double *alpha,
                                           const double *const A[],
                                           int64_t lda,
                                           double *const B[],
                                           int64_t ldb,
                                           int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCtrsmBatched(UPTKblasHandle_t handle,
                                        UPTKblasSideMode_t side,
                                        UPTKblasFillMode_t uplo,
                                        UPTKblasOperation_t trans,
                                        UPTKblasDiagType_t diag,
                                        int m,
                                        int n,
                                        const cuComplex *alpha,
                                        const cuComplex *const A[],
                                        int lda,
                                        cuComplex *const B[],
                                        int ldb,
                                        int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCtrsmBatched_64(UPTKblasHandle_t handle,
                                           UPTKblasSideMode_t side,
                                           UPTKblasFillMode_t uplo,
                                           UPTKblasOperation_t trans,
                                           UPTKblasDiagType_t diag,
                                           int64_t m,
                                           int64_t n,
                                           const cuComplex *alpha,
                                           const cuComplex *const A[],
                                           int64_t lda,
                                           cuComplex *const B[],
                                           int64_t ldb,
                                           int64_t batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrsmBatched(UPTKblasHandle_t handle,
                                        UPTKblasSideMode_t side,
                                        UPTKblasFillMode_t uplo,
                                        UPTKblasOperation_t trans,
                                        UPTKblasDiagType_t diag,
                                        int m,
                                        int n,
                                        const cuDoubleComplex *alpha,
                                        const cuDoubleComplex *const A[],
                                        int lda,
                                        cuDoubleComplex *const B[],
                                        int ldb,
                                        int batchCount);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrsmBatched_64(UPTKblasHandle_t handle,
                                           UPTKblasSideMode_t side,
                                           UPTKblasFillMode_t uplo,
                                           UPTKblasOperation_t trans,
                                           UPTKblasDiagType_t diag,
                                           int64_t m,
                                           int64_t n,
                                           const cuDoubleComplex *alpha,
                                           const cuDoubleComplex *const A[],
                                           int64_t lda,
                                           cuDoubleComplex *const B[],
                                           int64_t ldb,
                                           int64_t batchCount);

  /* DGMM */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSdgmm(UPTKblasHandle_t handle,
                                 UPTKblasSideMode_t mode,
                                 int m,
                                 int n,
                                 const float *A,
                                 int lda,
                                 const float *x,
                                 int incx,
                                 float *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSdgmm_64(UPTKblasHandle_t handle,
                                    UPTKblasSideMode_t mode,
                                    int64_t m,
                                    int64_t n,
                                    const float *A,
                                    int64_t lda,
                                    const float *x,
                                    int64_t incx,
                                    float *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDdgmm(UPTKblasHandle_t handle,
                                 UPTKblasSideMode_t mode,
                                 int m,
                                 int n,
                                 const double *A,
                                 int lda,
                                 const double *x,
                                 int incx,
                                 double *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDdgmm_64(UPTKblasHandle_t handle,
                                    UPTKblasSideMode_t mode,
                                    int64_t m,
                                    int64_t n,
                                    const double *A,
                                    int64_t lda,
                                    const double *x,
                                    int64_t incx,
                                    double *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCdgmm(UPTKblasHandle_t handle,
                                 UPTKblasSideMode_t mode,
                                 int m,
                                 int n,
                                 const cuComplex *A,
                                 int lda,
                                 const cuComplex *x,
                                 int incx,
                                 cuComplex *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCdgmm_64(UPTKblasHandle_t handle,
                                    UPTKblasSideMode_t mode,
                                    int64_t m,
                                    int64_t n,
                                    const cuComplex *A,
                                    int64_t lda,
                                    const cuComplex *x,
                                    int64_t incx,
                                    cuComplex *C,
                                    int64_t ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZdgmm(UPTKblasHandle_t handle,
                                 UPTKblasSideMode_t mode,
                                 int m,
                                 int n,
                                 const cuDoubleComplex *A,
                                 int lda,
                                 const cuDoubleComplex *x,
                                 int incx,
                                 cuDoubleComplex *C,
                                 int ldc);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZdgmm_64(UPTKblasHandle_t handle,
                                    UPTKblasSideMode_t mode,
                                    int64_t m,
                                    int64_t n,
                                    const cuDoubleComplex *A,
                                    int64_t lda,
                                    const cuDoubleComplex *x,
                                    int64_t incx,
                                    cuDoubleComplex *C,
                                    int64_t ldc);

  /* Batched - MATINV*/

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSmatinvBatched(UPTKblasHandle_t handle,
                                          int n,
                                          const float *const A[],
                                          int lda,
                                          float *const Ainv[],
                                          int lda_inv,
                                          int *info,
                                          int batchSize);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDmatinvBatched(UPTKblasHandle_t handle,
                                          int n,
                                          const double *const A[],
                                          int lda,
                                          double *const Ainv[],
                                          int lda_inv,
                                          int *info,
                                          int batchSize);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCmatinvBatched(UPTKblasHandle_t handle,
                                          int n,
                                          const cuComplex *const A[],
                                          int lda,
                                          cuComplex *const Ainv[],
                                          int lda_inv,
                                          int *info,
                                          int batchSize);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZmatinvBatched(UPTKblasHandle_t handle,
                                          int n,
                                          const cuDoubleComplex *const A[],
                                          int lda,
                                          cuDoubleComplex *const Ainv[],
                                          int lda_inv,
                                          int *info,
                                          int batchSize);

  /* Batch QR Factorization */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgeqrfBatched(UPTKblasHandle_t handle,
                                         int m,
                                         int n,
                                         float *const Aarray[],
                                         int lda,
                                         float *const TauArray[],
                                         int *info,
                                         int batchSize);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgeqrfBatched(UPTKblasHandle_t handle,
                                         int m,
                                         int n,
                                         double *const Aarray[],
                                         int lda,
                                         double *const TauArray[],
                                         int *info,
                                         int batchSize);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgeqrfBatched(UPTKblasHandle_t handle,
                                         int m,
                                         int n,
                                         cuComplex *const Aarray[],
                                         int lda,
                                         cuComplex *const TauArray[],
                                         int *info,
                                         int batchSize);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgeqrfBatched(UPTKblasHandle_t handle,
                                         int m,
                                         int n,
                                         cuDoubleComplex *const Aarray[],
                                         int lda,
                                         cuDoubleComplex *const TauArray[],
                                         int *info,
                                         int batchSize);

  /* Least Square Min only m >= n and Non-transpose supported */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgelsBatched(UPTKblasHandle_t handle,
                                        UPTKblasOperation_t trans,
                                        int m,
                                        int n,
                                        int nrhs,
                                        float *const Aarray[],
                                        int lda,
                                        float *const Carray[],
                                        int ldc,
                                        int *info,
                                        int *devInfoArray,
                                        int batchSize);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgelsBatched(UPTKblasHandle_t handle,
                                        UPTKblasOperation_t trans,
                                        int m,
                                        int n,
                                        int nrhs,
                                        double *const Aarray[],
                                        int lda,
                                        double *const Carray[],
                                        int ldc,
                                        int *info,
                                        int *devInfoArray,
                                        int batchSize);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgelsBatched(UPTKblasHandle_t handle,
                                        UPTKblasOperation_t trans,
                                        int m,
                                        int n,
                                        int nrhs,
                                        cuComplex *const Aarray[],
                                        int lda,
                                        cuComplex *const Carray[],
                                        int ldc,
                                        int *info,
                                        int *devInfoArray,
                                        int batchSize);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgelsBatched(UPTKblasHandle_t handle,
                                        UPTKblasOperation_t trans,
                                        int m,
                                        int n,
                                        int nrhs,
                                        cuDoubleComplex *const Aarray[],
                                        int lda,
                                        cuDoubleComplex *const Carray[],
                                        int ldc,
                                        int *info,
                                        int *devInfoArray,
                                        int batchSize);

  /* TPTTR : Triangular Pack format to Triangular format */

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasStpttr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const float *AP, float *A, int lda);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDtpttr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const double *AP, double *A, int lda);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasCtpttr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuComplex *AP, cuComplex *A, int lda);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZtpttr(
      UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuDoubleComplex *AP, cuDoubleComplex *A, int lda);

  /* TRTTP : Triangular format to Triangular Pack format */

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasStrttp(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const float *A, int lda, float *AP);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDtrttp(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const double *A, int lda, double *AP);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasCtrttp(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuComplex *A, int lda, cuComplex *AP);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrttp(
      UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *AP);

  /* Batched LU - GETRF*/

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasSgetrfBatched(UPTKblasHandle_t handle, int n, float *const A[], int lda, int *P, int *info, int batchSize);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasDgetrfBatched(UPTKblasHandle_t handle, int n, double *const A[], int lda, int *P, int *info, int batchSize);

  UPTKBLASAPI UPTKblasStatus_t
  UPTKblasCgetrfBatched(UPTKblasHandle_t handle, int n, cuComplex *const A[], int lda, int *P, int *info, int batchSize);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgetrfBatched(
      UPTKblasHandle_t handle, int n, cuDoubleComplex *const A[], int lda, int *P, int *info, int batchSize);

  /* Batched inversion based on LU factorization from getrf */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgetriBatched(UPTKblasHandle_t handle,
                                         int n,
                                         const float *const A[],
                                         int lda,
                                         const int *P,
                                         float *const C[],
                                         int ldc,
                                         int *info,
                                         int batchSize);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgetriBatched(UPTKblasHandle_t handle,
                                         int n,
                                         const double *const A[],
                                         int lda,
                                         const int *P,
                                         double *const C[],
                                         int ldc,
                                         int *info,
                                         int batchSize);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgetriBatched(UPTKblasHandle_t handle,
                                         int n,
                                         const cuComplex *const A[],
                                         int lda,
                                         const int *P,
                                         cuComplex *const C[],
                                         int ldc,
                                         int *info,
                                         int batchSize);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgetriBatched(UPTKblasHandle_t handle,
                                         int n,
                                         const cuDoubleComplex *const A[],
                                         int lda,
                                         const int *P,
                                         cuDoubleComplex *const C[],
                                         int ldc,
                                         int *info,
                                         int batchSize);

  /* Batched solver based on LU factorization from getrf */

  UPTKBLASAPI UPTKblasStatus_t UPTKblasSgetrsBatched(UPTKblasHandle_t handle,
                                         UPTKblasOperation_t trans,
                                         int n,
                                         int nrhs,
                                         const float *const Aarray[],
                                         int lda,
                                         const int *devIpiv,
                                         float *const Barray[],
                                         int ldb,
                                         int *info,
                                         int batchSize);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasDgetrsBatched(UPTKblasHandle_t handle,
                                         UPTKblasOperation_t trans,
                                         int n,
                                         int nrhs,
                                         const double *const Aarray[],
                                         int lda,
                                         const int *devIpiv,
                                         double *const Barray[],
                                         int ldb,
                                         int *info,
                                         int batchSize);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasCgetrsBatched(UPTKblasHandle_t handle,
                                         UPTKblasOperation_t trans,
                                         int n,
                                         int nrhs,
                                         const cuComplex *const Aarray[],
                                         int lda,
                                         const int *devIpiv,
                                         cuComplex *const Barray[],
                                         int ldb,
                                         int *info,
                                         int batchSize);

  UPTKBLASAPI UPTKblasStatus_t UPTKblasZgetrsBatched(UPTKblasHandle_t handle,
                                         UPTKblasOperation_t trans,
                                         int n,
                                         int nrhs,
                                         const cuDoubleComplex *const Aarray[],
                                         int lda,
                                         const int *devIpiv,
                                         cuDoubleComplex *const Barray[],
                                         int ldb,
                                         int *info,
                                         int batchSize);
  /* }}} cuBLAS Exported API */

#if defined(__cplusplus)
}

static inline UPTKBLASAPI UPTKblasStatus_t UPTKblasMigrateComputeType(UPTKblasHandle_t handle,
                                                          UPTKDataType_t dataType,
                                                          UPTKblasComputeType_t *computeType)
{
  UPTKblasMath_t mathMode = UPTKBLAS_DEFAULT_MATH;
  UPTKBLASAPI UPTKblasStatus_t status = UPTKBLAS_STATUS_SUCCESS;

  status = UPTKblasGetMathMode(handle, &mathMode);
  if (status != UPTKBLAS_STATUS_SUCCESS)
  {
    return status;
  }

  bool isPedantic = ((mathMode & 0xf) == UPTKBLAS_PEDANTIC_MATH);

  switch (dataType)
  {
  case UPTK_R_32F:
  case UPTK_C_32F:
    *computeType = isPedantic ? UPTKBLAS_COMPUTE_32F_PEDANTIC : UPTKBLAS_COMPUTE_32F;
    return UPTKBLAS_STATUS_SUCCESS;
  case UPTK_R_64F:
  case UPTK_C_64F:
    *computeType = isPedantic ? UPTKBLAS_COMPUTE_64F_PEDANTIC : UPTKBLAS_COMPUTE_64F;
    return UPTKBLAS_STATUS_SUCCESS;
  case UPTK_R_16F:
    *computeType = isPedantic ? UPTKBLAS_COMPUTE_16F_PEDANTIC : UPTKBLAS_COMPUTE_16F;
    return UPTKBLAS_STATUS_SUCCESS;
  case UPTK_R_32I:
    *computeType = isPedantic ? UPTKBLAS_COMPUTE_32I_PEDANTIC : UPTKBLAS_COMPUTE_32I;
    return UPTKBLAS_STATUS_SUCCESS;
  default:
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
  }
}
/* wrappers to accept old code with UPTKDataType computeType when referenced from c++ code */
static inline UPTKBLASAPI UPTKblasStatus_t UPTKblasGemmEx(UPTKblasHandle_t handle,
                                              UPTKblasOperation_t transa,
                                              UPTKblasOperation_t transb,
                                              int m,
                                              int n,
                                              int k,
                                              const void *alpha, /* host or device pointer */
                                              const void *A,
                                              UPTKDataType Atype,
                                              int lda,
                                              const void *B,
                                              UPTKDataType Btype,
                                              int ldb,
                                              const void *beta, /* host or device pointer */
                                              void *C,
                                              UPTKDataType Ctype,
                                              int ldc,
                                              UPTKDataType computeType,
                                              UPTKblasGemmAlgo_t algo)
{
  UPTKblasComputeType_t migratedComputeType = UPTKBLAS_COMPUTE_32F;
  UPTKBLASAPI UPTKblasStatus_t status = UPTKBLAS_STATUS_SUCCESS;
  status = UPTKblasMigrateComputeType(handle, computeType, &migratedComputeType);
  if (status != UPTKBLAS_STATUS_SUCCESS)
  {
    return status;
  }

  return UPTKblasGemmEx(handle,
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        alpha,
                        A,
                        Atype,
                        lda,
                        B,
                        Btype,
                        ldb,
                        beta,
                        C,
                        Ctype,
                        ldc,
                        migratedComputeType,
                        algo);
}

static inline UPTKBLASAPI UPTKblasStatus_t UPTKblasGemmBatchedEx(UPTKblasHandle_t handle,
                                                     UPTKblasOperation_t transa,
                                                     UPTKblasOperation_t transb,
                                                     int m,
                                                     int n,
                                                     int k,
                                                     const void *alpha, /* host or device pointer */
                                                     const void *const Aarray[],
                                                     UPTKDataType Atype,
                                                     int lda,
                                                     const void *const Barray[],
                                                     UPTKDataType Btype,
                                                     int ldb,
                                                     const void *beta, /* host or device pointer */
                                                     void *const Carray[],
                                                     UPTKDataType Ctype,
                                                     int ldc,
                                                     int batchCount,
                                                     UPTKDataType computeType,
                                                     UPTKblasGemmAlgo_t algo)
{
  UPTKblasComputeType_t migratedComputeType;
  UPTKBLASAPI UPTKblasStatus_t status;
  status = UPTKblasMigrateComputeType(handle, computeType, &migratedComputeType);
  if (status != UPTKBLAS_STATUS_SUCCESS)
  {
    return status;
  }

  return UPTKblasGemmBatchedEx(handle,
                               transa,
                               transb,
                               m,
                               n,
                               k,
                               alpha,
                               Aarray,
                               Atype,
                               lda,
                               Barray,
                               Btype,
                               ldb,
                               beta,
                               Carray,
                               Ctype,
                               ldc,
                               batchCount,
                               migratedComputeType,
                               algo);
}

static inline UPTKBLASAPI UPTKblasStatus_t UPTKblasGemmStridedBatchedEx(UPTKblasHandle_t handle,
                                                            UPTKblasOperation_t transa,
                                                            UPTKblasOperation_t transb,
                                                            int m,
                                                            int n,
                                                            int k,
                                                            const void *alpha, /* host or device pointer */
                                                            const void *A,
                                                            UPTKDataType Atype,
                                                            int lda,
                                                            long long int strideA, /* purposely signed */
                                                            const void *B,
                                                            UPTKDataType Btype,
                                                            int ldb,
                                                            long long int strideB,
                                                            const void *beta, /* host or device pointer */
                                                            void *C,
                                                            UPTKDataType Ctype,
                                                            int ldc,
                                                            long long int strideC,
                                                            int batchCount,
                                                            UPTKDataType computeType,
                                                            UPTKblasGemmAlgo_t algo)
{
  UPTKblasComputeType_t migratedComputeType;
  UPTKBLASAPI UPTKblasStatus_t status;
  status = UPTKblasMigrateComputeType(handle, computeType, &migratedComputeType);
  if (status != UPTKBLAS_STATUS_SUCCESS)
  {
    return status;
  }

  return UPTKblasGemmStridedBatchedEx(handle,
                                      transa,
                                      transb,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      Atype,
                                      lda,
                                      strideA,
                                      B,
                                      Btype,
                                      ldb,
                                      strideB,
                                      beta,
                                      C,
                                      Ctype,
                                      ldc,
                                      strideC,
                                      batchCount,
                                      migratedComputeType,
                                      algo);
}
#endif /* __cplusplus */

#endif /* !defined(UPTKBLAS_API_H_) */

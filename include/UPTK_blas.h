/*
 * Copyright 1993-2022 NVIDIA Corporation. All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

/*
 * This is the public header file for the CUBLAS library, defining the API
 *
 * CUBLAS is an implementation of BLAS (Basic Linear Algebra Subroutines)
 * on top of the CUDA runtime.
 */

#if !defined(UPTKBLAS_API_H_)
#define UPTKBLAS_API_H_

#define UPTKBLASAPI

#include "cuComplex.h" /* import complex data type */

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <UPTK_library_types.h>
#include "UPTK_driver_types.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#define UPTKBLAS_VER_MAJOR 1
#define UPTKBLAS_VER_MINOR 0
#define UPTKBLAS_VER_PATCH 0
#define UPTKBLAS_VER_BUILD 0
#define UPTKBLAS_VERSION (UPTKBLAS_VER_MAJOR * 10000 + UPTKBLAS_VER_MINOR * 100 + UPTKBLAS_VER_PATCH)

/* CUBLAS status type returns */
typedef enum {
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

typedef enum { UPTKBLAS_FILL_MODE_LOWER = 0, UPTKBLAS_FILL_MODE_UPPER = 1, UPTKBLAS_FILL_MODE_FULL = 2 } UPTKblasFillMode_t;

typedef enum { UPTKBLAS_DIAG_NON_UNIT = 0, UPTKBLAS_DIAG_UNIT = 1 } UPTKblasDiagType_t;

typedef enum { UPTKBLAS_SIDE_LEFT = 0, UPTKBLAS_SIDE_RIGHT = 1 } UPTKblasSideMode_t;

typedef enum {
  UPTKBLAS_OP_N = 0,
  UPTKBLAS_OP_T = 1,
  UPTKBLAS_OP_C = 2,
  UPTKBLAS_OP_HERMITAN = 2, /* synonym if UPTKBLAS_OP_C */
  UPTKBLAS_OP_CONJG = 3     /* conjugate, placeholder - not supported in the current release */
} UPTKblasOperation_t;

typedef enum { UPTKBLAS_POINTER_MODE_HOST = 0, UPTKBLAS_POINTER_MODE_DEVICE = 1 } UPTKblasPointerMode_t;

typedef enum { UPTKBLAS_ATOMICS_NOT_ALLOWED = 0, UPTKBLAS_ATOMICS_ALLOWED = 1 } UPTKblasAtomicsMode_t;

/*For different GEMM algorithm */
typedef enum {
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
  UPTKBLAS_GEMM_ALGO18 = 18,  // sliced 32x32
  UPTKBLAS_GEMM_ALGO19 = 19,  // sliced 64x32
  UPTKBLAS_GEMM_ALGO20 = 20,  // sliced 128x32
  UPTKBLAS_GEMM_ALGO21 = 21,  // sliced 32x32  -splitK
  UPTKBLAS_GEMM_ALGO22 = 22,  // sliced 64x32  -splitK
  UPTKBLAS_GEMM_ALGO23 = 23,  // sliced 128x32 -splitK
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
typedef enum {
  UPTKBLAS_DEFAULT_MATH = 0,

  /* deprecated, same effect as using UPTKBLAS_COMPUTE_32F_FAST_16F, will be removed in a future release */
  UPTKBLAS_TENSOR_OP_MATH = 1,

  /* same as using matching _PEDANTIC compute type when using cublas<T>routine calls or cublasEx() calls with
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
typedef enum {
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
typedef struct UPTKblasContext* UPTKblasHandle_t;

UPTKblasStatus_t UPTKBLASAPI UPTKblasCreate(UPTKblasHandle_t* handle);
UPTKblasStatus_t UPTKBLASAPI UPTKblasDestroy(UPTKblasHandle_t handle);

UPTKblasStatus_t UPTKBLASAPI UPTKblasGetVersion(UPTKblasHandle_t handle, int* version);
UPTKblasStatus_t UPTKBLASAPI UPTKblasGetProperty(libraryPropertyType type, int* value);
size_t UPTKBLASAPI UPTKblasGetCudartVersion(void);

UPTKblasStatus_t UPTKBLASAPI UPTKblasSetWorkspace(UPTKblasHandle_t handle,
                                                            void* workspace,
                                                            size_t workspaceSizeInBytes);

UPTKblasStatus_t UPTKBLASAPI UPTKblasSetStream(UPTKblasHandle_t handle, UPTKStream_t streamId);
UPTKblasStatus_t UPTKBLASAPI UPTKblasGetStream(UPTKblasHandle_t handle, UPTKStream_t* streamId);

UPTKblasStatus_t UPTKBLASAPI UPTKblasGetPointerMode(UPTKblasHandle_t handle, UPTKblasPointerMode_t* mode);
UPTKblasStatus_t UPTKBLASAPI UPTKblasSetPointerMode(UPTKblasHandle_t handle, UPTKblasPointerMode_t mode);

UPTKblasStatus_t UPTKBLASAPI UPTKblasGetAtomicsMode(UPTKblasHandle_t handle, UPTKblasAtomicsMode_t* mode);
UPTKblasStatus_t UPTKBLASAPI UPTKblasSetAtomicsMode(UPTKblasHandle_t handle, UPTKblasAtomicsMode_t mode);

UPTKblasStatus_t UPTKBLASAPI UPTKblasGetMathMode(UPTKblasHandle_t handle, UPTKblasMath_t* mode);
UPTKblasStatus_t UPTKBLASAPI UPTKblasSetMathMode(UPTKblasHandle_t handle, UPTKblasMath_t mode);

const char* UPTKBLASAPI UPTKblasGetStatusName(UPTKblasStatus_t status);
const char* UPTKBLASAPI UPTKblasGetStatusString(UPTKblasStatus_t status);

/*
 * UPTKblasStatus_t
 * UPTKblasSetVector (int n, int elemSize, const void *x, int incx,
 *                  void *y, int incy)
 *
 * copies n elements from a vector x in CPU memory space to a vector y
 * in GPU memory space. Elements in both vectors are assumed to have a
 * size of elemSize bytes. Storage spacing between consecutive elements
 * is incx for the source vector x and incy for the destination vector
 * y. In general, y points to an object, or part of an object, allocated
 * via UPTKblasAlloc(). Column major format for two-dimensional matrices
 * is assumed throughout CUBLAS. Therefore, if the increment for a vector
 * is equal to 1, this access a column vector while using an increment
 * equal to the leading dimension of the respective matrix accesses a
 * row vector.
 *
 * Return Values
 * -------------
 * UPTKBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
 * UPTKBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
 * UPTKBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory
 * UPTKBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSetVector(int n, int elemSize, const void* x, int incx, void* devicePtr, int incy);

/*
 * UPTKblasStatus_t
 * UPTKblasGetVector (int n, int elemSize, const void *x, int incx,
 *                  void *y, int incy)
 *
 * copies n elements from a vector x in GPU memory space to a vector y
 * in CPU memory space. Elements in both vectors are assumed to have a
 * size of elemSize bytes. Storage spacing between consecutive elements
 * is incx for the source vector x and incy for the destination vector
 * y. In general, x points to an object, or part of an object, allocated
 * via UPTKblasAlloc(). Column major format for two-dimensional matrices
 * is assumed throughout CUBLAS. Therefore, if the increment for a vector
 * is equal to 1, this access a column vector while using an increment
 * equal to the leading dimension of the respective matrix accesses a
 * row vector.
 *
 * Return Values
 * -------------
 * UPTKBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
 * UPTKBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
 * UPTKBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory
 * UPTKBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
UPTKblasStatus_t UPTKBLASAPI UPTKblasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy);

/*
 * UPTKblasStatus_t
 * UPTKblasSetMatrix (int rows, int cols, int elemSize, const void *A,
 *                  int lda, void *B, int ldb)
 *
 * copies a tile of rows x cols elements from a matrix A in CPU memory
 * space to a matrix B in GPU memory space. Each element requires storage
 * of elemSize bytes. Both matrices are assumed to be stored in column
 * major format, with the leading dimension (i.e. number of rows) of
 * source matrix A provided in lda, and the leading dimension of matrix B
 * provided in ldb. In general, B points to an object, or part of an
 * object, that was allocated via UPTKblasAlloc().
 *
 * Return Values
 * -------------
 * UPTKBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * UPTKBLAS_STATUS_INVALID_VALUE    if rows or cols < 0, or elemSize, lda, or
 *                                ldb <= 0
 * UPTKBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
 * UPTKBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb);

/*
 * UPTKblasStatus_t
 * UPTKblasGetMatrix (int rows, int cols, int elemSize, const void *A,
 *                  int lda, void *B, int ldb)
 *
 * copies a tile of rows x cols elements from a matrix A in GPU memory
 * space to a matrix B in CPU memory space. Each element requires storage
 * of elemSize bytes. Both matrices are assumed to be stored in column
 * major format, with the leading dimension (i.e. number of rows) of
 * source matrix A provided in lda, and the leading dimension of matrix B
 * provided in ldb. In general, A points to an object, or part of an
 * object, that was allocated via UPTKblasAlloc().
 *
 * Return Values
 * -------------
 * UPTKBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * UPTKBLAS_STATUS_INVALID_VALUE    if rows, cols, eleSize, lda, or ldb <= 0
 * UPTKBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
 * UPTKBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
UPTKblasStatus_t UPTKBLASAPI UPTKblasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb);

/*
 * UPTKblasStatus
 * UPTKblasSetVectorAsync ( int n, int elemSize, const void *x, int incx,
 *                       void *y, int incy, UPTKStream_t stream );
 *
 * UPTKblasSetVectorAsync has the same functionnality as UPTKblasSetVector
 * but the transfer is done asynchronously within the CUDA stream passed
 * in parameter.
 *
 * Return Values
 * -------------
 * UPTKBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
 * UPTKBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
 * UPTKBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory
 * UPTKBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSetVectorAsync(
    int n, int elemSize, const void* hostPtr, int incx, void* devicePtr, int incy, UPTKStream_t stream);
/*
 * UPTKblasStatus
 * UPTKblasGetVectorAsync( int n, int elemSize, const void *x, int incx,
 *                       void *y, int incy, UPTKStream_t stream)
 *
 * UPTKblasGetVectorAsync has the same functionnality as UPTKblasGetVector
 * but the transfer is done asynchronously within the CUDA stream passed
 * in parameter.
 *
 * Return Values
 * -------------
 * UPTKBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
 * UPTKBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
 * UPTKBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory
 * UPTKBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
UPTKblasStatus_t UPTKBLASAPI UPTKblasGetVectorAsync(
    int n, int elemSize, const void* devicePtr, int incx, void* hostPtr, int incy, UPTKStream_t stream);

/*
 * UPTKblasStatus_t
 * UPTKblasSetMatrixAsync (int rows, int cols, int elemSize, const void *A,
 *                       int lda, void *B, int ldb, UPTKStream_t stream)
 *
 * UPTKblasSetMatrixAsync has the same functionnality as UPTKblasSetMatrix
 * but the transfer is done asynchronously within the CUDA stream passed
 * in parameter.
 *
 * Return Values
 * -------------
 * UPTKBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * UPTKBLAS_STATUS_INVALID_VALUE    if rows or cols < 0, or elemSize, lda, or
 *                                ldb <= 0
 * UPTKBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
 * UPTKBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
UPTKblasStatus_t UPTKBLASAPI
UPTKblasSetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, UPTKStream_t stream);

/*
 * UPTKblasStatus_t
 * UPTKblasGetMatrixAsync (int rows, int cols, int elemSize, const void *A,
 *                       int lda, void *B, int ldb, UPTKStream_t stream)
 *
 * UPTKblasGetMatrixAsync has the same functionnality as UPTKblasGetMatrix
 * but the transfer is done asynchronously within the CUDA stream passed
 * in parameter.
 *
 * Return Values
 * -------------
 * UPTKBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * UPTKBLAS_STATUS_INVALID_VALUE    if rows, cols, eleSize, lda, or ldb <= 0
 * UPTKBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
 * UPTKBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
UPTKblasStatus_t UPTKBLASAPI
UPTKblasGetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, UPTKStream_t stream);

/* ---------------- CUBLAS BLAS1 functions ---------------- */
UPTKblasStatus_t UPTKBLASAPI UPTKblasNrm2Ex(UPTKblasHandle_t handle,
                                                   int n,
                                                   const void* x,
                                                   UPTKDataType xType,
                                                   int incx,
                                                   void* result,
                                                   UPTKDataType resultType,
                                                   UPTKDataType executionType); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI
UPTKblasSnrm2(UPTKblasHandle_t handle, int n, const float* x, int incx, float* result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI
UPTKblasDnrm2(UPTKblasHandle_t handle, int n, const double* x, int incx, double* result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI
UPTKblasScnrm2(UPTKblasHandle_t handle, int n, const cuComplex* x, int incx, float* result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasDznrm2(
    UPTKblasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasDotEx(UPTKblasHandle_t handle,
                                                  int n,
                                                  const void* x,
                                                  UPTKDataType xType,
                                                  int incx,
                                                  const void* y,
                                                  UPTKDataType yType,
                                                  int incy,
                                                  void* result,
                                                  UPTKDataType resultType,
                                                  UPTKDataType executionType);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDotcEx(UPTKblasHandle_t handle,
                                                   int n,
                                                   const void* x,
                                                   UPTKDataType xType,
                                                   int incx,
                                                   const void* y,
                                                   UPTKDataType yType,
                                                   int incy,
                                                   void* result,
                                                   UPTKDataType resultType,
                                                   UPTKDataType executionType);

UPTKblasStatus_t UPTKBLASAPI UPTKblasSdot(UPTKblasHandle_t handle,
                                                    int n,
                                                    const float* x,
                                                    int incx,
                                                    const float* y,
                                                    int incy,
                                                    float* result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasDdot(UPTKblasHandle_t handle,
                                                    int n,
                                                    const double* x,
                                                    int incx,
                                                    const double* y,
                                                    int incy,
                                                    double* result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasCdotu(UPTKblasHandle_t handle,
                                                     int n,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* y,
                                                     int incy,
                                                     cuComplex* result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasCdotc(UPTKblasHandle_t handle,
                                                     int n,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* y,
                                                     int incy,
                                                     cuComplex* result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasZdotu(UPTKblasHandle_t handle,
                                                     int n,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* y,
                                                     int incy,
                                                     cuDoubleComplex* result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasZdotc(UPTKblasHandle_t handle,
                                                     int n,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* y,
                                                     int incy,
                                                     cuDoubleComplex* result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasScalEx(UPTKblasHandle_t handle,
                                                   int n,
                                                   const void* alpha, /* host or device pointer */
                                                   UPTKDataType alphaType,
                                                   void* x,
                                                   UPTKDataType xType,
                                                   int incx,
                                                   UPTKDataType executionType);

UPTKblasStatus_t UPTKBLASAPI UPTKblasSscal(UPTKblasHandle_t handle,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     float* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDscal(UPTKblasHandle_t handle,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     double* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCscal(UPTKblasHandle_t handle,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     cuComplex* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCsscal(UPTKblasHandle_t handle,
                                                      int n,
                                                      const float* alpha, /* host or device pointer */
                                                      cuComplex* x,
                                                      int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZscal(UPTKblasHandle_t handle,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     cuDoubleComplex* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZdscal(UPTKblasHandle_t handle,
                                                      int n,
                                                      const double* alpha, /* host or device pointer */
                                                      cuDoubleComplex* x,
                                                      int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasAxpyEx(UPTKblasHandle_t handle,
                                                   int n,
                                                   const void* alpha, /* host or device pointer */
                                                   UPTKDataType alphaType,
                                                   const void* x,
                                                   UPTKDataType xType,
                                                   int incx,
                                                   void* y,
                                                   UPTKDataType yType,
                                                   int incy,
                                                   UPTKDataType executiontype);

UPTKblasStatus_t UPTKBLASAPI UPTKblasSaxpy(UPTKblasHandle_t handle,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* x,
                                                     int incx,
                                                     float* y,
                                                     int incy);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDaxpy(UPTKblasHandle_t handle,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* x,
                                                     int incx,
                                                     double* y,
                                                     int incy);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCaxpy(UPTKblasHandle_t handle,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* x,
                                                     int incx,
                                                     cuComplex* y,
                                                     int incy);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZaxpy(UPTKblasHandle_t handle,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     cuDoubleComplex* y,
                                                     int incy);

UPTKblasStatus_t UPTKBLASAPI
UPTKblasScopy(UPTKblasHandle_t handle, int n, const float* x, int incx, float* y, int incy);

UPTKblasStatus_t UPTKBLASAPI
UPTKblasDcopy(UPTKblasHandle_t handle, int n, const double* x, int incx, double* y, int incy);

UPTKblasStatus_t UPTKBLASAPI
UPTKblasCcopy(UPTKblasHandle_t handle, int n, const cuComplex* x, int incx, cuComplex* y, int incy);

UPTKblasStatus_t UPTKBLASAPI
UPTKblasZcopy(UPTKblasHandle_t handle, int n, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);

UPTKblasStatus_t UPTKBLASAPI
UPTKblasSswap(UPTKblasHandle_t handle, int n, float* x, int incx, float* y, int incy);

UPTKblasStatus_t UPTKBLASAPI
UPTKblasDswap(UPTKblasHandle_t handle, int n, double* x, int incx, double* y, int incy);

UPTKblasStatus_t UPTKBLASAPI
UPTKblasCswap(UPTKblasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy);

UPTKblasStatus_t UPTKBLASAPI
UPTKblasZswap(UPTKblasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);

UPTKblasStatus_t UPTKBLASAPI
UPTKblasIsamax(UPTKblasHandle_t handle, int n, const float* x, int incx, int* result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI
UPTKblasIdamax(UPTKblasHandle_t handle, int n, const double* x, int incx, int* result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI
UPTKblasIcamax(UPTKblasHandle_t handle, int n, const cuComplex* x, int incx, int* result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI
UPTKblasIsamin(UPTKblasHandle_t handle, int n, const float* x, int incx, int* result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI
UPTKblasIzamax(UPTKblasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI
UPTKblasIdamin(UPTKblasHandle_t handle, int n, const double* x, int incx, int* result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI
UPTKblasIcamin(UPTKblasHandle_t handle, int n, const cuComplex* x, int incx, int* result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasIzamin(
    UPTKblasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasIaminEx(
    UPTKblasHandle_t handle, int n, const void* x, UPTKDataType xType, int incx, int* result /* host or device pointer */
);

UPTKblasStatus_t UPTKBLASAPI
UPTKblasSasum(UPTKblasHandle_t handle, int n, const float* x, int incx, float* result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI
UPTKblasDasum(UPTKblasHandle_t handle, int n, const double* x, int incx, double* result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI
UPTKblasScasum(UPTKblasHandle_t handle, int n, const cuComplex* x, int incx, float* result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasDzasum(
    UPTKblasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasSrot(UPTKblasHandle_t handle,
                                                    int n,
                                                    float* x,
                                                    int incx,
                                                    float* y,
                                                    int incy,
                                                    const float* c,  /* host or device pointer */
                                                    const float* s); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasDrot(UPTKblasHandle_t handle,
                                                    int n,
                                                    double* x,
                                                    int incx,
                                                    double* y,
                                                    int incy,
                                                    const double* c,  /* host or device pointer */
                                                    const double* s); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasCrot(UPTKblasHandle_t handle,
                                                    int n,
                                                    cuComplex* x,
                                                    int incx,
                                                    cuComplex* y,
                                                    int incy,
                                                    const float* c,      /* host or device pointer */
                                                    const cuComplex* s); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasCsrot(UPTKblasHandle_t handle,
                                                     int n,
                                                     cuComplex* x,
                                                     int incx,
                                                     cuComplex* y,
                                                     int incy,
                                                     const float* c,  /* host or device pointer */
                                                     const float* s); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasZrot(UPTKblasHandle_t handle,
                                                    int n,
                                                    cuDoubleComplex* x,
                                                    int incx,
                                                    cuDoubleComplex* y,
                                                    int incy,
                                                    const double* c,           /* host or device pointer */
                                                    const cuDoubleComplex* s); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasZdrot(UPTKblasHandle_t handle,
                                                     int n,
                                                     cuDoubleComplex* x,
                                                     int incx,
                                                     cuDoubleComplex* y,
                                                     int incy,
                                                     const double* c,  /* host or device pointer */
                                                     const double* s); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasRotEx(UPTKblasHandle_t handle,
                                                  int n,
                                                  void* x,
                                                  UPTKDataType xType,
                                                  int incx,
                                                  void* y,
                                                  UPTKDataType yType,
                                                  int incy,
                                                  const void* c, /* host or device pointer */
                                                  const void* s,
                                                  UPTKDataType csType,
                                                  UPTKDataType executiontype);

UPTKblasStatus_t UPTKBLASAPI UPTKblasSrotg(UPTKblasHandle_t handle,
                                                     float* a,  /* host or device pointer */
                                                     float* b,  /* host or device pointer */
                                                     float* c,  /* host or device pointer */
                                                     float* s); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasDrotg(UPTKblasHandle_t handle,
                                                     double* a,  /* host or device pointer */
                                                     double* b,  /* host or device pointer */
                                                     double* c,  /* host or device pointer */
                                                     double* s); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasCrotg(UPTKblasHandle_t handle,
                                                     cuComplex* a,  /* host or device pointer */
                                                     cuComplex* b,  /* host or device pointer */
                                                     float* c,      /* host or device pointer */
                                                     cuComplex* s); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasZrotg(UPTKblasHandle_t handle,
                                                     cuDoubleComplex* a,  /* host or device pointer */
                                                     cuDoubleComplex* b,  /* host or device pointer */
                                                     double* c,           /* host or device pointer */
                                                     cuDoubleComplex* s); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasSrotm(UPTKblasHandle_t handle,
                                                     int n,
                                                     float* x,
                                                     int incx,
                                                     float* y,
                                                     int incy,
                                                     const float* param); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasDrotm(UPTKblasHandle_t handle,
                                                     int n,
                                                     double* x,
                                                     int incx,
                                                     double* y,
                                                     int incy,
                                                     const double* param); /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasSrotmg(UPTKblasHandle_t handle,
                                                      float* d1,       /* host or device pointer */
                                                      float* d2,       /* host or device pointer */
                                                      float* x1,       /* host or device pointer */
                                                      const float* y1, /* host or device pointer */
                                                      float* param);   /* host or device pointer */

UPTKblasStatus_t UPTKBLASAPI UPTKblasDrotmg(UPTKblasHandle_t handle,
                                                      double* d1,       /* host or device pointer */
                                                      double* d2,       /* host or device pointer */
                                                      double* x1,       /* host or device pointer */
                                                      const double* y1, /* host or device pointer */
                                                      double* param);   /* host or device pointer */
/* --------------- CUBLAS BLAS2 functions  ---------------- */

/* GEMV */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSgemv(UPTKblasHandle_t handle,
                                                     UPTKblasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* x,
                                                     int incx,
                                                     const float* beta, /* host or device pointer */
                                                     float* y,
                                                     int incy);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDgemv(UPTKblasHandle_t handle,
                                                     UPTKblasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* x,
                                                     int incx,
                                                     const double* beta, /* host or device pointer */
                                                     double* y,
                                                     int incy);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCgemv(UPTKblasHandle_t handle,
                                                     UPTKblasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* y,
                                                     int incy);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZgemv(UPTKblasHandle_t handle,
                                                     UPTKblasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* y,
                                                     int incy);
/* GBMV */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSgbmv(UPTKblasHandle_t handle,
                                                     UPTKblasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     int kl,
                                                     int ku,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* x,
                                                     int incx,
                                                     const float* beta, /* host or device pointer */
                                                     float* y,
                                                     int incy);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDgbmv(UPTKblasHandle_t handle,
                                                     UPTKblasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     int kl,
                                                     int ku,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* x,
                                                     int incx,
                                                     const double* beta, /* host or device pointer */
                                                     double* y,
                                                     int incy);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCgbmv(UPTKblasHandle_t handle,
                                                     UPTKblasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     int kl,
                                                     int ku,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* y,
                                                     int incy);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZgbmv(UPTKblasHandle_t handle,
                                                     UPTKblasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     int kl,
                                                     int ku,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* y,
                                                     int incy);

/* TRMV */
UPTKblasStatus_t UPTKBLASAPI UPTKblasStrmv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     const float* A,
                                                     int lda,
                                                     float* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDtrmv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     const double* A,
                                                     int lda,
                                                     double* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCtrmv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     const cuComplex* A,
                                                     int lda,
                                                     cuComplex* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZtrmv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     cuDoubleComplex* x,
                                                     int incx);

/* TBMV */
UPTKblasStatus_t UPTKBLASAPI UPTKblasStbmv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const float* A,
                                                     int lda,
                                                     float* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDtbmv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const double* A,
                                                     int lda,
                                                     double* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCtbmv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const cuComplex* A,
                                                     int lda,
                                                     cuComplex* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZtbmv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     cuDoubleComplex* x,
                                                     int incx);

/* TPMV */
UPTKblasStatus_t UPTKBLASAPI UPTKblasStpmv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     const float* AP,
                                                     float* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDtpmv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     const double* AP,
                                                     double* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCtpmv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     const cuComplex* AP,
                                                     cuComplex* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZtpmv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     const cuDoubleComplex* AP,
                                                     cuDoubleComplex* x,
                                                     int incx);

/* TRSV */
UPTKblasStatus_t UPTKBLASAPI UPTKblasStrsv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     const float* A,
                                                     int lda,
                                                     float* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDtrsv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     const double* A,
                                                     int lda,
                                                     double* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCtrsv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     const cuComplex* A,
                                                     int lda,
                                                     cuComplex* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZtrsv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     cuDoubleComplex* x,
                                                     int incx);

/* TPSV */
UPTKblasStatus_t UPTKBLASAPI UPTKblasStpsv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     const float* AP,
                                                     float* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDtpsv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     const double* AP,
                                                     double* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCtpsv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     const cuComplex* AP,
                                                     cuComplex* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZtpsv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     const cuDoubleComplex* AP,
                                                     cuDoubleComplex* x,
                                                     int incx);
/* TBSV */
UPTKblasStatus_t UPTKBLASAPI UPTKblasStbsv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const float* A,
                                                     int lda,
                                                     float* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDtbsv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const double* A,
                                                     int lda,
                                                     double* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCtbsv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const cuComplex* A,
                                                     int lda,
                                                     cuComplex* x,
                                                     int incx);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZtbsv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     cuDoubleComplex* x,
                                                     int incx);

/* SYMV/HEMV */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSsymv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* x,
                                                     int incx,
                                                     const float* beta, /* host or device pointer */
                                                     float* y,
                                                     int incy);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDsymv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* x,
                                                     int incx,
                                                     const double* beta, /* host or device pointer */
                                                     double* y,
                                                     int incy);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCsymv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* y,
                                                     int incy);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZsymv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* y,
                                                     int incy);

UPTKblasStatus_t UPTKBLASAPI UPTKblasChemv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* y,
                                                     int incy);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZhemv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* y,
                                                     int incy);

/* SBMV/HBMV */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSsbmv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     int k,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* x,
                                                     int incx,
                                                     const float* beta, /* host or device pointer */
                                                     float* y,
                                                     int incy);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDsbmv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     int k,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* x,
                                                     int incx,
                                                     const double* beta, /* host or device pointer */
                                                     double* y,
                                                     int incy);

UPTKblasStatus_t UPTKBLASAPI UPTKblasChbmv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     int k,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* y,
                                                     int incy);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZhbmv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     int k,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* y,
                                                     int incy);

/* SPMV/HPMV */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSspmv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* AP,
                                                     const float* x,
                                                     int incx,
                                                     const float* beta, /* host or device pointer */
                                                     float* y,
                                                     int incy);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDspmv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* AP,
                                                     const double* x,
                                                     int incx,
                                                     const double* beta, /* host or device pointer */
                                                     double* y,
                                                     int incy);

UPTKblasStatus_t UPTKBLASAPI UPTKblasChpmv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* AP,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* y,
                                                     int incy);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZhpmv(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* AP,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* y,
                                                     int incy);

/* GER */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSger(UPTKblasHandle_t handle,
                                                    int m,
                                                    int n,
                                                    const float* alpha, /* host or device pointer */
                                                    const float* x,
                                                    int incx,
                                                    const float* y,
                                                    int incy,
                                                    float* A,
                                                    int lda);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDger(UPTKblasHandle_t handle,
                                                    int m,
                                                    int n,
                                                    const double* alpha, /* host or device pointer */
                                                    const double* x,
                                                    int incx,
                                                    const double* y,
                                                    int incy,
                                                    double* A,
                                                    int lda);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCgeru(UPTKblasHandle_t handle,
                                                     int m,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* y,
                                                     int incy,
                                                     cuComplex* A,
                                                     int lda);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCgerc(UPTKblasHandle_t handle,
                                                     int m,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* y,
                                                     int incy,
                                                     cuComplex* A,
                                                     int lda);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZgeru(UPTKblasHandle_t handle,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* y,
                                                     int incy,
                                                     cuDoubleComplex* A,
                                                     int lda);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZgerc(UPTKblasHandle_t handle,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* y,
                                                     int incy,
                                                     cuDoubleComplex* A,
                                                     int lda);

/* SYR/HER */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSsyr(UPTKblasHandle_t handle,
                                                    UPTKblasFillMode_t uplo,
                                                    int n,
                                                    const float* alpha, /* host or device pointer */
                                                    const float* x,
                                                    int incx,
                                                    float* A,
                                                    int lda);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDsyr(UPTKblasHandle_t handle,
                                                    UPTKblasFillMode_t uplo,
                                                    int n,
                                                    const double* alpha, /* host or device pointer */
                                                    const double* x,
                                                    int incx,
                                                    double* A,
                                                    int lda);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCsyr(UPTKblasHandle_t handle,
                                                    UPTKblasFillMode_t uplo,
                                                    int n,
                                                    const cuComplex* alpha, /* host or device pointer */
                                                    const cuComplex* x,
                                                    int incx,
                                                    cuComplex* A,
                                                    int lda);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZsyr(UPTKblasHandle_t handle,
                                                    UPTKblasFillMode_t uplo,
                                                    int n,
                                                    const cuDoubleComplex* alpha, /* host or device pointer */
                                                    const cuDoubleComplex* x,
                                                    int incx,
                                                    cuDoubleComplex* A,
                                                    int lda);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCher(UPTKblasHandle_t handle,
                                                    UPTKblasFillMode_t uplo,
                                                    int n,
                                                    const float* alpha, /* host or device pointer */
                                                    const cuComplex* x,
                                                    int incx,
                                                    cuComplex* A,
                                                    int lda);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZher(UPTKblasHandle_t handle,
                                                    UPTKblasFillMode_t uplo,
                                                    int n,
                                                    const double* alpha, /* host or device pointer */
                                                    const cuDoubleComplex* x,
                                                    int incx,
                                                    cuDoubleComplex* A,
                                                    int lda);

/* SPR/HPR */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSspr(UPTKblasHandle_t handle,
                                                    UPTKblasFillMode_t uplo,
                                                    int n,
                                                    const float* alpha, /* host or device pointer */
                                                    const float* x,
                                                    int incx,
                                                    float* AP);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDspr(UPTKblasHandle_t handle,
                                                    UPTKblasFillMode_t uplo,
                                                    int n,
                                                    const double* alpha, /* host or device pointer */
                                                    const double* x,
                                                    int incx,
                                                    double* AP);

UPTKblasStatus_t UPTKBLASAPI UPTKblasChpr(UPTKblasHandle_t handle,
                                                    UPTKblasFillMode_t uplo,
                                                    int n,
                                                    const float* alpha, /* host or device pointer */
                                                    const cuComplex* x,
                                                    int incx,
                                                    cuComplex* AP);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZhpr(UPTKblasHandle_t handle,
                                                    UPTKblasFillMode_t uplo,
                                                    int n,
                                                    const double* alpha, /* host or device pointer */
                                                    const cuDoubleComplex* x,
                                                    int incx,
                                                    cuDoubleComplex* AP);

/* SYR2/HER2 */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSsyr2(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* x,
                                                     int incx,
                                                     const float* y,
                                                     int incy,
                                                     float* A,
                                                     int lda);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDsyr2(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* x,
                                                     int incx,
                                                     const double* y,
                                                     int incy,
                                                     double* A,
                                                     int lda);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCsyr2(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* y,
                                                     int incy,
                                                     cuComplex* A,
                                                     int lda);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZsyr2(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* y,
                                                     int incy,
                                                     cuDoubleComplex* A,
                                                     int lda);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCher2(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* y,
                                                     int incy,
                                                     cuComplex* A,
                                                     int lda);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZher2(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* y,
                                                     int incy,
                                                     cuDoubleComplex* A,
                                                     int lda);

/* SPR2/HPR2 */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSspr2(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* x,
                                                     int incx,
                                                     const float* y,
                                                     int incy,
                                                     float* AP);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDspr2(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* x,
                                                     int incx,
                                                     const double* y,
                                                     int incy,
                                                     double* AP);

UPTKblasStatus_t UPTKBLASAPI UPTKblasChpr2(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* y,
                                                     int incy,
                                                     cuComplex* AP);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZhpr2(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* y,
                                                     int incy,
                                                     cuDoubleComplex* AP);
/* BATCH GEMV */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSgemvBatched(UPTKblasHandle_t handle,
                                                         UPTKblasOperation_t trans,
                                                         int m,
                                                         int n,
                                                         const float* alpha, /* host or device pointer */
                                                         const float* const Aarray[],
                                                         int lda,
                                                         const float* const xarray[],
                                                         int incx,
                                                         const float* beta, /* host or device pointer */
                                                         float* const yarray[],
                                                         int incy,
                                                         int batchCount);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDgemvBatched(UPTKblasHandle_t handle,
                                                         UPTKblasOperation_t trans,
                                                         int m,
                                                         int n,
                                                         const double* alpha, /* host or device pointer */
                                                         const double* const Aarray[],
                                                         int lda,
                                                         const double* const xarray[],
                                                         int incx,
                                                         const double* beta, /* host or device pointer */
                                                         double* const yarray[],
                                                         int incy,
                                                         int batchCount);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCgemvBatched(UPTKblasHandle_t handle,
                                                         UPTKblasOperation_t trans,
                                                         int m,
                                                         int n,
                                                         const cuComplex* alpha, /* host or device pointer */
                                                         const cuComplex* const Aarray[],
                                                         int lda,
                                                         const cuComplex* const xarray[],
                                                         int incx,
                                                         const cuComplex* beta, /* host or device pointer */
                                                         cuComplex* const yarray[],
                                                         int incy,
                                                         int batchCount);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZgemvBatched(UPTKblasHandle_t handle,
                                                         UPTKblasOperation_t trans,
                                                         int m,
                                                         int n,
                                                         const cuDoubleComplex* alpha, /* host or device pointer */
                                                         const cuDoubleComplex* const Aarray[],
                                                         int lda,
                                                         const cuDoubleComplex* const xarray[],
                                                         int incx,
                                                         const cuDoubleComplex* beta, /* host or device pointer */
                                                         cuDoubleComplex* const yarray[],
                                                         int incy,
                                                         int batchCount);

#if defined(__cplusplus)
UPTKblasStatus_t UPTKBLASAPI UPTKblasHSHgemvBatched(UPTKblasHandle_t handle,
                                                           UPTKblasOperation_t trans,
                                                           int m,
                                                           int n,
                                                           const float* alpha, /* host or device pointer */
                                                           const __half* const Aarray[],
                                                           int lda,
                                                           const __half* const xarray[],
                                                           int incx,
                                                           const float* beta, /* host or device pointer */
                                                           __half* const yarray[],
                                                           int incy,
                                                           int batchCount);

UPTKblasStatus_t UPTKBLASAPI UPTKblasHSSgemvBatched(UPTKblasHandle_t handle,
                                                           UPTKblasOperation_t trans,
                                                           int m,
                                                           int n,
                                                           const float* alpha, /* host or device pointer */
                                                           const __half* const Aarray[],
                                                           int lda,
                                                           const __half* const xarray[],
                                                           int incx,
                                                           const float* beta, /* host or device pointer */
                                                           float* const yarray[],
                                                           int incy,
                                                           int batchCount);

UPTKblasStatus_t UPTKBLASAPI UPTKblasTSTgemvBatched(UPTKblasHandle_t handle,
                                                           UPTKblasOperation_t trans,
                                                           int m,
                                                           int n,
                                                           const float* alpha, /* host or device pointer */
                                                           const __nv_bfloat16* const Aarray[],
                                                           int lda,
                                                           const __nv_bfloat16* const xarray[],
                                                           int incx,
                                                           const float* beta, /* host or device pointer */
                                                           __nv_bfloat16* const yarray[],
                                                           int incy,
                                                           int batchCount);

UPTKblasStatus_t UPTKBLASAPI UPTKblasTSSgemvBatched(UPTKblasHandle_t handle,
                                                           UPTKblasOperation_t trans,
                                                           int m,
                                                           int n,
                                                           const float* alpha, /* host or device pointer */
                                                           const __nv_bfloat16* const Aarray[],
                                                           int lda,
                                                           const __nv_bfloat16* const xarray[],
                                                           int incx,
                                                           const float* beta, /* host or device pointer */
                                                           float* const yarray[],
                                                           int incy,
                                                           int batchCount);
#endif

UPTKblasStatus_t UPTKBLASAPI UPTKblasSgemvStridedBatched(UPTKblasHandle_t handle,
                                                                UPTKblasOperation_t trans,
                                                                int m,
                                                                int n,
                                                                const float* alpha, /* host or device pointer */
                                                                const float* A,
                                                                int lda,
                                                                long long int strideA, /* purposely signed */
                                                                const float* x,
                                                                int incx,
                                                                long long int stridex,
                                                                const float* beta, /* host or device pointer */
                                                                float* y,
                                                                int incy,
                                                                long long int stridey,
                                                                int batchCount);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDgemvStridedBatched(UPTKblasHandle_t handle,
                                                                UPTKblasOperation_t trans,
                                                                int m,
                                                                int n,
                                                                const double* alpha, /* host or device pointer */
                                                                const double* A,
                                                                int lda,
                                                                long long int strideA, /* purposely signed */
                                                                const double* x,
                                                                int incx,
                                                                long long int stridex,
                                                                const double* beta, /* host or device pointer */
                                                                double* y,
                                                                int incy,
                                                                long long int stridey,
                                                                int batchCount);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCgemvStridedBatched(UPTKblasHandle_t handle,
                                                                UPTKblasOperation_t trans,
                                                                int m,
                                                                int n,
                                                                const cuComplex* alpha, /* host or device pointer */
                                                                const cuComplex* A,
                                                                int lda,
                                                                long long int strideA, /* purposely signed */
                                                                const cuComplex* x,
                                                                int incx,
                                                                long long int stridex,
                                                                const cuComplex* beta, /* host or device pointer */
                                                                cuComplex* y,
                                                                int incy,
                                                                long long int stridey,
                                                                int batchCount);

UPTKblasStatus_t UPTKBLASAPI
UPTKblasZgemvStridedBatched(UPTKblasHandle_t handle,
                          UPTKblasOperation_t trans,
                          int m,
                          int n,
                          const cuDoubleComplex* alpha, /* host or device pointer */
                          const cuDoubleComplex* A,
                          int lda,
                          long long int strideA, /* purposely signed */
                          const cuDoubleComplex* x,
                          int incx,
                          long long int stridex,
                          const cuDoubleComplex* beta, /* host or device pointer */
                          cuDoubleComplex* y,
                          int incy,
                          long long int stridey,
                          int batchCount);

#if defined(__cplusplus)
UPTKblasStatus_t UPTKBLASAPI UPTKblasHSHgemvStridedBatched(UPTKblasHandle_t handle,
                                                                  UPTKblasOperation_t trans,
                                                                  int m,
                                                                  int n,
                                                                  const float* alpha, /* host or device pointer */
                                                                  const __half* A,
                                                                  int lda,
                                                                  long long int strideA, /* purposely signed */
                                                                  const __half* x,
                                                                  int incx,
                                                                  long long int stridex,
                                                                  const float* beta, /* host or device pointer */
                                                                  __half* y,
                                                                  int incy,
                                                                  long long int stridey,
                                                                  int batchCount);

UPTKblasStatus_t UPTKBLASAPI UPTKblasHSSgemvStridedBatched(UPTKblasHandle_t handle,
                                                                  UPTKblasOperation_t trans,
                                                                  int m,
                                                                  int n,
                                                                  const float* alpha, /* host or device pointer */
                                                                  const __half* A,
                                                                  int lda,
                                                                  long long int strideA, /* purposely signed */
                                                                  const __half* x,
                                                                  int incx,
                                                                  long long int stridex,
                                                                  const float* beta, /* host or device pointer */
                                                                  float* y,
                                                                  int incy,
                                                                  long long int stridey,
                                                                  int batchCount);

UPTKblasStatus_t UPTKBLASAPI UPTKblasTSTgemvStridedBatched(UPTKblasHandle_t handle,
                                                                  UPTKblasOperation_t trans,
                                                                  int m,
                                                                  int n,
                                                                  const float* alpha, /* host or device pointer */
                                                                  const __nv_bfloat16* A,
                                                                  int lda,
                                                                  long long int strideA, /* purposely signed */
                                                                  const __nv_bfloat16* x,
                                                                  int incx,
                                                                  long long int stridex,
                                                                  const float* beta, /* host or device pointer */
                                                                  __nv_bfloat16* y,
                                                                  int incy,
                                                                  long long int stridey,
                                                                  int batchCount);

UPTKblasStatus_t UPTKBLASAPI UPTKblasTSSgemvStridedBatched(UPTKblasHandle_t handle,
                                                                  UPTKblasOperation_t trans,
                                                                  int m,
                                                                  int n,
                                                                  const float* alpha, /* host or device pointer */
                                                                  const __nv_bfloat16* A,
                                                                  int lda,
                                                                  long long int strideA, /* purposely signed */
                                                                  const __nv_bfloat16* x,
                                                                  int incx,
                                                                  long long int stridex,
                                                                  const float* beta, /* host or device pointer */
                                                                  float* y,
                                                                  int incy,
                                                                  long long int stridey,
                                                                  int batchCount);
#endif
/* ---------------- CUBLAS BLAS3 functions ---------------- */

/* GEMM */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSgemm(UPTKblasHandle_t handle,
                                                     UPTKblasOperation_t transa,
                                                     UPTKblasOperation_t transb,
                                                     int m,
                                                     int n,
                                                     int k,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* B,
                                                     int ldb,
                                                     const float* beta, /* host or device pointer */
                                                     float* C,
                                                     int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDgemm(UPTKblasHandle_t handle,
                                                     UPTKblasOperation_t transa,
                                                     UPTKblasOperation_t transb,
                                                     int m,
                                                     int n,
                                                     int k,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* B,
                                                     int ldb,
                                                     const double* beta, /* host or device pointer */
                                                     double* C,
                                                     int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCgemm(UPTKblasHandle_t handle,
                                                     UPTKblasOperation_t transa,
                                                     UPTKblasOperation_t transb,
                                                     int m,
                                                     int n,
                                                     int k,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* B,
                                                     int ldb,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* C,
                                                     int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCgemm3m(UPTKblasHandle_t handle,
                                                    UPTKblasOperation_t transa,
                                                    UPTKblasOperation_t transb,
                                                    int m,
                                                    int n,
                                                    int k,
                                                    const cuComplex* alpha, /* host or device pointer */
                                                    const cuComplex* A,
                                                    int lda,
                                                    const cuComplex* B,
                                                    int ldb,
                                                    const cuComplex* beta, /* host or device pointer */
                                                    cuComplex* C,
                                                    int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZgemm(UPTKblasHandle_t handle,
                                                     UPTKblasOperation_t transa,
                                                     UPTKblasOperation_t transb,
                                                     int m,
                                                     int n,
                                                     int k,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* B,
                                                     int ldb,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* C,
                                                     int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZgemm3m(UPTKblasHandle_t handle,
                                                    UPTKblasOperation_t transa,
                                                    UPTKblasOperation_t transb,
                                                    int m,
                                                    int n,
                                                    int k,
                                                    const cuDoubleComplex* alpha, /* host or device pointer */
                                                    const cuDoubleComplex* A,
                                                    int lda,
                                                    const cuDoubleComplex* B,
                                                    int ldb,
                                                    const cuDoubleComplex* beta, /* host or device pointer */
                                                    cuDoubleComplex* C,
                                                    int ldc);

#if defined(__cplusplus)
UPTKblasStatus_t UPTKBLASAPI UPTKblasHgemm(UPTKblasHandle_t handle,
                                                  UPTKblasOperation_t transa,
                                                  UPTKblasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  int k,
                                                  const __half* alpha, /* host or device pointer */
                                                  const __half* A,
                                                  int lda,
                                                  const __half* B,
                                                  int ldb,
                                                  const __half* beta, /* host or device pointer */
                                                  __half* C,
                                                  int ldc);
#endif
/* IO in FP16/FP32, computation in float */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSgemmEx(UPTKblasHandle_t handle,
                                                    UPTKblasOperation_t transa,
                                                    UPTKblasOperation_t transb,
                                                    int m,
                                                    int n,
                                                    int k,
                                                    const float* alpha, /* host or device pointer */
                                                    const void* A,
                                                    UPTKDataType Atype,
                                                    int lda,
                                                    const void* B,
                                                    UPTKDataType Btype,
                                                    int ldb,
                                                    const float* beta, /* host or device pointer */
                                                    void* C,
                                                    UPTKDataType Ctype,
                                                    int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasGemmEx(UPTKblasHandle_t handle,
                                                   UPTKblasOperation_t transa,
                                                   UPTKblasOperation_t transb,
                                                   int m,
                                                   int n,
                                                   int k,
                                                   const void* alpha, /* host or device pointer */
                                                   const void* A,
                                                   UPTKDataType Atype,
                                                   int lda,
                                                   const void* B,
                                                   UPTKDataType Btype,
                                                   int ldb,
                                                   const void* beta, /* host or device pointer */
                                                   void* C,
                                                   UPTKDataType Ctype,
                                                   int ldc,
                                                   UPTKblasComputeType_t computeType,
                                                   UPTKblasGemmAlgo_t algo);

/* IO in Int8 complex/cuComplex, computation in cuComplex */
UPTKblasStatus_t UPTKBLASAPI UPTKblasCgemmEx(UPTKblasHandle_t handle,
                                                    UPTKblasOperation_t transa,
                                                    UPTKblasOperation_t transb,
                                                    int m,
                                                    int n,
                                                    int k,
                                                    const cuComplex* alpha,
                                                    const void* A,
                                                    UPTKDataType Atype,
                                                    int lda,
                                                    const void* B,
                                                    UPTKDataType Btype,
                                                    int ldb,
                                                    const cuComplex* beta,
                                                    void* C,
                                                    UPTKDataType Ctype,
                                                    int ldc);

/* SYRK */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSsyrk(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     int n,
                                                     int k,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* beta, /* host or device pointer */
                                                     float* C,
                                                     int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDsyrk(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     int n,
                                                     int k,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* beta, /* host or device pointer */
                                                     double* C,
                                                     int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCsyrk(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     int n,
                                                     int k,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* C,
                                                     int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZsyrk(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     int n,
                                                     int k,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* C,
                                                     int ldc);
/* IO in Int8 complex/cuComplex, computation in cuComplex */
UPTKblasStatus_t UPTKBLASAPI UPTKblasCsyrkEx(UPTKblasHandle_t handle,
                                                    UPTKblasFillMode_t uplo,
                                                    UPTKblasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const cuComplex* alpha, /* host or device pointer */
                                                    const void* A,
                                                    UPTKDataType Atype,
                                                    int lda,
                                                    const cuComplex* beta, /* host or device pointer */
                                                    void* C,
                                                    UPTKDataType Ctype,
                                                    int ldc);

/* IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math */
UPTKblasStatus_t UPTKBLASAPI UPTKblasCsyrk3mEx(UPTKblasHandle_t handle,
                                                      UPTKblasFillMode_t uplo,
                                                      UPTKblasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuComplex* alpha,
                                                      const void* A,
                                                      UPTKDataType Atype,
                                                      int lda,
                                                      const cuComplex* beta,
                                                      void* C,
                                                      UPTKDataType Ctype,
                                                      int ldc);

/* HERK */
UPTKblasStatus_t UPTKBLASAPI UPTKblasCherk(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     int n,
                                                     int k,
                                                     const float* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const float* beta, /* host or device pointer */
                                                     cuComplex* C,
                                                     int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZherk(UPTKblasHandle_t handle,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     int n,
                                                     int k,
                                                     const double* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const double* beta, /* host or device pointer */
                                                     cuDoubleComplex* C,
                                                     int ldc);

/* IO in Int8 complex/cuComplex, computation in cuComplex */
UPTKblasStatus_t UPTKBLASAPI UPTKblasCherkEx(UPTKblasHandle_t handle,
                                                    UPTKblasFillMode_t uplo,
                                                    UPTKblasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const float* alpha, /* host or device pointer */
                                                    const void* A,
                                                    UPTKDataType Atype,
                                                    int lda,
                                                    const float* beta, /* host or device pointer */
                                                    void* C,
                                                    UPTKDataType Ctype,
                                                    int ldc);

/* IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math */
UPTKblasStatus_t UPTKBLASAPI UPTKblasCherk3mEx(UPTKblasHandle_t handle,
                                                      UPTKblasFillMode_t uplo,
                                                      UPTKblasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const float* alpha,
                                                      const void* A,
                                                      UPTKDataType Atype,
                                                      int lda,
                                                      const float* beta,
                                                      void* C,
                                                      UPTKDataType Ctype,
                                                      int ldc);

/* SYR2K */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSsyr2k(UPTKblasHandle_t handle,
                                                      UPTKblasFillMode_t uplo,
                                                      UPTKblasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const float* alpha, /* host or device pointer */
                                                      const float* A,
                                                      int lda,
                                                      const float* B,
                                                      int ldb,
                                                      const float* beta, /* host or device pointer */
                                                      float* C,
                                                      int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDsyr2k(UPTKblasHandle_t handle,
                                                      UPTKblasFillMode_t uplo,
                                                      UPTKblasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const double* alpha, /* host or device pointer */
                                                      const double* A,
                                                      int lda,
                                                      const double* B,
                                                      int ldb,
                                                      const double* beta, /* host or device pointer */
                                                      double* C,
                                                      int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCsyr2k(UPTKblasHandle_t handle,
                                                      UPTKblasFillMode_t uplo,
                                                      UPTKblasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuComplex* alpha, /* host or device pointer */
                                                      const cuComplex* A,
                                                      int lda,
                                                      const cuComplex* B,
                                                      int ldb,
                                                      const cuComplex* beta, /* host or device pointer */
                                                      cuComplex* C,
                                                      int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZsyr2k(UPTKblasHandle_t handle,
                                                      UPTKblasFillMode_t uplo,
                                                      UPTKblasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex* alpha, /* host or device pointer */
                                                      const cuDoubleComplex* A,
                                                      int lda,
                                                      const cuDoubleComplex* B,
                                                      int ldb,
                                                      const cuDoubleComplex* beta, /* host or device pointer */
                                                      cuDoubleComplex* C,
                                                      int ldc);
/* HER2K */
UPTKblasStatus_t UPTKBLASAPI UPTKblasCher2k(UPTKblasHandle_t handle,
                                                      UPTKblasFillMode_t uplo,
                                                      UPTKblasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuComplex* alpha, /* host or device pointer */
                                                      const cuComplex* A,
                                                      int lda,
                                                      const cuComplex* B,
                                                      int ldb,
                                                      const float* beta, /* host or device pointer */
                                                      cuComplex* C,
                                                      int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZher2k(UPTKblasHandle_t handle,
                                                      UPTKblasFillMode_t uplo,
                                                      UPTKblasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex* alpha, /* host or device pointer */
                                                      const cuDoubleComplex* A,
                                                      int lda,
                                                      const cuDoubleComplex* B,
                                                      int ldb,
                                                      const double* beta, /* host or device pointer */
                                                      cuDoubleComplex* C,
                                                      int ldc);
/* SYRKX : eXtended SYRK*/
UPTKblasStatus_t UPTKBLASAPI UPTKblasSsyrkx(UPTKblasHandle_t handle,
                                                   UPTKblasFillMode_t uplo,
                                                   UPTKblasOperation_t trans,
                                                   int n,
                                                   int k,
                                                   const float* alpha, /* host or device pointer */
                                                   const float* A,
                                                   int lda,
                                                   const float* B,
                                                   int ldb,
                                                   const float* beta, /* host or device pointer */
                                                   float* C,
                                                   int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDsyrkx(UPTKblasHandle_t handle,
                                                   UPTKblasFillMode_t uplo,
                                                   UPTKblasOperation_t trans,
                                                   int n,
                                                   int k,
                                                   const double* alpha, /* host or device pointer */
                                                   const double* A,
                                                   int lda,
                                                   const double* B,
                                                   int ldb,
                                                   const double* beta, /* host or device pointer */
                                                   double* C,
                                                   int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCsyrkx(UPTKblasHandle_t handle,
                                                   UPTKblasFillMode_t uplo,
                                                   UPTKblasOperation_t trans,
                                                   int n,
                                                   int k,
                                                   const cuComplex* alpha, /* host or device pointer */
                                                   const cuComplex* A,
                                                   int lda,
                                                   const cuComplex* B,
                                                   int ldb,
                                                   const cuComplex* beta, /* host or device pointer */
                                                   cuComplex* C,
                                                   int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZsyrkx(UPTKblasHandle_t handle,
                                                   UPTKblasFillMode_t uplo,
                                                   UPTKblasOperation_t trans,
                                                   int n,
                                                   int k,
                                                   const cuDoubleComplex* alpha, /* host or device pointer */
                                                   const cuDoubleComplex* A,
                                                   int lda,
                                                   const cuDoubleComplex* B,
                                                   int ldb,
                                                   const cuDoubleComplex* beta, /* host or device pointer */
                                                   cuDoubleComplex* C,
                                                   int ldc);
/* HERKX : eXtended HERK */
UPTKblasStatus_t UPTKBLASAPI UPTKblasCherkx(UPTKblasHandle_t handle,
                                                   UPTKblasFillMode_t uplo,
                                                   UPTKblasOperation_t trans,
                                                   int n,
                                                   int k,
                                                   const cuComplex* alpha, /* host or device pointer */
                                                   const cuComplex* A,
                                                   int lda,
                                                   const cuComplex* B,
                                                   int ldb,
                                                   const float* beta, /* host or device pointer */
                                                   cuComplex* C,
                                                   int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZherkx(UPTKblasHandle_t handle,
                                                   UPTKblasFillMode_t uplo,
                                                   UPTKblasOperation_t trans,
                                                   int n,
                                                   int k,
                                                   const cuDoubleComplex* alpha, /* host or device pointer */
                                                   const cuDoubleComplex* A,
                                                   int lda,
                                                   const cuDoubleComplex* B,
                                                   int ldb,
                                                   const double* beta, /* host or device pointer */
                                                   cuDoubleComplex* C,
                                                   int ldc);
/* SYMM */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSsymm(UPTKblasHandle_t handle,
                                                     UPTKblasSideMode_t side,
                                                     UPTKblasFillMode_t uplo,
                                                     int m,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* B,
                                                     int ldb,
                                                     const float* beta, /* host or device pointer */
                                                     float* C,
                                                     int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDsymm(UPTKblasHandle_t handle,
                                                     UPTKblasSideMode_t side,
                                                     UPTKblasFillMode_t uplo,
                                                     int m,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* B,
                                                     int ldb,
                                                     const double* beta, /* host or device pointer */
                                                     double* C,
                                                     int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCsymm(UPTKblasHandle_t handle,
                                                     UPTKblasSideMode_t side,
                                                     UPTKblasFillMode_t uplo,
                                                     int m,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* B,
                                                     int ldb,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* C,
                                                     int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZsymm(UPTKblasHandle_t handle,
                                                     UPTKblasSideMode_t side,
                                                     UPTKblasFillMode_t uplo,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* B,
                                                     int ldb,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* C,
                                                     int ldc);

/* HEMM */
UPTKblasStatus_t UPTKBLASAPI UPTKblasChemm(UPTKblasHandle_t handle,
                                                     UPTKblasSideMode_t side,
                                                     UPTKblasFillMode_t uplo,
                                                     int m,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* B,
                                                     int ldb,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* C,
                                                     int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZhemm(UPTKblasHandle_t handle,
                                                     UPTKblasSideMode_t side,
                                                     UPTKblasFillMode_t uplo,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* B,
                                                     int ldb,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* C,
                                                     int ldc);

/* TRSM */
UPTKblasStatus_t UPTKBLASAPI UPTKblasStrsm(UPTKblasHandle_t handle,
                                                     UPTKblasSideMode_t side,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     float* B,
                                                     int ldb);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDtrsm(UPTKblasHandle_t handle,
                                                     UPTKblasSideMode_t side,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     double* B,
                                                     int ldb);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCtrsm(UPTKblasHandle_t handle,
                                                     UPTKblasSideMode_t side,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     cuComplex* B,
                                                     int ldb);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZtrsm(UPTKblasHandle_t handle,
                                                     UPTKblasSideMode_t side,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     cuDoubleComplex* B,
                                                     int ldb);

/* TRMM */
UPTKblasStatus_t UPTKBLASAPI UPTKblasStrmm(UPTKblasHandle_t handle,
                                                     UPTKblasSideMode_t side,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* B,
                                                     int ldb,
                                                     float* C,
                                                     int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDtrmm(UPTKblasHandle_t handle,
                                                     UPTKblasSideMode_t side,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* B,
                                                     int ldb,
                                                     double* C,
                                                     int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCtrmm(UPTKblasHandle_t handle,
                                                     UPTKblasSideMode_t side,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* B,
                                                     int ldb,
                                                     cuComplex* C,
                                                     int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZtrmm(UPTKblasHandle_t handle,
                                                     UPTKblasSideMode_t side,
                                                     UPTKblasFillMode_t uplo,
                                                     UPTKblasOperation_t trans,
                                                     UPTKblasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* B,
                                                     int ldb,
                                                     cuDoubleComplex* C,
                                                     int ldc);
/* BATCH GEMM */
#if defined(__cplusplus)
UPTKblasStatus_t UPTKBLASAPI UPTKblasHgemmBatched(UPTKblasHandle_t handle,
                                                         UPTKblasOperation_t transa,
                                                         UPTKblasOperation_t transb,
                                                         int m,
                                                         int n,
                                                         int k,
                                                         const __half* alpha, /* host or device pointer */
                                                         const __half* const Aarray[],
                                                         int lda,
                                                         const __half* const Barray[],
                                                         int ldb,
                                                         const __half* beta, /* host or device pointer */
                                                         __half* const Carray[],
                                                         int ldc,
                                                         int batchCount);
#endif
UPTKblasStatus_t UPTKBLASAPI UPTKblasSgemmBatched(UPTKblasHandle_t handle,
                                                         UPTKblasOperation_t transa,
                                                         UPTKblasOperation_t transb,
                                                         int m,
                                                         int n,
                                                         int k,
                                                         const float* alpha, /* host or device pointer */
                                                         const float* const Aarray[],
                                                         int lda,
                                                         const float* const Barray[],
                                                         int ldb,
                                                         const float* beta, /* host or device pointer */
                                                         float* const Carray[],
                                                         int ldc,
                                                         int batchCount);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDgemmBatched(UPTKblasHandle_t handle,
                                                         UPTKblasOperation_t transa,
                                                         UPTKblasOperation_t transb,
                                                         int m,
                                                         int n,
                                                         int k,
                                                         const double* alpha, /* host or device pointer */
                                                         const double* const Aarray[],
                                                         int lda,
                                                         const double* const Barray[],
                                                         int ldb,
                                                         const double* beta, /* host or device pointer */
                                                         double* const Carray[],
                                                         int ldc,
                                                         int batchCount);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCgemmBatched(UPTKblasHandle_t handle,
                                                         UPTKblasOperation_t transa,
                                                         UPTKblasOperation_t transb,
                                                         int m,
                                                         int n,
                                                         int k,
                                                         const cuComplex* alpha, /* host or device pointer */
                                                         const cuComplex* const Aarray[],
                                                         int lda,
                                                         const cuComplex* const Barray[],
                                                         int ldb,
                                                         const cuComplex* beta, /* host or device pointer */
                                                         cuComplex* const Carray[],
                                                         int ldc,
                                                         int batchCount);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZgemmBatched(UPTKblasHandle_t handle,
                                                         UPTKblasOperation_t transa,
                                                         UPTKblasOperation_t transb,
                                                         int m,
                                                         int n,
                                                         int k,
                                                         const cuDoubleComplex* alpha, /* host or device pointer */
                                                         const cuDoubleComplex* const Aarray[],
                                                         int lda,
                                                         const cuDoubleComplex* const Barray[],
                                                         int ldb,
                                                         const cuDoubleComplex* beta, /* host or device pointer */
                                                         cuDoubleComplex* const Carray[],
                                                         int ldc,
                                                         int batchCount);

UPTKblasStatus_t UPTKBLASAPI UPTKblasGemmBatchedEx(UPTKblasHandle_t handle,
                                                          UPTKblasOperation_t transa,
                                                          UPTKblasOperation_t transb,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const void* alpha, /* host or device pointer */
                                                          const void* const Aarray[],
                                                          UPTKDataType Atype,
                                                          int lda,
                                                          const void* const Barray[],
                                                          UPTKDataType Btype,
                                                          int ldb,
                                                          const void* beta, /* host or device pointer */
                                                          void* const Carray[],
                                                          UPTKDataType Ctype,
                                                          int ldc,
                                                          int batchCount,
                                                          UPTKblasComputeType_t computeType,
                                                          UPTKblasGemmAlgo_t algo);

UPTKblasStatus_t UPTKBLASAPI UPTKblasGemmStridedBatchedEx(UPTKblasHandle_t handle,
                                                                 UPTKblasOperation_t transa,
                                                                 UPTKblasOperation_t transb,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const void* alpha, /* host or device pointer */
                                                                 const void* A,
                                                                 UPTKDataType Atype,
                                                                 int lda,
                                                                 long long int strideA, /* purposely signed */
                                                                 const void* B,
                                                                 UPTKDataType Btype,
                                                                 int ldb,
                                                                 long long int strideB,
                                                                 const void* beta, /* host or device pointer */
                                                                 void* C,
                                                                 UPTKDataType Ctype,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount,
                                                                 UPTKblasComputeType_t computeType,
                                                                 UPTKblasGemmAlgo_t algo);

UPTKblasStatus_t UPTKBLASAPI UPTKblasSgemmStridedBatched(UPTKblasHandle_t handle,
                                                                UPTKblasOperation_t transa,
                                                                UPTKblasOperation_t transb,
                                                                int m,
                                                                int n,
                                                                int k,
                                                                const float* alpha, /* host or device pointer */
                                                                const float* A,
                                                                int lda,
                                                                long long int strideA, /* purposely signed */
                                                                const float* B,
                                                                int ldb,
                                                                long long int strideB,
                                                                const float* beta, /* host or device pointer */
                                                                float* C,
                                                                int ldc,
                                                                long long int strideC,
                                                                int batchCount);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDgemmStridedBatched(UPTKblasHandle_t handle,
                                                                UPTKblasOperation_t transa,
                                                                UPTKblasOperation_t transb,
                                                                int m,
                                                                int n,
                                                                int k,
                                                                const double* alpha, /* host or device pointer */
                                                                const double* A,
                                                                int lda,
                                                                long long int strideA, /* purposely signed */
                                                                const double* B,
                                                                int ldb,
                                                                long long int strideB,
                                                                const double* beta, /* host or device pointer */
                                                                double* C,
                                                                int ldc,
                                                                long long int strideC,
                                                                int batchCount);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCgemmStridedBatched(UPTKblasHandle_t handle,
                                                                UPTKblasOperation_t transa,
                                                                UPTKblasOperation_t transb,
                                                                int m,
                                                                int n,
                                                                int k,
                                                                const cuComplex* alpha, /* host or device pointer */
                                                                const cuComplex* A,
                                                                int lda,
                                                                long long int strideA, /* purposely signed */
                                                                const cuComplex* B,
                                                                int ldb,
                                                                long long int strideB,
                                                                const cuComplex* beta, /* host or device pointer */
                                                                cuComplex* C,
                                                                int ldc,
                                                                long long int strideC,
                                                                int batchCount);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCgemm3mStridedBatched(UPTKblasHandle_t handle,
                                                                  UPTKblasOperation_t transa,
                                                                  UPTKblasOperation_t transb,
                                                                  int m,
                                                                  int n,
                                                                  int k,
                                                                  const cuComplex* alpha, /* host or device pointer */
                                                                  const cuComplex* A,
                                                                  int lda,
                                                                  long long int strideA, /* purposely signed */
                                                                  const cuComplex* B,
                                                                  int ldb,
                                                                  long long int strideB,
                                                                  const cuComplex* beta, /* host or device pointer */
                                                                  cuComplex* C,
                                                                  int ldc,
                                                                  long long int strideC,
                                                                  int batchCount);

UPTKblasStatus_t UPTKBLASAPI
UPTKblasZgemmStridedBatched(UPTKblasHandle_t handle,
                          UPTKblasOperation_t transa,
                          UPTKblasOperation_t transb,
                          int m,
                          int n,
                          int k,
                          const cuDoubleComplex* alpha, /* host or device pointer */
                          const cuDoubleComplex* A,
                          int lda,
                          long long int strideA, /* purposely signed */
                          const cuDoubleComplex* B,
                          int ldb,
                          long long int strideB,
                          const cuDoubleComplex* beta, /* host or device poi */
                          cuDoubleComplex* C,
                          int ldc,
                          long long int strideC,
                          int batchCount);

#if defined(__cplusplus)
UPTKblasStatus_t UPTKBLASAPI UPTKblasHgemmStridedBatched(UPTKblasHandle_t handle,
                                                                UPTKblasOperation_t transa,
                                                                UPTKblasOperation_t transb,
                                                                int m,
                                                                int n,
                                                                int k,
                                                                const __half* alpha, /* host or device pointer */
                                                                const __half* A,
                                                                int lda,
                                                                long long int strideA, /* purposely signed */
                                                                const __half* B,
                                                                int ldb,
                                                                long long int strideB,
                                                                const __half* beta, /* host or device pointer */
                                                                __half* C,
                                                                int ldc,
                                                                long long int strideC,
                                                                int batchCount);
#endif
/* ---------------- CUBLAS BLAS-like extension ---------------- */
/* GEAM */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSgeam(UPTKblasHandle_t handle,
                                                  UPTKblasOperation_t transa,
                                                  UPTKblasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  const float* alpha, /* host or device pointer */
                                                  const float* A,
                                                  int lda,
                                                  const float* beta, /* host or device pointer */
                                                  const float* B,
                                                  int ldb,
                                                  float* C,
                                                  int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDgeam(UPTKblasHandle_t handle,
                                                  UPTKblasOperation_t transa,
                                                  UPTKblasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  const double* alpha, /* host or device pointer */
                                                  const double* A,
                                                  int lda,
                                                  const double* beta, /* host or device pointer */
                                                  const double* B,
                                                  int ldb,
                                                  double* C,
                                                  int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCgeam(UPTKblasHandle_t handle,
                                                  UPTKblasOperation_t transa,
                                                  UPTKblasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  const cuComplex* alpha, /* host or device pointer */
                                                  const cuComplex* A,
                                                  int lda,
                                                  const cuComplex* beta, /* host or device pointer */
                                                  const cuComplex* B,
                                                  int ldb,
                                                  cuComplex* C,
                                                  int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZgeam(UPTKblasHandle_t handle,
                                                  UPTKblasOperation_t transa,
                                                  UPTKblasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  const cuDoubleComplex* alpha, /* host or device pointer */
                                                  const cuDoubleComplex* A,
                                                  int lda,
                                                  const cuDoubleComplex* beta, /* host or device pointer */
                                                  const cuDoubleComplex* B,
                                                  int ldb,
                                                  cuDoubleComplex* C,
                                                  int ldc);

/* Batched LU - GETRF*/
UPTKblasStatus_t UPTKBLASAPI UPTKblasSgetrfBatched(UPTKblasHandle_t handle,
                                                          int n,
                                                          float* const A[], /*Device pointer*/
                                                          int lda,
                                                          int* P,    /*Device Pointer*/
                                                          int* info, /*Device Pointer*/
                                                          int batchSize);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDgetrfBatched(UPTKblasHandle_t handle,
                                                          int n,
                                                          double* const A[], /*Device pointer*/
                                                          int lda,
                                                          int* P,    /*Device Pointer*/
                                                          int* info, /*Device Pointer*/
                                                          int batchSize);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCgetrfBatched(UPTKblasHandle_t handle,
                                                          int n,
                                                          cuComplex* const A[], /*Device pointer*/
                                                          int lda,
                                                          int* P,    /*Device Pointer*/
                                                          int* info, /*Device Pointer*/
                                                          int batchSize);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZgetrfBatched(UPTKblasHandle_t handle,
                                                          int n,
                                                          cuDoubleComplex* const A[], /*Device pointer*/
                                                          int lda,
                                                          int* P,    /*Device Pointer*/
                                                          int* info, /*Device Pointer*/
                                                          int batchSize);

/* Batched inversion based on LU factorization from getrf */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSgetriBatched(UPTKblasHandle_t handle,
                                                          int n,
                                                          const float* const A[], /*Device pointer*/
                                                          int lda,
                                                          const int* P,     /*Device pointer*/
                                                          float* const C[], /*Device pointer*/
                                                          int ldc,
                                                          int* info,
                                                          int batchSize);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDgetriBatched(UPTKblasHandle_t handle,
                                                          int n,
                                                          const double* const A[], /*Device pointer*/
                                                          int lda,
                                                          const int* P,      /*Device pointer*/
                                                          double* const C[], /*Device pointer*/
                                                          int ldc,
                                                          int* info,
                                                          int batchSize);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCgetriBatched(UPTKblasHandle_t handle,
                                                          int n,
                                                          const cuComplex* const A[], /*Device pointer*/
                                                          int lda,
                                                          const int* P,         /*Device pointer*/
                                                          cuComplex* const C[], /*Device pointer*/
                                                          int ldc,
                                                          int* info,
                                                          int batchSize);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZgetriBatched(UPTKblasHandle_t handle,
                                                          int n,
                                                          const cuDoubleComplex* const A[], /*Device pointer*/
                                                          int lda,
                                                          const int* P,               /*Device pointer*/
                                                          cuDoubleComplex* const C[], /*Device pointer*/
                                                          int ldc,
                                                          int* info,
                                                          int batchSize);

/* Batched solver based on LU factorization from getrf */

UPTKblasStatus_t UPTKBLASAPI UPTKblasSgetrsBatched(UPTKblasHandle_t handle,
                                                          UPTKblasOperation_t trans,
                                                          int n,
                                                          int nrhs,
                                                          const float* const Aarray[],
                                                          int lda,
                                                          const int* devIpiv,
                                                          float* const Barray[],
                                                          int ldb,
                                                          int* info,
                                                          int batchSize);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDgetrsBatched(UPTKblasHandle_t handle,
                                                          UPTKblasOperation_t trans,
                                                          int n,
                                                          int nrhs,
                                                          const double* const Aarray[],
                                                          int lda,
                                                          const int* devIpiv,
                                                          double* const Barray[],
                                                          int ldb,
                                                          int* info,
                                                          int batchSize);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCgetrsBatched(UPTKblasHandle_t handle,
                                                          UPTKblasOperation_t trans,
                                                          int n,
                                                          int nrhs,
                                                          const cuComplex* const Aarray[],
                                                          int lda,
                                                          const int* devIpiv,
                                                          cuComplex* const Barray[],
                                                          int ldb,
                                                          int* info,
                                                          int batchSize);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZgetrsBatched(UPTKblasHandle_t handle,
                                                          UPTKblasOperation_t trans,
                                                          int n,
                                                          int nrhs,
                                                          const cuDoubleComplex* const Aarray[],
                                                          int lda,
                                                          const int* devIpiv,
                                                          cuDoubleComplex* const Barray[],
                                                          int ldb,
                                                          int* info,
                                                          int batchSize);

/* TRSM - Batched Triangular Solver */
UPTKblasStatus_t UPTKBLASAPI UPTKblasStrsmBatched(UPTKblasHandle_t handle,
                                                         UPTKblasSideMode_t side,
                                                         UPTKblasFillMode_t uplo,
                                                         UPTKblasOperation_t trans,
                                                         UPTKblasDiagType_t diag,
                                                         int m,
                                                         int n,
                                                         const float* alpha, /*Host or Device Pointer*/
                                                         const float* const A[],
                                                         int lda,
                                                         float* const B[],
                                                         int ldb,
                                                         int batchCount);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDtrsmBatched(UPTKblasHandle_t handle,
                                                         UPTKblasSideMode_t side,
                                                         UPTKblasFillMode_t uplo,
                                                         UPTKblasOperation_t trans,
                                                         UPTKblasDiagType_t diag,
                                                         int m,
                                                         int n,
                                                         const double* alpha, /*Host or Device Pointer*/
                                                         const double* const A[],
                                                         int lda,
                                                         double* const B[],
                                                         int ldb,
                                                         int batchCount);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCtrsmBatched(UPTKblasHandle_t handle,
                                                         UPTKblasSideMode_t side,
                                                         UPTKblasFillMode_t uplo,
                                                         UPTKblasOperation_t trans,
                                                         UPTKblasDiagType_t diag,
                                                         int m,
                                                         int n,
                                                         const cuComplex* alpha, /*Host or Device Pointer*/
                                                         const cuComplex* const A[],
                                                         int lda,
                                                         cuComplex* const B[],
                                                         int ldb,
                                                         int batchCount);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZtrsmBatched(UPTKblasHandle_t handle,
                                                         UPTKblasSideMode_t side,
                                                         UPTKblasFillMode_t uplo,
                                                         UPTKblasOperation_t trans,
                                                         UPTKblasDiagType_t diag,
                                                         int m,
                                                         int n,
                                                         const cuDoubleComplex* alpha, /*Host or Device Pointer*/
                                                         const cuDoubleComplex* const A[],
                                                         int lda,
                                                         cuDoubleComplex* const B[],
                                                         int ldb,
                                                         int batchCount);

/* Batched - MATINV*/
UPTKblasStatus_t UPTKBLASAPI UPTKblasSmatinvBatched(UPTKblasHandle_t handle,
                                                           int n,
                                                           const float* const A[], /*Device pointer*/
                                                           int lda,
                                                           float* const Ainv[], /*Device pointer*/
                                                           int lda_inv,
                                                           int* info, /*Device Pointer*/
                                                           int batchSize);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDmatinvBatched(UPTKblasHandle_t handle,
                                                           int n,
                                                           const double* const A[], /*Device pointer*/
                                                           int lda,
                                                           double* const Ainv[], /*Device pointer*/
                                                           int lda_inv,
                                                           int* info, /*Device Pointer*/
                                                           int batchSize);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCmatinvBatched(UPTKblasHandle_t handle,
                                                           int n,
                                                           const cuComplex* const A[], /*Device pointer*/
                                                           int lda,
                                                           cuComplex* const Ainv[], /*Device pointer*/
                                                           int lda_inv,
                                                           int* info, /*Device Pointer*/
                                                           int batchSize);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZmatinvBatched(UPTKblasHandle_t handle,
                                                           int n,
                                                           const cuDoubleComplex* const A[], /*Device pointer*/
                                                           int lda,
                                                           cuDoubleComplex* const Ainv[], /*Device pointer*/
                                                           int lda_inv,
                                                           int* info, /*Device Pointer*/
                                                           int batchSize);

/* Batch QR Factorization */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSgeqrfBatched(UPTKblasHandle_t handle,
                                                          int m,
                                                          int n,
                                                          float* const Aarray[], /*Device pointer*/
                                                          int lda,
                                                          float* const TauArray[], /*Device pointer*/
                                                          int* info,
                                                          int batchSize);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDgeqrfBatched(UPTKblasHandle_t handle,
                                                          int m,
                                                          int n,
                                                          double* const Aarray[], /*Device pointer*/
                                                          int lda,
                                                          double* const TauArray[], /*Device pointer*/
                                                          int* info,
                                                          int batchSize);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCgeqrfBatched(UPTKblasHandle_t handle,
                                                          int m,
                                                          int n,
                                                          cuComplex* const Aarray[], /*Device pointer*/
                                                          int lda,
                                                          cuComplex* const TauArray[], /*Device pointer*/
                                                          int* info,
                                                          int batchSize);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZgeqrfBatched(UPTKblasHandle_t handle,
                                                          int m,
                                                          int n,
                                                          cuDoubleComplex* const Aarray[], /*Device pointer*/
                                                          int lda,
                                                          cuDoubleComplex* const TauArray[], /*Device pointer*/
                                                          int* info,
                                                          int batchSize);
/* Least Square Min only m >= n and Non-transpose supported */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSgelsBatched(UPTKblasHandle_t handle,
                                                         UPTKblasOperation_t trans,
                                                         int m,
                                                         int n,
                                                         int nrhs,
                                                         float* const Aarray[], /*Device pointer*/
                                                         int lda,
                                                         float* const Carray[], /*Device pointer*/
                                                         int ldc,
                                                         int* info,
                                                         int* devInfoArray, /*Device pointer*/
                                                         int batchSize);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDgelsBatched(UPTKblasHandle_t handle,
                                                         UPTKblasOperation_t trans,
                                                         int m,
                                                         int n,
                                                         int nrhs,
                                                         double* const Aarray[], /*Device pointer*/
                                                         int lda,
                                                         double* const Carray[], /*Device pointer*/
                                                         int ldc,
                                                         int* info,
                                                         int* devInfoArray, /*Device pointer*/
                                                         int batchSize);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCgelsBatched(UPTKblasHandle_t handle,
                                                         UPTKblasOperation_t trans,
                                                         int m,
                                                         int n,
                                                         int nrhs,
                                                         cuComplex* const Aarray[], /*Device pointer*/
                                                         int lda,
                                                         cuComplex* const Carray[], /*Device pointer*/
                                                         int ldc,
                                                         int* info,
                                                         int* devInfoArray,
                                                         int batchSize);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZgelsBatched(UPTKblasHandle_t handle,
                                                         UPTKblasOperation_t trans,
                                                         int m,
                                                         int n,
                                                         int nrhs,
                                                         cuDoubleComplex* const Aarray[], /*Device pointer*/
                                                         int lda,
                                                         cuDoubleComplex* const Carray[], /*Device pointer*/
                                                         int ldc,
                                                         int* info,
                                                         int* devInfoArray,
                                                         int batchSize);
/* DGMM */
UPTKblasStatus_t UPTKBLASAPI UPTKblasSdgmm(UPTKblasHandle_t handle,
                                                  UPTKblasSideMode_t mode,
                                                  int m,
                                                  int n,
                                                  const float* A,
                                                  int lda,
                                                  const float* x,
                                                  int incx,
                                                  float* C,
                                                  int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasDdgmm(UPTKblasHandle_t handle,
                                                  UPTKblasSideMode_t mode,
                                                  int m,
                                                  int n,
                                                  const double* A,
                                                  int lda,
                                                  const double* x,
                                                  int incx,
                                                  double* C,
                                                  int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasCdgmm(UPTKblasHandle_t handle,
                                                  UPTKblasSideMode_t mode,
                                                  int m,
                                                  int n,
                                                  const cuComplex* A,
                                                  int lda,
                                                  const cuComplex* x,
                                                  int incx,
                                                  cuComplex* C,
                                                  int ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZdgmm(UPTKblasHandle_t handle,
                                                  UPTKblasSideMode_t mode,
                                                  int m,
                                                  int n,
                                                  const cuDoubleComplex* A,
                                                  int lda,
                                                  const cuDoubleComplex* x,
                                                  int incx,
                                                  cuDoubleComplex* C,
                                                  int ldc);

/* TPTTR : Triangular Pack format to Triangular format */
UPTKblasStatus_t UPTKBLASAPI
UPTKblasStpttr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const float* AP, float* A, int lda);

UPTKblasStatus_t UPTKBLASAPI
UPTKblasDtpttr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const double* AP, double* A, int lda);

UPTKblasStatus_t UPTKBLASAPI
UPTKblasCtpttr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuComplex* AP, cuComplex* A, int lda);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZtpttr(
    UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuDoubleComplex* AP, cuDoubleComplex* A, int lda);
/* TRTTP : Triangular format to Triangular Pack format */
UPTKblasStatus_t UPTKBLASAPI
UPTKblasStrttp(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const float* A, int lda, float* AP);

UPTKblasStatus_t UPTKBLASAPI
UPTKblasDtrttp(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const double* A, int lda, double* AP);

UPTKblasStatus_t UPTKBLASAPI
UPTKblasCtrttp(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuComplex* A, int lda, cuComplex* AP);

UPTKblasStatus_t UPTKBLASAPI UPTKblasZtrttp(
    UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* AP);

#if defined(__cplusplus)
}

static inline UPTKblasStatus_t UPTKblasMigrateComputeType(UPTKblasHandle_t handle,
                                                      UPTKDataType_t dataType,
                                                      UPTKblasComputeType_t* computeType) {
  UPTKblasMath_t mathMode = UPTKBLAS_DEFAULT_MATH;
  UPTKblasStatus_t status = UPTKBLAS_STATUS_SUCCESS;

  status = UPTKblasGetMathMode(handle, &mathMode);
  if (status != UPTKBLAS_STATUS_SUCCESS) {
    return status;
  }

  bool isPedantic = ((mathMode & 0xf) == UPTKBLAS_PEDANTIC_MATH);

  switch (dataType) {
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
static inline UPTKblasStatus_t UPTKblasGemmEx(UPTKblasHandle_t handle,
                                          UPTKblasOperation_t transa,
                                          UPTKblasOperation_t transb,
                                          int m,
                                          int n,
                                          int k,
                                          const void* alpha, /* host or device pointer */
                                          const void* A,
                                          UPTKDataType Atype,
                                          int lda,
                                          const void* B,
                                          UPTKDataType Btype,
                                          int ldb,
                                          const void* beta, /* host or device pointer */
                                          void* C,
                                          UPTKDataType Ctype,
                                          int ldc,
                                          UPTKDataType computeType,
                                          UPTKblasGemmAlgo_t algo) {
  UPTKblasComputeType_t migratedComputeType = UPTKBLAS_COMPUTE_32F;
  UPTKblasStatus_t status = UPTKBLAS_STATUS_SUCCESS;
  status = UPTKblasMigrateComputeType(handle, computeType, &migratedComputeType);
  if (status != UPTKBLAS_STATUS_SUCCESS) {
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

static inline UPTKblasStatus_t UPTKblasGemmBatchedEx(UPTKblasHandle_t handle,
                                                 UPTKblasOperation_t transa,
                                                 UPTKblasOperation_t transb,
                                                 int m,
                                                 int n,
                                                 int k,
                                                 const void* alpha, /* host or device pointer */
                                                 const void* const Aarray[],
                                                 UPTKDataType Atype,
                                                 int lda,
                                                 const void* const Barray[],
                                                 UPTKDataType Btype,
                                                 int ldb,
                                                 const void* beta, /* host or device pointer */
                                                 void* const Carray[],
                                                 UPTKDataType Ctype,
                                                 int ldc,
                                                 int batchCount,
                                                 UPTKDataType computeType,
                                                 UPTKblasGemmAlgo_t algo) {
  UPTKblasComputeType_t migratedComputeType;
  UPTKblasStatus_t status;
  status = UPTKblasMigrateComputeType(handle, computeType, &migratedComputeType);
  if (status != UPTKBLAS_STATUS_SUCCESS) {
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

static inline UPTKblasStatus_t UPTKblasGemmStridedBatchedEx(UPTKblasHandle_t handle,
                                                        UPTKblasOperation_t transa,
                                                        UPTKblasOperation_t transb,
                                                        int m,
                                                        int n,
                                                        int k,
                                                        const void* alpha, /* host or device pointer */
                                                        const void* A,
                                                        UPTKDataType Atype,
                                                        int lda,
                                                        long long int strideA, /* purposely signed */
                                                        const void* B,
                                                        UPTKDataType Btype,
                                                        int ldb,
                                                        long long int strideB,
                                                        const void* beta, /* host or device pointer */
                                                        void* C,
                                                        UPTKDataType Ctype,
                                                        int ldc,
                                                        long long int strideC,
                                                        int batchCount,
                                                        UPTKDataType computeType,
                                                        UPTKblasGemmAlgo_t algo) {
  UPTKblasComputeType_t migratedComputeType;
  UPTKblasStatus_t status;
  status = UPTKblasMigrateComputeType(handle, computeType, &migratedComputeType);
  if (status != UPTKBLAS_STATUS_SUCCESS) {
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

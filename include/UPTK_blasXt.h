/*
 * Copyright 1993-2019 NVIDIA Corporation. All rights reserved.
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

/*   UPTKblasXt : Host API, Out of Core and Multi-GPU BLAS Library

*/

#if !defined(UPTKBLAS_XT_H_)
#define UPTKBLAS_XT_H_

#include "cuComplex.h" /* import complex data type */

#include "UPTK_blas.h"
#include "UPTK_driver_types.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

struct UPTKblasXtContext;
typedef struct UPTKblasXtContext* UPTKblasXtHandle_t;

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCreate(UPTKblasXtHandle_t* handle);
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtDestroy(UPTKblasXtHandle_t handle);
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtGetNumBoards(int nbDevices, int deviceId[], int* nbBoards);
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtMaxBoards(int* nbGpuBoards);
/* This routine selects the Gpus that the user want to use for CUBLAS-XT */
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtDeviceSelect(UPTKblasXtHandle_t handle, int nbDevices, int deviceId[]);

/* This routine allows to change the dimension of the tiles ( blockDim x blockDim ) */
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtSetBlockDim(UPTKblasXtHandle_t handle, int blockDim);
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtGetBlockDim(UPTKblasXtHandle_t handle, int* blockDim);

typedef enum { UPTKBLASXT_PINNING_DISABLED = 0, UPTKBLASXT_PINNING_ENABLED = 1 } UPTKblasXtPinnedMemMode_t;
/* This routine allows to CUBLAS-XT to pin the Host memory if it find out that some of the matrix passed
   are not pinned : Pinning/Unpinning the Host memory is still a costly operation
   It is better if the user controls the memory on its own (by pinning/unpinning oly when necessary)
*/
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtGetPinningMemMode(UPTKblasXtHandle_t handle, UPTKblasXtPinnedMemMode_t* mode);
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtSetPinningMemMode(UPTKblasXtHandle_t handle, UPTKblasXtPinnedMemMode_t mode);

/* This routines is to provide a CPU Blas routines, used for too small sizes or hybrid computation */
typedef enum {
  UPTKBLASXT_FLOAT = 0,
  UPTKBLASXT_DOUBLE = 1,
  UPTKBLASXT_COMPLEX = 2,
  UPTKBLASXT_DOUBLECOMPLEX = 3,
} UPTKblasXtOpType_t;

typedef enum {
  UPTKBLASXT_GEMM = 0,
  UPTKBLASXT_SYRK = 1,
  UPTKBLASXT_HERK = 2,
  UPTKBLASXT_SYMM = 3,
  UPTKBLASXT_HEMM = 4,
  UPTKBLASXT_TRSM = 5,
  UPTKBLASXT_SYR2K = 6,
  UPTKBLASXT_HER2K = 7,

  UPTKBLASXT_SPMM = 8,
  UPTKBLASXT_SYRKX = 9,
  UPTKBLASXT_HERKX = 10,
  UPTKBLASXT_TRMM = 11,
  UPTKBLASXT_ROUTINE_MAX = 12,
} UPTKblasXtBlasOp_t;

/* Currently only 32-bit integer BLAS routines are supported */
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtSetCpuRoutine(UPTKblasXtHandle_t handle,
                                                  UPTKblasXtBlasOp_t blasOp,
                                                  UPTKblasXtOpType_t type,
                                                  void* blasFunctor);

/* Specified the percentage of work that should done by the CPU, default is 0 (no work) */
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtSetCpuRatio(UPTKblasXtHandle_t handle,
                                                UPTKblasXtBlasOp_t blasOp,
                                                UPTKblasXtOpType_t type,
                                                float ratio);

/* GEMM */
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtSgemm(UPTKblasXtHandle_t handle,
                                          UPTKblasOperation_t transa,
                                          UPTKblasOperation_t transb,
                                          size_t m,
                                          size_t n,
                                          size_t k,
                                          const float* alpha,
                                          const float* A,
                                          size_t lda,
                                          const float* B,
                                          size_t ldb,
                                          const float* beta,
                                          float* C,
                                          size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtDgemm(UPTKblasXtHandle_t handle,
                                          UPTKblasOperation_t transa,
                                          UPTKblasOperation_t transb,
                                          size_t m,
                                          size_t n,
                                          size_t k,
                                          const double* alpha,
                                          const double* A,
                                          size_t lda,
                                          const double* B,
                                          size_t ldb,
                                          const double* beta,
                                          double* C,
                                          size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCgemm(UPTKblasXtHandle_t handle,
                                          UPTKblasOperation_t transa,
                                          UPTKblasOperation_t transb,
                                          size_t m,
                                          size_t n,
                                          size_t k,
                                          const cuComplex* alpha,
                                          const cuComplex* A,
                                          size_t lda,
                                          const cuComplex* B,
                                          size_t ldb,
                                          const cuComplex* beta,
                                          cuComplex* C,
                                          size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZgemm(UPTKblasXtHandle_t handle,
                                          UPTKblasOperation_t transa,
                                          UPTKblasOperation_t transb,
                                          size_t m,
                                          size_t n,
                                          size_t k,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* A,
                                          size_t lda,
                                          const cuDoubleComplex* B,
                                          size_t ldb,
                                          const cuDoubleComplex* beta,
                                          cuDoubleComplex* C,
                                          size_t ldc);
/* ------------------------------------------------------- */
/* SYRK */
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtSsyrk(UPTKblasXtHandle_t handle,
                                          UPTKblasFillMode_t uplo,
                                          UPTKblasOperation_t trans,
                                          size_t n,
                                          size_t k,
                                          const float* alpha,
                                          const float* A,
                                          size_t lda,
                                          const float* beta,
                                          float* C,
                                          size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtDsyrk(UPTKblasXtHandle_t handle,
                                          UPTKblasFillMode_t uplo,
                                          UPTKblasOperation_t trans,
                                          size_t n,
                                          size_t k,
                                          const double* alpha,
                                          const double* A,
                                          size_t lda,
                                          const double* beta,
                                          double* C,
                                          size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCsyrk(UPTKblasXtHandle_t handle,
                                          UPTKblasFillMode_t uplo,
                                          UPTKblasOperation_t trans,
                                          size_t n,
                                          size_t k,
                                          const cuComplex* alpha,
                                          const cuComplex* A,
                                          size_t lda,
                                          const cuComplex* beta,
                                          cuComplex* C,
                                          size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZsyrk(UPTKblasXtHandle_t handle,
                                          UPTKblasFillMode_t uplo,
                                          UPTKblasOperation_t trans,
                                          size_t n,
                                          size_t k,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* A,
                                          size_t lda,
                                          const cuDoubleComplex* beta,
                                          cuDoubleComplex* C,
                                          size_t ldc);
/* -------------------------------------------------------------------- */
/* HERK */
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCherk(UPTKblasXtHandle_t handle,
                                          UPTKblasFillMode_t uplo,
                                          UPTKblasOperation_t trans,
                                          size_t n,
                                          size_t k,
                                          const float* alpha,
                                          const cuComplex* A,
                                          size_t lda,
                                          const float* beta,
                                          cuComplex* C,
                                          size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZherk(UPTKblasXtHandle_t handle,
                                          UPTKblasFillMode_t uplo,
                                          UPTKblasOperation_t trans,
                                          size_t n,
                                          size_t k,
                                          const double* alpha,
                                          const cuDoubleComplex* A,
                                          size_t lda,
                                          const double* beta,
                                          cuDoubleComplex* C,
                                          size_t ldc);
/* -------------------------------------------------------------------- */
/* SYR2K */
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtSsyr2k(UPTKblasXtHandle_t handle,
                                           UPTKblasFillMode_t uplo,
                                           UPTKblasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const float* alpha,
                                           const float* A,
                                           size_t lda,
                                           const float* B,
                                           size_t ldb,
                                           const float* beta,
                                           float* C,
                                           size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtDsyr2k(UPTKblasXtHandle_t handle,
                                           UPTKblasFillMode_t uplo,
                                           UPTKblasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const double* alpha,
                                           const double* A,
                                           size_t lda,
                                           const double* B,
                                           size_t ldb,
                                           const double* beta,
                                           double* C,
                                           size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCsyr2k(UPTKblasXtHandle_t handle,
                                           UPTKblasFillMode_t uplo,
                                           UPTKblasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuComplex* alpha,
                                           const cuComplex* A,
                                           size_t lda,
                                           const cuComplex* B,
                                           size_t ldb,
                                           const cuComplex* beta,
                                           cuComplex* C,
                                           size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZsyr2k(UPTKblasXtHandle_t handle,
                                           UPTKblasFillMode_t uplo,
                                           UPTKblasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuDoubleComplex* alpha,
                                           const cuDoubleComplex* A,
                                           size_t lda,
                                           const cuDoubleComplex* B,
                                           size_t ldb,
                                           const cuDoubleComplex* beta,
                                           cuDoubleComplex* C,
                                           size_t ldc);
/* -------------------------------------------------------------------- */
/* HERKX : variant extension of HERK */
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCherkx(UPTKblasXtHandle_t handle,
                                           UPTKblasFillMode_t uplo,
                                           UPTKblasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuComplex* alpha,
                                           const cuComplex* A,
                                           size_t lda,
                                           const cuComplex* B,
                                           size_t ldb,
                                           const float* beta,
                                           cuComplex* C,
                                           size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZherkx(UPTKblasXtHandle_t handle,
                                           UPTKblasFillMode_t uplo,
                                           UPTKblasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuDoubleComplex* alpha,
                                           const cuDoubleComplex* A,
                                           size_t lda,
                                           const cuDoubleComplex* B,
                                           size_t ldb,
                                           const double* beta,
                                           cuDoubleComplex* C,
                                           size_t ldc);

/* -------------------------------------------------------------------- */
/* TRSM */
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtStrsm(UPTKblasXtHandle_t handle,
                                          UPTKblasSideMode_t side,
                                          UPTKblasFillMode_t uplo,
                                          UPTKblasOperation_t trans,
                                          UPTKblasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const float* alpha,
                                          const float* A,
                                          size_t lda,
                                          float* B,
                                          size_t ldb);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtDtrsm(UPTKblasXtHandle_t handle,
                                          UPTKblasSideMode_t side,
                                          UPTKblasFillMode_t uplo,
                                          UPTKblasOperation_t trans,
                                          UPTKblasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const double* alpha,
                                          const double* A,
                                          size_t lda,
                                          double* B,
                                          size_t ldb);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCtrsm(UPTKblasXtHandle_t handle,
                                          UPTKblasSideMode_t side,
                                          UPTKblasFillMode_t uplo,
                                          UPTKblasOperation_t trans,
                                          UPTKblasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const cuComplex* alpha,
                                          const cuComplex* A,
                                          size_t lda,
                                          cuComplex* B,
                                          size_t ldb);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZtrsm(UPTKblasXtHandle_t handle,
                                          UPTKblasSideMode_t side,
                                          UPTKblasFillMode_t uplo,
                                          UPTKblasOperation_t trans,
                                          UPTKblasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* A,
                                          size_t lda,
                                          cuDoubleComplex* B,
                                          size_t ldb);
/* -------------------------------------------------------------------- */
/* SYMM : Symmetric Multiply Matrix*/
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtSsymm(UPTKblasXtHandle_t handle,
                                          UPTKblasSideMode_t side,
                                          UPTKblasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const float* alpha,
                                          const float* A,
                                          size_t lda,
                                          const float* B,
                                          size_t ldb,
                                          const float* beta,
                                          float* C,
                                          size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtDsymm(UPTKblasXtHandle_t handle,
                                          UPTKblasSideMode_t side,
                                          UPTKblasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const double* alpha,
                                          const double* A,
                                          size_t lda,
                                          const double* B,
                                          size_t ldb,
                                          const double* beta,
                                          double* C,
                                          size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCsymm(UPTKblasXtHandle_t handle,
                                          UPTKblasSideMode_t side,
                                          UPTKblasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const cuComplex* alpha,
                                          const cuComplex* A,
                                          size_t lda,
                                          const cuComplex* B,
                                          size_t ldb,
                                          const cuComplex* beta,
                                          cuComplex* C,
                                          size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZsymm(UPTKblasXtHandle_t handle,
                                          UPTKblasSideMode_t side,
                                          UPTKblasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* A,
                                          size_t lda,
                                          const cuDoubleComplex* B,
                                          size_t ldb,
                                          const cuDoubleComplex* beta,
                                          cuDoubleComplex* C,
                                          size_t ldc);
/* -------------------------------------------------------------------- */
/* HEMM : Hermitian Matrix Multiply */
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtChemm(UPTKblasXtHandle_t handle,
                                          UPTKblasSideMode_t side,
                                          UPTKblasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const cuComplex* alpha,
                                          const cuComplex* A,
                                          size_t lda,
                                          const cuComplex* B,
                                          size_t ldb,
                                          const cuComplex* beta,
                                          cuComplex* C,
                                          size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZhemm(UPTKblasXtHandle_t handle,
                                          UPTKblasSideMode_t side,
                                          UPTKblasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* A,
                                          size_t lda,
                                          const cuDoubleComplex* B,
                                          size_t ldb,
                                          const cuDoubleComplex* beta,
                                          cuDoubleComplex* C,
                                          size_t ldc);

/* -------------------------------------------------------------------- */
/* SYRKX : variant extension of SYRK  */
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtSsyrkx(UPTKblasXtHandle_t handle,
                                           UPTKblasFillMode_t uplo,
                                           UPTKblasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const float* alpha,
                                           const float* A,
                                           size_t lda,
                                           const float* B,
                                           size_t ldb,
                                           const float* beta,
                                           float* C,
                                           size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtDsyrkx(UPTKblasXtHandle_t handle,
                                           UPTKblasFillMode_t uplo,
                                           UPTKblasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const double* alpha,
                                           const double* A,
                                           size_t lda,
                                           const double* B,
                                           size_t ldb,
                                           const double* beta,
                                           double* C,
                                           size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCsyrkx(UPTKblasXtHandle_t handle,
                                           UPTKblasFillMode_t uplo,
                                           UPTKblasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuComplex* alpha,
                                           const cuComplex* A,
                                           size_t lda,
                                           const cuComplex* B,
                                           size_t ldb,
                                           const cuComplex* beta,
                                           cuComplex* C,
                                           size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZsyrkx(UPTKblasXtHandle_t handle,
                                           UPTKblasFillMode_t uplo,
                                           UPTKblasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuDoubleComplex* alpha,
                                           const cuDoubleComplex* A,
                                           size_t lda,
                                           const cuDoubleComplex* B,
                                           size_t ldb,
                                           const cuDoubleComplex* beta,
                                           cuDoubleComplex* C,
                                           size_t ldc);
/* -------------------------------------------------------------------- */
/* HER2K : variant extension of HERK  */
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCher2k(UPTKblasXtHandle_t handle,
                                           UPTKblasFillMode_t uplo,
                                           UPTKblasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuComplex* alpha,
                                           const cuComplex* A,
                                           size_t lda,
                                           const cuComplex* B,
                                           size_t ldb,
                                           const float* beta,
                                           cuComplex* C,
                                           size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZher2k(UPTKblasXtHandle_t handle,
                                           UPTKblasFillMode_t uplo,
                                           UPTKblasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuDoubleComplex* alpha,
                                           const cuDoubleComplex* A,
                                           size_t lda,
                                           const cuDoubleComplex* B,
                                           size_t ldb,
                                           const double* beta,
                                           cuDoubleComplex* C,
                                           size_t ldc);

/* -------------------------------------------------------------------- */
/* SPMM : Symmetric Packed Multiply Matrix*/
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtSspmm(UPTKblasXtHandle_t handle,
                                          UPTKblasSideMode_t side,
                                          UPTKblasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const float* alpha,
                                          const float* AP,
                                          const float* B,
                                          size_t ldb,
                                          const float* beta,
                                          float* C,
                                          size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtDspmm(UPTKblasXtHandle_t handle,
                                          UPTKblasSideMode_t side,
                                          UPTKblasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const double* alpha,
                                          const double* AP,
                                          const double* B,
                                          size_t ldb,
                                          const double* beta,
                                          double* C,
                                          size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCspmm(UPTKblasXtHandle_t handle,
                                          UPTKblasSideMode_t side,
                                          UPTKblasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const cuComplex* alpha,
                                          const cuComplex* AP,
                                          const cuComplex* B,
                                          size_t ldb,
                                          const cuComplex* beta,
                                          cuComplex* C,
                                          size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZspmm(UPTKblasXtHandle_t handle,
                                          UPTKblasSideMode_t side,
                                          UPTKblasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* AP,
                                          const cuDoubleComplex* B,
                                          size_t ldb,
                                          const cuDoubleComplex* beta,
                                          cuDoubleComplex* C,
                                          size_t ldc);

/* -------------------------------------------------------------------- */
/* TRMM */
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtStrmm(UPTKblasXtHandle_t handle,
                                          UPTKblasSideMode_t side,
                                          UPTKblasFillMode_t uplo,
                                          UPTKblasOperation_t trans,
                                          UPTKblasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const float* alpha,
                                          const float* A,
                                          size_t lda,
                                          const float* B,
                                          size_t ldb,
                                          float* C,
                                          size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtDtrmm(UPTKblasXtHandle_t handle,
                                          UPTKblasSideMode_t side,
                                          UPTKblasFillMode_t uplo,
                                          UPTKblasOperation_t trans,
                                          UPTKblasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const double* alpha,
                                          const double* A,
                                          size_t lda,
                                          const double* B,
                                          size_t ldb,
                                          double* C,
                                          size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCtrmm(UPTKblasXtHandle_t handle,
                                          UPTKblasSideMode_t side,
                                          UPTKblasFillMode_t uplo,
                                          UPTKblasOperation_t trans,
                                          UPTKblasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const cuComplex* alpha,
                                          const cuComplex* A,
                                          size_t lda,
                                          const cuComplex* B,
                                          size_t ldb,
                                          cuComplex* C,
                                          size_t ldc);

UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZtrmm(UPTKblasXtHandle_t handle,
                                          UPTKblasSideMode_t side,
                                          UPTKblasFillMode_t uplo,
                                          UPTKblasOperation_t trans,
                                          UPTKblasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* A,
                                          size_t lda,
                                          const cuDoubleComplex* B,
                                          size_t ldb,
                                          cuDoubleComplex* C,
                                          size_t ldc);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* !defined(UPTKBLAS_XT_H_) */

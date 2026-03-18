#if !defined(UPTKSPARSE_H_)
#define UPTKSPARSE_H_

#include <cuComplex.h>        
#include <UPTK_runtime_api.h>
#include <UPTK_library_types.h>
#include <stdint.h>
#include <stdio.h>

#if defined(__cplusplus)
#   include <cuda_fp16.h>     // __half
#endif // defined(__cplusplus)


#define UPTKSPARSE_VER_MAJOR 1
#define UPTKSPARSE_VER_MINOR 0
#define UPTKSPARSE_VER_PATCH 0
#define UPTKSPARSE_VER_BUILD 0
#define UPTKSPARSE_VERSION (UPTKSPARSE_VER_MAJOR * 1000 + \
                          UPTKSPARSE_VER_MINOR *  100 + \
                          UPTKSPARSE_VER_PATCH)

//------------------------------------------------------------------------------
#define UPTKSPARSEAPI
#if defined(__cplusplus)
extern "C" {
#endif // defined(__cplusplus)

//##############################################################################
//# OPAQUE DATA STRUCTURES
//##############################################################################

struct UPTKsparseContext;
typedef struct UPTKsparseContext* UPTKsparseHandle_t;

struct UPTKsparseMatDescr;
typedef struct UPTKsparseMatDescr* UPTKsparseMatDescr_t;

struct UPTKbsrsv2Info;
typedef struct UPTKbsrsv2Info* UPTKbsrsv2Info_t;

struct UPTKbsrsm2Info;
typedef struct UPTKbsrsm2Info* UPTKbsrsm2Info_t;

struct UPTKcsric02Info;
typedef struct UPTKcsric02Info* UPTKcsric02Info_t;

struct UPTKbsric02Info;
typedef struct UPTKbsric02Info* UPTKbsric02Info_t;

struct UPTKcsrilu02Info;
typedef struct UPTKcsrilu02Info* UPTKcsrilu02Info_t;

struct UPTKbsrilu02Info;
typedef struct UPTKbsrilu02Info* UPTKbsrilu02Info_t;

struct csru2csrInfo;
typedef struct UPTKcsru2csrInfo* UPTKcsru2csrInfo_t;

struct UPTKsparseColorInfo;
typedef struct UPTKsparseColorInfo* UPTKsparseColorInfo_t;

struct UPTKpruneInfo;
typedef struct UPTKpruneInfo* UPTKpruneInfo_t;

//##############################################################################
//# ENUMERATORS
//##############################################################################

typedef enum {
    UPTKSPARSE_STATUS_SUCCESS                   = 0,
    UPTKSPARSE_STATUS_NOT_INITIALIZED           = 1,
    UPTKSPARSE_STATUS_ALLOC_FAILED              = 2,
    UPTKSPARSE_STATUS_INVALID_VALUE             = 3,
    UPTKSPARSE_STATUS_ARCH_MISMATCH             = 4,
    UPTKSPARSE_STATUS_MAPPING_ERROR             = 5,
    UPTKSPARSE_STATUS_EXECUTION_FAILED          = 6,
    UPTKSPARSE_STATUS_INTERNAL_ERROR            = 7,
    UPTKSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
    UPTKSPARSE_STATUS_ZERO_PIVOT                = 9,
    UPTKSPARSE_STATUS_NOT_SUPPORTED             = 10,
    UPTKSPARSE_STATUS_INSUFFICIENT_RESOURCES    = 11
} UPTKsparseStatus_t;

typedef enum {
    UPTKSPARSE_POINTER_MODE_HOST   = 0,
    UPTKSPARSE_POINTER_MODE_DEVICE = 1
} UPTKsparsePointerMode_t;

typedef enum {
    UPTKSPARSE_ACTION_SYMBOLIC = 0,
    UPTKSPARSE_ACTION_NUMERIC  = 1
} UPTKsparseAction_t;

typedef enum {
    UPTKSPARSE_MATRIX_TYPE_GENERAL    = 0,
    UPTKSPARSE_MATRIX_TYPE_SYMMETRIC  = 1,
    UPTKSPARSE_MATRIX_TYPE_HERMITIAN  = 2,
    UPTKSPARSE_MATRIX_TYPE_TRIANGULAR = 3
} UPTKsparseMatrixType_t;

typedef enum {
    UPTKSPARSE_FILL_MODE_LOWER = 0,
    UPTKSPARSE_FILL_MODE_UPPER = 1
} UPTKsparseFillMode_t;

typedef enum {
    UPTKSPARSE_DIAG_TYPE_NON_UNIT = 0,
    UPTKSPARSE_DIAG_TYPE_UNIT     = 1
} UPTKsparseDiagType_t;

typedef enum {
    UPTKSPARSE_INDEX_BASE_ZERO = 0,
    UPTKSPARSE_INDEX_BASE_ONE  = 1
} UPTKsparseIndexBase_t;

typedef enum {
    UPTKSPARSE_OPERATION_NON_TRANSPOSE       = 0,
    UPTKSPARSE_OPERATION_TRANSPOSE           = 1,
    UPTKSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
} UPTKsparseOperation_t;

typedef enum {
    UPTKSPARSE_DIRECTION_ROW    = 0,
    UPTKSPARSE_DIRECTION_COLUMN = 1
} UPTKsparseDirection_t;

typedef enum {
    UPTKSPARSE_SOLVE_POLICY_NO_LEVEL = 0,
    UPTKSPARSE_SOLVE_POLICY_USE_LEVEL = 1
} UPTKsparseSolvePolicy_t;

typedef enum {
    UPTKSPARSE_COLOR_ALG0 = 0, // default
    UPTKSPARSE_COLOR_ALG1 = 1
} UPTKsparseColorAlg_t;

//##############################################################################
//# INITIALIZATION AND MANAGEMENT ROUTINES
//##############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreate(UPTKsparseHandle_t* handle);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroy(UPTKsparseHandle_t handle);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseGetVersion(UPTKsparseHandle_t handle,
                   int*             version);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseGetProperty(libraryPropertyType type,
                    int*                value);

const char* UPTKSPARSEAPI
UPTKsparseGetErrorName(UPTKsparseStatus_t status);

const char* UPTKSPARSEAPI
UPTKsparseGetErrorString(UPTKsparseStatus_t status);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSetStream(UPTKsparseHandle_t handle,
                  UPTKStream_t     streamId);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseGetStream(UPTKsparseHandle_t handle,
                  UPTKStream_t*    streamId);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseGetPointerMode(UPTKsparseHandle_t       handle,
                       UPTKsparsePointerMode_t* mode);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSetPointerMode(UPTKsparseHandle_t      handle,
                       UPTKsparsePointerMode_t mode);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateMatDescr(UPTKsparseMatDescr_t* descrA);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyMatDescr(UPTKsparseMatDescr_t descrA);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSetMatType(UPTKsparseMatDescr_t   descrA,
                   UPTKsparseMatrixType_t type);

UPTKsparseMatrixType_t UPTKSPARSEAPI
UPTKsparseGetMatType(const UPTKsparseMatDescr_t descrA);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSetMatFillMode(UPTKsparseMatDescr_t descrA,
                       UPTKsparseFillMode_t fillMode);

UPTKsparseFillMode_t UPTKSPARSEAPI
UPTKsparseGetMatFillMode(const UPTKsparseMatDescr_t descrA);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSetMatDiagType(UPTKsparseMatDescr_t descrA,
                       UPTKsparseDiagType_t diagType);

UPTKsparseDiagType_t UPTKSPARSEAPI
UPTKsparseGetMatDiagType(const UPTKsparseMatDescr_t descrA);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSetMatIndexBase(UPTKsparseMatDescr_t  descrA,
                        UPTKsparseIndexBase_t base);

UPTKsparseIndexBase_t UPTKSPARSEAPI
UPTKsparseGetMatIndexBase(const UPTKsparseMatDescr_t descrA);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateCsric02Info(UPTKcsric02Info_t* info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyCsric02Info(UPTKcsric02Info_t info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateBsric02Info(UPTKbsric02Info_t* info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyBsric02Info(UPTKbsric02Info_t info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateCsrilu02Info(UPTKcsrilu02Info_t* info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyCsrilu02Info(UPTKcsrilu02Info_t info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateBsrilu02Info(UPTKbsrilu02Info_t* info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyBsrilu02Info(UPTKbsrilu02Info_t info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateBsrsv2Info(UPTKbsrsv2Info_t* info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyBsrsv2Info(UPTKbsrsv2Info_t info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateBsrsm2Info(UPTKbsrsm2Info_t* info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyBsrsm2Info(UPTKbsrsm2Info_t info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateCsru2csrInfo(UPTKcsru2csrInfo_t* info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyCsru2csrInfo(UPTKcsru2csrInfo_t info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateColorInfo(UPTKsparseColorInfo_t* info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyColorInfo(UPTKsparseColorInfo_t info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreatePruneInfo(UPTKpruneInfo_t* info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyPruneInfo(UPTKpruneInfo_t info);

//##############################################################################
//# SPARSE LEVEL 2 ROUTINES
//##############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSgemvi(UPTKsparseHandle_t    handle,
               UPTKsparseOperation_t transA,
               int                 m,
               int                 n,
               const float*        alpha,
               const float*        A,
               int                 lda,
               int                 nnz,
               const float*        xVal,
               const int*          xInd,
               const float*        beta,
               float*              y,
               UPTKsparseIndexBase_t idxBase,
               void*               pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSgemvi_bufferSize(UPTKsparseHandle_t    handle,
                          UPTKsparseOperation_t transA,
                          int                 m,
                          int                 n,
                          int                 nnz,
                          int*                pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgemvi(UPTKsparseHandle_t    handle,
               UPTKsparseOperation_t transA,
               int                 m,
               int                 n,
               const double*       alpha,
               const double*       A,
               int                 lda,
               int                 nnz,
               const double*       xVal,
               const int*          xInd,
               const double*       beta,
               double*             y,
               UPTKsparseIndexBase_t idxBase,
               void*               pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgemvi_bufferSize(UPTKsparseHandle_t    handle,
                          UPTKsparseOperation_t transA,
                          int                 m,
                          int                 n,
                          int                 nnz,
                          int*                pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgemvi(UPTKsparseHandle_t    handle,
               UPTKsparseOperation_t transA,
               int                 m,
               int                 n,
               const cuComplex*    alpha,
               const cuComplex*    A,
               int                 lda,
               int                 nnz,
               const cuComplex*    xVal,
               const int*          xInd,
               const cuComplex*    beta,
               cuComplex*          y,
               UPTKsparseIndexBase_t idxBase,
               void*               pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgemvi_bufferSize(UPTKsparseHandle_t    handle,
                          UPTKsparseOperation_t transA,
                          int                 m,
                          int                 n,
                          int                 nnz,
                          int*                pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgemvi(UPTKsparseHandle_t       handle,
               UPTKsparseOperation_t    transA,
               int                    m,
               int                    n,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* A,
               int                    lda,
               int                    nnz,
               const cuDoubleComplex* xVal,
               const int*             xInd,
               const cuDoubleComplex* beta,
               cuDoubleComplex*       y,
               UPTKsparseIndexBase_t    idxBase,
               void*                  pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgemvi_bufferSize(UPTKsparseHandle_t    handle,
                          UPTKsparseOperation_t transA,
                          int                 m,
                          int                 n,
                          int                 nnz,
                          int*                pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsrmv(UPTKsparseHandle_t         handle,
               UPTKsparseDirection_t      dirA,
               UPTKsparseOperation_t      transA,
               int                      mb,
               int                      nb,
               int                      nnzb,
               const float*             alpha,
               const UPTKsparseMatDescr_t descrA,
               const float*             bsrSortedValA,
               const int*               bsrSortedRowPtrA,
               const int*               bsrSortedColIndA,
               int                      blockDim,
               const float*             x,
               const float*             beta,
               float*                   y);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsrmv(UPTKsparseHandle_t         handle,
               UPTKsparseDirection_t      dirA,
               UPTKsparseOperation_t      transA,
               int                      mb,
               int                      nb,
               int                      nnzb,
               const double*            alpha,
               const UPTKsparseMatDescr_t descrA,
               const double*            bsrSortedValA,
               const int*               bsrSortedRowPtrA,
               const int*               bsrSortedColIndA,
               int                      blockDim,
               const double*            x,
               const double*            beta,
               double*                  y);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsrmv(UPTKsparseHandle_t         handle,
               UPTKsparseDirection_t      dirA,
               UPTKsparseOperation_t      transA,
               int                      mb,
               int                      nb,
               int                      nnzb,
               const cuComplex*         alpha,
               const UPTKsparseMatDescr_t descrA,
               const cuComplex*         bsrSortedValA,
               const int*               bsrSortedRowPtrA,
               const int*               bsrSortedColIndA,
               int                      blockDim,
               const cuComplex*         x,
               const cuComplex*         beta,
               cuComplex*               y);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsrmv(UPTKsparseHandle_t         handle,
               UPTKsparseDirection_t      dirA,
               UPTKsparseOperation_t      transA,
               int                      mb,
               int                      nb,
               int                      nnzb,
               const cuDoubleComplex*   alpha,
               const UPTKsparseMatDescr_t descrA,
               const cuDoubleComplex*   bsrSortedValA,
               const int*               bsrSortedRowPtrA,
               const int*               bsrSortedColIndA,
               int                      blockDim,
               const cuDoubleComplex*   x,
               const cuDoubleComplex*   beta,
               cuDoubleComplex*         y);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsrxmv(UPTKsparseHandle_t         handle,
                UPTKsparseDirection_t      dirA,
                UPTKsparseOperation_t      transA,
                int                      sizeOfMask,
                int                      mb,
                int                      nb,
                int                      nnzb,
                const float*             alpha,
                const UPTKsparseMatDescr_t descrA,
                const float*             bsrSortedValA,
                const int*               bsrSortedMaskPtrA,
                const int*               bsrSortedRowPtrA,
                const int*               bsrSortedEndPtrA,
                const int*               bsrSortedColIndA,
                int                      blockDim,
                const float*             x,
                const float*             beta,
                float*                   y);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsrxmv(UPTKsparseHandle_t         handle,
                UPTKsparseDirection_t      dirA,
                UPTKsparseOperation_t      transA,
                int                      sizeOfMask,
                int                      mb,
                int                      nb,
                int                      nnzb,
                const double*            alpha,
                const UPTKsparseMatDescr_t descrA,
                const double*            bsrSortedValA,
                const int*               bsrSortedMaskPtrA,
                const int*               bsrSortedRowPtrA,
                const int*               bsrSortedEndPtrA,
                const int*               bsrSortedColIndA,
                int                      blockDim,
                const double*            x,
                const double*            beta,
                double*                  y);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsrxmv(UPTKsparseHandle_t         handle,
                UPTKsparseDirection_t      dirA,
                UPTKsparseOperation_t      transA,
                int                      sizeOfMask,
                int                      mb,
                int                      nb,
                int                      nnzb,
                const cuComplex*         alpha,
                const UPTKsparseMatDescr_t descrA,
                const cuComplex*         bsrSortedValA,
                const int*               bsrSortedMaskPtrA,
                const int*               bsrSortedRowPtrA,
                const int*               bsrSortedEndPtrA,
                const int*               bsrSortedColIndA,
                int                      blockDim,
                const cuComplex*         x,
                const cuComplex*         beta,
                cuComplex*               y);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsrxmv(UPTKsparseHandle_t         handle,
                UPTKsparseDirection_t      dirA,
                UPTKsparseOperation_t      transA,
                int                      sizeOfMask,
                int                      mb,
                int                      nb,
                int                      nnzb,
                const cuDoubleComplex*   alpha,
                const UPTKsparseMatDescr_t descrA,
                const cuDoubleComplex*   bsrSortedValA,
                const int*               bsrSortedMaskPtrA,
                const int*               bsrSortedRowPtrA,
                const int*               bsrSortedEndPtrA,
                const int*               bsrSortedColIndA,
                int                      blockDim,
                const cuDoubleComplex*   x,
                const cuDoubleComplex*   beta,
                cuDoubleComplex*         y);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXbsrsv2_zeroPivot(UPTKsparseHandle_t handle,
                          UPTKbsrsv2Info_t     info,
                          int*             position);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsrsv2_bufferSize(UPTKsparseHandle_t         handle,
                           UPTKsparseDirection_t      dirA,
                           UPTKsparseOperation_t      transA,
                           int                      mb,
                           int                      nnzb,
                           const UPTKsparseMatDescr_t descrA,
                           float*                   bsrSortedValA,
                           const int*               bsrSortedRowPtrA,
                           const int*               bsrSortedColIndA,
                           int                      blockDim,
                           UPTKbsrsv2Info_t             info,
                           int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsrsv2_bufferSize(UPTKsparseHandle_t         handle,
                           UPTKsparseDirection_t      dirA,
                           UPTKsparseOperation_t      transA,
                           int                      mb,
                           int                      nnzb,
                           const UPTKsparseMatDescr_t descrA,
                           double*                  bsrSortedValA,
                           const int*               bsrSortedRowPtrA,
                           const int*               bsrSortedColIndA,
                           int                      blockDim,
                           UPTKbsrsv2Info_t             info,
                           int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsrsv2_bufferSize(UPTKsparseHandle_t         handle,
                           UPTKsparseDirection_t      dirA,
                           UPTKsparseOperation_t      transA,
                           int                      mb,
                           int                      nnzb,
                           const UPTKsparseMatDescr_t descrA,
                           cuComplex*               bsrSortedValA,
                           const int*               bsrSortedRowPtrA,
                           const int*               bsrSortedColIndA,
                           int                      blockDim,
                           UPTKbsrsv2Info_t             info,
                           int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsrsv2_bufferSize(UPTKsparseHandle_t         handle,
                           UPTKsparseDirection_t      dirA,
                           UPTKsparseOperation_t      transA,
                           int                      mb,
                           int                      nnzb,
                           const UPTKsparseMatDescr_t descrA,
                           cuDoubleComplex*         bsrSortedValA,
                           const int*               bsrSortedRowPtrA,
                           const int*               bsrSortedColIndA,
                           int                      blockDim,
                           UPTKbsrsv2Info_t             info,
                           int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsrsv2_bufferSizeExt(UPTKsparseHandle_t         handle,
                              UPTKsparseDirection_t      dirA,
                              UPTKsparseOperation_t      transA,
                              int                      mb,
                              int                      nnzb,
                              const UPTKsparseMatDescr_t descrA,
                              float*                   bsrSortedValA,
                              const int*               bsrSortedRowPtrA,
                              const int*               bsrSortedColIndA,
                              int                      blockSize,
                              UPTKbsrsv2Info_t             info,
                              size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsrsv2_bufferSizeExt(UPTKsparseHandle_t         handle,
                              UPTKsparseDirection_t      dirA,
                              UPTKsparseOperation_t      transA,
                              int                      mb,
                              int                      nnzb,
                              const UPTKsparseMatDescr_t descrA,
                              double*                  bsrSortedValA,
                              const int*               bsrSortedRowPtrA,
                              const int*               bsrSortedColIndA,
                              int                      blockSize,
                              UPTKbsrsv2Info_t             info,
                              size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsrsv2_bufferSizeExt(UPTKsparseHandle_t         handle,
                              UPTKsparseDirection_t      dirA,
                              UPTKsparseOperation_t      transA,
                              int                      mb,
                              int                      nnzb,
                              const UPTKsparseMatDescr_t descrA,
                              cuComplex*               bsrSortedValA,
                              const int*               bsrSortedRowPtrA,
                              const int*               bsrSortedColIndA,
                              int                      blockSize,
                              UPTKbsrsv2Info_t             info,
                              size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsrsv2_bufferSizeExt(UPTKsparseHandle_t         handle,
                              UPTKsparseDirection_t      dirA,
                              UPTKsparseOperation_t      transA,
                              int                      mb,
                              int                      nnzb,
                              const UPTKsparseMatDescr_t descrA,
                              cuDoubleComplex*         bsrSortedValA,
                              const int*               bsrSortedRowPtrA,
                              const int*               bsrSortedColIndA,
                              int                      blockSize,
                              UPTKbsrsv2Info_t             info,
                              size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsrsv2_analysis(UPTKsparseHandle_t         handle,
                         UPTKsparseDirection_t      dirA,
                         UPTKsparseOperation_t      transA,
                         int                      mb,
                         int                      nnzb,
                         const UPTKsparseMatDescr_t descrA,
                         const float*             bsrSortedValA,
                         const int*               bsrSortedRowPtrA,
                         const int*               bsrSortedColIndA,
                         int                      blockDim,
                         UPTKbsrsv2Info_t             info,
                         UPTKsparseSolvePolicy_t    policy,
                         void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsrsv2_analysis(UPTKsparseHandle_t         handle,
                         UPTKsparseDirection_t      dirA,
                         UPTKsparseOperation_t      transA,
                         int                      mb,
                         int                      nnzb,
                         const UPTKsparseMatDescr_t descrA,
                         const double*            bsrSortedValA,
                         const int*               bsrSortedRowPtrA,
                         const int*               bsrSortedColIndA,
                         int                      blockDim,
                         UPTKbsrsv2Info_t             info,
                         UPTKsparseSolvePolicy_t    policy,
                         void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsrsv2_analysis(UPTKsparseHandle_t         handle,
                         UPTKsparseDirection_t      dirA,
                         UPTKsparseOperation_t      transA,
                         int                      mb,
                         int                      nnzb,
                         const UPTKsparseMatDescr_t descrA,
                         const cuComplex*         bsrSortedValA,
                         const int*               bsrSortedRowPtrA,
                         const int*               bsrSortedColIndA,
                         int                      blockDim,
                         UPTKbsrsv2Info_t             info,
                         UPTKsparseSolvePolicy_t    policy,
                         void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsrsv2_analysis(UPTKsparseHandle_t         handle,
                         UPTKsparseDirection_t      dirA,
                         UPTKsparseOperation_t      transA,
                         int                      mb,
                         int                      nnzb,
                         const UPTKsparseMatDescr_t descrA,
                         const cuDoubleComplex*   bsrSortedValA,
                         const int*               bsrSortedRowPtrA,
                         const int*               bsrSortedColIndA,
                         int                      blockDim,
                         UPTKbsrsv2Info_t             info,
                         UPTKsparseSolvePolicy_t    policy,
                         void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsrsv2_solve(UPTKsparseHandle_t         handle,
                      UPTKsparseDirection_t      dirA,
                      UPTKsparseOperation_t      transA,
                      int                      mb,
                      int                      nnzb,
                      const float*             alpha,
                      const UPTKsparseMatDescr_t descrA,
                      const float*             bsrSortedValA,
                      const int*               bsrSortedRowPtrA,
                      const int*               bsrSortedColIndA,
                      int                      blockDim,
                      UPTKbsrsv2Info_t             info,
                      const float*             f,
                      float*                   x,
                      UPTKsparseSolvePolicy_t    policy,
                      void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsrsv2_solve(UPTKsparseHandle_t         handle,
                      UPTKsparseDirection_t      dirA,
                      UPTKsparseOperation_t      transA,
                      int                      mb,
                      int                      nnzb,
                      const double*            alpha,
                      const UPTKsparseMatDescr_t descrA,
                      const double*            bsrSortedValA,
                      const int*               bsrSortedRowPtrA,
                      const int*               bsrSortedColIndA,
                      int                      blockDim,
                      UPTKbsrsv2Info_t             info,
                      const double*            f,
                      double*                  x,
                      UPTKsparseSolvePolicy_t    policy,
                      void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsrsv2_solve(UPTKsparseHandle_t         handle,
                      UPTKsparseDirection_t      dirA,
                      UPTKsparseOperation_t      transA,
                      int                      mb,
                      int                      nnzb,
                      const cuComplex*         alpha,
                      const UPTKsparseMatDescr_t descrA,
                      const cuComplex*         bsrSortedValA,
                      const int*               bsrSortedRowPtrA,
                      const int*               bsrSortedColIndA,
                      int                      blockDim,
                      UPTKbsrsv2Info_t             info,
                      const cuComplex*         f,
                      cuComplex*               x,
                      UPTKsparseSolvePolicy_t    policy,
                      void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsrsv2_solve(UPTKsparseHandle_t         handle,
                      UPTKsparseDirection_t      dirA,
                      UPTKsparseOperation_t      transA,
                      int                      mb,
                      int                      nnzb,
                      const cuDoubleComplex*   alpha,
                      const UPTKsparseMatDescr_t descrA,
                      const cuDoubleComplex*   bsrSortedValA,
                      const int*               bsrSortedRowPtrA,
                      const int*               bsrSortedColIndA,
                      int                      blockDim,
                      UPTKbsrsv2Info_t             info,
                      const cuDoubleComplex*   f,
                      cuDoubleComplex*         x,
                      UPTKsparseSolvePolicy_t    policy,
                      void*                    pBuffer);

//##############################################################################
//# SPARSE LEVEL 3 ROUTINES
//##############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsrmm(UPTKsparseHandle_t         handle,
               UPTKsparseDirection_t      dirA,
               UPTKsparseOperation_t      transA,
               UPTKsparseOperation_t      transB,
               int                      mb,
               int                      n,
               int                      kb,
               int                      nnzb,
               const float*             alpha,
               const UPTKsparseMatDescr_t descrA,
               const float* bsrSortedValA,
               const int*   bsrSortedRowPtrA,
               const int*   bsrSortedColIndA,
               const int    blockSize,
               const float* B,
               const int    ldb,
               const float* beta,
               float*       C,
               int          ldc);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsrmm(UPTKsparseHandle_t         handle,
               UPTKsparseDirection_t      dirA,
               UPTKsparseOperation_t      transA,
               UPTKsparseOperation_t      transB,
               int                      mb,
               int                      n,
               int                      kb,
               int                      nnzb,
               const double*            alpha,
               const UPTKsparseMatDescr_t descrA,
               const double* bsrSortedValA,
               const int*    bsrSortedRowPtrA,
               const int*    bsrSortedColIndA,
               const int     blockSize,
               const double* B,
               const int     ldb,
               const double* beta,
               double*       C,
               int           ldc);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsrmm(UPTKsparseHandle_t         handle,
               UPTKsparseDirection_t      dirA,
               UPTKsparseOperation_t      transA,
               UPTKsparseOperation_t      transB,
               int                      mb,
               int                      n,
               int                      kb,
               int                      nnzb,
               const cuComplex*         alpha,
               const UPTKsparseMatDescr_t descrA,
               const cuComplex* bsrSortedValA,
               const int*       bsrSortedRowPtrA,
               const int*       bsrSortedColIndA,
               const int        blockSize,
               const cuComplex* B,
               const int        ldb,
               const cuComplex* beta,
               cuComplex*       C,
               int              ldc);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsrmm(UPTKsparseHandle_t         handle,
                UPTKsparseDirection_t      dirA,
                UPTKsparseOperation_t      transA,
                UPTKsparseOperation_t      transB,
                int                      mb,
                int                      n,
                int                      kb,
                int                      nnzb,
                const cuDoubleComplex*   alpha,
                const UPTKsparseMatDescr_t descrA,
                const cuDoubleComplex*   bsrSortedValA,
                const int*               bsrSortedRowPtrA,
                const int*               bsrSortedColIndA,
                const int                blockSize,
                const cuDoubleComplex*   B,
                const int                ldb,
                const cuDoubleComplex*   beta,
                cuDoubleComplex*         C,
                int                      ldc);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXbsrsm2_zeroPivot(UPTKsparseHandle_t handle,
                          UPTKbsrsm2Info_t     info,
                          int*             position);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsrsm2_bufferSize(UPTKsparseHandle_t         handle,
                           UPTKsparseDirection_t      dirA,
                           UPTKsparseOperation_t      transA,
                           UPTKsparseOperation_t      transXY,
                           int                      mb,
                           int                      n,
                           int                      nnzb,
                           const UPTKsparseMatDescr_t descrA,
                           float*                   bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockSize,
                           UPTKbsrsm2Info_t             info,
                           int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsrsm2_bufferSize(UPTKsparseHandle_t         handle,
                           UPTKsparseDirection_t      dirA,
                           UPTKsparseOperation_t      transA,
                           UPTKsparseOperation_t      transXY,
                           int                      mb,
                           int                      n,
                           int                      nnzb,
                           const UPTKsparseMatDescr_t descrA,
                           double*                  bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockSize,
                           UPTKbsrsm2Info_t             info,
                           int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsrsm2_bufferSize(UPTKsparseHandle_t         handle,
                           UPTKsparseDirection_t      dirA,
                           UPTKsparseOperation_t      transA,
                           UPTKsparseOperation_t      transXY,
                           int                      mb,
                           int                      n,
                           int                      nnzb,
                           const UPTKsparseMatDescr_t descrA,
                           cuComplex*               bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockSize,
                           UPTKbsrsm2Info_t             info,
                           int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsrsm2_bufferSize(UPTKsparseHandle_t         handle,
                           UPTKsparseDirection_t      dirA,
                           UPTKsparseOperation_t      transA,
                           UPTKsparseOperation_t      transXY,
                           int                      mb,
                           int                      n,
                           int                      nnzb,
                           const UPTKsparseMatDescr_t descrA,
                           cuDoubleComplex*         bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockSize,
                           UPTKbsrsm2Info_t             info,
                           int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsrsm2_analysis(UPTKsparseHandle_t         handle,
                         UPTKsparseDirection_t      dirA,
                         UPTKsparseOperation_t      transA,
                         UPTKsparseOperation_t      transXY,
                         int                      mb,
                         int                      n,
                         int                      nnzb,
                         const UPTKsparseMatDescr_t descrA,
                         const float*             bsrSortedVal,
                         const int*               bsrSortedRowPtr,
                         const int*               bsrSortedColInd,
                         int                      blockSize,
                         UPTKbsrsm2Info_t             info,
                         UPTKsparseSolvePolicy_t    policy,
                         void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsrsm2_analysis(UPTKsparseHandle_t         handle,
                         UPTKsparseDirection_t      dirA,
                         UPTKsparseOperation_t      transA,
                         UPTKsparseOperation_t      transXY,
                         int                      mb,
                         int                      n,
                         int                      nnzb,
                         const UPTKsparseMatDescr_t descrA,
                         const double*            bsrSortedVal,
                         const int*               bsrSortedRowPtr,
                         const int*               bsrSortedColInd,
                         int                      blockSize,
                         UPTKbsrsm2Info_t             info,
                         UPTKsparseSolvePolicy_t    policy,
                         void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsrsm2_analysis(UPTKsparseHandle_t         handle,
                         UPTKsparseDirection_t      dirA,
                         UPTKsparseOperation_t      transA,
                         UPTKsparseOperation_t      transXY,
                         int                      mb,
                         int                      n,
                         int                      nnzb,
                         const UPTKsparseMatDescr_t descrA,
                         const cuComplex*         bsrSortedVal,
                         const int*               bsrSortedRowPtr,
                         const int*               bsrSortedColInd,
                         int                      blockSize,
                         UPTKbsrsm2Info_t             info,
                         UPTKsparseSolvePolicy_t    policy,
                         void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsrsm2_analysis(UPTKsparseHandle_t         handle,
                         UPTKsparseDirection_t      dirA,
                         UPTKsparseOperation_t      transA,
                         UPTKsparseOperation_t      transXY,
                         int                      mb,
                         int                      n,
                         int                      nnzb,
                         const UPTKsparseMatDescr_t descrA,
                         const cuDoubleComplex*   bsrSortedVal,
                         const int*               bsrSortedRowPtr,
                         const int*               bsrSortedColInd,
                         int                      blockSize,
                         UPTKbsrsm2Info_t             info,
                         UPTKsparseSolvePolicy_t    policy,
                         void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsrsm2_solve(UPTKsparseHandle_t         handle,
                      UPTKsparseDirection_t      dirA,
                      UPTKsparseOperation_t      transA,
                      UPTKsparseOperation_t      transXY,
                      int                      mb,
                      int                      n,
                      int                      nnzb,
                      const float*             alpha,
                      const UPTKsparseMatDescr_t descrA,
                      const float*             bsrSortedVal,
                      const int*               bsrSortedRowPtr,
                      const int*               bsrSortedColInd,
                      int                      blockSize,
                      UPTKbsrsm2Info_t             info,
                      const float*             B,
                      int                      ldb,
                      float*                   X,
                      int                      ldx,
                      UPTKsparseSolvePolicy_t    policy,
                      void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsrsm2_solve(UPTKsparseHandle_t         handle,
                      UPTKsparseDirection_t      dirA,
                      UPTKsparseOperation_t      transA,
                      UPTKsparseOperation_t      transXY,
                      int                      mb,
                      int                      n,
                      int                      nnzb,
                      const double*            alpha,
                      const UPTKsparseMatDescr_t descrA,
                      const double*            bsrSortedVal,
                      const int*               bsrSortedRowPtr,
                      const int*               bsrSortedColInd,
                      int                      blockSize,
                      UPTKbsrsm2Info_t             info,
                      const double*            B,
                      int                      ldb,
                      double*                  X,
                      int                      ldx,
                      UPTKsparseSolvePolicy_t    policy,
                      void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsrsm2_solve(UPTKsparseHandle_t         handle,
                      UPTKsparseDirection_t      dirA,
                      UPTKsparseOperation_t      transA,
                      UPTKsparseOperation_t      transXY,
                      int                      mb,
                      int                      n,
                      int                      nnzb,
                      const cuComplex*         alpha,
                      const UPTKsparseMatDescr_t descrA,
                      const cuComplex*         bsrSortedVal,
                      const int*               bsrSortedRowPtr,
                      const int*               bsrSortedColInd,
                      int                      blockSize,
                      UPTKbsrsm2Info_t             info,
                      const cuComplex*         B,
                      int                      ldb,
                      cuComplex*               X,
                      int                      ldx,
                      UPTKsparseSolvePolicy_t    policy,
                      void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsrsm2_solve(UPTKsparseHandle_t         handle,
                      UPTKsparseDirection_t      dirA,
                      UPTKsparseOperation_t      transA,
                      UPTKsparseOperation_t      transXY,
                      int                      mb,
                      int                      n,
                      int                      nnzb,
                      const cuDoubleComplex*   alpha,
                      const UPTKsparseMatDescr_t descrA,
                      const cuDoubleComplex*   bsrSortedVal,
                      const int*               bsrSortedRowPtr,
                      const int*               bsrSortedColInd,
                      int                      blockSize,
                      UPTKbsrsm2Info_t             info,
                      const cuDoubleComplex*   B,
                      int                      ldb,
                      cuDoubleComplex*         X,
                      int                      ldx,
                      UPTKsparseSolvePolicy_t    policy,
                      void*                    pBuffer);

//##############################################################################
//# PRECONDITIONERS
//##############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsrilu02_numericBoost(UPTKsparseHandle_t handle,
                               UPTKcsrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               float*           boost_val);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsrilu02_numericBoost(UPTKsparseHandle_t handle,
                               UPTKcsrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               double*          boost_val);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsrilu02_numericBoost(UPTKsparseHandle_t handle,
                               UPTKcsrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               cuComplex*       boost_val);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsrilu02_numericBoost(UPTKsparseHandle_t handle,
                               UPTKcsrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               cuDoubleComplex* boost_val);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXcsrilu02_zeroPivot(UPTKsparseHandle_t handle,
                            UPTKcsrilu02Info_t   info,
                            int*             position);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsrilu02_bufferSize(UPTKsparseHandle_t         handle,
                             int                      m,
                             int                      nnz,
                             const UPTKsparseMatDescr_t descrA,
                             float*                   csrSortedValA,
                             const int*               csrSortedRowPtrA,
                             const int*               csrSortedColIndA,
                             UPTKcsrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsrilu02_bufferSize(UPTKsparseHandle_t         handle,
                             int                      m,
                             int                      nnz,
                             const UPTKsparseMatDescr_t descrA,
                             double*                  csrSortedValA,
                             const int*               csrSortedRowPtrA,
                             const int*               csrSortedColIndA,
                             UPTKcsrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsrilu02_bufferSize(UPTKsparseHandle_t         handle,
                             int                      m,
                             int                      nnz,
                             const UPTKsparseMatDescr_t descrA,
                             cuComplex*               csrSortedValA,
                             const int*               csrSortedRowPtrA,
                             const int*               csrSortedColIndA,
                             UPTKcsrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsrilu02_bufferSize(UPTKsparseHandle_t         handle,
                             int                      m,
                             int                      nnz,
                             const UPTKsparseMatDescr_t descrA,
                             cuDoubleComplex*         csrSortedValA,
                             const int*               csrSortedRowPtrA,
                             const int*               csrSortedColIndA,
                             UPTKcsrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsrilu02_bufferSizeExt(UPTKsparseHandle_t         handle,
                                int                      m,
                                int                      nnz,
                                const UPTKsparseMatDescr_t descrA,
                                float*                   csrSortedVal,
                                const int*               csrSortedRowPtr,
                                const int*               csrSortedColInd,
                                UPTKcsrilu02Info_t           info,
                                size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsrilu02_bufferSizeExt(UPTKsparseHandle_t         handle,
                                int                      m,
                                int                      nnz,
                                const UPTKsparseMatDescr_t descrA,
                                double*                  csrSortedVal,
                                const int*               csrSortedRowPtr,
                                const int*               csrSortedColInd,
                                UPTKcsrilu02Info_t           info,
                                size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsrilu02_bufferSizeExt(UPTKsparseHandle_t         handle,
                                int                      m,
                                int                      nnz,
                                const UPTKsparseMatDescr_t descrA,
                                cuComplex*               csrSortedVal,
                                const int*               csrSortedRowPtr,
                                const int*               csrSortedColInd,
                                UPTKcsrilu02Info_t           info,
                                size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsrilu02_bufferSizeExt(UPTKsparseHandle_t         handle,
                                int                      m,
                                int                      nnz,
                                const UPTKsparseMatDescr_t descrA,
                                cuDoubleComplex*         csrSortedVal,
                                const int*               csrSortedRowPtr,
                                const int*               csrSortedColInd,
                                UPTKcsrilu02Info_t           info,
                                size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsrilu02_analysis(UPTKsparseHandle_t         handle,
                           int                      m,
                           int                      nnz,
                           const UPTKsparseMatDescr_t descrA,
                           const float*             csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           UPTKcsrilu02Info_t           info,
                           UPTKsparseSolvePolicy_t    policy,
                           void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsrilu02_analysis(UPTKsparseHandle_t         handle,
                           int                      m,
                           int                      nnz,
                           const UPTKsparseMatDescr_t descrA,
                           const double*            csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           UPTKcsrilu02Info_t           info,
                           UPTKsparseSolvePolicy_t    policy,
                           void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsrilu02_analysis(UPTKsparseHandle_t         handle,
                           int                      m,
                           int                      nnz,
                           const UPTKsparseMatDescr_t descrA,
                           const cuComplex*         csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           UPTKcsrilu02Info_t           info,
                           UPTKsparseSolvePolicy_t    policy,
                           void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsrilu02_analysis(UPTKsparseHandle_t         handle,
                           int                      m,
                           int                      nnz,
                           const UPTKsparseMatDescr_t descrA,
                           const cuDoubleComplex*   csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           UPTKcsrilu02Info_t           info,
                           UPTKsparseSolvePolicy_t    policy,
                           void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsrilu02(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      nnz,
                  const UPTKsparseMatDescr_t descrA,
                  float*                   csrSortedValA_valM,
                  const int*            csrSortedRowPtrA,
                  const int*            csrSortedColIndA,
                  UPTKcsrilu02Info_t        info,
                  UPTKsparseSolvePolicy_t policy,
                  void*                 pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsrilu02(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      nnz,
                  const UPTKsparseMatDescr_t descrA,
                  double*                  csrSortedValA_valM,
                  const int*            csrSortedRowPtrA,
                  const int*            csrSortedColIndA,
                  UPTKcsrilu02Info_t        info,
                  UPTKsparseSolvePolicy_t policy,
                  void*                 pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsrilu02(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      nnz,
                  const UPTKsparseMatDescr_t descrA,
                  cuComplex*               csrSortedValA_valM,
                  const int*            csrSortedRowPtrA,
                  const int*            csrSortedColIndA,
                  UPTKcsrilu02Info_t        info,
                  UPTKsparseSolvePolicy_t policy,
                  void*                 pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsrilu02(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      nnz,
                  const UPTKsparseMatDescr_t descrA,
                  cuDoubleComplex*         csrSortedValA_valM,
                  const int*            csrSortedRowPtrA,
                  const int*            csrSortedColIndA,
                  UPTKcsrilu02Info_t        info,
                  UPTKsparseSolvePolicy_t policy,
                  void*                 pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsrilu02_numericBoost(UPTKsparseHandle_t handle,
                               UPTKbsrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               float*           boost_val);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsrilu02_numericBoost(UPTKsparseHandle_t handle,
                               UPTKbsrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               double*          boost_val);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsrilu02_numericBoost(UPTKsparseHandle_t handle,
                               UPTKbsrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               cuComplex*       boost_val);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsrilu02_numericBoost(UPTKsparseHandle_t handle,
                               UPTKbsrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               cuDoubleComplex* boost_val);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXbsrilu02_zeroPivot(UPTKsparseHandle_t handle,
                            UPTKbsrilu02Info_t   info,
                            int*             position);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsrilu02_bufferSize(UPTKsparseHandle_t         handle,
                             UPTKsparseDirection_t      dirA,
                             int                      mb,
                             int                      nnzb,
                             const UPTKsparseMatDescr_t descrA,
                             float*                   bsrSortedVal,
                             const int*               bsrSortedRowPtr,
                             const int*               bsrSortedColInd,
                             int                      blockDim,
                             UPTKbsrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsrilu02_bufferSize(UPTKsparseHandle_t         handle,
                             UPTKsparseDirection_t      dirA,
                             int                      mb,
                             int                      nnzb,
                             const UPTKsparseMatDescr_t descrA,
                             double*                  bsrSortedVal,
                             const int*               bsrSortedRowPtr,
                             const int*               bsrSortedColInd,
                             int                      blockDim,
                             UPTKbsrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsrilu02_bufferSize(UPTKsparseHandle_t         handle,
                             UPTKsparseDirection_t      dirA,
                             int                      mb,
                             int                      nnzb,
                             const UPTKsparseMatDescr_t descrA,
                             cuComplex*               bsrSortedVal,
                             const int*               bsrSortedRowPtr,
                             const int*               bsrSortedColInd,
                             int                      blockDim,
                             UPTKbsrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsrilu02_bufferSize(UPTKsparseHandle_t         handle,
                             UPTKsparseDirection_t      dirA,
                             int                      mb,
                             int                      nnzb,
                             const UPTKsparseMatDescr_t descrA,
                             cuDoubleComplex*         bsrSortedVal,
                             const int*               bsrSortedRowPtr,
                             const int*               bsrSortedColInd,
                             int                      blockDim,
                             UPTKbsrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsrilu02_analysis(UPTKsparseHandle_t         handle,
                           UPTKsparseDirection_t      dirA,
                           int                      mb,
                           int                      nnzb,
                           const UPTKsparseMatDescr_t descrA,
                           float*                   bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockDim,
                           UPTKbsrilu02Info_t           info,
                           UPTKsparseSolvePolicy_t    policy,
                           void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsrilu02_analysis(UPTKsparseHandle_t         handle,
                           UPTKsparseDirection_t      dirA,
                           int                      mb,
                           int                      nnzb,
                           const UPTKsparseMatDescr_t descrA,
                           double*                  bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockDim,
                           UPTKbsrilu02Info_t           info,
                           UPTKsparseSolvePolicy_t    policy,
                           void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsrilu02_analysis(UPTKsparseHandle_t         handle,
                           UPTKsparseDirection_t      dirA,
                           int                      mb,
                           int                      nnzb,
                           const UPTKsparseMatDescr_t descrA,
                           cuComplex*               bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockDim,
                           UPTKbsrilu02Info_t           info,
                           UPTKsparseSolvePolicy_t    policy,
                           void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsrilu02_analysis(UPTKsparseHandle_t         handle,
                           UPTKsparseDirection_t      dirA,
                           int                      mb,
                           int                      nnzb,
                           const UPTKsparseMatDescr_t descrA,
                           cuDoubleComplex*         bsrSortedVal,
                           const int*               bsrSortedRowPtr,
                           const int*               bsrSortedColInd,
                           int                      blockDim,
                           UPTKbsrilu02Info_t           info,
                           UPTKsparseSolvePolicy_t    policy,
                           void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsrilu02(UPTKsparseHandle_t         handle,
                  UPTKsparseDirection_t      dirA,
                  int                      mb,
                  int                      nnzb,
                  const UPTKsparseMatDescr_t descrA,
                  float*                   bsrSortedVal,
                  const int*               bsrSortedRowPtr,
                  const int*               bsrSortedColInd,
                  int                      blockDim,
                  UPTKbsrilu02Info_t           info,
                  UPTKsparseSolvePolicy_t    policy,
                  void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsrilu02(UPTKsparseHandle_t         handle,
                  UPTKsparseDirection_t      dirA,
                  int                      mb,
                  int                      nnzb,
                  const UPTKsparseMatDescr_t descrA,
                  double*                  bsrSortedVal,
                  const int*               bsrSortedRowPtr,
                  const int*               bsrSortedColInd,
                  int                      blockDim,
                  UPTKbsrilu02Info_t           info,
                  UPTKsparseSolvePolicy_t    policy,
                  void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsrilu02(UPTKsparseHandle_t         handle,
                  UPTKsparseDirection_t      dirA,
                  int                      mb,
                  int                      nnzb,
                  const UPTKsparseMatDescr_t descrA,
                  cuComplex*               bsrSortedVal,
                  const int*               bsrSortedRowPtr,
                  const int*               bsrSortedColInd,
                  int                      blockDim,
                  UPTKbsrilu02Info_t           info,
                  UPTKsparseSolvePolicy_t    policy,
                  void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsrilu02(UPTKsparseHandle_t         handle,
                  UPTKsparseDirection_t      dirA,
                  int                      mb,
                  int                      nnzb,
                  const UPTKsparseMatDescr_t descrA,
                  cuDoubleComplex*         bsrSortedVal,
                  const int*               bsrSortedRowPtr,
                  const int*               bsrSortedColInd,
                  int                      blockDim,
                  UPTKbsrilu02Info_t           info,
                  UPTKsparseSolvePolicy_t    policy,
                  void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXcsric02_zeroPivot(UPTKsparseHandle_t handle,
                           UPTKcsric02Info_t    info,
                           int*             position);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsric02_bufferSize(UPTKsparseHandle_t         handle,
                            int                      m,
                            int                      nnz,
                            const UPTKsparseMatDescr_t descrA,
                            float*                   csrSortedValA,
                            const int*               csrSortedRowPtrA,
                            const int*               csrSortedColIndA,
                            UPTKcsric02Info_t            info,
                            int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsric02_bufferSize(UPTKsparseHandle_t         handle,
                            int                      m,
                            int                      nnz,
                            const UPTKsparseMatDescr_t descrA,
                            double*                  csrSortedValA,
                            const int*               csrSortedRowPtrA,
                            const int*               csrSortedColIndA,
                            UPTKcsric02Info_t            info,
                            int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsric02_bufferSize(UPTKsparseHandle_t         handle,
                            int                      m,
                            int                      nnz,
                            const UPTKsparseMatDescr_t descrA,
                            cuComplex*               csrSortedValA,
                            const int*               csrSortedRowPtrA,
                            const int*               csrSortedColIndA,
                            UPTKcsric02Info_t            info,
                            int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsric02_bufferSize(UPTKsparseHandle_t         handle,
                            int                      m,
                            int                      nnz,
                            const UPTKsparseMatDescr_t descrA,
                            cuDoubleComplex*         csrSortedValA,
                            const int*               csrSortedRowPtrA,
                            const int*               csrSortedColIndA,
                            UPTKcsric02Info_t            info,
                            int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsric02_bufferSizeExt(UPTKsparseHandle_t         handle,
                               int                      m,
                               int                      nnz,
                               const UPTKsparseMatDescr_t descrA,
                               float*                   csrSortedVal,
                               const int*               csrSortedRowPtr,
                               const int*               csrSortedColInd,
                               UPTKcsric02Info_t            info,
                               size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsric02_bufferSizeExt(UPTKsparseHandle_t         handle,
                               int                      m,
                               int                      nnz,
                               const UPTKsparseMatDescr_t descrA,
                               double*                  csrSortedVal,
                               const int*               csrSortedRowPtr,
                               const int*               csrSortedColInd,
                               UPTKcsric02Info_t            info,
                               size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsric02_bufferSizeExt(UPTKsparseHandle_t         handle,
                               int                      m,
                               int                      nnz,
                               const UPTKsparseMatDescr_t descrA,
                               cuComplex*               csrSortedVal,
                               const int*               csrSortedRowPtr,
                               const int*               csrSortedColInd,
                               UPTKcsric02Info_t            info,
                               size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsric02_bufferSizeExt(UPTKsparseHandle_t         handle,
                               int                      m,
                               int                      nnz,
                               const UPTKsparseMatDescr_t descrA,
                               cuDoubleComplex*         csrSortedVal,
                               const int*               csrSortedRowPtr,
                               const int*               csrSortedColInd,
                               UPTKcsric02Info_t            info,
                               size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsric02_analysis(UPTKsparseHandle_t         handle,
                          int                      m,
                          int                      nnz,
                          const UPTKsparseMatDescr_t descrA,
                          const float*             csrSortedValA,
                          const int*               csrSortedRowPtrA,
                          const int*               csrSortedColIndA,
                          UPTKcsric02Info_t            info,
                          UPTKsparseSolvePolicy_t    policy,
                          void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsric02_analysis(UPTKsparseHandle_t         handle,
                          int                      m,
                          int                      nnz,
                          const UPTKsparseMatDescr_t descrA,
                          const double*            csrSortedValA,
                          const int*               csrSortedRowPtrA,
                          const int*               csrSortedColIndA,
                          UPTKcsric02Info_t            info,
                          UPTKsparseSolvePolicy_t    policy,
                          void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsric02_analysis(UPTKsparseHandle_t         handle,
                          int                      m,
                          int                      nnz,
                          const UPTKsparseMatDescr_t descrA,
                          const cuComplex*         csrSortedValA,
                          const int*               csrSortedRowPtrA,
                          const int*               csrSortedColIndA,
                          UPTKcsric02Info_t            info,
                          UPTKsparseSolvePolicy_t    policy,
                          void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsric02_analysis(UPTKsparseHandle_t         handle,
                          int                      m,
                          int                      nnz,
                          const UPTKsparseMatDescr_t descrA,
                          const cuDoubleComplex*   csrSortedValA,
                          const int*               csrSortedRowPtrA,
                          const int*               csrSortedColIndA,
                          UPTKcsric02Info_t            info,
                          UPTKsparseSolvePolicy_t    policy,
                          void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsric02(UPTKsparseHandle_t         handle,
                 int                      m,
                 int                      nnz,
                 const UPTKsparseMatDescr_t descrA,
                 float*                   csrSortedValA_valM,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 UPTKcsric02Info_t            info,
                 UPTKsparseSolvePolicy_t    policy,
                 void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsric02(UPTKsparseHandle_t         handle,
                 int                      m,
                 int                      nnz,
                 const UPTKsparseMatDescr_t descrA,
                 double*                  csrSortedValA_valM,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 UPTKcsric02Info_t            info,
                 UPTKsparseSolvePolicy_t    policy,
                 void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsric02(UPTKsparseHandle_t         handle,
                 int                      m,
                 int                      nnz,
                 const UPTKsparseMatDescr_t descrA,
                 cuComplex*               csrSortedValA_valM,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 UPTKcsric02Info_t            info,
                 UPTKsparseSolvePolicy_t    policy,
                 void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsric02(UPTKsparseHandle_t         handle,
                 int                      m,
                 int                      nnz,
                 const UPTKsparseMatDescr_t descrA,
                 cuDoubleComplex*         csrSortedValA_valM,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 UPTKcsric02Info_t            info,
                 UPTKsparseSolvePolicy_t    policy,
                 void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXbsric02_zeroPivot(UPTKsparseHandle_t handle,
                           UPTKbsric02Info_t    info,
                           int*             position);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsric02_bufferSize(UPTKsparseHandle_t         handle,
                            UPTKsparseDirection_t      dirA,
                            int                      mb,
                            int                      nnzb,
                            const UPTKsparseMatDescr_t descrA,
                            float*                   bsrSortedVal,
                            const int*               bsrSortedRowPtr,
                            const int*               bsrSortedColInd,
                            int                      blockDim,
                            UPTKbsric02Info_t            info,
                            int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsric02_bufferSize(UPTKsparseHandle_t         handle,
                            UPTKsparseDirection_t      dirA,
                            int                      mb,
                            int                      nnzb,
                            const UPTKsparseMatDescr_t descrA,
                            double*                  bsrSortedVal,
                            const int*               bsrSortedRowPtr,
                            const int*               bsrSortedColInd,
                            int                      blockDim,
                            UPTKbsric02Info_t            info,
                            int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsric02_bufferSize(UPTKsparseHandle_t         handle,
                            UPTKsparseDirection_t      dirA,
                            int                      mb,
                            int                      nnzb,
                            const UPTKsparseMatDescr_t descrA,
                            cuComplex*               bsrSortedVal,
                            const int*               bsrSortedRowPtr,
                            const int*               bsrSortedColInd,
                            int                      blockDim,
                            UPTKbsric02Info_t            info,
                            int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsric02_bufferSize(UPTKsparseHandle_t         handle,
                            UPTKsparseDirection_t      dirA,
                            int                      mb,
                            int                      nnzb,
                            const UPTKsparseMatDescr_t descrA,
                            cuDoubleComplex*         bsrSortedVal,
                            const int*               bsrSortedRowPtr,
                            const int*               bsrSortedColInd,
                            int                      blockDim,
                            UPTKbsric02Info_t            info,
                            int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsric02_analysis(UPTKsparseHandle_t         handle,
                          UPTKsparseDirection_t      dirA,
                          int                      mb,
                          int                      nnzb,
                          const UPTKsparseMatDescr_t descrA,
                          const float*             bsrSortedVal,
                          const int*               bsrSortedRowPtr,
                          const int*               bsrSortedColInd,
                          int                      blockDim,
                          UPTKbsric02Info_t            info,
                          UPTKsparseSolvePolicy_t    policy,
                          void*                    pInputBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsric02_analysis(UPTKsparseHandle_t         handle,
                          UPTKsparseDirection_t      dirA,
                          int                      mb,
                          int                      nnzb,
                          const UPTKsparseMatDescr_t descrA,
                          const double*            bsrSortedVal,
                          const int*               bsrSortedRowPtr,
                          const int*               bsrSortedColInd,
                          int                      blockDim,
                          UPTKbsric02Info_t            info,
                          UPTKsparseSolvePolicy_t    policy,
                          void*                    pInputBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsric02_analysis(UPTKsparseHandle_t         handle,
                          UPTKsparseDirection_t      dirA,
                          int                      mb,
                          int                      nnzb,
                          const UPTKsparseMatDescr_t descrA,
                          const cuComplex*         bsrSortedVal,
                          const int*               bsrSortedRowPtr,
                          const int*               bsrSortedColInd,
                          int                      blockDim,
                          UPTKbsric02Info_t            info,
                          UPTKsparseSolvePolicy_t    policy,
                          void*                    pInputBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsric02_analysis(UPTKsparseHandle_t         handle,
                          UPTKsparseDirection_t      dirA,
                          int                      mb,
                          int                      nnzb,
                          const UPTKsparseMatDescr_t descrA,
                          const cuDoubleComplex*   bsrSortedVal,
                          const int*               bsrSortedRowPtr,
                          const int*               bsrSortedColInd,
                          int                      blockDim,
                          UPTKbsric02Info_t            info,
                          UPTKsparseSolvePolicy_t    policy,
                          void*                    pInputBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsric02(UPTKsparseHandle_t         handle,
                 UPTKsparseDirection_t      dirA,
                 int                      mb,
                 int                      nnzb,
                 const UPTKsparseMatDescr_t descrA,
                 float*                   bsrSortedVal,
                 const int*               bsrSortedRowPtr,
                 const int*               bsrSortedColInd,
                 int                      blockDim,
                 UPTKbsric02Info_t            info,
                 UPTKsparseSolvePolicy_t    policy,
                 void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsric02(UPTKsparseHandle_t         handle,
                 UPTKsparseDirection_t      dirA,
                 int                      mb,
                 int                      nnzb,
                 const UPTKsparseMatDescr_t descrA,
                 double*                  bsrSortedVal,
                 const int*               bsrSortedRowPtr,
                 const int*               bsrSortedColInd,
                 int                      blockDim,
                 UPTKbsric02Info_t            info,
                 UPTKsparseSolvePolicy_t    policy,
                 void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsric02(UPTKsparseHandle_t         handle,
                 UPTKsparseDirection_t      dirA,
                 int                      mb,
                 int                      nnzb,
                 const UPTKsparseMatDescr_t descrA,
                 cuComplex*               bsrSortedVal,
                 const int*               bsrSortedRowPtr,
                 const int*
                      bsrSortedColInd,
                 int                      blockDim,
                 UPTKbsric02Info_t            info,
                 UPTKsparseSolvePolicy_t    policy,
                 void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsric02(UPTKsparseHandle_t         handle,
                 UPTKsparseDirection_t      dirA,
                 int                      mb,
                 int                      nnzb,
                 const UPTKsparseMatDescr_t descrA,
                 cuDoubleComplex*         bsrSortedVal,
                 const int*               bsrSortedRowPtr,
                 const int*               bsrSortedColInd,
                 int                      blockDim,
                 UPTKbsric02Info_t            info,
                 UPTKsparseSolvePolicy_t    policy,
                 void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSgtsv2_bufferSizeExt(UPTKsparseHandle_t handle,
                             int              m,
                             int              n,
                             const float*     dl,
                             const float*     d,
                             const float*     du,
                             const float*     B,
                             int              ldb,
                             size_t*          bufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgtsv2_bufferSizeExt(UPTKsparseHandle_t handle,
                             int              m,
                             int              n,
                             const double*    dl,
                             const double*    d,
                             const double*    du,
                             const double*    B,
                             int              ldb,
                             size_t*          bufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgtsv2_bufferSizeExt(UPTKsparseHandle_t handle,
                             int              m,
                             int              n,
                             const cuComplex* dl,
                             const cuComplex* d,
                             const cuComplex* du,
                             const cuComplex* B,
                             int              ldb,
                             size_t*          bufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgtsv2_bufferSizeExt(UPTKsparseHandle_t       handle,
                             int                    m,
                             int                    n,
                             const cuDoubleComplex* dl,
                             const cuDoubleComplex* d,
                             const cuDoubleComplex* du,
                             const cuDoubleComplex* B,
                             int                    ldb,
                             size_t*                bufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSgtsv2(UPTKsparseHandle_t handle,
               int              m,
               int              n,
               const float*     dl,
               const float*     d,
               const float*     du,
               float*           B,
               int              ldb,
               void*            pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgtsv2(UPTKsparseHandle_t handle,
               int              m,
               int              n,
               const double*    dl,
               const double*    d,
               const double*    du,
               double*          B,
               int              ldb,
               void*            pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgtsv2(UPTKsparseHandle_t handle,
               int              m,
               int              n,
               const cuComplex* dl,
               const cuComplex* d,
               const cuComplex* du,
               cuComplex*       B,
               int              ldb,
               void*            pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgtsv2(UPTKsparseHandle_t       handle,
               int                    m,
               int                    n,
               const cuDoubleComplex* dl,
               const cuDoubleComplex* d,
               const cuDoubleComplex* du,
               cuDoubleComplex*       B,
               int                    ldb,
               void*                  pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSgtsv2_nopivot_bufferSizeExt(UPTKsparseHandle_t handle,
                                     int              m,
                                     int              n,
                                     const float*     dl,
                                     const float*     d,
                                     const float*     du,
                                     const float*     B,
                                     int              ldb,
                                     size_t*          bufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgtsv2_nopivot_bufferSizeExt(UPTKsparseHandle_t handle,
                                     int              m,
                                     int              n,
                                     const double*    dl,
                                     const double*    d,
                                     const double*    du,
                                     const double*    B,
                                     int              ldb,
                                     size_t*          bufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgtsv2_nopivot_bufferSizeExt(UPTKsparseHandle_t handle,
                                     int              m,
                                     int              n,
                                     const cuComplex* dl,
                                     const cuComplex* d,
                                     const cuComplex* du,
                                     const cuComplex* B,
                                     int              ldb,
                                     size_t*          bufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgtsv2_nopivot_bufferSizeExt(UPTKsparseHandle_t       handle,
                                     int                    m,
                                     int                    n,
                                     const cuDoubleComplex* dl,
                                     const cuDoubleComplex* d,
                                     const cuDoubleComplex* du,
                                     const cuDoubleComplex* B,
                                     int                    ldb,
                                     size_t*                bufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSgtsv2_nopivot(UPTKsparseHandle_t handle,
                       int              m,
                       int              n,
                       const float*     dl,
                       const float*     d,
                       const float*     du,
                       float*           B,
                       int              ldb,
                       void*            pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgtsv2_nopivot(UPTKsparseHandle_t handle,
                       int              m,
                       int              n,
                       const double*    dl,
                       const double*    d,
                       const double*    du,
                       double*          B,
                       int              ldb,
                       void*            pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgtsv2_nopivot(UPTKsparseHandle_t handle,
                       int              m,
                       int              n,
                       const cuComplex* dl,
                       const cuComplex* d,
                       const cuComplex* du,
                       cuComplex*       B,
                       int              ldb,
                       void*            pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgtsv2_nopivot(UPTKsparseHandle_t       handle,
                       int                    m,
                       int                    n,
                       const cuDoubleComplex* dl,
                       const cuDoubleComplex* d,
                       const cuDoubleComplex* du,
                       cuDoubleComplex*       B,
                       int                    ldb,
                       void*                  pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSgtsv2StridedBatch_bufferSizeExt(UPTKsparseHandle_t handle,
                                         int              m,
                                         const float*     dl,
                                         const float*     d,
                                         const float*     du,
                                         const float*     x,
                                         int              batchCount,
                                         int              batchStride,
                                         size_t*          bufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgtsv2StridedBatch_bufferSizeExt(UPTKsparseHandle_t handle,
                                         int              m,
                                         const double*    dl,
                                         const double*    d,
                                         const double*    du,
                                         const double*    x,
                                         int              batchCount,
                                         int              batchStride,
                                         size_t*          bufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgtsv2StridedBatch_bufferSizeExt(UPTKsparseHandle_t handle,
                                         int              m,
                                         const cuComplex* dl,
                                         const cuComplex* d,
                                         const cuComplex* du,
                                         const cuComplex* x,
                                         int              batchCount,
                                         int              batchStride,
                                         size_t*          bufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgtsv2StridedBatch_bufferSizeExt(UPTKsparseHandle_t       handle,
                                         int                    m,
                                         const cuDoubleComplex* dl,
                                         const cuDoubleComplex* d,
                                         const cuDoubleComplex* du,
                                         const cuDoubleComplex* x,
                                         int                    batchCount,
                                         int                    batchStride,
                                         size_t* bufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSgtsv2StridedBatch(UPTKsparseHandle_t handle,
                           int              m,
                           const float*     dl,
                           const float*     d,
                           const float*     du,
                           float*           x,
                           int              batchCount,
                           int              batchStride,
                           void*            pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgtsv2StridedBatch(UPTKsparseHandle_t handle,
                           int              m,
                           const double*    dl,
                           const double*    d,
                           const double*    du,
                           double*          x,
                           int              batchCount,
                           int              batchStride,
                           void*            pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgtsv2StridedBatch(UPTKsparseHandle_t handle,
                           int              m,
                           const cuComplex* dl,
                           const cuComplex* d,
                           const cuComplex* du,
                           cuComplex*       x,
                           int              batchCount,
                           int              batchStride,
                           void*            pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgtsv2StridedBatch(UPTKsparseHandle_t       handle,
                           int                    m,
                           const cuDoubleComplex* dl,
                           const cuDoubleComplex* d,
                           const cuDoubleComplex* du,
                           cuDoubleComplex*       x,
                           int                    batchCount,
                           int                    batchStride,
                           void*                  pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSgtsvInterleavedBatch_bufferSizeExt(UPTKsparseHandle_t handle,
                                            int              algo,
                                            int              m,
                                            const float*     dl,
                                            const float*     d,
                                            const float*     du,
                                            const float*     x,
                                            int              batchCount,
                                            size_t*         pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgtsvInterleavedBatch_bufferSizeExt(UPTKsparseHandle_t handle,
                                         int              algo,
                                         int              m,
                                         const double*    dl,
                                         const double*    d,
                                         const double*    du,
                                         const double*    x,
                                         int              batchCount,
                                         size_t*          pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgtsvInterleavedBatch_bufferSizeExt(UPTKsparseHandle_t handle,
                                            int              algo,
                                            int              m,
                                            const cuComplex* dl,
                                            const cuComplex* d,
                                            const cuComplex* du,
                                            const cuComplex* x,
                                            int              batchCount,
                                            size_t*         pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgtsvInterleavedBatch_bufferSizeExt(UPTKsparseHandle_t       handle,
                                            int                    algo,
                                            int                    m,
                                            const cuDoubleComplex* dl,
                                            const cuDoubleComplex* d,
                                            const cuDoubleComplex* du,
                                            const cuDoubleComplex* x,
                                            int                    batchCount,
                                            size_t*        pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSgtsvInterleavedBatch(UPTKsparseHandle_t handle,
                              int              algo,
                              int              m,
                              float*           dl,
                              float*           d,
                              float*           du,
                              float*           x,
                              int              batchCount,
                              void*            pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgtsvInterleavedBatch(UPTKsparseHandle_t handle,
                              int              algo,
                              int              m,
                              double*          dl,
                              double*          d,
                              double*          du,
                              double*          x,
                              int              batchCount,
                              void*            pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgtsvInterleavedBatch(UPTKsparseHandle_t handle,
                              int              algo,
                              int              m,
                              cuComplex*       dl,
                              cuComplex*       d,
                              cuComplex*       du,
                              cuComplex*       x,
                              int              batchCount,
                              void*            pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgtsvInterleavedBatch(UPTKsparseHandle_t handle,
                              int              algo,
                              int              m,
                              cuDoubleComplex* dl,
                              cuDoubleComplex* d,
                              cuDoubleComplex* du,
                              cuDoubleComplex* x,
                              int              batchCount,
                              void*            pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSgpsvInterleavedBatch_bufferSizeExt(UPTKsparseHandle_t handle,
                                            int              algo,
                                            int              m,
                                            const float*     ds,
                                            const float*     dl,
                                            const float*     d,
                                            const float*     du,
                                            const float*     dw,
                                            const float*     x,
                                            int              batchCount,
                                            size_t*         pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgpsvInterleavedBatch_bufferSizeExt(UPTKsparseHandle_t handle,
                                            int              algo,
                                            int              m,
                                            const double*    ds,
                                            const double*    dl,
                                            const double*    d,
                                            const double*    du,
                                            const double*    dw,
                                            const double*    x,
                                            int              batchCount,
                                            size_t*         pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgpsvInterleavedBatch_bufferSizeExt(UPTKsparseHandle_t handle,
                                            int              algo,
                                            int              m,
                                            const cuComplex* ds,
                                            const cuComplex* dl,
                                            const cuComplex* d,
                                            const cuComplex* du,
                                            const cuComplex* dw,
                                            const cuComplex* x,
                                            int              batchCount,
                                            size_t*         pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgpsvInterleavedBatch_bufferSizeExt(UPTKsparseHandle_t       handle,
                                            int                    algo,
                                            int                    m,
                                            const cuDoubleComplex* ds,
                                            const cuDoubleComplex* dl,
                                            const cuDoubleComplex* d,
                                            const cuDoubleComplex* du,
                                            const cuDoubleComplex* dw,
                                            const cuDoubleComplex* x,
                                            int                    batchCount,
                                            size_t*         pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSgpsvInterleavedBatch(UPTKsparseHandle_t handle,
                              int              algo,
                              int              m,
                              float*           ds,
                              float*           dl,
                              float*           d,
                              float*           du,
                              float*           dw,
                              float*           x,
                              int              batchCount,
                              void*            pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgpsvInterleavedBatch(UPTKsparseHandle_t handle,
                              int              algo,
                              int              m,
                              double*          ds,
                              double*          dl,
                              double*          d,
                              double*          du,
                              double*          dw,
                              double*          x,
                              int              batchCount,
                              void*            pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgpsvInterleavedBatch(UPTKsparseHandle_t handle,
                              int              algo,
                              int              m,
                              cuComplex*       ds,
                              cuComplex*       dl,
                              cuComplex*       d,
                              cuComplex*       du,
                              cuComplex*       dw,
                              cuComplex*       x,
                              int              batchCount,
                              void*            pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgpsvInterleavedBatch(UPTKsparseHandle_t handle,
                              int              algo,
                              int              m,
                              cuDoubleComplex* ds,
                              cuDoubleComplex* dl,
                              cuDoubleComplex* d,
                              cuDoubleComplex* du,
                              cuDoubleComplex* dw,
                              cuDoubleComplex* x,
                              int              batchCount,
                              void*            pBuffer);

//##############################################################################
//# EXTRA ROUTINES
//##############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsrgeam2_bufferSizeExt(UPTKsparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                const float*             alpha,
                                const UPTKsparseMatDescr_t descrA,
                                int                      nnzA,
                                const float*             csrSortedValA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const float*             beta,
                                const UPTKsparseMatDescr_t descrB,
                                int                      nnzB,
                                const float*             csrSortedValB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const UPTKsparseMatDescr_t descrC,
                                const float*             csrSortedValC,
                                const int*               csrSortedRowPtrC,
                                const int*               csrSortedColIndC,
                                size_t*                  pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsrgeam2_bufferSizeExt(UPTKsparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                const double*            alpha,
                                const UPTKsparseMatDescr_t descrA,
                                int                      nnzA,
                                const double*            csrSortedValA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const double*            beta,
                                const UPTKsparseMatDescr_t descrB,
                                int                      nnzB,
                                const double*            csrSortedValB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const UPTKsparseMatDescr_t descrC,
                                const double*            csrSortedValC,
                                const int*               csrSortedRowPtrC,
                                const int*               csrSortedColIndC,
                                size_t*                  pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsrgeam2_bufferSizeExt(UPTKsparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                const cuComplex*         alpha,
                                const UPTKsparseMatDescr_t descrA,
                                int                      nnzA,
                                const cuComplex*         csrSortedValA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const cuComplex*         beta,
                                const UPTKsparseMatDescr_t descrB,
                                int                      nnzB,
                                const cuComplex*         csrSortedValB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const UPTKsparseMatDescr_t descrC,
                                const cuComplex*         csrSortedValC,
                                const int*               csrSortedRowPtrC,
                                const int*               csrSortedColIndC,
                                size_t*                  pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsrgeam2_bufferSizeExt(UPTKsparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                const cuDoubleComplex*   alpha,
                                const UPTKsparseMatDescr_t descrA,
                                int                      nnzA,
                                const cuDoubleComplex*   csrSortedValA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const cuDoubleComplex*   beta,
                                const UPTKsparseMatDescr_t descrB,
                                int                      nnzB,
                                const cuDoubleComplex*   csrSortedValB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const UPTKsparseMatDescr_t descrC,
                                const cuDoubleComplex*   csrSortedValC,
                                const int*               csrSortedRowPtrC,
                                const int*               csrSortedColIndC,
                                size_t*                  pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXcsrgeam2Nnz(UPTKsparseHandle_t         handle,
                     int                      m,
                     int                      n,
                     const UPTKsparseMatDescr_t descrA,
                     int                      nnzA,
                     const int*               csrSortedRowPtrA,
                     const int*               csrSortedColIndA,
                     const UPTKsparseMatDescr_t descrB,
                     int                      nnzB,
                     const int*               csrSortedRowPtrB,
                     const int*               csrSortedColIndB,
                     const UPTKsparseMatDescr_t descrC,
                     int*                     csrSortedRowPtrC,
                     int*                     nnzTotalDevHostPtr,
                     void*                    workspace);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsrgeam2(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  const float*             alpha,
                  const UPTKsparseMatDescr_t descrA,
                  int                      nnzA,
                  const float*             csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const float*             beta,
                  const UPTKsparseMatDescr_t descrB,
                  int                      nnzB,
                  const float*             csrSortedValB,
                  const int*               csrSortedRowPtrB,
                  const int*               csrSortedColIndB,
                  const UPTKsparseMatDescr_t descrC,
                  float*                   csrSortedValC,
                  int*                     csrSortedRowPtrC,
                  int*                     csrSortedColIndC,
                  void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsrgeam2(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  const double*            alpha,
                  const UPTKsparseMatDescr_t descrA,
                  int                      nnzA,
                  const double*            csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const double*            beta,
                  const UPTKsparseMatDescr_t descrB,
                  int                      nnzB,
                  const double*            csrSortedValB,
                  const int*               csrSortedRowPtrB,
                  const int*               csrSortedColIndB,
                  const UPTKsparseMatDescr_t descrC,
                  double*                  csrSortedValC,
                  int*                     csrSortedRowPtrC,
                  int*                     csrSortedColIndC,
                  void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsrgeam2(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  const cuComplex*         alpha,
                  const UPTKsparseMatDescr_t descrA,
                  int                      nnzA,
                  const cuComplex*         csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const cuComplex*         beta,
                  const UPTKsparseMatDescr_t descrB,
                  int                      nnzB,
                  const cuComplex*         csrSortedValB,
                  const int*               csrSortedRowPtrB,
                  const int*               csrSortedColIndB,
                  const UPTKsparseMatDescr_t descrC,
                  cuComplex*               csrSortedValC,
                  int*                     csrSortedRowPtrC,
                  int*                     csrSortedColIndC,
                  void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsrgeam2(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  const cuDoubleComplex*   alpha,
                  const UPTKsparseMatDescr_t descrA,
                  int                      nnzA,
                  const cuDoubleComplex*   csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const cuDoubleComplex*   beta,
                  const UPTKsparseMatDescr_t descrB,
                  int                      nnzB,
                  const cuDoubleComplex*   csrSortedValB,
                  const int*               csrSortedRowPtrB,
                  const int*               csrSortedColIndB,
                  const UPTKsparseMatDescr_t descrC,
                  cuDoubleComplex*         csrSortedValC,
                  int*                     csrSortedRowPtrC,
                  int*                     csrSortedColIndC,
                  void*                    pBuffer);

//##############################################################################
//# SPARSE MATRIX REORDERING
//##############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsrcolor(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      nnz,
                  const UPTKsparseMatDescr_t descrA,
                  const float*              csrSortedValA,
                  const int*                csrSortedRowPtrA,
                  const int*                csrSortedColIndA,
                  const float*              fractionToColor,
                  int*                      ncolors,
                  int*                      coloring,
                  int*                      reordering,
                  const UPTKsparseColorInfo_t info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsrcolor(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      nnz,
                  const UPTKsparseMatDescr_t descrA,
                  const double*            csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const double*            fractionToColor,
                  int*                     ncolors,
                  int*                     coloring,
                  int*                     reordering,
                  const UPTKsparseColorInfo_t info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsrcolor(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      nnz,
                  const UPTKsparseMatDescr_t descrA,
                  const cuComplex*          csrSortedValA,
                  const int*                csrSortedRowPtrA,
                  const int*                csrSortedColIndA,
                  const float*              fractionToColor,
                  int*                      ncolors,
                  int*                      coloring,
                  int*                      reordering,
                  const UPTKsparseColorInfo_t info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsrcolor(UPTKsparseHandle_t          handle,
                  int                       m,
                  int                       nnz,
                  const UPTKsparseMatDescr_t  descrA,
                  const cuDoubleComplex*    csrSortedValA,
                  const int*                csrSortedRowPtrA,
                  const int*                csrSortedColIndA,
                  const double*             fractionToColor,
                  int*                      ncolors,
                  int*                      coloring,
                  int*                      reordering,
                  const UPTKsparseColorInfo_t info);

//##############################################################################
//# SPARSE FORMAT CONVERSION
//##############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSnnz(UPTKsparseHandle_t         handle,
             UPTKsparseDirection_t      dirA,
             int                      m,
             int                      n,
             const UPTKsparseMatDescr_t descrA,
             const float*             A,
             int                      lda,
             int*                     nnzPerRowCol,
             int*                     nnzTotalDevHostPtr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDnnz(UPTKsparseHandle_t         handle,
             UPTKsparseDirection_t      dirA,
             int                      m,
             int                      n,
             const UPTKsparseMatDescr_t descrA,
             const double*            A,
             int                      lda,
             int*                     nnzPerRowCol,
             int*                     nnzTotalDevHostPtr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCnnz(UPTKsparseHandle_t         handle,
             UPTKsparseDirection_t      dirA,
             int                      m,
             int                      n,
             const UPTKsparseMatDescr_t descrA,
             const cuComplex*         A,
             int                      lda,
             int*                     nnzPerRowCol,
             int*                     nnzTotalDevHostPtr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZnnz(UPTKsparseHandle_t         handle,
             UPTKsparseDirection_t      dirA,
             int                      m,
             int                      n,
             const UPTKsparseMatDescr_t descrA,
             const cuDoubleComplex*   A,
             int                      lda,
             int*                     nnzPerRowCol,
             int*                     nnzTotalDevHostPtr);

//##############################################################################
//# SPARSE FORMAT CONVERSION
//##############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSnnz_compress(UPTKsparseHandle_t         handle,
                      int                      m,
                      const UPTKsparseMatDescr_t descr,
                      const float*             csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      int*                     nnzPerRow,
                      int*                     nnzC,
                      float                    tol);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDnnz_compress(UPTKsparseHandle_t         handle,
                      int                      m,
                      const UPTKsparseMatDescr_t descr,
                      const double*            csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      int*                     nnzPerRow,
                      int*                     nnzC,
                      double                   tol);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCnnz_compress(UPTKsparseHandle_t         handle,
                      int                      m,
                      const UPTKsparseMatDescr_t descr,
                      const cuComplex*         csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      int*                     nnzPerRow,
                      int*                     nnzC,
                      cuComplex                tol);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZnnz_compress(UPTKsparseHandle_t         handle,
                      int                      m,
                      const UPTKsparseMatDescr_t descr,
                      const cuDoubleComplex*   csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      int*                     nnzPerRow,
                      int*                     nnzC,
                      cuDoubleComplex          tol);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsr2csr_compress(UPTKsparseHandle_t         handle,
                          int                      m,
                          int                      n,
                          const UPTKsparseMatDescr_t descrA,
                          const float*             csrSortedValA,
                          const int*               csrSortedColIndA,
                          const int*               csrSortedRowPtrA,
                          int                      nnzA,
                          const int*               nnzPerRow,
                          float*                   csrSortedValC,
                          int*                     csrSortedColIndC,
                          int*                     csrSortedRowPtrC,
                          float                    tol);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsr2csr_compress(UPTKsparseHandle_t         handle,
                          int                      m,
                          int                      n,
                          const UPTKsparseMatDescr_t descrA,
                          const double*            csrSortedValA,
                          const int*               csrSortedColIndA,
                          const int*               csrSortedRowPtrA,
                          int                      nnzA,
                          const int*               nnzPerRow,
                          double*                  csrSortedValC,
                          int*                     csrSortedColIndC,
                          int*                     csrSortedRowPtrC,
                          double                   tol);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsr2csr_compress(UPTKsparseHandle_t         handle,
                          int                      m,
                          int                      n,
                          const UPTKsparseMatDescr_t descrA,
                          const cuComplex*         csrSortedValA,
                          const int*               csrSortedColIndA,
                          const int*               csrSortedRowPtrA,
                          int                      nnzA,
                          const int*               nnzPerRow,
                          cuComplex*               csrSortedValC,
                          int*                     csrSortedColIndC,
                          int*                     csrSortedRowPtrC,
                          cuComplex                tol);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsr2csr_compress(UPTKsparseHandle_t         handle,
                          int                      m,
                          int                      n,
                          const UPTKsparseMatDescr_t descrA,
                          const cuDoubleComplex*   csrSortedValA,
                          const int*               csrSortedColIndA,
                          const int*               csrSortedRowPtrA,
                          int                      nnzA,
                          const int*               nnzPerRow,
                          cuDoubleComplex*         csrSortedValC,
                          int*                     csrSortedColIndC,
                          int*                     csrSortedRowPtrC,
                          cuDoubleComplex          tol);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXcoo2csr(UPTKsparseHandle_t    handle,
                 const int*          cooRowInd,
                 int                 nnz,
                 int                 m,
                 int*                csrSortedRowPtr,
                 UPTKsparseIndexBase_t idxBase);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXcsr2coo(UPTKsparseHandle_t    handle,
                 const int*          csrSortedRowPtr,
                 int                 nnz,
                 int                 m,
                 int*                cooRowInd,
                 UPTKsparseIndexBase_t idxBase);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXcsr2bsrNnz(UPTKsparseHandle_t         handle,
                    UPTKsparseDirection_t      dirA,
                    int                      m,
                    int                      n,
                    const UPTKsparseMatDescr_t descrA,
                    const int*               csrSortedRowPtrA,
                    const int*               csrSortedColIndA,
                    int                      blockDim,
                    const UPTKsparseMatDescr_t descrC,
                    int*                     bsrSortedRowPtrC,
                    int*                     nnzTotalDevHostPtr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsr2bsr(UPTKsparseHandle_t         handle,
                 UPTKsparseDirection_t      dirA,
                 int                      m,
                 int                      n,
                 const UPTKsparseMatDescr_t descrA,
                 const float*             csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 int                      blockDim,
                 const UPTKsparseMatDescr_t descrC,
                 float*                   bsrSortedValC,
                 int*                     bsrSortedRowPtrC,
                 int*                     bsrSortedColIndC);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsr2bsr(UPTKsparseHandle_t         handle,
                 UPTKsparseDirection_t      dirA,
                 int                      m,
                 int                      n,
                 const UPTKsparseMatDescr_t descrA,
                 const double*            csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 int                      blockDim,
                 const UPTKsparseMatDescr_t descrC,
                 double*                  bsrSortedValC,
                 int*                     bsrSortedRowPtrC,
                 int*                     bsrSortedColIndC);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsr2bsr(UPTKsparseHandle_t         handle,
                 UPTKsparseDirection_t      dirA,
                 int                      m,
                 int                      n,
                 const UPTKsparseMatDescr_t descrA,
                 const cuComplex*         csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 int                      blockDim,
                 const UPTKsparseMatDescr_t descrC,
                 cuComplex*               bsrSortedValC,
                 int*                     bsrSortedRowPtrC,
                 int*                     bsrSortedColIndC);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsr2bsr(UPTKsparseHandle_t         handle,
                 UPTKsparseDirection_t      dirA,
                 int                      m,
                 int                      n,
                 const UPTKsparseMatDescr_t descrA,
                 const cuDoubleComplex*   csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 int                      blockDim,
                 const UPTKsparseMatDescr_t descrC,
                 cuDoubleComplex*         bsrSortedValC,
                 int*                     bsrSortedRowPtrC,
                 int*                     bsrSortedColIndC);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsr2csr(UPTKsparseHandle_t         handle,
                 UPTKsparseDirection_t      dirA,
                 int                      mb,
                 int                      nb,
                 const UPTKsparseMatDescr_t descrA,
                 const float*             bsrSortedValA,
                 const int*               bsrSortedRowPtrA,
                 const int*               bsrSortedColIndA,
                 int                      blockDim,
                 const UPTKsparseMatDescr_t descrC,
                 float*                   csrSortedValC,
                 int*                     csrSortedRowPtrC,
                 int*                     csrSortedColIndC);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsr2csr(UPTKsparseHandle_t         handle,
                 UPTKsparseDirection_t      dirA,
                 int                      mb,
                 int                      nb,
                 const UPTKsparseMatDescr_t descrA,
                 const double*            bsrSortedValA,
                 const int*               bsrSortedRowPtrA,
                 const int*               bsrSortedColIndA,
                 int                      blockDim,
                 const UPTKsparseMatDescr_t descrC,
                 double*                  csrSortedValC,
                 int*                     csrSortedRowPtrC,
                 int*                     csrSortedColIndC);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsr2csr(UPTKsparseHandle_t         handle,
                 UPTKsparseDirection_t      dirA,
                 int                      mb,
                 int                      nb,
                 const UPTKsparseMatDescr_t descrA,
                 const cuComplex*         bsrSortedValA,
                 const int*               bsrSortedRowPtrA,
                 const int*               bsrSortedColIndA,
                 int                      blockDim,
                 const UPTKsparseMatDescr_t descrC,
                 cuComplex*               csrSortedValC,
                 int*                     csrSortedRowPtrC,
                 int*                     csrSortedColIndC);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsr2csr(UPTKsparseHandle_t         handle,
                 UPTKsparseDirection_t      dirA,
                 int                      mb,
                 int                      nb,
                 const UPTKsparseMatDescr_t descrA,
                 const cuDoubleComplex*   bsrSortedValA,
                 const int*               bsrSortedRowPtrA,
                 const int*               bsrSortedColIndA,
                 int                      blockDim,
                 const UPTKsparseMatDescr_t descrC,
                 cuDoubleComplex*         csrSortedValC,
                 int*                     csrSortedRowPtrC,
                 int*                     csrSortedColIndC);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSgebsr2gebsc_bufferSize(UPTKsparseHandle_t handle,
                                int              mb,
                                int              nb,
                                int              nnzb,
                                const float*     bsrSortedVal,
                                const int*       bsrSortedRowPtr,
                                const int*       bsrSortedColInd,
                                int              rowBlockDim,
                                int              colBlockDim,
                                int*             pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgebsr2gebsc_bufferSize(UPTKsparseHandle_t handle,
                                int              mb,
                                int              nb,
                                int              nnzb,
                                const double*    bsrSortedVal,
                                const int*       bsrSortedRowPtr,
                                const int*       bsrSortedColInd,
                                int              rowBlockDim,
                                int              colBlockDim,
                                int*             pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgebsr2gebsc_bufferSize(UPTKsparseHandle_t handle,
                                int              mb,
                                int              nb,
                                int              nnzb,
                                const cuComplex* bsrSortedVal,
                                const int*       bsrSortedRowPtr,
                                const int*       bsrSortedColInd,
                                int              rowBlockDim,
                                int              colBlockDim,
                                int*             pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgebsr2gebsc_bufferSize(UPTKsparseHandle_t       handle,
                                int                    mb,
                                int                    nb,
                                int                    nnzb,
                                const cuDoubleComplex* bsrSortedVal,
                                const int*             bsrSortedRowPtr,
                                const int*             bsrSortedColInd,
                                int                    rowBlockDim,
                                int                    colBlockDim,
                                int*                   pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSgebsr2gebsc(UPTKsparseHandle_t handle,
                     int              mb,
                     int              nb,
                     int              nnzb,
                     const float*     bsrSortedVal,
                     const int* bsrSortedRowPtr,
                     const int* bsrSortedColInd,
                     int        rowBlockDim,
                     int        colBlockDim,
                     float*     bscVal,
                     int*       bscRowInd,
                     int*       bscColPtr,
                     UPTKsparseAction_t copyValues,
                     UPTKsparseIndexBase_t idxBase,
                     void*               pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgebsr2gebsc(UPTKsparseHandle_t    handle,
                     int                 mb,
                     int                 nb,
                     int                 nnzb,
                     const double*       bsrSortedVal,
                     const int*          bsrSortedRowPtr,
                     const int*          bsrSortedColInd,
                     int                 rowBlockDim,
                     int                 colBlockDim,
                     double*             bscVal,
                     int*                bscRowInd,
                     int*                bscColPtr,
                     UPTKsparseAction_t    copyValues,
                     UPTKsparseIndexBase_t idxBase,
                     void*               pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgebsr2gebsc(UPTKsparseHandle_t    handle,
                     int                 mb,
                     int                 nb,
                     int                 nnzb,
                     const cuComplex*    bsrSortedVal,
                     const int*          bsrSortedRowPtr,
                     const int*          bsrSortedColInd,
                     int                 rowBlockDim,
                     int                 colBlockDim,
                     cuComplex*          bscVal,
                     int*                bscRowInd,
                     int*                bscColPtr,
                     UPTKsparseAction_t    copyValues,
                     UPTKsparseIndexBase_t idxBase,
                     void*               pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgebsr2gebsc(UPTKsparseHandle_t       handle,
                     int                    mb,
                     int                    nb,
                     int                    nnzb,
                     const cuDoubleComplex* bsrSortedVal,
                     const int*             bsrSortedRowPtr,
                     const int*             bsrSortedColInd,
                     int                    rowBlockDim,
                     int                    colBlockDim,
                     cuDoubleComplex*       bscVal,
                     int*                   bscRowInd,
                     int*                   bscColPtr,
                     UPTKsparseAction_t       copyValues,
                     UPTKsparseIndexBase_t    idxBase,
                     void*                  pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSgebsr2csr(UPTKsparseHandle_t         handle,
                   UPTKsparseDirection_t      dirA,
                   int                      mb,
                   int                      nb,
                   const UPTKsparseMatDescr_t descrA,
                   const float*             bsrSortedValA,
                   const int*               bsrSortedRowPtrA,
                   const int*               bsrSortedColIndA,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   const UPTKsparseMatDescr_t descrC,
                   float*                   csrSortedValC,
                   int*                     csrSortedRowPtrC,
                   int*                     csrSortedColIndC);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgebsr2csr(UPTKsparseHandle_t         handle,
                   UPTKsparseDirection_t      dirA,
                   int                      mb,
                   int                      nb,
                   const UPTKsparseMatDescr_t descrA,
                   const double*            bsrSortedValA,
                   const int*               bsrSortedRowPtrA,
                   const int*               bsrSortedColIndA,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   const UPTKsparseMatDescr_t descrC,
                   double*                  csrSortedValC,
                   int*                     csrSortedRowPtrC,
                   int*                     csrSortedColIndC);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgebsr2csr(UPTKsparseHandle_t         handle,
                   UPTKsparseDirection_t      dirA,
                   int                      mb,
                   int                      nb,
                   const UPTKsparseMatDescr_t descrA,
                   const cuComplex*         bsrSortedValA,
                   const int*               bsrSortedRowPtrA,
                   const int*               bsrSortedColIndA,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   const UPTKsparseMatDescr_t descrC,
                   cuComplex*               csrSortedValC,
                   int*                     csrSortedRowPtrC,
                   int*                     csrSortedColIndC);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgebsr2csr(UPTKsparseHandle_t         handle,
                   UPTKsparseDirection_t      dirA,
                   int                      mb,
                   int                      nb,
                   const UPTKsparseMatDescr_t descrA,
                   const cuDoubleComplex*   bsrSortedValA,
                   const int*               bsrSortedRowPtrA,
                   const int*               bsrSortedColIndA,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   const UPTKsparseMatDescr_t descrC,
                   cuDoubleComplex*         csrSortedValC,
                   int*                     csrSortedRowPtrC,
                   int*                     csrSortedColIndC);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsr2gebsr_bufferSize(UPTKsparseHandle_t         handle,
                              UPTKsparseDirection_t      dirA,
                              int                      m,
                              int                      n,
                              const UPTKsparseMatDescr_t descrA,
                              const float*             csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              int                      rowBlockDim,
                              int                      colBlockDim,
                              int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsr2gebsr_bufferSize(UPTKsparseHandle_t         handle,
                              UPTKsparseDirection_t      dirA,
                              int                      m,
                              int                      n,
                              const UPTKsparseMatDescr_t descrA,
                              const double*            csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              int                      rowBlockDim,
                              int                      colBlockDim,
                              int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsr2gebsr_bufferSize(UPTKsparseHandle_t         handle,
                              UPTKsparseDirection_t      dirA,
                              int                      m,
                              int                      n,
                              const UPTKsparseMatDescr_t descrA,
                              const cuComplex*         csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              int                      rowBlockDim,
                              int                      colBlockDim,
                              int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsr2gebsr_bufferSize(UPTKsparseHandle_t         handle,
                              UPTKsparseDirection_t      dirA,
                              int                      m,
                              int                      n,
                              const UPTKsparseMatDescr_t descrA,
                              const cuDoubleComplex*   csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              int                      rowBlockDim,
                              int                      colBlockDim,
                              int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXcsr2gebsrNnz(UPTKsparseHandle_t         handle,
                      UPTKsparseDirection_t      dirA,
                      int                      m,
                      int                      n,
                      const UPTKsparseMatDescr_t descrA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      const UPTKsparseMatDescr_t descrC,
                      int*                     bsrSortedRowPtrC,
                      int                      rowBlockDim,
                      int                      colBlockDim,
                      int*                     nnzTotalDevHostPtr,
                      void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsr2gebsr(UPTKsparseHandle_t         handle,
                   UPTKsparseDirection_t      dirA,
                   int                      m,
                   int                      n,
                   const UPTKsparseMatDescr_t descrA,
                   const float*             csrSortedValA,
                   const int*               csrSortedRowPtrA,
                   const int*               csrSortedColIndA,
                   const UPTKsparseMatDescr_t descrC,
                   float*                   bsrSortedValC,
                   int*                     bsrSortedRowPtrC,
                   int*                     bsrSortedColIndC,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsr2gebsr(UPTKsparseHandle_t         handle,
                   UPTKsparseDirection_t      dirA,
                   int                      m,
                   int                      n,
                   const UPTKsparseMatDescr_t descrA,
                   const double*            csrSortedValA,
                   const int*               csrSortedRowPtrA,
                   const int*               csrSortedColIndA,
                   const UPTKsparseMatDescr_t descrC,
                   double*                  bsrSortedValC,
                   int*                     bsrSortedRowPtrC,
                   int*                     bsrSortedColIndC,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsr2gebsr(UPTKsparseHandle_t         handle,
                   UPTKsparseDirection_t      dirA,
                   int                      m,
                   int                      n,
                   const UPTKsparseMatDescr_t descrA,
                   const cuComplex*         csrSortedValA,
                   const int*               csrSortedRowPtrA,
                   const int*               csrSortedColIndA,
                   const UPTKsparseMatDescr_t descrC,
                   cuComplex*               bsrSortedValC,
                   int*                     bsrSortedRowPtrC,
                   int*                     bsrSortedColIndC,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsr2gebsr(UPTKsparseHandle_t         handle,
                   UPTKsparseDirection_t      dirA,
                   int                      m,
                   int                      n,
                   const UPTKsparseMatDescr_t descrA,
                   const cuDoubleComplex*   csrSortedValA,
                   const int*               csrSortedRowPtrA,
                   const int*               csrSortedColIndA,
                   const UPTKsparseMatDescr_t descrC,
                   cuDoubleComplex*         bsrSortedValC,
                   int*                     bsrSortedRowPtrC,
                   int*                     bsrSortedColIndC,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSgebsr2gebsr_bufferSize(UPTKsparseHandle_t         handle,
                                UPTKsparseDirection_t      dirA,
                                int                      mb,
                                int                      nb,
                                int                      nnzb,
                                const UPTKsparseMatDescr_t descrA,
                                const float*             bsrSortedValA,
                                const int*               bsrSortedRowPtrA,
                                const int*               bsrSortedColIndA,
                                int                      rowBlockDimA,
                                int                      colBlockDimA,
                                int                      rowBlockDimC,
                                int                      colBlockDimC,
                                int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgebsr2gebsr_bufferSize(UPTKsparseHandle_t         handle,
                                UPTKsparseDirection_t      dirA,
                                int                      mb,
                                int                      nb,
                                int                      nnzb,
                                const UPTKsparseMatDescr_t descrA,
                                const double*            bsrSortedValA,
                                const int*               bsrSortedRowPtrA,
                                const int*               bsrSortedColIndA,
                                int                      rowBlockDimA,
                                int                      colBlockDimA,
                                int                      rowBlockDimC,
                                int                      colBlockDimC,
                                int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgebsr2gebsr_bufferSize(UPTKsparseHandle_t         handle,
                                UPTKsparseDirection_t      dirA,
                                int                      mb,
                                int                      nb,
                                int                      nnzb,
                                const UPTKsparseMatDescr_t descrA,
                                const cuComplex*         bsrSortedValA,
                                const int*               bsrSortedRowPtrA,
                                const int*               bsrSortedColIndA,
                                int                      rowBlockDimA,
                                int                      colBlockDimA,
                                int                      rowBlockDimC,
                                int                      colBlockDimC,
                                int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgebsr2gebsr_bufferSize(UPTKsparseHandle_t         handle,
                                UPTKsparseDirection_t      dirA,
                                int                      mb,
                                int                      nb,
                                int                      nnzb,
                                const UPTKsparseMatDescr_t descrA,
                                const cuDoubleComplex*   bsrSortedValA,
                                const int*               bsrSortedRowPtrA,
                                const int*               bsrSortedColIndA,
                                int                      rowBlockDimA,
                                int                      colBlockDimA,
                                int                      rowBlockDimC,
                                int                      colBlockDimC,
                                int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXgebsr2gebsrNnz(UPTKsparseHandle_t         handle,
                        UPTKsparseDirection_t      dirA,
                        int                      mb,
                        int                      nb,
                        int                      nnzb,
                        const UPTKsparseMatDescr_t descrA,
                        const int*               bsrSortedRowPtrA,
                        const int*               bsrSortedColIndA,
                        int                      rowBlockDimA,
                        int                      colBlockDimA,
                        const UPTKsparseMatDescr_t descrC,
                        int*                     bsrSortedRowPtrC,
                        int                      rowBlockDimC,
                        int                      colBlockDimC,
                        int*                     nnzTotalDevHostPtr,
                        void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSgebsr2gebsr(UPTKsparseHandle_t         handle,
                     UPTKsparseDirection_t      dirA,
                     int                      mb,
                     int                      nb,
                     int                      nnzb,
                     const UPTKsparseMatDescr_t descrA,
                     const float*             bsrSortedValA,
                     const int*               bsrSortedRowPtrA,
                     const int*               bsrSortedColIndA,
                     int                      rowBlockDimA,
                     int                      colBlockDimA,
                     const UPTKsparseMatDescr_t descrC,
                     float*                   bsrSortedValC,
                     int*                     bsrSortedRowPtrC,
                     int*                     bsrSortedColIndC,
                     int                      rowBlockDimC,
                     int                      colBlockDimC,
                     void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgebsr2gebsr(UPTKsparseHandle_t         handle,
                     UPTKsparseDirection_t      dirA,
                     int                      mb,
                     int                      nb,
                     int                      nnzb,
                     const UPTKsparseMatDescr_t descrA,
                     const double*            bsrSortedValA,
                     const int*               bsrSortedRowPtrA,
                     const int*               bsrSortedColIndA,
                     int                      rowBlockDimA,
                     int                      colBlockDimA,
                     const UPTKsparseMatDescr_t descrC,
                     double*                  bsrSortedValC,
                     int*                     bsrSortedRowPtrC,
                     int*                     bsrSortedColIndC,
                     int                      rowBlockDimC,
                     int                      colBlockDimC,
                     void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgebsr2gebsr(UPTKsparseHandle_t         handle,
                     UPTKsparseDirection_t      dirA,
                     int                      mb,
                     int                      nb,
                     int                      nnzb,
                     const UPTKsparseMatDescr_t descrA,
                     const cuComplex*         bsrSortedValA,
                     const int*               bsrSortedRowPtrA,
                     const int*               bsrSortedColIndA,
                     int                      rowBlockDimA,
                     int                      colBlockDimA,
                     const UPTKsparseMatDescr_t descrC,
                     cuComplex*               bsrSortedValC,
                     int*                     bsrSortedRowPtrC,
                     int*                     bsrSortedColIndC,
                     int                      rowBlockDimC,
                     int                      colBlockDimC,
                     void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgebsr2gebsr(UPTKsparseHandle_t         handle,
                     UPTKsparseDirection_t      dirA,
                     int                      mb,
                     int                      nb,
                     int                      nnzb,
                     const UPTKsparseMatDescr_t descrA,
                     const cuDoubleComplex*   bsrSortedValA,
                     const int*               bsrSortedRowPtrA,
                     const int*               bsrSortedColIndA,
                     int                      rowBlockDimA,
                     int                      colBlockDimA,
                     const UPTKsparseMatDescr_t descrC,
                     cuDoubleComplex*         bsrSortedValC,
                     int*                     bsrSortedRowPtrC,
                     int*                     bsrSortedColIndC,
                     int                      rowBlockDimC,
                     int                      colBlockDimC,
                     void*                    pBuffer);

//##############################################################################
//# SPARSE MATRIX SORTING
//##############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateIdentityPermutation(UPTKsparseHandle_t handle,
                                  int              n,
                                  int*             p);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXcoosort_bufferSizeExt(UPTKsparseHandle_t handle,
                               int              m,
                               int              n,
                               int              nnz,
                               const int*       cooRowsA,
                               const int*       cooColsA,
                               size_t*          pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXcoosortByRow(UPTKsparseHandle_t handle,
                      int              m,
                      int              n,
                      int              nnz,
                      int*             cooRowsA,
                      int*             cooColsA,
                      int*             P,
                      void*            pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXcoosortByColumn(UPTKsparseHandle_t handle,
                         int              m,
                         int              n,
                         int              nnz,
                         int*             cooRowsA,
                         int*             cooColsA,
                         int*             P,
                         void*            pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXcsrsort_bufferSizeExt(UPTKsparseHandle_t handle,
                               int              m,
                               int              n,
                               int              nnz,
                               const int*       csrRowPtrA,
                               const int*       csrColIndA,
                               size_t*          pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXcsrsort(UPTKsparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 int                      nnz,
                 const UPTKsparseMatDescr_t descrA,
                 const int*               csrRowPtrA,
                 int*                     csrColIndA,
                 int*                     P,
                 void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXcscsort_bufferSizeExt(UPTKsparseHandle_t handle,
                               int              m,
                               int              n,
                               int              nnz,
                               const int*       cscColPtrA,
                               const int*       cscRowIndA,
                               size_t*          pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXcscsort(UPTKsparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 int                      nnz,
                 const UPTKsparseMatDescr_t descrA,
                 const int*               cscColPtrA,
                 int*                     cscRowIndA,
                 int*                     P,
                 void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsru2csr_bufferSizeExt(UPTKsparseHandle_t handle,
                                int              m,
                                int              n,
                                int              nnz,
                                float*           csrVal,
                                const int*       csrRowPtr,
                                int*             csrColInd,
                                UPTKcsru2csrInfo_t   info,
                                size_t*          pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsru2csr_bufferSizeExt(UPTKsparseHandle_t handle,
                                int              m,
                                int              n,
                                int              nnz,
                                double*          csrVal,
                                const int*       csrRowPtr,
                                int*             csrColInd,
                                UPTKcsru2csrInfo_t   info,
                                size_t*          pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsru2csr_bufferSizeExt(UPTKsparseHandle_t handle,
                                int              m,
                                int              n,
                                int              nnz,
                                cuComplex*       csrVal,
                                const int*       csrRowPtr,
                                int*             csrColInd,
                                UPTKcsru2csrInfo_t   info,
                                size_t*          pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsru2csr_bufferSizeExt(UPTKsparseHandle_t handle,
                                int              m,
                                int              n,
                                int              nnz,
                                cuDoubleComplex* csrVal,
                                const int*       csrRowPtr,
                                int*             csrColInd,
                                UPTKcsru2csrInfo_t   info,
                                size_t*          pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsru2csr(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const UPTKsparseMatDescr_t descrA,
                  float*                   csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  UPTKcsru2csrInfo_t           info,
                  void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsru2csr(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const UPTKsparseMatDescr_t descrA,
                  double*                  csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  UPTKcsru2csrInfo_t           info,
                  void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsru2csr(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const UPTKsparseMatDescr_t descrA,
                  cuComplex*               csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  UPTKcsru2csrInfo_t           info,
                  void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsru2csr(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const UPTKsparseMatDescr_t descrA,
                  cuDoubleComplex*         csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  UPTKcsru2csrInfo_t           info,
                  void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsr2csru(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const UPTKsparseMatDescr_t descrA,
                  float*                   csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  UPTKcsru2csrInfo_t           info,
                  void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsr2csru(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const UPTKsparseMatDescr_t descrA,
                  double*                  csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  UPTKcsru2csrInfo_t           info,
                  void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsr2csru(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const UPTKsparseMatDescr_t descrA,
                  cuComplex*               csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  UPTKcsru2csrInfo_t           info,
                  void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsr2csru(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      nnz,
                  const UPTKsparseMatDescr_t descrA,
                  cuDoubleComplex*         csrVal,
                  const int*               csrRowPtr,
                  int*                     csrColInd,
                  UPTKcsru2csrInfo_t           info,
                  void*                    pBuffer);


UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpruneDense2csr_bufferSizeExt(UPTKsparseHandle_t         handle,
                                      int                      m,
                                      int                      n,
                                      const float*             A,
                                      int                      lda,
                                      const float*             threshold,
                                      const UPTKsparseMatDescr_t descrC,
                                      const float*             csrSortedValC,
                                      const int*               csrSortedRowPtrC,
                                      const int*               csrSortedColIndC,
                                      size_t* pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDpruneDense2csr_bufferSizeExt(UPTKsparseHandle_t         handle,
                                      int                      m,
                                      int                      n,
                                      const double*            A,
                                      int                      lda,
                                      const double*            threshold,
                                      const UPTKsparseMatDescr_t descrC,
                                      const double*            csrSortedValC,
                                      const int*               csrSortedRowPtrC,
                                      const int*               csrSortedColIndC,
                                      size_t*               pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpruneDense2csrNnz(UPTKsparseHandle_t         handle,
                           int                      m,
                           int                      n,
                           const float*             A,
                           int                      lda,
                           const float*             threshold,
                           const UPTKsparseMatDescr_t descrC,
                           int*                     csrRowPtrC,
                           int*                     nnzTotalDevHostPtr,
                           void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDpruneDense2csrNnz(UPTKsparseHandle_t         handle,
                           int                      m,
                           int                      n,
                           const double*            A,
                           int                      lda,
                           const double*            threshold,
                           const UPTKsparseMatDescr_t descrC,
                           int*                     csrSortedRowPtrC,
                           int*                     nnzTotalDevHostPtr,
                           void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpruneDense2csr(UPTKsparseHandle_t         handle,
                        int                      m,
                        int                      n,
                        const float*             A,
                        int                      lda,
                        const float*             threshold,
                        const UPTKsparseMatDescr_t descrC,
                        float*                   csrSortedValC,
                        const int*               csrSortedRowPtrC,
                        int*                     csrSortedColIndC,
                        void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDpruneDense2csr(UPTKsparseHandle_t         handle,
                        int                      m,
                        int                      n,
                        const double*            A,
                        int                      lda,
                        const double*            threshold,
                        const UPTKsparseMatDescr_t descrC,
                        double*                  csrSortedValC,
                        const int*               csrSortedRowPtrC,
                        int*                     csrSortedColIndC,
                        void*                    pBuffer);


UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpruneCsr2csr_bufferSizeExt(UPTKsparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    int                      nnzA,
                                    const UPTKsparseMatDescr_t descrA,
                                    const float*             csrSortedValA,
                                    const int*               csrSortedRowPtrA,
                                    const int*               csrSortedColIndA,
                                    const float*             threshold,
                                    const UPTKsparseMatDescr_t descrC,
                                    const float*             csrSortedValC,
                                    const int*               csrSortedRowPtrC,
                                    const int*               csrSortedColIndC,
                                    size_t*                 pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDpruneCsr2csr_bufferSizeExt(UPTKsparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    int                      nnzA,
                                    const UPTKsparseMatDescr_t descrA,
                                    const double*            csrSortedValA,
                                    const int*               csrSortedRowPtrA,
                                    const int*               csrSortedColIndA,
                                    const double*            threshold,
                                    const UPTKsparseMatDescr_t descrC,
                                    const double*            csrSortedValC,
                                    const int*               csrSortedRowPtrC,
                                    const int*               csrSortedColIndC,
                                    size_t*                 pBufferSizeInBytes);


UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpruneCsr2csrNnz(UPTKsparseHandle_t         handle,
                         int                      m,
                         int                      n,
                         int                      nnzA,
                         const UPTKsparseMatDescr_t descrA,
                         const float*             csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         const float*             threshold,
                         const UPTKsparseMatDescr_t descrC,
                         int*                     csrSortedRowPtrC,
                         int*                     nnzTotalDevHostPtr,
                         void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDpruneCsr2csrNnz(UPTKsparseHandle_t         handle,
                          int                      m,
                          int                      n,
                          int                      nnzA,
                          const UPTKsparseMatDescr_t descrA,
                          const double*            csrSortedValA,
                          const int*               csrSortedRowPtrA,
                          const int*               csrSortedColIndA,
                          const double*            threshold,
                          const UPTKsparseMatDescr_t descrC,
                          int*                     csrSortedRowPtrC,
                          int*                     nnzTotalDevHostPtr,
                          void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpruneCsr2csr(UPTKsparseHandle_t         handle,
                      int                      m,
                      int                      n,
                      int                      nnzA,
                      const UPTKsparseMatDescr_t descrA,
                      const float*             csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      const float*             threshold,
                      const UPTKsparseMatDescr_t descrC,
                      float*                   csrSortedValC,
                      const int*               csrSortedRowPtrC,
                      int*                     csrSortedColIndC,
                      void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDpruneCsr2csr(UPTKsparseHandle_t         handle,
                      int                      m,
                      int                      n,
                      int                      nnzA,
                      const UPTKsparseMatDescr_t descrA,
                      const double*            csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      const double*            threshold,
                      const UPTKsparseMatDescr_t descrC,
                      double*                  csrSortedValC,
                      const int*               csrSortedRowPtrC,
                      int*                     csrSortedColIndC,
                      void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpruneDense2csrByPercentage_bufferSizeExt(
                                   UPTKsparseHandle_t         handle,
                                   int                      m,
                                   int                      n,
                                   const float*             A,
                                   int                      lda,
                                   float                    percentage,
                                   const UPTKsparseMatDescr_t descrC,
                                   const float*             csrSortedValC,
                                   const int*               csrSortedRowPtrC,
                                   const int*               csrSortedColIndC,
                                   UPTKpruneInfo_t              info,
                                   size_t*                  pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDpruneDense2csrByPercentage_bufferSizeExt(
                                   UPTKsparseHandle_t         handle,
                                   int                      m,
                                   int                      n,
                                   const double*            A,
                                   int                      lda,
                                   float                    percentage,
                                   const UPTKsparseMatDescr_t descrC,
                                   const double*            csrSortedValC,
                                   const int*               csrSortedRowPtrC,
                                   const int*               csrSortedColIndC,
                                   UPTKpruneInfo_t              info,
                                   size_t*                  pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpruneDense2csrNnzByPercentage(
                                    UPTKsparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const float*             A,
                                    int                      lda,
                                    float                    percentage,
                                    const UPTKsparseMatDescr_t descrC,
                                    int*                     csrRowPtrC,
                                    int*                     nnzTotalDevHostPtr,
                                    UPTKpruneInfo_t              info,
                                    void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDpruneDense2csrNnzByPercentage(
                                    UPTKsparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const double*            A,
                                    int                      lda,
                                    float                    percentage,
                                    const UPTKsparseMatDescr_t descrC,
                                    int*                     csrRowPtrC,
                                    int*                     nnzTotalDevHostPtr,
                                    UPTKpruneInfo_t              info,
                                    void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpruneDense2csrByPercentage(UPTKsparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const float*             A,
                                    int                      lda,
                                    float                    percentage,
                                    const UPTKsparseMatDescr_t descrC,
                                    float*                   csrSortedValC,
                                    const int*               csrSortedRowPtrC,
                                    int*                     csrSortedColIndC,
                                    UPTKpruneInfo_t              info,
                                    void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDpruneDense2csrByPercentage(UPTKsparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const double*            A,
                                    int                      lda,
                                    float                    percentage,
                                    const UPTKsparseMatDescr_t descrC,
                                    double*                  csrSortedValC,
                                    const int*               csrSortedRowPtrC,
                                    int*                     csrSortedColIndC,
                                    UPTKpruneInfo_t              info,
                                    void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpruneCsr2csrByPercentage_bufferSizeExt(
                                   UPTKsparseHandle_t         handle,
                                   int                      m,
                                   int                      n,
                                   int                      nnzA,
                                   const UPTKsparseMatDescr_t descrA,
                                   const float*             csrSortedValA,
                                   const int*               csrSortedRowPtrA,
                                   const int*               csrSortedColIndA,
                                   float                    percentage,
                                   const UPTKsparseMatDescr_t descrC,
                                   const float*             csrSortedValC,
                                   const int*               csrSortedRowPtrC,
                                   const int*               csrSortedColIndC,
                                   UPTKpruneInfo_t              info,
                                   size_t*                  pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDpruneCsr2csrByPercentage_bufferSizeExt(
                                   UPTKsparseHandle_t         handle,
                                   int                      m,
                                   int                      n,
                                   int                      nnzA,
                                   const UPTKsparseMatDescr_t descrA,
                                   const double*            csrSortedValA,
                                   const int*               csrSortedRowPtrA,
                                   const int*               csrSortedColIndA,
                                   float                    percentage,
                                   const UPTKsparseMatDescr_t descrC,
                                   const double*            csrSortedValC,
                                   const int*               csrSortedRowPtrC,
                                   const int*               csrSortedColIndC,
                                   UPTKpruneInfo_t              info,
                                   size_t*                  pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpruneCsr2csrNnzByPercentage(
                                    UPTKsparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    int                      nnzA,
                                    const UPTKsparseMatDescr_t descrA,
                                    const float*             csrSortedValA,
                                    const int*               csrSortedRowPtrA,
                                    const int*               csrSortedColIndA,
                                    float                    percentage,
                                    const UPTKsparseMatDescr_t descrC,
                                    int*                     csrSortedRowPtrC,
                                    int*                     nnzTotalDevHostPtr,
                                    UPTKpruneInfo_t              info,
                                    void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDpruneCsr2csrNnzByPercentage(
                                    UPTKsparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    int                      nnzA,
                                    const UPTKsparseMatDescr_t descrA,
                                    const double*            csrSortedValA,
                                    const int*               csrSortedRowPtrA,
                                    const int*               csrSortedColIndA,
                                    float                    percentage,
                                    const UPTKsparseMatDescr_t descrC,
                                    int*                     csrSortedRowPtrC,
                                    int*                     nnzTotalDevHostPtr,
                                    UPTKpruneInfo_t              info,
                                    void*                    pBuffer);


UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpruneCsr2csrByPercentage(UPTKsparseHandle_t         handle,
                                  int                      m,
                                  int                      n,
                                  int                      nnzA,
                                  const UPTKsparseMatDescr_t descrA,
                                  const float*             csrSortedValA,
                                  const int*               csrSortedRowPtrA,
                                  const int*               csrSortedColIndA,
                                  float                    percentage,
                                  const UPTKsparseMatDescr_t descrC,
                                  float*                   csrSortedValC,
                                  const int*               csrSortedRowPtrC,
                                  int*                     csrSortedColIndC,
                                  UPTKpruneInfo_t              info,
                                  void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDpruneCsr2csrByPercentage(UPTKsparseHandle_t         handle,
                                  int                      m,
                                  int                      n,
                                  int                      nnzA,
                                  const UPTKsparseMatDescr_t descrA,
                                  const double*            csrSortedValA,
                                  const int*               csrSortedRowPtrA,
                                  const int*               csrSortedColIndA,
                                  float                    percentage,
                                  const UPTKsparseMatDescr_t descrC,
                                  double*                  csrSortedValC,
                                  const int*               csrSortedRowPtrC,
                                  int*                     csrSortedColIndC,
                                  UPTKpruneInfo_t              info,
                                  void*                    pBuffer);

//##############################################################################
//# CSR2CSC
//##############################################################################

typedef enum {
    UPTKSPARSE_CSR2CSC_ALG_DEFAULT = 1,
    UPTKSPARSE_CSR2CSC_ALG1 = 1
} UPTKsparseCsr2CscAlg_t;

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCsr2cscEx2(UPTKsparseHandle_t     handle,
                   int                  m,
                   int                  n,
                   int                  nnz,
                   const void*          csrVal,
                   const int*           csrRowPtr,
                   const int*           csrColInd,
                   void*                cscVal,
                   int*                 cscColPtr,
                   int*                 cscRowInd,
                   UPTKDataType         valType,
                   UPTKsparseAction_t     copyValues,
                   UPTKsparseIndexBase_t  idxBase,
                   UPTKsparseCsr2CscAlg_t alg,
                   void*                buffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCsr2cscEx2_bufferSize(UPTKsparseHandle_t     handle,
                              int                  m,
                              int                  n,
                              int                  nnz,
                              const void*          csrVal,
                              const int*           csrRowPtr,
                              const int*           csrColInd,
                              void*                cscVal,
                              int*                 cscColPtr,
                              int*                 cscRowInd,
                              UPTKDataType         valType,
                              UPTKsparseAction_t     copyValues,
                              UPTKsparseIndexBase_t  idxBase,
                              UPTKsparseCsr2CscAlg_t alg,
                              size_t*              bufferSize);

// #############################################################################
// # GENERIC APIs - Enumerators and Opaque Data Structures
// #############################################################################

typedef enum {
    UPTKSPARSE_FORMAT_CSR            = 1, ///< Compressed Sparse Row (CSR)
    UPTKSPARSE_FORMAT_CSC            = 2, ///< Compressed Sparse Column (CSC)
    UPTKSPARSE_FORMAT_COO            = 3, ///< Coordinate (COO) - Structure of Arrays
    UPTKSPARSE_FORMAT_BLOCKED_ELL    = 5, ///< Blocked ELL
    UPTKSPARSE_FORMAT_BSR            = 6, ///< Blocked Compressed Sparse Row (BSR)
    UPTKSPARSE_FORMAT_SLICED_ELLPACK = 7 ///< Sliced ELL
} UPTKsparseFormat_t;

typedef enum {
    UPTKSPARSE_ORDER_COL = 1, ///< Column-Major Order - Matrix memory layout
    UPTKSPARSE_ORDER_ROW = 2  ///< Row-Major Order - Matrix memory layout
} UPTKsparseOrder_t;

typedef enum {
    UPTKSPARSE_INDEX_16U = 1, ///< 16-bit unsigned integer for matrix/vector
                            ///< indices
    UPTKSPARSE_INDEX_32I = 2, ///< 32-bit signed integer for matrix/vector indices
    UPTKSPARSE_INDEX_64I = 3  ///< 64-bit signed integer for matrix/vector indices
} UPTKsparseIndexType_t;

//------------------------------------------------------------------------------

struct UPTKsparseSpVecDescr;
struct UPTKsparseDnVecDescr;
struct UPTKsparseSpMatDescr;
struct UPTKsparseDnMatDescr;

typedef struct UPTKsparseSpVecDescr* UPTKsparseSpVecDescr_t;
typedef struct UPTKsparseDnVecDescr* UPTKsparseDnVecDescr_t;
typedef struct UPTKsparseSpMatDescr* UPTKsparseSpMatDescr_t;
typedef struct UPTKsparseDnMatDescr* UPTKsparseDnMatDescr_t;

typedef struct UPTKsparseSpVecDescr const* UPTKsparseConstSpVecDescr_t;
typedef struct UPTKsparseDnVecDescr const* UPTKsparseConstDnVecDescr_t;
typedef struct UPTKsparseSpMatDescr const* UPTKsparseConstSpMatDescr_t;
typedef struct UPTKsparseDnMatDescr const* UPTKsparseConstDnMatDescr_t;

// #############################################################################
// # SPARSE VECTOR DESCRIPTOR
// #############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateSpVec(UPTKsparseSpVecDescr_t* spVecDescr,
                    int64_t               size,
                    int64_t               nnz,
                    void*                 indices,
                    void*                 values,
                    UPTKsparseIndexType_t   idxType,
                    UPTKsparseIndexBase_t   idxBase,
                    UPTKDataType          valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateConstSpVec(UPTKsparseConstSpVecDescr_t* spVecDescr,
                         int64_t                    size,
                         int64_t                    nnz,
                         const void*                indices,
                         const void*                values,
                         UPTKsparseIndexType_t        idxType,
                         UPTKsparseIndexBase_t        idxBase,
                         UPTKDataType               valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroySpVec(UPTKsparseConstSpVecDescr_t spVecDescr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpVecGet(UPTKsparseSpVecDescr_t spVecDescr,
                 int64_t*             size,
                 int64_t*             nnz,
                 void**               indices,
                 void**               values,
                 UPTKsparseIndexType_t* idxType,
                 UPTKsparseIndexBase_t* idxBase,
                 UPTKDataType*        valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstSpVecGet(UPTKsparseConstSpVecDescr_t spVecDescr,
                      int64_t*             size,
                      int64_t*             nnz,
                      const void**         indices,
                      const void**         values,
                      UPTKsparseIndexType_t* idxType,
                      UPTKsparseIndexBase_t* idxBase,
                      UPTKDataType*        valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpVecGetIndexBase(UPTKsparseConstSpVecDescr_t spVecDescr,
                          UPTKsparseIndexBase_t*      idxBase);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpVecGetValues(UPTKsparseSpVecDescr_t spVecDescr,
                       void**               values);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstSpVecGetValues(UPTKsparseConstSpVecDescr_t spVecDescr,
                            const void**              values);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpVecSetValues(UPTKsparseSpVecDescr_t spVecDescr,
                       void*                values);

// #############################################################################
// # DENSE VECTOR DESCRIPTOR
// #############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateDnVec(UPTKsparseDnVecDescr_t* dnVecDescr,
                    int64_t               size,
                    void*                 values,
                    UPTKDataType          valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateConstDnVec(UPTKsparseConstDnVecDescr_t* dnVecDescr,
                         int64_t                    size,
                         const void*                values,
                         UPTKDataType               valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyDnVec(UPTKsparseConstDnVecDescr_t dnVecDescr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDnVecGet(UPTKsparseDnVecDescr_t dnVecDescr,
                 int64_t*             size,
                 void**               values,
                 UPTKDataType*        valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstDnVecGet(UPTKsparseConstDnVecDescr_t dnVecDescr,
                      int64_t*                  size,
                      const void**              values,
                      UPTKDataType*             valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDnVecGetValues(UPTKsparseDnVecDescr_t dnVecDescr,
                       void**               values);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstDnVecGetValues(UPTKsparseConstDnVecDescr_t dnVecDescr,
                            const void**              values);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDnVecSetValues(UPTKsparseDnVecDescr_t dnVecDescr,
                       void*                values);

// #############################################################################
// # SPARSE MATRIX DESCRIPTOR
// #############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroySpMat(UPTKsparseConstSpMatDescr_t spMatDescr);

 UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMatGetFormat(UPTKsparseConstSpMatDescr_t spMatDescr,
                       UPTKsparseFormat_t*         format);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMatGetIndexBase(UPTKsparseConstSpMatDescr_t spMatDescr,
                          UPTKsparseIndexBase_t*      idxBase);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMatGetValues(UPTKsparseSpMatDescr_t spMatDescr,
                       void**               values);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstSpMatGetValues(UPTKsparseConstSpMatDescr_t spMatDescr,
                            const void**               values);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMatSetValues(UPTKsparseSpMatDescr_t spMatDescr,
                       void*                values);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMatGetSize(UPTKsparseConstSpMatDescr_t spMatDescr,
                     int64_t*                  rows,
                     int64_t*                  cols,
                     int64_t*                  nnz);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMatGetStridedBatch(UPTKsparseConstSpMatDescr_t spMatDescr,
                             int*                      batchCount);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCooSetStridedBatch(UPTKsparseSpMatDescr_t spMatDescr,
                           int                  batchCount,
                           int64_t              batchStride);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCsrSetStridedBatch(UPTKsparseSpMatDescr_t spMatDescr,
                           int                  batchCount,
                           int64_t              offsetsBatchStride,
                           int64_t              columnsValuesBatchStride);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseBsrSetStridedBatch(UPTKsparseSpMatDescr_t spMatDescr,
                           int                  batchCount,
                           int64_t              offsetsBatchStride,
                           int64_t              columnsBatchStride,
                           int64_t              ValuesBatchStride);

typedef enum {
    UPTKSPARSE_SPMAT_FILL_MODE,
    UPTKSPARSE_SPMAT_DIAG_TYPE
} UPTKsparseSpMatAttribute_t;

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMatGetAttribute(UPTKsparseConstSpMatDescr_t spMatDescr,
                          UPTKsparseSpMatAttribute_t  attribute,
                          void*                     data,
                          size_t                    dataSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMatSetAttribute(UPTKsparseSpMatDescr_t     spMatDescr,
                          UPTKsparseSpMatAttribute_t attribute,
                          void*                    data,
                          size_t                   dataSize);

//------------------------------------------------------------------------------
// ### CSR ###

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateCsr(UPTKsparseSpMatDescr_t* spMatDescr,
                  int64_t               rows,
                  int64_t               cols,
                  int64_t               nnz,
                  void*                 csrRowOffsets,
                  void*                 csrColInd,
                  void*                 csrValues,
                  UPTKsparseIndexType_t   csrRowOffsetsType,
                  UPTKsparseIndexType_t   csrColIndType,
                  UPTKsparseIndexBase_t   idxBase,
                  UPTKDataType          valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateConstCsr(UPTKsparseConstSpMatDescr_t* spMatDescr,
                       int64_t                    rows,
                       int64_t                    cols,
                       int64_t                    nnz,
                       const void*                csrRowOffsets,
                       const void*                csrColInd,
                       const void*                csrValues,
                       UPTKsparseIndexType_t        csrRowOffsetsType,
                       UPTKsparseIndexType_t        csrColIndType,
                       UPTKsparseIndexBase_t        idxBase,
                       UPTKDataType               valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateCsc(UPTKsparseSpMatDescr_t* spMatDescr,
                  int64_t               rows,
                  int64_t               cols,
                  int64_t               nnz,
                  void*                 cscColOffsets,
                  void*                 cscRowInd,
                  void*                 cscValues,
                  UPTKsparseIndexType_t   cscColOffsetsType,
                  UPTKsparseIndexType_t   cscRowIndType,
                  UPTKsparseIndexBase_t   idxBase,
                  UPTKDataType          valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateConstCsc(UPTKsparseConstSpMatDescr_t* spMatDescr,
                       int64_t                    rows,
                       int64_t                    cols,
                       int64_t                    nnz,
                       const void*                cscColOffsets,
                       const void*                cscRowInd,
                       const void*                cscValues,
                       UPTKsparseIndexType_t        cscColOffsetsType,
                       UPTKsparseIndexType_t        cscRowIndType,
                       UPTKsparseIndexBase_t        idxBase,
                       UPTKDataType               valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCsrGet(UPTKsparseSpMatDescr_t spMatDescr,
               int64_t*             rows,
               int64_t*             cols,
               int64_t*             nnz,
               void**               csrRowOffsets,
               void**               csrColInd,
               void**               csrValues,
               UPTKsparseIndexType_t* csrRowOffsetsType,
               UPTKsparseIndexType_t* csrColIndType,
               UPTKsparseIndexBase_t* idxBase,
               UPTKDataType*        valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstCsrGet(UPTKsparseConstSpMatDescr_t spMatDescr,
                    int64_t*                  rows,
                    int64_t*                  cols,
                    int64_t*                  nnz,
                    const void**              csrRowOffsets,
                    const void**              csrColInd,
                    const void**              csrValues,
                    UPTKsparseIndexType_t*      csrRowOffsetsType,
                    UPTKsparseIndexType_t*      csrColIndType,
                    UPTKsparseIndexBase_t*      idxBase,
                    UPTKDataType*             valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCscGet(UPTKsparseSpMatDescr_t spMatDescr,
               int64_t*             rows,
               int64_t*             cols,
               int64_t*             nnz,
               void**               cscColOffsets,
               void**               cscRowInd,
               void**               cscValues,
               UPTKsparseIndexType_t* cscColOffsetsType,
               UPTKsparseIndexType_t* cscRowIndType,
               UPTKsparseIndexBase_t* idxBase,
               UPTKDataType*        valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstCscGet(UPTKsparseConstSpMatDescr_t spMatDescr,
                    int64_t*                  rows,
                    int64_t*                  cols,
                    int64_t*                  nnz,
                    const void**              cscColOffsets,
                    const void**              cscRowInd,
                    const void**              cscValues,
                    UPTKsparseIndexType_t*      cscColOffsetsType,
                    UPTKsparseIndexType_t*      cscRowIndType,
                    UPTKsparseIndexBase_t*      idxBase,
                    UPTKDataType*             valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCsrSetPointers(UPTKsparseSpMatDescr_t spMatDescr,
                       void*                csrRowOffsets,
                       void*                csrColInd,
                       void*                csrValues);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCscSetPointers(UPTKsparseSpMatDescr_t spMatDescr,
                       void*                cscColOffsets,
                       void*                cscRowInd,
                       void*                cscValues);

//------------------------------------------------------------------------------
// ### BSR ###

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateBsr(UPTKsparseSpMatDescr_t* spMatDescr,
                  int64_t               brows,
                  int64_t               bcols,
                  int64_t               bnnz,
                  int64_t               rowBlockSize,
                  int64_t               colBlockSize,
                  void*                 bsrRowOffsets,
                  void*                 bsrColInd,
                  void*                 bsrValues,
                  UPTKsparseIndexType_t   bsrRowOffsetsType,
                  UPTKsparseIndexType_t   bsrColIndType,
                  UPTKsparseIndexBase_t   idxBase,
                  UPTKDataType          valueType,
                  UPTKsparseOrder_t       order);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateConstBsr(UPTKsparseConstSpMatDescr_t* spMatDescr,
                       int64_t                    brows,
                       int64_t                    bcols,
                       int64_t                    bnnz,
                       int64_t                    rowBlockDim,
                       int64_t                    colBlockDim,
                       const void*                bsrRowOffsets,
                       const void*                bsrColInd,
                       const void*                bsrValues,
                       UPTKsparseIndexType_t        bsrRowOffsetsType,
                       UPTKsparseIndexType_t        bsrColIndType,
                       UPTKsparseIndexBase_t        idxBase,
                       UPTKDataType               valueType,
                       UPTKsparseOrder_t            order);

//------------------------------------------------------------------------------
// ### COO ###

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateCoo(UPTKsparseSpMatDescr_t* spMatDescr,
                  int64_t               rows,
                  int64_t               cols,
                  int64_t               nnz,
                  void*                 cooRowInd,
                  void*                 cooColInd,
                  void*                 cooValues,
                  UPTKsparseIndexType_t   cooIdxType,
                  UPTKsparseIndexBase_t   idxBase,
                  UPTKDataType          valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateConstCoo(UPTKsparseConstSpMatDescr_t* spMatDescr,
                       int64_t                    rows,
                       int64_t                    cols,
                       int64_t                    nnz,
                       const void*                cooRowInd,
                       const void*                cooColInd,
                       const void*                cooValues,
                       UPTKsparseIndexType_t        cooIdxType,
                       UPTKsparseIndexBase_t        idxBase,
                       UPTKDataType               valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCooGet(UPTKsparseSpMatDescr_t spMatDescr,
               int64_t*             rows,
               int64_t*             cols,
               int64_t*             nnz,
               void**               cooRowInd,  // COO row indices
               void**               cooColInd,  // COO column indices
               void**               cooValues,  // COO values
               UPTKsparseIndexType_t* idxType,
               UPTKsparseIndexBase_t* idxBase,
               UPTKDataType*        valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstCooGet(UPTKsparseConstSpMatDescr_t spMatDescr,
                    int64_t*                  rows,
                    int64_t*                  cols,
                    int64_t*                  nnz,
                    const void**              cooRowInd,  // COO row indices
                    const void**              cooColInd,  // COO column indices
                    const void**              cooValues,  // COO values
                    UPTKsparseIndexType_t*      idxType,
                    UPTKsparseIndexBase_t*      idxBase,
                    UPTKDataType*             valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCooSetPointers(UPTKsparseSpMatDescr_t spMatDescr,
                       void*                cooRows,
                       void*                cooColumns,
                       void*                cooValues);

//------------------------------------------------------------------------------
// ### BLOCKED ELL ###

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateBlockedEll(UPTKsparseSpMatDescr_t* spMatDescr,
                         int64_t               rows,
                         int64_t               cols,
                         int64_t               ellBlockSize,
                         int64_t               ellCols,
                         void*                 ellColInd,
                         void*                 ellValue,
                         UPTKsparseIndexType_t   ellIdxType,
                         UPTKsparseIndexBase_t   idxBase,
                         UPTKDataType          valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateConstBlockedEll(UPTKsparseConstSpMatDescr_t* spMatDescr,
                              int64_t                    rows,
                              int64_t                    cols,
                              int64_t                    ellBlockSize,
                              int64_t                    ellCols,
                              const void*                ellColInd,
                              const void*                ellValue,
                              UPTKsparseIndexType_t        ellIdxType,
                              UPTKsparseIndexBase_t        idxBase,
                              UPTKDataType               valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseBlockedEllGet(UPTKsparseSpMatDescr_t spMatDescr,
                      int64_t*             rows,
                      int64_t*             cols,
                      int64_t*             ellBlockSize,
                      int64_t*             ellCols,
                      void**               ellColInd,
                      void**               ellValue,
                      UPTKsparseIndexType_t* ellIdxType,
                      UPTKsparseIndexBase_t* idxBase,
                      UPTKDataType*        valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstBlockedEllGet(UPTKsparseConstSpMatDescr_t spMatDescr,
                           int64_t*                  rows,
                           int64_t*                  cols,
                           int64_t*                  ellBlockSize,
                           int64_t*                  ellCols,
                           const void**              ellColInd,
                           const void**              ellValue,
                           UPTKsparseIndexType_t*      ellIdxType,
                           UPTKsparseIndexBase_t*      idxBase,
                           UPTKDataType*             valueType);

//------------------------------------------------------------------------------
// ### Sliced ELLPACK ###

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateSlicedEll(UPTKsparseSpMatDescr_t*   spMatDescr,
                        int64_t                 rows,
                        int64_t                 cols,
                        int64_t                 nnz,
                        int64_t                 sellValuesSize,
                        int64_t                 sliceSize,
	                void*                   sellSliceOffsets,
                        void*                   sellColInd,
                        void*                   sellValues,
			UPTKsparseIndexType_t     sellSliceOffsetsType,
                        UPTKsparseIndexType_t     sellColIndType,
                        UPTKsparseIndexBase_t     idxBase,
                        UPTKDataType            valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateConstSlicedEll(UPTKsparseConstSpMatDescr_t* spMatDescr,
                             int64_t                    rows,
                             int64_t                    cols,
                             int64_t                    nnz,
                             int64_t                    sellValuesSize,
                             int64_t                    sliceSize,
                             const void*                sellSliceOffsets,
                             const void*                sellColInd,
                             const void*                sellValues,
                             UPTKsparseIndexType_t        sellSliceOffsetsType,
                             UPTKsparseIndexType_t        sellColIndType,
                             UPTKsparseIndexBase_t        idxBase,
                             UPTKDataType               valueType);

// #############################################################################
// # DENSE MATRIX DESCRIPTOR
// #############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateDnMat(UPTKsparseDnMatDescr_t* dnMatDescr,
                    int64_t               rows,
                    int64_t               cols,
                    int64_t               ld,
                    void*                 values,
                    UPTKDataType          valueType,
                    UPTKsparseOrder_t       order);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateConstDnMat(UPTKsparseConstDnMatDescr_t* dnMatDescr,
                         int64_t                    rows,
                         int64_t                    cols,
                         int64_t                    ld,
                         const void*                values,
                         UPTKDataType               valueType,
                         UPTKsparseOrder_t            order);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyDnMat(UPTKsparseConstDnMatDescr_t dnMatDescr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDnMatGet(UPTKsparseDnMatDescr_t dnMatDescr,
                 int64_t*             rows,
                 int64_t*             cols,
                 int64_t*             ld,
                 void**               values,
                 UPTKDataType*        type,
                 UPTKsparseOrder_t*     order);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstDnMatGet(UPTKsparseConstDnMatDescr_t dnMatDescr,
                      int64_t*                  rows,
                      int64_t*                  cols,
                      int64_t*                  ld,
                      const void**              values,
                      UPTKDataType*             type,
                      UPTKsparseOrder_t*          order);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDnMatGetValues(UPTKsparseDnMatDescr_t dnMatDescr,
                       void**               values);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstDnMatGetValues(UPTKsparseConstDnMatDescr_t dnMatDescr,
                            const void**              values);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDnMatSetValues(UPTKsparseDnMatDescr_t dnMatDescr,
                       void*                values);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDnMatSetStridedBatch(UPTKsparseDnMatDescr_t dnMatDescr,
                             int                  batchCount,
                             int64_t              batchStride);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDnMatGetStridedBatch(UPTKsparseConstDnMatDescr_t dnMatDescr,
                             int*                      batchCount,
                             int64_t*                  batchStride);

// #############################################################################
// # VECTOR-VECTOR OPERATIONS
// #############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseAxpby(UPTKsparseHandle_t          handle,
              const void*               alpha,
              UPTKsparseConstSpVecDescr_t vecX,
              const void*               beta,
              UPTKsparseDnVecDescr_t      vecY);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseGather(UPTKsparseHandle_t          handle,
               UPTKsparseConstDnVecDescr_t vecY,
               UPTKsparseSpVecDescr_t      vecX);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScatter(UPTKsparseHandle_t          handle,
                UPTKsparseConstSpVecDescr_t vecX,
                UPTKsparseDnVecDescr_t      vecY);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseRot(UPTKsparseHandle_t     handle,
            const void*          c_coeff,
            const void*          s_coeff,
            UPTKsparseSpVecDescr_t vecX,
            UPTKsparseDnVecDescr_t vecY);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpVV_bufferSize(UPTKsparseHandle_t          handle,
                        UPTKsparseOperation_t       opX,
                        UPTKsparseConstSpVecDescr_t vecX,
                        UPTKsparseConstDnVecDescr_t vecY,
                        const void*               result,
                        UPTKDataType              computeType,
                        size_t*                   bufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpVV(UPTKsparseHandle_t          handle,
             UPTKsparseOperation_t       opX,
             UPTKsparseConstSpVecDescr_t vecX,
             UPTKsparseConstDnVecDescr_t vecY,
             void*                     result,
             UPTKDataType              computeType,
             void*                     externalBuffer);

// #############################################################################
// # SPARSE TO DENSE
// #############################################################################

typedef enum {
    UPTKSPARSE_SPARSETODENSE_ALG_DEFAULT = 0
} UPTKsparseSparseToDenseAlg_t;

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSparseToDense_bufferSize(UPTKsparseHandle_t           handle,
                                 UPTKsparseConstSpMatDescr_t  matA,
                                 UPTKsparseDnMatDescr_t       matB,
                                 UPTKsparseSparseToDenseAlg_t alg,
                                 size_t*                    bufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSparseToDense(UPTKsparseHandle_t           handle,
                      UPTKsparseConstSpMatDescr_t  matA,
                      UPTKsparseDnMatDescr_t       matB,
                      UPTKsparseSparseToDenseAlg_t alg,
                      void*                      externalBuffer);

// #############################################################################
// # DENSE TO SPARSE
// #############################################################################

typedef enum {
    UPTKSPARSE_DENSETOSPARSE_ALG_DEFAULT = 0
} UPTKsparseDenseToSparseAlg_t;

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDenseToSparse_bufferSize(UPTKsparseHandle_t           handle,
                                 UPTKsparseConstDnMatDescr_t  matA,
                                 UPTKsparseSpMatDescr_t       matB,
                                 UPTKsparseDenseToSparseAlg_t alg,
                                 size_t*                    bufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDenseToSparse_analysis(UPTKsparseHandle_t           handle,
                               UPTKsparseConstDnMatDescr_t  matA,
                               UPTKsparseSpMatDescr_t       matB,
                               UPTKsparseDenseToSparseAlg_t alg,
                               void*                      externalBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDenseToSparse_convert(UPTKsparseHandle_t           handle,
                              UPTKsparseConstDnMatDescr_t  matA,
                              UPTKsparseSpMatDescr_t       matB,
                              UPTKsparseDenseToSparseAlg_t alg,
                              void*                      externalBuffer);

// #############################################################################
// # SPARSE MATRIX-VECTOR MULTIPLICATION
// #############################################################################

typedef enum {
    UPTKSPARSE_SPMV_ALG_DEFAULT = 0,
    UPTKSPARSE_SPMV_CSR_ALG1    = 2,
    UPTKSPARSE_SPMV_CSR_ALG2    = 3,
    UPTKSPARSE_SPMV_COO_ALG1    = 1,
    UPTKSPARSE_SPMV_COO_ALG2    = 4,
    UPTKSPARSE_SPMV_SELL_ALG1   = 5
} UPTKsparseSpMVAlg_t;

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMV(UPTKsparseHandle_t          handle,
             UPTKsparseOperation_t       opA,
             const void*               alpha,
             UPTKsparseConstSpMatDescr_t matA,
             UPTKsparseConstDnVecDescr_t vecX,
             const void*               beta,
             UPTKsparseDnVecDescr_t      vecY,
             UPTKDataType              computeType,
             UPTKsparseSpMVAlg_t         alg,
             void*                     externalBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMV_bufferSize(UPTKsparseHandle_t          handle,
                        UPTKsparseOperation_t       opA,
                        const void*               alpha,
                        UPTKsparseConstSpMatDescr_t matA,
                        UPTKsparseConstDnVecDescr_t vecX,
                        const void*               beta,
                        UPTKsparseDnVecDescr_t      vecY,
                        UPTKDataType              computeType,
                        UPTKsparseSpMVAlg_t         alg,
                        size_t*                   bufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMV_preprocess(UPTKsparseHandle_t          handle,
                        UPTKsparseOperation_t       opA,
                        const void*               alpha,
                        UPTKsparseConstSpMatDescr_t matA,
                        UPTKsparseConstDnVecDescr_t vecX,
                        const void*               beta,
                        UPTKsparseDnVecDescr_t      vecY,
                        UPTKDataType              computeType,
                        UPTKsparseSpMVAlg_t         alg,
                        void*                     externalBuffer);
// #############################################################################
// # SPARSE TRIANGULAR VECTOR SOLVE
// #############################################################################

typedef enum {
    UPTKSPARSE_SPSV_ALG_DEFAULT = 0,
} UPTKsparseSpSVAlg_t;

typedef enum {
    UPTKSPARSE_SPSV_UPDATE_GENERAL  = 0,
    UPTKSPARSE_SPSV_UPDATE_DIAGONAL = 1
} UPTKsparseSpSVUpdate_t;

struct UPTKsparseSpSVDescr;
typedef struct UPTKsparseSpSVDescr* UPTKsparseSpSVDescr_t;

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSV_createDescr(UPTKsparseSpSVDescr_t* descr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSV_destroyDescr(UPTKsparseSpSVDescr_t descr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSV_bufferSize(UPTKsparseHandle_t          handle,
                        UPTKsparseOperation_t       opA,
                        const void*               alpha,
                        UPTKsparseConstSpMatDescr_t matA,
                        UPTKsparseConstDnVecDescr_t vecX,
                        UPTKsparseDnVecDescr_t      vecY,
                        UPTKDataType              computeType,
                        UPTKsparseSpSVAlg_t         alg,
                        UPTKsparseSpSVDescr_t       spsvDescr,
                        size_t*                   bufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSV_analysis(UPTKsparseHandle_t          handle,
                      UPTKsparseOperation_t       opA,
                      const void*               alpha,
                      UPTKsparseConstSpMatDescr_t matA,
                      UPTKsparseConstDnVecDescr_t vecX,
                      UPTKsparseDnVecDescr_t      vecY,
                      UPTKDataType              computeType,
                      UPTKsparseSpSVAlg_t         alg,
                      UPTKsparseSpSVDescr_t       spsvDescr,
                      void*                     externalBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSV_solve(UPTKsparseHandle_t          handle,
                   UPTKsparseOperation_t       opA,
                   const void*               alpha,
                   UPTKsparseConstSpMatDescr_t matA,
                   UPTKsparseConstDnVecDescr_t vecX,
                   UPTKsparseDnVecDescr_t      vecY,
                   UPTKDataType              computeType,
                   UPTKsparseSpSVAlg_t         alg,
                   UPTKsparseSpSVDescr_t       spsvDescr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSV_updateMatrix(UPTKsparseHandle_t      handle,
				          UPTKsparseSpSVDescr_t   spsvDescr,
                          void*                 newValues,
                          UPTKsparseSpSVUpdate_t  updatePart);



// #############################################################################
// # SPARSE TRIANGULAR MATRIX SOLVE
// #############################################################################

typedef enum {
    UPTKSPARSE_SPSM_ALG_DEFAULT = 0,
} UPTKsparseSpSMAlg_t;

typedef enum {
    UPTKSPARSE_SPSM_UPDATE_GENERAL  = 0,
    UPTKSPARSE_SPSM_UPDATE_DIAGONAL = 1
} UPTKsparseSpSMUpdate_t;

struct UPTKsparseSpSMDescr;
typedef struct UPTKsparseSpSMDescr* UPTKsparseSpSMDescr_t;

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSM_createDescr(UPTKsparseSpSMDescr_t* descr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSM_destroyDescr(UPTKsparseSpSMDescr_t descr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSM_bufferSize(UPTKsparseHandle_t          handle,
                        UPTKsparseOperation_t       opA,
                        UPTKsparseOperation_t       opB,
                        const void*               alpha,
                        UPTKsparseConstSpMatDescr_t matA,
                        UPTKsparseConstDnMatDescr_t matB,
                        UPTKsparseDnMatDescr_t      matC,
                        UPTKDataType              computeType,
                        UPTKsparseSpSMAlg_t         alg,
                        UPTKsparseSpSMDescr_t       spsmDescr,
                        size_t*                   bufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSM_analysis(UPTKsparseHandle_t          handle,
                      UPTKsparseOperation_t       opA,
                      UPTKsparseOperation_t       opB,
                      const void*               alpha,
                      UPTKsparseConstSpMatDescr_t matA,
                      UPTKsparseConstDnMatDescr_t matB,
                      UPTKsparseDnMatDescr_t      matC,
                      UPTKDataType              computeType,
                      UPTKsparseSpSMAlg_t         alg,
                      UPTKsparseSpSMDescr_t       spsmDescr,
                      void*                     externalBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSM_solve(UPTKsparseHandle_t          handle,
                   UPTKsparseOperation_t       opA,
                   UPTKsparseOperation_t       opB,
                   const void*               alpha,
                   UPTKsparseConstSpMatDescr_t matA,
                   UPTKsparseConstDnMatDescr_t matB,
                   UPTKsparseDnMatDescr_t      matC,
                   UPTKDataType              computeType,
                   UPTKsparseSpSMAlg_t         alg,
                   UPTKsparseSpSMDescr_t       spsmDescr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSM_updateMatrix(UPTKsparseHandle_t      handle,
				          UPTKsparseSpSMDescr_t   spsmDescr,
                          void*                 newValues,
                          UPTKsparseSpSMUpdate_t  updatePart);

// #############################################################################
// # SPARSE MATRIX-MATRIX MULTIPLICATION
// #############################################################################

typedef enum {
    UPTKSPARSE_SPMM_ALG_DEFAULT      = 0,
    UPTKSPARSE_SPMM_COO_ALG1         = 1,
    UPTKSPARSE_SPMM_COO_ALG2         = 2,
    UPTKSPARSE_SPMM_COO_ALG3         = 3,
    UPTKSPARSE_SPMM_COO_ALG4         = 5,
    UPTKSPARSE_SPMM_CSR_ALG1         = 4,
    UPTKSPARSE_SPMM_CSR_ALG2         = 6,
    UPTKSPARSE_SPMM_CSR_ALG3         = 12,
    UPTKSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13,
    UPTKSPARSE_SPMM_BSR_ALG1         = 14
} UPTKsparseSpMMAlg_t;

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMM_bufferSize(UPTKsparseHandle_t          handle,
                        UPTKsparseOperation_t       opA,
                        UPTKsparseOperation_t       opB,
                        const void*               alpha,
                        UPTKsparseConstSpMatDescr_t matA,
                        UPTKsparseConstDnMatDescr_t matB,
                        const void*               beta,
                        UPTKsparseDnMatDescr_t      matC,
                        UPTKDataType              computeType,
                        UPTKsparseSpMMAlg_t         alg,
                        size_t*                   bufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMM_preprocess(UPTKsparseHandle_t          handle,
                        UPTKsparseOperation_t       opA,
                        UPTKsparseOperation_t       opB,
                        const void*               alpha,
                        UPTKsparseConstSpMatDescr_t matA,
                        UPTKsparseConstDnMatDescr_t matB,
                        const void*               beta,
                        UPTKsparseDnMatDescr_t      matC,
                        UPTKDataType              computeType,
                        UPTKsparseSpMMAlg_t         alg,
                        void*                     externalBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMM(UPTKsparseHandle_t          handle,
             UPTKsparseOperation_t       opA,
             UPTKsparseOperation_t       opB,
             const void*               alpha,
             UPTKsparseConstSpMatDescr_t matA,
             UPTKsparseConstDnMatDescr_t matB,
             const void*               beta,
             UPTKsparseDnMatDescr_t      matC,
             UPTKDataType              computeType,
             UPTKsparseSpMMAlg_t         alg,
             void*                     externalBuffer);

// #############################################################################
// # SPARSE MATRIX - SPARSE MATRIX MULTIPLICATION (SpGEMM)
// #############################################################################

typedef enum {
    UPTKSPARSE_SPGEMM_DEFAULT                 = 0,
    UPTKSPARSE_SPGEMM_CSR_ALG_DETERMINITIC    = 1,
    UPTKSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC = 2,
    UPTKSPARSE_SPGEMM_ALG1                    = 3,
    UPTKSPARSE_SPGEMM_ALG2                    = 4,
    UPTKSPARSE_SPGEMM_ALG3                    = 5
} UPTKsparseSpGEMMAlg_t;

struct UPTKsparseSpGEMMDescr;
typedef struct UPTKsparseSpGEMMDescr* UPTKsparseSpGEMMDescr_t;

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMM_createDescr(UPTKsparseSpGEMMDescr_t* descr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMM_destroyDescr(UPTKsparseSpGEMMDescr_t descr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMM_workEstimation(UPTKsparseHandle_t          handle,
                              UPTKsparseOperation_t       opA,
                              UPTKsparseOperation_t       opB,
                              const void*               alpha,
                              UPTKsparseConstSpMatDescr_t matA,
                              UPTKsparseConstSpMatDescr_t matB,
                              const void*               beta,
                              UPTKsparseSpMatDescr_t      matC,
                              UPTKDataType              computeType,
                              UPTKsparseSpGEMMAlg_t       alg,
                              UPTKsparseSpGEMMDescr_t     spgemmDescr,
                              size_t*                   bufferSize1,
                              void*                     externalBuffer1);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMM_getNumProducts(UPTKsparseSpGEMMDescr_t spgemmDescr,
                              int64_t*              num_prods);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMM_estimateMemory(UPTKsparseHandle_t          handle,
                              UPTKsparseOperation_t       opA,
                              UPTKsparseOperation_t       opB,
                              const void*               alpha,
                              UPTKsparseConstSpMatDescr_t matA,
                              UPTKsparseConstSpMatDescr_t matB,
                              const void*               beta,
                              UPTKsparseSpMatDescr_t      matC,
                              UPTKDataType              computeType,
                              UPTKsparseSpGEMMAlg_t       alg,
                              UPTKsparseSpGEMMDescr_t     spgemmDescr,
                              float                     chunk_fraction,
                              size_t*                   bufferSize3,
                              void*                     externalBuffer3,
                              size_t*                   bufferSize2);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMM_compute(UPTKsparseHandle_t          handle,
                       UPTKsparseOperation_t       opA,
                       UPTKsparseOperation_t       opB,
                       const void*               alpha,
                       UPTKsparseConstSpMatDescr_t matA,
                       UPTKsparseConstSpMatDescr_t matB,
                       const void*               beta,
                       UPTKsparseSpMatDescr_t      matC,
                       UPTKDataType              computeType,
                       UPTKsparseSpGEMMAlg_t       alg,
                       UPTKsparseSpGEMMDescr_t     spgemmDescr,
                       size_t*                   bufferSize2,
                       void*                     externalBuffer2);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMM_copy(UPTKsparseHandle_t          handle,
                    UPTKsparseOperation_t       opA,
                    UPTKsparseOperation_t       opB,
                    const void*               alpha,
                    UPTKsparseConstSpMatDescr_t matA,
                    UPTKsparseConstSpMatDescr_t matB,
                    const void*               beta,
                    UPTKsparseSpMatDescr_t      matC,
                    UPTKDataType              computeType,
                    UPTKsparseSpGEMMAlg_t       alg,
                    UPTKsparseSpGEMMDescr_t     spgemmDescr);

// #############################################################################
// # SPARSE MATRIX - SPARSE MATRIX MULTIPLICATION (SpGEMM) STRUCTURE REUSE
// #############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMMreuse_workEstimation(UPTKsparseHandle_t          handle,
                                   UPTKsparseOperation_t       opA,
                                   UPTKsparseOperation_t       opB,
                                   UPTKsparseConstSpMatDescr_t matA,
                                   UPTKsparseConstSpMatDescr_t matB,
                                   UPTKsparseSpMatDescr_t      matC,
                                   UPTKsparseSpGEMMAlg_t       alg,
                                   UPTKsparseSpGEMMDescr_t     spgemmDescr,
                                   size_t*                   bufferSize1,
                                   void*                     externalBuffer1);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMMreuse_nnz(UPTKsparseHandle_t          handle,
                        UPTKsparseOperation_t       opA,
                        UPTKsparseOperation_t       opB,
                        UPTKsparseConstSpMatDescr_t matA,
                        UPTKsparseConstSpMatDescr_t matB,
                        UPTKsparseSpMatDescr_t      matC,
                        UPTKsparseSpGEMMAlg_t       alg,
                        UPTKsparseSpGEMMDescr_t     spgemmDescr,
                        size_t*                   bufferSize2,
                        void*                     externalBuffer2,
                        size_t*                   bufferSize3,
                        void*                     externalBuffer3,
                        size_t*                   bufferSize4,
                        void*                     externalBuffer4);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMMreuse_copy(UPTKsparseHandle_t          handle,
                         UPTKsparseOperation_t       opA,
                         UPTKsparseOperation_t       opB,
                         UPTKsparseConstSpMatDescr_t matA,
                         UPTKsparseConstSpMatDescr_t matB,
                         UPTKsparseSpMatDescr_t      matC,
                         UPTKsparseSpGEMMAlg_t       alg,
                         UPTKsparseSpGEMMDescr_t     spgemmDescr,
                         size_t*                   bufferSize5,
                         void*                     externalBuffer5);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMMreuse_compute(UPTKsparseHandle_t          handle,
                            UPTKsparseOperation_t       opA,
                            UPTKsparseOperation_t       opB,
                            const void*               alpha,
                            UPTKsparseConstSpMatDescr_t matA,
                            UPTKsparseConstSpMatDescr_t matB,
                            const void*               beta,
                            UPTKsparseSpMatDescr_t      matC,
                            UPTKDataType              computeType,
                            UPTKsparseSpGEMMAlg_t       alg,
                            UPTKsparseSpGEMMDescr_t     spgemmDescr);

// #############################################################################
// # SAMPLED DENSE-DENSE MATRIX MULTIPLICATION
// #############################################################################

typedef enum {
    UPTKSPARSE_SDDMM_ALG_DEFAULT = 0
} UPTKsparseSDDMMAlg_t;

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSDDMM_bufferSize(UPTKsparseHandle_t          handle,
                         UPTKsparseOperation_t       opA,
                         UPTKsparseOperation_t       opB,
                         const void*               alpha,
                         UPTKsparseConstDnMatDescr_t matA,
                         UPTKsparseConstDnMatDescr_t matB,
                         const void*               beta,
                         UPTKsparseSpMatDescr_t      matC,
                         UPTKDataType              computeType,
                         UPTKsparseSDDMMAlg_t        alg,
                         size_t*                   bufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSDDMM_preprocess(UPTKsparseHandle_t          handle,
                         UPTKsparseOperation_t       opA,
                         UPTKsparseOperation_t       opB,
                         const void*               alpha,
                         UPTKsparseConstDnMatDescr_t matA,
                         UPTKsparseConstDnMatDescr_t matB,
                         const void*               beta,
                         UPTKsparseSpMatDescr_t      matC,
                         UPTKDataType              computeType,
                         UPTKsparseSDDMMAlg_t        alg,
                         void*                     externalBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSDDMM(UPTKsparseHandle_t          handle,
              UPTKsparseOperation_t       opA,
              UPTKsparseOperation_t       opB,
              const void*               alpha,
              UPTKsparseConstDnMatDescr_t matA,
              UPTKsparseConstDnMatDescr_t matB,
              const void*               beta,
              UPTKsparseSpMatDescr_t      matC,
              UPTKDataType              computeType,
              UPTKsparseSDDMMAlg_t        alg,
              void*                     externalBuffer);

// #############################################################################
// # GENERIC APIs WITH CUSTOM OPERATORS (PREVIEW)
// #############################################################################

struct UPTKsparseSpMMOpPlan;
typedef struct UPTKsparseSpMMOpPlan*       UPTKsparseSpMMOpPlan_t;

typedef enum {
    UPTKSPARSE_SPMM_OP_ALG_DEFAULT
} UPTKsparseSpMMOpAlg_t;

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMMOp_createPlan(UPTKsparseHandle_t          handle,
                          UPTKsparseSpMMOpPlan_t*     plan,
                          UPTKsparseOperation_t       opA,
                          UPTKsparseOperation_t       opB,
                          UPTKsparseConstSpMatDescr_t matA,
                          UPTKsparseConstDnMatDescr_t matB,
                          UPTKsparseDnMatDescr_t      matC,
                          UPTKDataType              computeType,
                          UPTKsparseSpMMOpAlg_t       alg,
                          const void*               addOperationNvvmBuffer,
                          size_t                    addOperationBufferSize,
                          const void*               mulOperationNvvmBuffer,
                          size_t                    mulOperationBufferSize,
                          const void*               epilogueNvvmBuffer,
                          size_t                    epilogueBufferSize,
                          size_t*                   SpMMWorkspaceSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMMOp(UPTKsparseSpMMOpPlan_t plan,
               void*                externalBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMMOp_destroyPlan(UPTKsparseSpMMOpPlan_t plan);

//------------------------------------------------------------------------------

#if defined(__cplusplus)
} // extern "C"
#endif // defined(__cplusplus)

#endif // !defined(UPTKSPARSE_H_)

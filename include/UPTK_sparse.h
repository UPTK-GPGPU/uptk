#if !defined(UPTKSPARSE_H_)
#define UPTKSPARSE_H_

#include <cuComplex.h>
#include <cuda_fp16.h>
#include <driver_types.h>
#include <library_types.h>
#include <UPTK_library_types.h>
#include <stdint.h>
#include <stdio.h>

#define UPTKSPARSE_VER_MAJOR 11
#define UPTKSPARSE_VER_MINOR 7
#define UPTKSPARSE_VER_PATCH 5
#define UPTKSPARSE_VER_BUILD 86
#define UPTKSPARSE_VERSION (UPTKSPARSE_VER_MAJOR * 1000 + \
                          UPTKSPARSE_VER_MINOR *  100 + \
                          UPTKSPARSE_VER_PATCH)

#if !defined(UPTKSPARSEAPI)
#    if defined(_WIN32)
#        define UPTKSPARSEAPI __stdcall
#    else
#        define UPTKSPARSEAPI
#    endif
#endif

#if !defined(_MSC_VER)
#   define UPTKSPARSE_CPP_VERSION __cplusplus
#elif _MSC_FULL_VER >= 190024210
#   define UPTKSPARSE_CPP_VERSION _MSVC_LANG
#else
#   define UPTKSPARSE_CPP_VERSION 0
#endif

#if !defined(DISABLE_UPTKSPARSE_DEPRECATED)

#   if UPTKSPARSE_CPP_VERSION >= 201402L

#       define UPTKSPARSE_DEPRECATED(new_func)                                   \
            [[deprecated("please use " #new_func " instead")]]

#   elif defined(_MSC_VER)

#       define UPTKSPARSE_DEPRECATED(new_func)                                   \
            __declspec(deprecated("please use " #new_func " instead"))

#   elif defined(__INTEL_COMPILER) || defined(__clang__) ||                    \
         (defined(__GNUC__) &&                                                 \
          (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 5)))

#       define UPTKSPARSE_DEPRECATED(new_func)                                   \
            __attribute__((deprecated("please use " #new_func " instead")))

#   elif defined(__GNUC__) || defined(__xlc__)

#       define UPTKSPARSE_DEPRECATED(new_func)                                   \
            __attribute__((deprecated))

#   else

#       define UPTKSPARSE_DEPRECATED(new_func)

#   endif
#   if UPTKSPARSE_CPP_VERSION >= 201703L

#       define UPTKSPARSE_DEPRECATED_ENUM(new_enum)                              \
            [[deprecated("please use " #new_enum " instead")]]

#   elif defined(__clang__) ||                                                 \
         (defined(__GNUC__) && __GNUC__ >= 6 && !defined(__PGI))

#       define UPTKSPARSE_DEPRECATED_ENUM(new_enum)                              \
            __attribute__((deprecated("please use " #new_enum " instead")))

#   else

#       define UPTKSPARSE_DEPRECATED_ENUM(new_enum)

#   endif

#else

#   define UPTKSPARSE_DEPRECATED(new_func)
#   define UPTKSPARSE_DEPRECATED_ENUM(new_enum)

#endif

#undef UPTKSPARSE_CPP_VERSION
#if defined(__cplusplus)
extern "C" {
#endif
struct UPTKsparseContext;
typedef struct UPTKsparseContext* UPTKsparseHandle_t;

struct UPTKsparseMatDescr;
typedef struct UPTKsparseMatDescr* UPTKsparseMatDescr_t;

struct csrsv2Info;
typedef struct csrsv2Info* csrsv2Info_t;

struct csrsm2Info;
typedef struct csrsm2Info* csrsm2Info_t;

struct bsrsv2Info;
typedef struct bsrsv2Info* bsrsv2Info_t;

struct bsrsm2Info;
typedef struct bsrsm2Info* bsrsm2Info_t;

struct csric02Info;
typedef struct csric02Info* csric02Info_t;

struct bsric02Info;
typedef struct bsric02Info* bsric02Info_t;

struct csrilu02Info;
typedef struct csrilu02Info* csrilu02Info_t;

struct bsrilu02Info;
typedef struct bsrilu02Info* bsrilu02Info_t;

struct csrgemm2Info;
typedef struct csrgemm2Info* csrgemm2Info_t;

struct csru2csrInfo;
typedef struct csru2csrInfo* csru2csrInfo_t;

struct UPTKsparseColorInfo;
typedef struct UPTKsparseColorInfo* UPTKsparseColorInfo_t;

struct pruneInfo;
typedef struct pruneInfo* pruneInfo_t;
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
    UPTKSPARSE_COLOR_ALG0 = 0,
    UPTKSPARSE_COLOR_ALG1 = 1
} UPTKsparseColorAlg_t;

typedef enum {
    UPTKSPARSE_ALG_MERGE_PATH
} UPTKsparseAlgMode_t;
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

typedef void (*UPTKsparseLoggerCallback_t)(int         logLevel,
                                         const char* functionName,
                                         const char* message);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseLoggerSetCallback(UPTKsparseLoggerCallback_t callback);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseLoggerSetFile(FILE* file);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseLoggerOpenFile(const char* logFile);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseLoggerSetLevel(int level);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseLoggerSetMask(int mask);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseLoggerForceDisable(void);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateMatDescr(UPTKsparseMatDescr_t* descrA);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyMatDescr(UPTKsparseMatDescr_t descrA);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCopyMatDescr(UPTKsparseMatDescr_t       dest,
                     const UPTKsparseMatDescr_t src);

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

UPTKSPARSE_DEPRECATED(UPTKsparseSpSV)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateCsrsv2Info(csrsv2Info_t* info);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSV)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyCsrsv2Info(csrsv2Info_t info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateCsric02Info(csric02Info_t* info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyCsric02Info(csric02Info_t info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateBsric02Info(bsric02Info_t* info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyBsric02Info(bsric02Info_t info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateCsrilu02Info(csrilu02Info_t* info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyCsrilu02Info(csrilu02Info_t info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateBsrilu02Info(bsrilu02Info_t* info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyBsrilu02Info(bsrilu02Info_t info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateBsrsv2Info(bsrsv2Info_t* info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyBsrsv2Info(bsrsv2Info_t info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateBsrsm2Info(bsrsm2Info_t* info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyBsrsm2Info(bsrsm2Info_t info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateCsru2csrInfo(csru2csrInfo_t* info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyCsru2csrInfo(csru2csrInfo_t info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateColorInfo(UPTKsparseColorInfo_t* info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyColorInfo(UPTKsparseColorInfo_t info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSetColorAlgs(UPTKsparseColorInfo_t info,
                     UPTKsparseColorAlg_t  alg);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseGetColorAlgs(UPTKsparseColorInfo_t info,
                     UPTKsparseColorAlg_t* alg);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreatePruneInfo(pruneInfo_t* info);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyPruneInfo(pruneInfo_t info);

UPTKSPARSE_DEPRECATED(UPTKsparseAxpby)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSaxpyi(UPTKsparseHandle_t    handle,
               int                 nnz,
               const float*        alpha,
               const float*        xVal,
               const int*          xInd,
               float*              y,
               UPTKsparseIndexBase_t idxBase);

UPTKSPARSE_DEPRECATED(UPTKsparseAxpby)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDaxpyi(UPTKsparseHandle_t    handle,
               int                 nnz,
               const double*       alpha,
               const double*       xVal,
               const int*          xInd,
               double*             y,
               UPTKsparseIndexBase_t idxBase);

UPTKSPARSE_DEPRECATED(UPTKsparseAxpby)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCaxpyi(UPTKsparseHandle_t    handle,
               int                 nnz,
               const cuComplex*    alpha,
               const cuComplex*    xVal,
               const int*          xInd,
               cuComplex*          y,
               UPTKsparseIndexBase_t idxBase);

UPTKSPARSE_DEPRECATED(UPTKsparseAxpby)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZaxpyi(UPTKsparseHandle_t       handle,
               int                    nnz,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* xVal,
               const int*             xInd,
               cuDoubleComplex*       y,
               UPTKsparseIndexBase_t    idxBase);

UPTKSPARSE_DEPRECATED(UPTKsparseGather)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSgthr(UPTKsparseHandle_t    handle,
              int                 nnz,
              const float*        y,
              float*              xVal,
              const int*          xInd,
              UPTKsparseIndexBase_t idxBase);

UPTKSPARSE_DEPRECATED(UPTKsparseGather)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgthr(UPTKsparseHandle_t    handle,
              int                 nnz,
              const double*       y,
              double*             xVal,
              const int*          xInd,
              UPTKsparseIndexBase_t idxBase);

UPTKSPARSE_DEPRECATED(UPTKsparseGather)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgthr(UPTKsparseHandle_t    handle,
              int                 nnz,
              const cuComplex*    y,
              cuComplex*          xVal,
              const int*          xInd,
              UPTKsparseIndexBase_t idxBase);

UPTKSPARSE_DEPRECATED(UPTKsparseGather)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgthr(UPTKsparseHandle_t       handle,
              int                    nnz,
              const cuDoubleComplex* y,
              cuDoubleComplex*       xVal,
              const int*             xInd,
              UPTKsparseIndexBase_t    idxBase);

UPTKSPARSE_DEPRECATED(UPTKsparseGather)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSgthrz(UPTKsparseHandle_t    handle,
               int                 nnz,
               float*              y,
               float*              xVal,
               const int*          xInd,
               UPTKsparseIndexBase_t idxBase);

UPTKSPARSE_DEPRECATED(UPTKsparseGather)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgthrz(UPTKsparseHandle_t    handle,
               int                 nnz,
               double*             y,
               double*             xVal,
               const int*          xInd,
               UPTKsparseIndexBase_t idxBase);

UPTKSPARSE_DEPRECATED(UPTKsparseGather)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgthrz(UPTKsparseHandle_t    handle,
               int                 nnz,
               cuComplex*          y,
               cuComplex*          xVal,
               const int*          xInd,
               UPTKsparseIndexBase_t idxBase);

UPTKSPARSE_DEPRECATED(UPTKsparseGather)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgthrz(UPTKsparseHandle_t    handle,
               int                 nnz,
               cuDoubleComplex*    y,
               cuDoubleComplex*    xVal,
               const int*          xInd,
               UPTKsparseIndexBase_t idxBase);

UPTKSPARSE_DEPRECATED(UPTKsparseScatter)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSsctr(UPTKsparseHandle_t    handle,
              int                 nnz,
              const float*        xVal,
              const int*          xInd,
              float*              y,
              UPTKsparseIndexBase_t idxBase);

UPTKSPARSE_DEPRECATED(UPTKsparseScatter)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDsctr(UPTKsparseHandle_t    handle,
              int                 nnz,
              const double*       xVal,
              const int*          xInd,
              double*             y,
              UPTKsparseIndexBase_t idxBase);

UPTKSPARSE_DEPRECATED(UPTKsparseScatter)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCsctr(UPTKsparseHandle_t    handle,
              int                 nnz,
              const cuComplex*    xVal,
              const int*          xInd,
              cuComplex*          y,
              UPTKsparseIndexBase_t idxBase);

UPTKSPARSE_DEPRECATED(UPTKsparseScatter)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZsctr(UPTKsparseHandle_t       handle,
              int                    nnz,
              const cuDoubleComplex* xVal,
              const int*             xInd,
              cuDoubleComplex*       y,
              UPTKsparseIndexBase_t    idxBase);

UPTKSPARSE_DEPRECATED(UPTKsparseRot)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSroti(UPTKsparseHandle_t    handle,
              int                 nnz,
              float*              xVal,
              const int*          xInd,
              float*              y,
              const float*        c,
              const float*        s,
              UPTKsparseIndexBase_t idxBase);

UPTKSPARSE_DEPRECATED(UPTKsparseRot)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDroti(UPTKsparseHandle_t    handle,
              int                 nnz,
              double*             xVal,
              const int*          xInd,
              double*             y,
              const double*       c,
              const double*       s,
              UPTKsparseIndexBase_t idxBase);

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

UPTKSPARSE_DEPRECATED(UPTKsparseSpMV)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCsrmvEx_bufferSize(UPTKsparseHandle_t         handle,
                           UPTKsparseAlgMode_t        alg,
                           UPTKsparseOperation_t      transA,
                           int                      m,
                           int                      n,
                           int                      nnz,
                           const void*              alpha,
                           UPTKDataType             alphatype,
                           const UPTKsparseMatDescr_t descrA,
                           const void*              csrValA,
                           UPTKDataType             csrValAtype,
                           const int*               csrRowPtrA,
                           const int*               csrColIndA,
                           const void*              x,
                           UPTKDataType             xtype,
                           const void*              beta,
                           UPTKDataType             betatype,
                           void*                    y,
                           UPTKDataType             ytype,
                           UPTKDataType             executiontype,
                           size_t*                  bufferSizeInBytes);

UPTKSPARSE_DEPRECATED(UPTKsparseSpMV)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCsrmvEx(UPTKsparseHandle_t         handle,
                UPTKsparseAlgMode_t        alg,
                UPTKsparseOperation_t      transA,
                int                      m,
                int                      n,
                int                      nnz,
                const void*              alpha,
                UPTKDataType             alphatype,
                const UPTKsparseMatDescr_t descrA,
                const void*              csrValA,
                UPTKDataType             csrValAtype,
                const int*               csrRowPtrA,
                const int*               csrColIndA,
                const void*              x,
                UPTKDataType             xtype,
                const void*              beta,
                UPTKDataType             betatype,
                void*                    y,
                UPTKDataType             ytype,
                UPTKDataType             executiontype,
                void*                    buffer);

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

UPTKSPARSE_DEPRECATED(UPTKsparseSpSV)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXcsrsv2_zeroPivot(UPTKsparseHandle_t handle,
                          csrsv2Info_t     info,
                          int*             position);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSV)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsrsv2_bufferSize(UPTKsparseHandle_t         handle,
                           UPTKsparseOperation_t      transA,
                           int                      m,
                           int                      nnz,
                           const UPTKsparseMatDescr_t descrA,
                           float*                   csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           csrsv2Info_t             info,
                           int*                     pBufferSizeInBytes);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSV)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsrsv2_bufferSize(UPTKsparseHandle_t         handle,
                           UPTKsparseOperation_t      transA,
                           int                      m,
                           int                      nnz,
                           const UPTKsparseMatDescr_t descrA,
                           double*                  csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           csrsv2Info_t             info,
                           int*                     pBufferSizeInBytes);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSV)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsrsv2_bufferSize(UPTKsparseHandle_t         handle,
                           UPTKsparseOperation_t      transA,
                           int                      m,
                           int                      nnz,
                           const UPTKsparseMatDescr_t descrA,
                           cuComplex*               csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           csrsv2Info_t             info,
                           int*                     pBufferSizeInBytes);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSV)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsrsv2_bufferSize(UPTKsparseHandle_t         handle,
                           UPTKsparseOperation_t      transA,
                           int                      m,
                           int                      nnz,
                           const UPTKsparseMatDescr_t descrA,
                           cuDoubleComplex*         csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           csrsv2Info_t             info,
                           int*                     pBufferSizeInBytes);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSV)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsrsv2_bufferSizeExt(UPTKsparseHandle_t         handle,
                              UPTKsparseOperation_t      transA,
                              int                      m,
                              int                      nnz,
                              const UPTKsparseMatDescr_t descrA,
                              float*                   csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              csrsv2Info_t             info,
                              size_t*                  pBufferSize);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSV)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsrsv2_bufferSizeExt(UPTKsparseHandle_t         handle,
                              UPTKsparseOperation_t      transA,
                              int                      m,
                              int                      nnz,
                              const UPTKsparseMatDescr_t descrA,
                              double*                  csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              csrsv2Info_t             info,
                              size_t*                  pBufferSize);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSV)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsrsv2_bufferSizeExt(UPTKsparseHandle_t         handle,
                              UPTKsparseOperation_t      transA,
                              int                      m,
                              int                      nnz,
                              const UPTKsparseMatDescr_t descrA,
                              cuComplex*               csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              csrsv2Info_t             info,
                              size_t*                  pBufferSize);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSV)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsrsv2_bufferSizeExt(UPTKsparseHandle_t         handle,
                              UPTKsparseOperation_t      transA,
                              int                      m,
                              int                      nnz,
                              const UPTKsparseMatDescr_t descrA,
                              cuDoubleComplex*         csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              csrsv2Info_t             info,
                              size_t*                  pBufferSize);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSV)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsrsv2_analysis(UPTKsparseHandle_t         handle,
                         UPTKsparseOperation_t      transA,
                         int                      m,
                         int                      nnz,
                         const UPTKsparseMatDescr_t descrA,
                         const float*             csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         csrsv2Info_t             info,
                         UPTKsparseSolvePolicy_t    policy,
                         void*                    pBuffer);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSV)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsrsv2_analysis(UPTKsparseHandle_t         handle,
                         UPTKsparseOperation_t      transA,
                         int                      m,
                         int                      nnz,
                         const UPTKsparseMatDescr_t descrA,
                         const double*            csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         csrsv2Info_t             info,
                         UPTKsparseSolvePolicy_t    policy,
                         void*                    pBuffer);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSV)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsrsv2_analysis(UPTKsparseHandle_t         handle,
                         UPTKsparseOperation_t      transA,
                         int                      m,
                         int                      nnz,
                         const UPTKsparseMatDescr_t descrA,
                         const cuComplex*         csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         csrsv2Info_t             info,
                         UPTKsparseSolvePolicy_t    policy,
                         void*                    pBuffer);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSV)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsrsv2_analysis(UPTKsparseHandle_t         handle,
                         UPTKsparseOperation_t      transA,
                         int                      m,
                         int                      nnz,
                         const UPTKsparseMatDescr_t descrA,
                         const cuDoubleComplex*   csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         csrsv2Info_t             info,
                         UPTKsparseSolvePolicy_t    policy,
                         void*                    pBuffer);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSV)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsrsv2_solve(UPTKsparseHandle_t         handle,
                      UPTKsparseOperation_t      transA,
                      int                      m,
                      int                      nnz,
                      const float*             alpha,
                      const UPTKsparseMatDescr_t descrA,
                      const float*             csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      csrsv2Info_t             info,
                      const float*             f,
                      float*                   x,
                      UPTKsparseSolvePolicy_t    policy,
                      void*                    pBuffer);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSV)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsrsv2_solve(UPTKsparseHandle_t         handle,
                      UPTKsparseOperation_t      transA,
                      int                      m,
                      int                      nnz,
                      const double*            alpha,
                      const UPTKsparseMatDescr_t descrA,
                      const double*            csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      csrsv2Info_t             info,
                      const double*            f,
                      double*                  x,
                      UPTKsparseSolvePolicy_t    policy,
                      void*                    pBuffer);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSV)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsrsv2_solve(UPTKsparseHandle_t         handle,
                      UPTKsparseOperation_t      transA,
                      int                      m,
                      int                      nnz,
                      const cuComplex*         alpha,
                      const UPTKsparseMatDescr_t descrA,
                      const cuComplex*         csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      csrsv2Info_t             info,
                      const cuComplex*         f,
                      cuComplex*               x,
                      UPTKsparseSolvePolicy_t    policy,
                      void*                    pBuffer);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSV)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsrsv2_solve(UPTKsparseHandle_t         handle,
                      UPTKsparseOperation_t      transA,
                      int                      m,
                      int                      nnz,
                      const cuDoubleComplex*   alpha,
                      const UPTKsparseMatDescr_t descrA,
                      const cuDoubleComplex*   csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      csrsv2Info_t             info,
                      const cuDoubleComplex*   f,
                      cuDoubleComplex*         x,
                      UPTKsparseSolvePolicy_t    policy,
                      void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXbsrsv2_zeroPivot(UPTKsparseHandle_t handle,
                          bsrsv2Info_t     info,
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
                           bsrsv2Info_t             info,
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
                           bsrsv2Info_t             info,
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
                           bsrsv2Info_t             info,
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
                           bsrsv2Info_t             info,
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
                              bsrsv2Info_t             info,
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
                              bsrsv2Info_t             info,
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
                              bsrsv2Info_t             info,
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
                              bsrsv2Info_t             info,
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
                         bsrsv2Info_t             info,
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
                         bsrsv2Info_t             info,
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
                         bsrsv2Info_t             info,
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
                         bsrsv2Info_t             info,
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
                      bsrsv2Info_t             info,
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
                      bsrsv2Info_t             info,
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
                      bsrsv2Info_t             info,
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
                      bsrsv2Info_t             info,
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

UPTKSPARSE_DEPRECATED(UPTKsparseSpMM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSgemmi(UPTKsparseHandle_t handle,
               int              m,
               int              n,
               int              k,
               int              nnz,
               const float*     alpha,
               const float*     A,
               int              lda,
               const float*     cscValB,
               const int*       cscColPtrB,
               const int*       cscRowIndB,
               const float*     beta,
               float*           C,
               int              ldc);

UPTKSPARSE_DEPRECATED(UPTKsparseSpMM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgemmi(UPTKsparseHandle_t handle,
               int              m,
               int              n,
               int              k,
               int              nnz,
               const double*    alpha,
               const double*    A,
               int              lda,
               const double*    cscValB,
               const int*       cscColPtrB,
               const int*       cscRowIndB,
               const double*    beta,
               double*          C,
               int              ldc);

UPTKSPARSE_DEPRECATED(UPTKsparseSpMM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgemmi(UPTKsparseHandle_t handle,
               int              m,
               int              n,
               int              k,
               int              nnz,
               const cuComplex* alpha,
               const cuComplex* A,
               int              lda,
               const cuComplex* cscValB,
               const int*       cscColPtrB,
               const int*       cscRowIndB,
               const cuComplex* beta,
               cuComplex*       C,
               int              ldc);

UPTKSPARSE_DEPRECATED(UPTKsparseSpMM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgemmi(UPTKsparseHandle_t       handle,
               int                    m,
               int                    n,
               int                    k,
               int                    nnz,
               const cuDoubleComplex* alpha,
               const cuDoubleComplex* A,
               int                    lda,
               const cuDoubleComplex* cscValB,
               const int*             cscColPtrB,
               const int*             cscRowIndB,
               const cuDoubleComplex* beta,
               cuDoubleComplex*       C,
               int                    ldc);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateCsrsm2Info(csrsm2Info_t* info);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyCsrsm2Info(csrsm2Info_t info);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXcsrsm2_zeroPivot(UPTKsparseHandle_t handle,
                          csrsm2Info_t     info,
                          int* position);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsrsm2_bufferSizeExt(UPTKsparseHandle_t         handle,
                              int                      algo,
                              UPTKsparseOperation_t      transA,
                              UPTKsparseOperation_t      transB,
                              int                      m,
                              int                      nrhs,
                              int                      nnz,
                              const float*             alpha,
                              const UPTKsparseMatDescr_t descrA,
                              const float*             csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              const float*             B,
                              int                      ldb,
                              csrsm2Info_t             info,
                              UPTKsparseSolvePolicy_t    policy,
                              size_t*                  pBufferSize);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsrsm2_bufferSizeExt(UPTKsparseHandle_t         handle,
                              int                      algo,
                              UPTKsparseOperation_t      transA,
                              UPTKsparseOperation_t      transB,
                              int                      m,
                              int                      nrhs,
                              int                      nnz,
                              const double*            alpha,
                              const UPTKsparseMatDescr_t descrA,
                              const double*            csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              const double*            B,
                              int                      ldb,
                              csrsm2Info_t             info,
                              UPTKsparseSolvePolicy_t    policy,
                              size_t*                  pBufferSize);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsrsm2_bufferSizeExt(UPTKsparseHandle_t         handle,
                              int                      algo,
                              UPTKsparseOperation_t      transA,
                              UPTKsparseOperation_t      transB,
                              int                      m,
                              int                      nrhs,
                              int                      nnz,
                              const cuComplex*         alpha,
                              const UPTKsparseMatDescr_t descrA,
                              const cuComplex*         csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              const cuComplex*         B,
                              int                      ldb,
                              csrsm2Info_t             info,
                              UPTKsparseSolvePolicy_t    policy,
                              size_t*                  pBufferSize);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsrsm2_bufferSizeExt(UPTKsparseHandle_t         handle,
                              int                      algo,
                              UPTKsparseOperation_t      transA,
                              UPTKsparseOperation_t      transB,
                              int                      m,
                              int                      nrhs,
                              int                      nnz,
                              const cuDoubleComplex*   alpha,
                              const UPTKsparseMatDescr_t descrA,
                              const cuDoubleComplex*   csrSortedValA,
                              const int*               csrSortedRowPtrA,
                              const int*               csrSortedColIndA,
                              const cuDoubleComplex*   B,
                              int                      ldb,
                              csrsm2Info_t             info,
                              UPTKsparseSolvePolicy_t    policy,
                              size_t*                  pBufferSize);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsrsm2_analysis(UPTKsparseHandle_t         handle,
                         int                      algo,
                         UPTKsparseOperation_t      transA,
                         UPTKsparseOperation_t      transB,
                         int                      m,
                         int                      nrhs,
                         int                      nnz,
                         const float*             alpha,
                         const UPTKsparseMatDescr_t descrA,
                         const float*             csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         const float*             B,
                         int                      ldb,
                         csrsm2Info_t             info,
                         UPTKsparseSolvePolicy_t    policy,
                         void*                    pBuffer);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsrsm2_analysis(UPTKsparseHandle_t         handle,
                         int                      algo,
                         UPTKsparseOperation_t      transA,
                         UPTKsparseOperation_t      transB,
                         int                      m,
                         int                      nrhs,
                         int                      nnz,
                         const double*            alpha,
                         const UPTKsparseMatDescr_t descrA,
                         const double*            csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         const double*            B,
                         int                      ldb,
                         csrsm2Info_t             info,
                         UPTKsparseSolvePolicy_t    policy,
                         void*                    pBuffer);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsrsm2_analysis(UPTKsparseHandle_t         handle,
                         int                      algo,
                         UPTKsparseOperation_t      transA,
                         UPTKsparseOperation_t      transB,
                         int                      m,
                         int                      nrhs,
                         int                      nnz,
                         const cuComplex*         alpha,
                         const UPTKsparseMatDescr_t descrA,
                         const cuComplex*         csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         const cuComplex*         B,
                         int                      ldb,
                         csrsm2Info_t             info,
                         UPTKsparseSolvePolicy_t    policy,
                         void*                    pBuffer);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsrsm2_analysis(UPTKsparseHandle_t         handle,
                         int                      algo,
                         UPTKsparseOperation_t      transA,
                         UPTKsparseOperation_t      transB,
                         int                      m,
                         int                      nrhs,
                         int                      nnz,
                         const cuDoubleComplex*   alpha,
                         const UPTKsparseMatDescr_t descrA,
                         const cuDoubleComplex*   csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         const cuDoubleComplex*   B,
                         int                      ldb,
                         csrsm2Info_t             info,
                         UPTKsparseSolvePolicy_t    policy,
                         void*                    pBuffer);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsrsm2_solve(UPTKsparseHandle_t         handle,
                      int                      algo,
                      UPTKsparseOperation_t      transA,
                      UPTKsparseOperation_t      transB,
                      int                      m,
                      int                      nrhs,
                      int                      nnz,
                      const float*             alpha,
                      const UPTKsparseMatDescr_t descrA,
                      const float*             csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      float*                   B,
                      int                      ldb,
                      csrsm2Info_t             info,
                      UPTKsparseSolvePolicy_t    policy,
                      void*                    pBuffer);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsrsm2_solve(UPTKsparseHandle_t         handle,
                      int                      algo,
                      UPTKsparseOperation_t      transA,
                      UPTKsparseOperation_t      transB,
                      int                      m,
                      int                      nrhs,
                      int                      nnz,
                      const double*            alpha,
                      const UPTKsparseMatDescr_t descrA,
                      const double*            csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      double*                  B,
                      int                      ldb,
                      csrsm2Info_t             info,
                      UPTKsparseSolvePolicy_t    policy,
                      void*                    pBuffer);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsrsm2_solve(UPTKsparseHandle_t         handle,
                      int                      algo,
                      UPTKsparseOperation_t      transA,
                      UPTKsparseOperation_t      transB,
                      int                      m,
                      int                      nrhs,
                      int                      nnz,
                      const cuComplex*         alpha,
                      const UPTKsparseMatDescr_t descrA,
                      const cuComplex*         csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      cuComplex*               B,
                      int                      ldb,
                      csrsm2Info_t             info,
                      UPTKsparseSolvePolicy_t    policy,
                      void*                    pBuffer);

UPTKSPARSE_DEPRECATED(UPTKsparseSpSM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsrsm2_solve(UPTKsparseHandle_t         handle,
                      int                      algo,
                      UPTKsparseOperation_t      transA,
                      UPTKsparseOperation_t      transB,
                      int                      m,
                      int                      nrhs,
                      int                      nnz,
                      const cuDoubleComplex*   alpha,
                      const UPTKsparseMatDescr_t descrA,
                      const cuDoubleComplex*   csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      cuDoubleComplex*         B,
                      int                      ldb,
                      csrsm2Info_t             info,
                      UPTKsparseSolvePolicy_t    policy,
                      void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXbsrsm2_zeroPivot(UPTKsparseHandle_t handle,
                          bsrsm2Info_t     info,
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
                           bsrsm2Info_t             info,
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
                           bsrsm2Info_t             info,
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
                           bsrsm2Info_t             info,
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
                           bsrsm2Info_t             info,
                           int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsrsm2_bufferSizeExt(UPTKsparseHandle_t         handle,
                              UPTKsparseDirection_t      dirA,
                              UPTKsparseOperation_t      transA,
                              UPTKsparseOperation_t      transB,
                              int                      mb,
                              int                      n,
                              int                      nnzb,
                              const UPTKsparseMatDescr_t descrA,
                              float*                   bsrSortedVal,
                              const int*               bsrSortedRowPtr,
                              const int*               bsrSortedColInd,
                              int                      blockSize,
                              bsrsm2Info_t             info,
                              size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsrsm2_bufferSizeExt(UPTKsparseHandle_t         handle,
                              UPTKsparseDirection_t      dirA,
                              UPTKsparseOperation_t      transA,
                              UPTKsparseOperation_t      transB,
                              int                      mb,
                              int                      n,
                              int                      nnzb,
                              const UPTKsparseMatDescr_t descrA,
                              double*                  bsrSortedVal,
                              const int*               bsrSortedRowPtr,
                              const int*               bsrSortedColInd,
                              int                      blockSize,
                              bsrsm2Info_t             info,
                              size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsrsm2_bufferSizeExt(UPTKsparseHandle_t         handle,
                              UPTKsparseDirection_t      dirA,
                              UPTKsparseOperation_t      transA,
                              UPTKsparseOperation_t      transB,
                              int                      mb,
                              int                      n,
                              int                      nnzb,
                              const UPTKsparseMatDescr_t descrA,
                              cuComplex*               bsrSortedVal,
                              const int*               bsrSortedRowPtr,
                              const int*               bsrSortedColInd,
                              int                      blockSize,
                              bsrsm2Info_t             info,
                              size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsrsm2_bufferSizeExt(UPTKsparseHandle_t         handle,
                              UPTKsparseDirection_t      dirA,
                              UPTKsparseOperation_t      transA,
                              UPTKsparseOperation_t      transB,
                              int                      mb,
                              int                      n,
                              int                      nnzb,
                              const UPTKsparseMatDescr_t descrA,
                              cuDoubleComplex*         bsrSortedVal,
                              const int*               bsrSortedRowPtr,
                              const int*               bsrSortedColInd,
                              int                      blockSize,
                              bsrsm2Info_t             info,
                              size_t*                  pBufferSize);

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
                         bsrsm2Info_t             info,
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
                         bsrsm2Info_t             info,
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
                         bsrsm2Info_t             info,
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
                         bsrsm2Info_t             info,
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
                      bsrsm2Info_t             info,
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
                      bsrsm2Info_t             info,
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
                      bsrsm2Info_t             info,
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
                      bsrsm2Info_t             info,
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
                               csrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               float*           boost_val);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsrilu02_numericBoost(UPTKsparseHandle_t handle,
                               csrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               double*          boost_val);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsrilu02_numericBoost(UPTKsparseHandle_t handle,
                               csrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               cuComplex*       boost_val);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsrilu02_numericBoost(UPTKsparseHandle_t handle,
                               csrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               cuDoubleComplex* boost_val);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXcsrilu02_zeroPivot(UPTKsparseHandle_t handle,
                            csrilu02Info_t   info,
                            int*             position);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsrilu02_bufferSize(UPTKsparseHandle_t         handle,
                             int                      m,
                             int                      nnz,
                             const UPTKsparseMatDescr_t descrA,
                             float*                   csrSortedValA,
                             const int*               csrSortedRowPtrA,
                             const int*               csrSortedColIndA,
                             csrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsrilu02_bufferSize(UPTKsparseHandle_t         handle,
                             int                      m,
                             int                      nnz,
                             const UPTKsparseMatDescr_t descrA,
                             double*                  csrSortedValA,
                             const int*               csrSortedRowPtrA,
                             const int*               csrSortedColIndA,
                             csrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsrilu02_bufferSize(UPTKsparseHandle_t         handle,
                             int                      m,
                             int                      nnz,
                             const UPTKsparseMatDescr_t descrA,
                             cuComplex*               csrSortedValA,
                             const int*               csrSortedRowPtrA,
                             const int*               csrSortedColIndA,
                             csrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsrilu02_bufferSize(UPTKsparseHandle_t         handle,
                             int                      m,
                             int                      nnz,
                             const UPTKsparseMatDescr_t descrA,
                             cuDoubleComplex*         csrSortedValA,
                             const int*               csrSortedRowPtrA,
                             const int*               csrSortedColIndA,
                             csrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsrilu02_bufferSizeExt(UPTKsparseHandle_t         handle,
                                int                      m,
                                int                      nnz,
                                const UPTKsparseMatDescr_t descrA,
                                float*                   csrSortedVal,
                                const int*               csrSortedRowPtr,
                                const int*               csrSortedColInd,
                                csrilu02Info_t           info,
                                size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsrilu02_bufferSizeExt(UPTKsparseHandle_t         handle,
                                int                      m,
                                int                      nnz,
                                const UPTKsparseMatDescr_t descrA,
                                double*                  csrSortedVal,
                                const int*               csrSortedRowPtr,
                                const int*               csrSortedColInd,
                                csrilu02Info_t           info,
                                size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsrilu02_bufferSizeExt(UPTKsparseHandle_t         handle,
                                int                      m,
                                int                      nnz,
                                const UPTKsparseMatDescr_t descrA,
                                cuComplex*               csrSortedVal,
                                const int*               csrSortedRowPtr,
                                const int*               csrSortedColInd,
                                csrilu02Info_t           info,
                                size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsrilu02_bufferSizeExt(UPTKsparseHandle_t         handle,
                                int                      m,
                                int                      nnz,
                                const UPTKsparseMatDescr_t descrA,
                                cuDoubleComplex*         csrSortedVal,
                                const int*               csrSortedRowPtr,
                                const int*               csrSortedColInd,
                                csrilu02Info_t           info,
                                size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsrilu02_analysis(UPTKsparseHandle_t         handle,
                           int                      m,
                           int                      nnz,
                           const UPTKsparseMatDescr_t descrA,
                           const float*             csrSortedValA,
                           const int*               csrSortedRowPtrA,
                           const int*               csrSortedColIndA,
                           csrilu02Info_t           info,
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
                           csrilu02Info_t           info,
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
                           csrilu02Info_t           info,
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
                           csrilu02Info_t           info,
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
                  csrilu02Info_t        info,
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
                  csrilu02Info_t        info,
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
                  csrilu02Info_t        info,
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
                  csrilu02Info_t        info,
                  UPTKsparseSolvePolicy_t policy,
                  void*                 pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsrilu02_numericBoost(UPTKsparseHandle_t handle,
                               bsrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               float*           boost_val);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsrilu02_numericBoost(UPTKsparseHandle_t handle,
                               bsrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               double*          boost_val);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsrilu02_numericBoost(UPTKsparseHandle_t handle,
                               bsrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               cuComplex*       boost_val);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsrilu02_numericBoost(UPTKsparseHandle_t handle,
                               bsrilu02Info_t   info,
                               int              enable_boost,
                               double*          tol,
                               cuDoubleComplex* boost_val);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXbsrilu02_zeroPivot(UPTKsparseHandle_t handle,
                            bsrilu02Info_t   info,
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
                             bsrilu02Info_t           info,
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
                             bsrilu02Info_t           info,
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
                             bsrilu02Info_t           info,
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
                             bsrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsrilu02_bufferSizeExt(UPTKsparseHandle_t         handle,
                                UPTKsparseDirection_t      dirA,
                                int                      mb,
                                int                      nnzb,
                                const UPTKsparseMatDescr_t descrA,
                                float*                   bsrSortedVal,
                                const int*               bsrSortedRowPtr,
                                const int*               bsrSortedColInd,
                                int                      blockSize,
                                bsrilu02Info_t           info,
                                size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsrilu02_bufferSizeExt(UPTKsparseHandle_t         handle,
                                UPTKsparseDirection_t      dirA,
                                int                      mb,
                                int                      nnzb,
                                const UPTKsparseMatDescr_t descrA,
                                double*                  bsrSortedVal,
                                const int*               bsrSortedRowPtr,
                                const int*               bsrSortedColInd,
                                int                      blockSize,
                                bsrilu02Info_t           info,
                                size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsrilu02_bufferSizeExt(UPTKsparseHandle_t         handle,
                                UPTKsparseDirection_t      dirA,
                                int                      mb,
                                int                      nnzb,
                                const UPTKsparseMatDescr_t descrA,
                                cuComplex*               bsrSortedVal,
                                const int*               bsrSortedRowPtr,
                                const int*               bsrSortedColInd,
                                int                      blockSize,
                                bsrilu02Info_t           info,
                                size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsrilu02_bufferSizeExt(UPTKsparseHandle_t         handle,
                               UPTKsparseDirection_t      dirA,
                               int                      mb,
                               int                      nnzb,
                               const UPTKsparseMatDescr_t descrA,
                               cuDoubleComplex*         bsrSortedVal,
                               const int*               bsrSortedRowPtr,
                               const int*               bsrSortedColInd,
                               int                      blockSize,
                               bsrilu02Info_t           info,
                               size_t*                  pBufferSize);

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
                           bsrilu02Info_t           info,
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
                           bsrilu02Info_t           info,
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
                           bsrilu02Info_t           info,
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
                           bsrilu02Info_t           info,
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
                  bsrilu02Info_t           info,
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
                  bsrilu02Info_t           info,
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
                  bsrilu02Info_t           info,
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
                  bsrilu02Info_t           info,
                  UPTKsparseSolvePolicy_t    policy,
                  void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXcsric02_zeroPivot(UPTKsparseHandle_t handle,
                           csric02Info_t    info,
                           int*             position);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsric02_bufferSize(UPTKsparseHandle_t         handle,
                            int                      m,
                            int                      nnz,
                            const UPTKsparseMatDescr_t descrA,
                            float*                   csrSortedValA,
                            const int*               csrSortedRowPtrA,
                            const int*               csrSortedColIndA,
                            csric02Info_t            info,
                            int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsric02_bufferSize(UPTKsparseHandle_t         handle,
                            int                      m,
                            int                      nnz,
                            const UPTKsparseMatDescr_t descrA,
                            double*                  csrSortedValA,
                            const int*               csrSortedRowPtrA,
                            const int*               csrSortedColIndA,
                            csric02Info_t            info,
                            int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsric02_bufferSize(UPTKsparseHandle_t         handle,
                            int                      m,
                            int                      nnz,
                            const UPTKsparseMatDescr_t descrA,
                            cuComplex*               csrSortedValA,
                            const int*               csrSortedRowPtrA,
                            const int*               csrSortedColIndA,
                            csric02Info_t            info,
                            int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsric02_bufferSize(UPTKsparseHandle_t         handle,
                            int                      m,
                            int                      nnz,
                            const UPTKsparseMatDescr_t descrA,
                            cuDoubleComplex*         csrSortedValA,
                            const int*               csrSortedRowPtrA,
                            const int*               csrSortedColIndA,
                            csric02Info_t            info,
                            int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsric02_bufferSizeExt(UPTKsparseHandle_t         handle,
                               int                      m,
                               int                      nnz,
                               const UPTKsparseMatDescr_t descrA,
                               float*                   csrSortedVal,
                               const int*               csrSortedRowPtr,
                               const int*               csrSortedColInd,
                               csric02Info_t            info,
                               size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsric02_bufferSizeExt(UPTKsparseHandle_t         handle,
                               int                      m,
                               int                      nnz,
                               const UPTKsparseMatDescr_t descrA,
                               double*                  csrSortedVal,
                               const int*               csrSortedRowPtr,
                               const int*               csrSortedColInd,
                               csric02Info_t            info,
                               size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsric02_bufferSizeExt(UPTKsparseHandle_t         handle,
                               int                      m,
                               int                      nnz,
                               const UPTKsparseMatDescr_t descrA,
                               cuComplex*               csrSortedVal,
                               const int*               csrSortedRowPtr,
                               const int*               csrSortedColInd,
                               csric02Info_t            info,
                               size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsric02_bufferSizeExt(UPTKsparseHandle_t         handle,
                               int                      m,
                               int                      nnz,
                               const UPTKsparseMatDescr_t descrA,
                               cuDoubleComplex*         csrSortedVal,
                               const int*               csrSortedRowPtr,
                               const int*               csrSortedColInd,
                               csric02Info_t            info,
                               size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsric02_analysis(UPTKsparseHandle_t         handle,
                          int                      m,
                          int                      nnz,
                          const UPTKsparseMatDescr_t descrA,
                          const float*             csrSortedValA,
                          const int*               csrSortedRowPtrA,
                          const int*               csrSortedColIndA,
                          csric02Info_t            info,
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
                          csric02Info_t            info,
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
                          csric02Info_t            info,
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
                          csric02Info_t            info,
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
                 csric02Info_t            info,
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
                 csric02Info_t            info,
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
                 csric02Info_t            info,
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
                 csric02Info_t            info,
                 UPTKsparseSolvePolicy_t    policy,
                 void*                    pBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXbsric02_zeroPivot(UPTKsparseHandle_t handle,
                           bsric02Info_t    info,
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
                            bsric02Info_t            info,
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
                            bsric02Info_t            info,
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
                            bsric02Info_t            info,
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
                            bsric02Info_t            info,
                            int*                     pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSbsric02_bufferSizeExt(UPTKsparseHandle_t         handle,
                               UPTKsparseDirection_t      dirA,
                               int                      mb,
                               int                      nnzb,
                               const UPTKsparseMatDescr_t descrA,
                               float*                   bsrSortedVal,
                               const int*               bsrSortedRowPtr,
                               const int*               bsrSortedColInd,
                               int                      blockSize,
                               bsric02Info_t            info,
                               size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDbsric02_bufferSizeExt(UPTKsparseHandle_t         handle,
                               UPTKsparseDirection_t      dirA,
                               int                      mb,
                               int                      nnzb,
                               const UPTKsparseMatDescr_t descrA,
                               double*                  bsrSortedVal,
                               const int*               bsrSortedRowPtr,
                               const int*               bsrSortedColInd,
                               int                      blockSize,
                               bsric02Info_t            info,
                               size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCbsric02_bufferSizeExt(UPTKsparseHandle_t         handle,
                               UPTKsparseDirection_t      dirA,
                               int                      mb,
                               int                      nnzb,
                               const UPTKsparseMatDescr_t descrA,
                               cuComplex*               bsrSortedVal,
                               const int*               bsrSortedRowPtr,
                               const int*               bsrSortedColInd,
                               int                      blockSize,
                               bsric02Info_t            info,
                               size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZbsric02_bufferSizeExt(UPTKsparseHandle_t         handle,
                               UPTKsparseDirection_t      dirA,
                               int                      mb,
                               int                      nnzb,
                               const UPTKsparseMatDescr_t descrA,
                               cuDoubleComplex*         bsrSortedVal,
                               const int*               bsrSortedRowPtr,
                               const int*               bsrSortedColInd,
                               int                      blockSize,
                               bsric02Info_t            info,
                               size_t*                  pBufferSize);

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
                          bsric02Info_t            info,
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
                          bsric02Info_t            info,
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
                          bsric02Info_t            info,
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
                          bsric02Info_t            info,
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
                 bsric02Info_t            info,
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
                 bsric02Info_t            info,
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
                 bsric02Info_t            info,
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
                 bsric02Info_t            info,
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

UPTKSPARSE_DEPRECATED(UPTKsparseSpGEMM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateCsrgemm2Info(csrgemm2Info_t* info);

UPTKSPARSE_DEPRECATED(UPTKsparseSpGEMM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroyCsrgemm2Info(csrgemm2Info_t info);

UPTKSPARSE_DEPRECATED(UPTKsparseSpGEMM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsrgemm2_bufferSizeExt(UPTKsparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                int                      k,
                                const float*             alpha,
                                const UPTKsparseMatDescr_t descrA,
                                int                      nnzA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const UPTKsparseMatDescr_t descrB,
                                int                      nnzB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const float*             beta,
                                const UPTKsparseMatDescr_t descrD,
                                int                      nnzD,
                                const int*               csrSortedRowPtrD,
                                const int*               csrSortedColIndD,
                                csrgemm2Info_t           info,
                                size_t*                  pBufferSizeInBytes);

UPTKSPARSE_DEPRECATED(UPTKsparseSpGEMM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsrgemm2_bufferSizeExt(UPTKsparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                int                      k,
                                const double*            alpha,
                                const UPTKsparseMatDescr_t descrA,
                                int                      nnzA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const UPTKsparseMatDescr_t descrB,
                                int                      nnzB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const double*            beta,
                                const UPTKsparseMatDescr_t descrD,
                                int                      nnzD,
                                const int*               csrSortedRowPtrD,
                                const int*               csrSortedColIndD,
                                csrgemm2Info_t           info,
                                size_t*                  pBufferSizeInBytes);

UPTKSPARSE_DEPRECATED(UPTKsparseSpGEMM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsrgemm2_bufferSizeExt(UPTKsparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                int                      k,
                                const cuComplex*         alpha,
                                const UPTKsparseMatDescr_t descrA,
                                int                      nnzA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const UPTKsparseMatDescr_t descrB,
                                int                      nnzB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const cuComplex*         beta,
                                const UPTKsparseMatDescr_t descrD,
                                int                      nnzD,
                                const int*               csrSortedRowPtrD,
                                const int*               csrSortedColIndD,
                                csrgemm2Info_t           info,
                                size_t*                  pBufferSizeInBytes);

UPTKSPARSE_DEPRECATED(UPTKsparseSpGEMM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsrgemm2_bufferSizeExt(UPTKsparseHandle_t         handle,
                                int                      m,
                                int                      n,
                                int                      k,
                                const cuDoubleComplex*   alpha,
                                const UPTKsparseMatDescr_t descrA,
                                int                      nnzA,
                                const int*               csrSortedRowPtrA,
                                const int*               csrSortedColIndA,
                                const UPTKsparseMatDescr_t descrB,
                                int                      nnzB,
                                const int*               csrSortedRowPtrB,
                                const int*               csrSortedColIndB,
                                const cuDoubleComplex*   beta,
                                const UPTKsparseMatDescr_t descrD,
                                int                      nnzD,
                                const int*               csrSortedRowPtrD,
                                const int*               csrSortedColIndD,
                                csrgemm2Info_t           info,
                                size_t*                  pBufferSizeInBytes);

UPTKSPARSE_DEPRECATED(UPTKsparseSpGEMM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseXcsrgemm2Nnz(UPTKsparseHandle_t         handle,
                     int                      m,
                     int                      n,
                     int                      k,
                     const UPTKsparseMatDescr_t descrA,
                     int                      nnzA,
                     const int*               csrSortedRowPtrA,
                     const int*               csrSortedColIndA,
                     const UPTKsparseMatDescr_t descrB,
                     int                      nnzB,
                     const int*               csrSortedRowPtrB,
                     const int*               csrSortedColIndB,
                     const UPTKsparseMatDescr_t descrD,
                     int                      nnzD,
                     const int*               csrSortedRowPtrD,
                     const int*               csrSortedColIndD,
                     const UPTKsparseMatDescr_t descrC,
                     int*                     csrSortedRowPtrC,
                     int*                     nnzTotalDevHostPtr,
                     const csrgemm2Info_t     info,
                     void*                    pBuffer);

UPTKSPARSE_DEPRECATED(UPTKsparseSpGEMM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsrgemm2(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      k,
                  const float*             alpha,
                  const UPTKsparseMatDescr_t descrA,
                  int                      nnzA,
                  const float*             csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const UPTKsparseMatDescr_t descrB,
                  int                      nnzB,
                  const float*             csrSortedValB,
                  const int*               csrSortedRowPtrB,
                  const int*               csrSortedColIndB,
                  const float*             beta,
                  const UPTKsparseMatDescr_t descrD,
                  int                      nnzD,
                  const float*             csrSortedValD,
                  const int*               csrSortedRowPtrD,
                  const int*               csrSortedColIndD,
                  const UPTKsparseMatDescr_t descrC,
                  float*                   csrSortedValC,
                  const int*               csrSortedRowPtrC,
                  int*                     csrSortedColIndC,
                  const csrgemm2Info_t     info,
                  void*                    pBuffer);

UPTKSPARSE_DEPRECATED(UPTKsparseSpGEMM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsrgemm2(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      k,
                  const double*            alpha,
                  const UPTKsparseMatDescr_t descrA,
                  int                      nnzA,
                  const double*            csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const UPTKsparseMatDescr_t descrB,
                  int                      nnzB,
                  const double*            csrSortedValB,
                  const int*               csrSortedRowPtrB,
                  const int*               csrSortedColIndB,
                  const double*            beta,
                  const UPTKsparseMatDescr_t descrD,
                  int                      nnzD,
                  const double*            csrSortedValD,
                  const int*               csrSortedRowPtrD,
                  const int*               csrSortedColIndD,
                  const UPTKsparseMatDescr_t descrC,
                  double*                  csrSortedValC,
                  const int*               csrSortedRowPtrC,
                  int*                     csrSortedColIndC,
                  const csrgemm2Info_t     info,
                  void*                    pBuffer);

UPTKSPARSE_DEPRECATED(UPTKsparseSpGEMM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsrgemm2(UPTKsparseHandle_t         handle,
                 int                      m,
                 int                      n,
                 int                      k,
                 const cuComplex*         alpha,
                 const UPTKsparseMatDescr_t descrA,
                 int                      nnzA,
                 const cuComplex*         csrSortedValA,
                 const int*               csrSortedRowPtrA,
                 const int*               csrSortedColIndA,
                 const UPTKsparseMatDescr_t descrB,
                 int                      nnzB,
                 const cuComplex*         csrSortedValB,
                 const int*               csrSortedRowPtrB,
                 const int*               csrSortedColIndB,
                 const cuComplex*         beta,
                 const UPTKsparseMatDescr_t descrD,
                 int                      nnzD,
                 const cuComplex*         csrSortedValD,
                 const int*               csrSortedRowPtrD,
                 const int*               csrSortedColIndD,
                 const UPTKsparseMatDescr_t descrC,
                 cuComplex*               csrSortedValC,
                 const int*               csrSortedRowPtrC,
                 int*                     csrSortedColIndC,
                 const csrgemm2Info_t     info,
                 void*                    pBuffer);

UPTKSPARSE_DEPRECATED(UPTKsparseSpGEMM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsrgemm2(UPTKsparseHandle_t         handle,
                  int                      m,
                  int                      n,
                  int                      k,
                  const cuDoubleComplex*   alpha,
                  const UPTKsparseMatDescr_t descrA,
                  int                      nnzA,
                  const cuDoubleComplex*   csrSortedValA,
                  const int*               csrSortedRowPtrA,
                  const int*               csrSortedColIndA,
                  const UPTKsparseMatDescr_t descrB,
                  int                      nnzB,
                  const cuDoubleComplex*   csrSortedValB,
                  const int*               csrSortedRowPtrB,
                  const int*               csrSortedColIndB,
                  const cuDoubleComplex*   beta,
                  const UPTKsparseMatDescr_t descrD,
                  int                      nnzD,
                  const cuDoubleComplex*   csrSortedValD,
                  const int*               csrSortedRowPtrD,
                  const int*               csrSortedColIndD,
                  const UPTKsparseMatDescr_t descrC,
                  cuDoubleComplex*         csrSortedValC,
                  const int*               csrSortedRowPtrC,
                  int*                     csrSortedColIndC,
                  const csrgemm2Info_t     info,
                  void*                    pBuffer);

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

UPTKSPARSE_DEPRECATED(UPTKsparseDenseToSparse)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSdense2csr(UPTKsparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const UPTKsparseMatDescr_t descrA,
                   const float*             A,
                   int                      lda,
                   const int*               nnzPerRow,
                   float*                   csrSortedValA,
                   int*                     csrSortedRowPtrA,
                   int*                     csrSortedColIndA);

UPTKSPARSE_DEPRECATED(UPTKsparseDenseToSparse)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDdense2csr(UPTKsparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const UPTKsparseMatDescr_t descrA,
                   const double*            A,
                   int                      lda,
                   const int*               nnzPerRow,
                   double*                  csrSortedValA,
                   int*                     csrSortedRowPtrA,
                   int*                     csrSortedColIndA);

UPTKSPARSE_DEPRECATED(UPTKsparseDenseToSparse)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCdense2csr(UPTKsparseHandle_t           handle,
                     int                      m,
                     int                      n,
                     const UPTKsparseMatDescr_t descrA,
                     const cuComplex*         A,
                     int                      lda,
                     const int*               nnzPerRow,
                     cuComplex*               csrSortedValA,
                     int*                     csrSortedRowPtrA,
                     int*                     csrSortedColIndA);

UPTKSPARSE_DEPRECATED(UPTKsparseDenseToSparse)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZdense2csr(UPTKsparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const UPTKsparseMatDescr_t descrA,
                   const cuDoubleComplex*   A,
                   int                      lda,
                   const int*               nnzPerRow,
                   cuDoubleComplex*         csrSortedValA,
                   int*                     csrSortedRowPtrA,
                   int*                     csrSortedColIndA);

UPTKSPARSE_DEPRECATED(UPTKsparseSparseToDense)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsr2dense(UPTKsparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const UPTKsparseMatDescr_t descrA,
                   const float*             csrSortedValA,
                   const int*               csrSortedRowPtrA,
                   const int*               csrSortedColIndA,
                   float*                   A,
                   int                      lda);

UPTKSPARSE_DEPRECATED(UPTKsparseSparseToDense)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsr2dense(UPTKsparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const UPTKsparseMatDescr_t descrA,
                   const double*            csrSortedValA,
                   const int*               csrSortedRowPtrA,
                   const int*               csrSortedColIndA,
                   double*                  A,
                   int                      lda);

UPTKSPARSE_DEPRECATED(UPTKsparseSparseToDense)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsr2dense(UPTKsparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const UPTKsparseMatDescr_t descrA,
                   const cuComplex*         csrSortedValA,
                   const int*               csrSortedRowPtrA,
                   const int*               csrSortedColIndA,
                   cuComplex*               A,
                   int                      lda);

UPTKSPARSE_DEPRECATED(UPTKsparseSparseToDense)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsr2dense(UPTKsparseHandle_t         handle,
                int                      m,
                int                      n,
                const UPTKsparseMatDescr_t descrA,
                const cuDoubleComplex*   csrSortedValA,
                const int*               csrSortedRowPtrA,
                const int*               csrSortedColIndA,
                cuDoubleComplex*         A,
                int                      lda);

UPTKSPARSE_DEPRECATED(UPTKsparseDenseToSparse)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSdense2csc(UPTKsparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const UPTKsparseMatDescr_t descrA,
                   const float*             A,
                   int                      lda,
                   const int*               nnzPerCol,
                   float*                   cscSortedValA,
                   int*                     cscSortedRowIndA,
                   int*                     cscSortedColPtrA);

UPTKSPARSE_DEPRECATED(UPTKsparseDenseToSparse)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDdense2csc(UPTKsparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const UPTKsparseMatDescr_t descrA,
                   const double*            A,
                   int                      lda,
                   const int*               nnzPerCol,
                   double*                  cscSortedValA,
                   int*                     cscSortedRowIndA,
                   int*                     cscSortedColPtrA);

UPTKSPARSE_DEPRECATED(UPTKsparseDenseToSparse)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCdense2csc(UPTKsparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const UPTKsparseMatDescr_t descrA,
                   const cuComplex*         A,
                   int                      lda,
                   const int*               nnzPerCol,
                   cuComplex*               cscSortedValA,
                   int*                     cscSortedRowIndA,
                   int*                     cscSortedColPtrA);

UPTKSPARSE_DEPRECATED(UPTKsparseDenseToSparse)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZdense2csc(UPTKsparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const UPTKsparseMatDescr_t descrA,
                   const cuDoubleComplex*   A,
                   int                      lda,
                   const int*               nnzPerCol,
                   cuDoubleComplex*         cscSortedValA,
                   int*                     cscSortedRowIndA,
                   int*                     cscSortedColPtrA);

UPTKSPARSE_DEPRECATED(UPTKsparseSparseToDense)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScsc2dense(UPTKsparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const UPTKsparseMatDescr_t descrA,
                   const float*             cscSortedValA,
                   const int*               cscSortedRowIndA,
                   const int*               cscSortedColPtrA,
                   float*                   A,
                   int                      lda);

UPTKSPARSE_DEPRECATED(UPTKsparseSparseToDense)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsc2dense(UPTKsparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const UPTKsparseMatDescr_t descrA,
                   const double*            cscSortedValA,
                   const int*               cscSortedRowIndA,
                   const int*               cscSortedColPtrA,
                   double*                  A,
                   int                      lda);

UPTKSPARSE_DEPRECATED(UPTKsparseSparseToDense)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsc2dense(UPTKsparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const UPTKsparseMatDescr_t descrA,
                   const cuComplex*         cscSortedValA,
                   const int*               cscSortedRowIndA,
                   const int*               cscSortedColPtrA,
                   cuComplex*               A,
                   int                      lda);

UPTKSPARSE_DEPRECATED(UPTKsparseSparseToDense)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsc2dense(UPTKsparseHandle_t         handle,
                   int                      m,
                   int                      n,
                   const UPTKsparseMatDescr_t descrA,
                   const cuDoubleComplex*   cscSortedValA,
                   const int*               cscSortedRowIndA,
                   const int*               cscSortedColPtrA,
                   cuDoubleComplex*         A,
                   int                      lda);

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
UPTKsparseSgebsr2gebsc_bufferSizeExt(UPTKsparseHandle_t handle,
                                   int              mb,
                                   int              nb,
                                   int              nnzb,
                                   const float*     bsrSortedVal,
                                   const int*       bsrSortedRowPtr,
                                   const int*       bsrSortedColInd,
                                   int              rowBlockDim,
                                   int              colBlockDim,
                                   size_t*          pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgebsr2gebsc_bufferSizeExt(UPTKsparseHandle_t handle,
                                   int              mb,
                                   int              nb,
                                   int              nnzb,
                                   const double*    bsrSortedVal,
                                   const int*       bsrSortedRowPtr,
                                   const int*       bsrSortedColInd,
                                   int              rowBlockDim,
                                   int              colBlockDim,
                                   size_t*          pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgebsr2gebsc_bufferSizeExt(UPTKsparseHandle_t handle,
                                   int              mb,
                                   int              nb,
                                   int              nnzb,
                                   const cuComplex* bsrSortedVal,
                                   const int*       bsrSortedRowPtr,
                                   const int*       bsrSortedColInd,
                                   int              rowBlockDim,
                                   int              colBlockDim,
                                   size_t*          pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgebsr2gebsc_bufferSizeExt(UPTKsparseHandle_t       handle,
                                   int                    mb,
                                   int                    nb,
                                   int                    nnzb,
                                   const cuDoubleComplex* bsrSortedVal,
                                   const int*             bsrSortedRowPtr,
                                   const int*             bsrSortedColInd,
                                   int                    rowBlockDim,
                                   int                    colBlockDim,
                                   size_t*                pBufferSize);

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
UPTKsparseXgebsr2csr(UPTKsparseHandle_t         handle,
                   UPTKsparseDirection_t      dirA,
                   int                      mb,
                   int                      nb,
                   const UPTKsparseMatDescr_t descrA,
                   const int*               bsrSortedRowPtrA,
                   const int*               bsrSortedColIndA,
                   int                      rowBlockDim,
                   int                      colBlockDim,
                   const UPTKsparseMatDescr_t descrC,
                   int*                     csrSortedRowPtrC,
                   int*                     csrSortedColIndC);

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
UPTKsparseScsr2gebsr_bufferSizeExt(UPTKsparseHandle_t         handle,
                                 UPTKsparseDirection_t      dirA,
                                 int                      m,
                                 int                      n,
                                 const UPTKsparseMatDescr_t descrA,
                                 const float*             csrSortedValA,
                                 const int*               csrSortedRowPtrA,
                                 const int*               csrSortedColIndA,
                                 int                      rowBlockDim,
                                 int                      colBlockDim,
                                 size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsr2gebsr_bufferSizeExt(UPTKsparseHandle_t         handle,
                                 UPTKsparseDirection_t      dirA,
                                 int                      m,
                                 int                      n,
                                 const UPTKsparseMatDescr_t descrA,
                                 const double*            csrSortedValA,
                                 const int*               csrSortedRowPtrA,
                                 const int*               csrSortedColIndA,
                                 int                      rowBlockDim,
                                 int                      colBlockDim,
                                 size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsr2gebsr_bufferSizeExt(UPTKsparseHandle_t         handle,
                                 UPTKsparseDirection_t      dirA,
                                 int                      m,
                                 int                      n,
                                 const UPTKsparseMatDescr_t descrA,
                                 const cuComplex*         csrSortedValA,
                                 const int*               csrSortedRowPtrA,
                                 const int*               csrSortedColIndA,
                                 int                      rowBlockDim,
                                 int                      colBlockDim,
                                 size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsr2gebsr_bufferSizeExt(UPTKsparseHandle_t         handle,
                                 UPTKsparseDirection_t      dirA,
                                 int                      m,
                                 int                      n,
                                 const UPTKsparseMatDescr_t descrA,
                                 const cuDoubleComplex*   csrSortedValA,
                                 const int*               csrSortedRowPtrA,
                                 const int*               csrSortedColIndA,
                                 int                      rowBlockDim,
                                 int                      colBlockDim,
                                 size_t*                  pBufferSize);

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
UPTKsparseSgebsr2gebsr_bufferSizeExt(UPTKsparseHandle_t         handle,
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
                                   size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDgebsr2gebsr_bufferSizeExt(UPTKsparseHandle_t         handle,
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
                                   size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCgebsr2gebsr_bufferSizeExt(UPTKsparseHandle_t         handle,
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
                                   size_t*                  pBufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZgebsr2gebsr_bufferSizeExt(UPTKsparseHandle_t         handle,
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
                                   size_t*                  pBufferSize);

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
                                csru2csrInfo_t   info,
                                size_t*          pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDcsru2csr_bufferSizeExt(UPTKsparseHandle_t handle,
                                int              m,
                                int              n,
                                int              nnz,
                                double*          csrVal,
                                const int*       csrRowPtr,
                                int*             csrColInd,
                                csru2csrInfo_t   info,
                                size_t*          pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCcsru2csr_bufferSizeExt(UPTKsparseHandle_t handle,
                                int              m,
                                int              n,
                                int              nnz,
                                cuComplex*       csrVal,
                                const int*       csrRowPtr,
                                int*             csrColInd,
                                csru2csrInfo_t   info,
                                size_t*          pBufferSizeInBytes);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseZcsru2csr_bufferSizeExt(UPTKsparseHandle_t handle,
                                int              m,
                                int              n,
                                int              nnz,
                                cuDoubleComplex* csrVal,
                                const int*       csrRowPtr,
                                int*             csrColInd,
                                csru2csrInfo_t   info,
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
                  csru2csrInfo_t           info,
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
                  csru2csrInfo_t           info,
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
                  csru2csrInfo_t           info,
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
                  csru2csrInfo_t           info,
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
                  csru2csrInfo_t           info,
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
                  csru2csrInfo_t           info,
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
                  csru2csrInfo_t           info,
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
                  csru2csrInfo_t           info,
                  void*                    pBuffer);

#if defined(__cplusplus)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseHpruneDense2csr_bufferSizeExt(UPTKsparseHandle_t         handle,
                                      int                      m,
                                      int                      n,
                                      const __half*            A,
                                      int                      lda,
                                      const __half*            threshold,
                                      const UPTKsparseMatDescr_t descrC,
                                      const __half*            csrSortedValC,
                                      const int*               csrSortedRowPtrC,
                                      const int*               csrSortedColIndC,
                                      size_t* pBufferSizeInBytes);
#endif // defined(__cplusplus)

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

#if defined(__cplusplus)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseHpruneDense2csrNnz(UPTKsparseHandle_t         handle,
                           int                      m,
                           int                      n,
                           const __half*            A,
                           int                      lda,
                           const __half*            threshold,
                           const UPTKsparseMatDescr_t descrC,
                           int*                     csrRowPtrC,
                           int*                     nnzTotalDevHostPtr,
                           void*                    pBuffer);
#endif // defined(__cplusplus)

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

#if defined(__cplusplus)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseHpruneDense2csr(UPTKsparseHandle_t         handle,
                        int                      m,
                        int                      n,
                        const __half*            A,
                        int                      lda,
                        const __half*            threshold,
                        const UPTKsparseMatDescr_t descrC,
                        __half*                  csrSortedValC,
                        const int*               csrSortedRowPtrC,
                        int*                     csrSortedColIndC,
                        void*                    pBuffer);
#endif // defined(__cplusplus)

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

#if defined(__cplusplus)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseHpruneCsr2csr_bufferSizeExt(UPTKsparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    int                      nnzA,
                                    const UPTKsparseMatDescr_t descrA,
                                    const __half*            csrSortedValA,
                                    const int*               csrSortedRowPtrA,
                                    const int*               csrSortedColIndA,
                                    const __half*            threshold,
                                    const UPTKsparseMatDescr_t descrC,
                                    const __half*            csrSortedValC,
                                    const int*               csrSortedRowPtrC,
                                    const int*               csrSortedColIndC,
                                    size_t* pBufferSizeInBytes);
#endif // defined(__cplusplus)

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

#if defined(__cplusplus)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseHpruneCsr2csrNnz(UPTKsparseHandle_t         handle,
                         int                      m,
                         int                      n,
                         int                      nnzA,
                         const UPTKsparseMatDescr_t descrA,
                         const __half*            csrSortedValA,
                         const int*               csrSortedRowPtrA,
                         const int*               csrSortedColIndA,
                         const __half*            threshold,
                         const UPTKsparseMatDescr_t descrC,
                         int*                     csrSortedRowPtrC,
                         int*                     nnzTotalDevHostPtr,
                         void*                    pBuffer);
#endif // defined(__cplusplus)

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

#if defined(__cplusplus)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseHpruneCsr2csr(UPTKsparseHandle_t         handle,
                      int                      m,
                      int                      n,
                      int                      nnzA,
                      const UPTKsparseMatDescr_t descrA,
                      const __half*            csrSortedValA,
                      const int*               csrSortedRowPtrA,
                      const int*               csrSortedColIndA,
                      const __half*            threshold,
                      const UPTKsparseMatDescr_t descrC,
                      __half*                  csrSortedValC,
                      const int*               csrSortedRowPtrC,
                      int*                     csrSortedColIndC,
                      void*                    pBuffer);
#endif // defined(__cplusplus)

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

#if defined(__cplusplus)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseHpruneDense2csrByPercentage_bufferSizeExt(
                                   UPTKsparseHandle_t         handle,
                                   int                      m,
                                   int                      n,
                                   const __half*            A,
                                   int                      lda,
                                   float                    percentage,
                                   const UPTKsparseMatDescr_t descrC,
                                   const __half*            csrSortedValC,
                                   const int*               csrSortedRowPtrC,
                                   const int*               csrSortedColIndC,
                                   pruneInfo_t              info,
                                   size_t*                  pBufferSizeInBytes);
#endif // defined(__cplusplus)

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
                                   pruneInfo_t              info,
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
                                   pruneInfo_t              info,
                                   size_t*                  pBufferSizeInBytes);

#if defined(__cplusplus)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseHpruneDense2csrNnzByPercentage(
                                    UPTKsparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const __half*            A,
                                    int                      lda,
                                    float                    percentage,
                                    const UPTKsparseMatDescr_t descrC,
                                    int*                     csrRowPtrC,
                                    int*                     nnzTotalDevHostPtr,
                                    pruneInfo_t              info,
                                    void*                    pBuffer);
#endif // defined(__cplusplus)

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
                                    pruneInfo_t              info,
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
                                    pruneInfo_t              info,
                                    void*                    pBuffer);

#if defined(__cplusplus)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseHpruneDense2csrByPercentage(UPTKsparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    const __half*            A,
                                    int                      lda,
                                    float                    percentage,
                                    const UPTKsparseMatDescr_t descrC,
                                    __half*                  csrSortedValC,
                                    const int*               csrSortedRowPtrC,
                                    int*                     csrSortedColIndC,
                                    pruneInfo_t              info,
                                    void*                    pBuffer);
#endif // defined(__cplusplus)

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
                                    pruneInfo_t              info,
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
                                    pruneInfo_t              info,
                                    void*                    pBuffer);

#if defined(__cplusplus)

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseHpruneCsr2csrByPercentage_bufferSizeExt(
                                   UPTKsparseHandle_t         handle,
                                   int                      m,
                                   int                      n,
                                   int                      nnzA,
                                   const UPTKsparseMatDescr_t descrA,
                                   const __half*            csrSortedValA,
                                   const int*               csrSortedRowPtrA,
                                   const int*               csrSortedColIndA,
                                   float                    percentage,
                                   const UPTKsparseMatDescr_t descrC,
                                   const __half*            csrSortedValC,
                                   const int*               csrSortedRowPtrC,
                                   const int*               csrSortedColIndC,
                                   pruneInfo_t              info,
                                   size_t*                  pBufferSizeInBytes);

#endif // defined(__cplusplus)

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
                                   pruneInfo_t              info,
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
                                   pruneInfo_t              info,
                                   size_t*                  pBufferSizeInBytes);

#if defined(__cplusplus)

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseHpruneCsr2csrNnzByPercentage(
                                    UPTKsparseHandle_t         handle,
                                    int                      m,
                                    int                      n,
                                    int                      nnzA,
                                    const UPTKsparseMatDescr_t descrA,
                                    const __half*            csrSortedValA,
                                    const int*               csrSortedRowPtrA,
                                    const int*               csrSortedColIndA,
                                    float                    percentage,
                                    const UPTKsparseMatDescr_t descrC,
                                    int*                     csrSortedRowPtrC,
                                    int*                     nnzTotalDevHostPtr,
                                    pruneInfo_t              info,
                                    void*                    pBuffer);

#endif // defined(__cplusplus)

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
                                    pruneInfo_t              info,
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
                                    pruneInfo_t              info,
                                    void*                    pBuffer);

#if defined(__cplusplus)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseHpruneCsr2csrByPercentage(UPTKsparseHandle_t         handle,
                                  int                      m,
                                  int                      n,
                                  int                      nnzA,
                                  const UPTKsparseMatDescr_t descrA,
                                  const __half*            csrSortedValA,
                                  const int*               csrSortedRowPtrA,
                                  const int*               csrSortedColIndA,
                                  float percentage, /* between 0 to 100 */
                                  const UPTKsparseMatDescr_t descrC,
                                  __half*                  csrSortedValC,
                                  const int*               csrSortedRowPtrC,
                                  int*                     csrSortedColIndC,
                                  pruneInfo_t              info,
                                  void*                    pBuffer);

#endif // defined(__cplusplus)

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
                                  pruneInfo_t              info,
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
                                  pruneInfo_t              info,
                                  void*                    pBuffer);

//##############################################################################
//# CSR2CSC
//##############################################################################

typedef enum {
    UPTKSPARSE_CSR2CSC_ALG1 = 1, // faster than V2 (in general), deterministc
    UPTKSPARSE_CSR2CSC_ALG2 = 2  // low memory requirement, non-deterministc
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
    UPTKSPARSE_FORMAT_CSR         = 1, ///< Compressed Sparse Row (CSR)
    UPTKSPARSE_FORMAT_CSC         = 2, ///< Compressed Sparse Column (CSC)
    UPTKSPARSE_FORMAT_COO         = 3, ///< Coordinate (COO) - Structure of Arrays
    UPTKSPARSE_FORMAT_COO_AOS     = 4, ///< Coordinate (COO) - Array of Structures
    UPTKSPARSE_FORMAT_BLOCKED_ELL = 5, ///< Blocked ELL
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
UPTKsparseDestroySpVec(UPTKsparseSpVecDescr_t spVecDescr);

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
UPTKsparseSpVecGetIndexBase(UPTKsparseSpVecDescr_t spVecDescr,
                          UPTKsparseIndexBase_t* idxBase);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpVecGetValues(UPTKsparseSpVecDescr_t spVecDescr,
                       void**               values);

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
UPTKsparseDestroyDnVec(UPTKsparseDnVecDescr_t dnVecDescr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDnVecGet(UPTKsparseDnVecDescr_t dnVecDescr,
                 int64_t*             size,
                 void**               values,
                 UPTKDataType*        valueType);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDnVecGetValues(UPTKsparseDnVecDescr_t dnVecDescr,
                       void**               values);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDnVecSetValues(UPTKsparseDnVecDescr_t dnVecDescr,
                       void*                values);

// #############################################################################
// # SPARSE MATRIX DESCRIPTOR
// #############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDestroySpMat(UPTKsparseSpMatDescr_t spMatDescr);

 UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMatGetFormat(UPTKsparseSpMatDescr_t spMatDescr,
                       UPTKsparseFormat_t*    format);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMatGetIndexBase(UPTKsparseSpMatDescr_t spMatDescr,
                          UPTKsparseIndexBase_t* idxBase);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMatGetValues(UPTKsparseSpMatDescr_t spMatDescr,
                       void**               values);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMatSetValues(UPTKsparseSpMatDescr_t spMatDescr,
                       void*                values);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMatGetSize(UPTKsparseSpMatDescr_t spMatDescr,
                     int64_t*             rows,
                     int64_t*             cols,
                     int64_t*             nnz);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMatSetStridedBatch(UPTKsparseSpMatDescr_t spMatDescr,
                             int                  batchCount);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMatGetStridedBatch(UPTKsparseSpMatDescr_t spMatDescr,
                             int*                 batchCount);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCooSetStridedBatch(UPTKsparseSpMatDescr_t spMatDescr,
                            int                 batchCount,
                            int64_t             batchStride);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCsrSetStridedBatch(UPTKsparseSpMatDescr_t spMatDescr,
                            int                 batchCount,
                            int64_t             offsetsBatchStride,
                            int64_t             columnsValuesBatchStride);

typedef enum {
    UPTKSPARSE_SPMAT_FILL_MODE,
    UPTKSPARSE_SPMAT_DIAG_TYPE
} UPTKsparseSpMatAttribute_t;

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMatGetAttribute(UPTKsparseSpMatDescr_t     spMatDescr,
                          UPTKsparseSpMatAttribute_t attribute,
                          void*                    data,
                          size_t                   dataSize);

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

UPTKSPARSE_DEPRECATED(UPTKsparseCreateCoo)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateCooAoS(UPTKsparseSpMatDescr_t* spMatDescr,
                     int64_t               rows,
                     int64_t               cols,
                     int64_t               nnz,
                     void*                 cooInd,
                     void*                 cooValues,
                     UPTKsparseIndexType_t   cooIdxType,
                     UPTKsparseIndexBase_t   idxBase,
                     UPTKDataType          valueType);

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

UPTKSPARSE_DEPRECATED(UPTKsparseCooGet)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCooAoSGet(UPTKsparseSpMatDescr_t spMatDescr,
                  int64_t*             rows,
                  int64_t*             cols,
                  int64_t*             nnz,
                  void**               cooInd,     // COO indices
                  void**               cooValues,  // COO values
                  UPTKsparseIndexType_t* idxType,
                  UPTKsparseIndexBase_t* idxBase,
                  UPTKDataType*        valueType);

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
UPTKsparseDestroyDnMat(UPTKsparseDnMatDescr_t dnMatDescr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDnMatGet(UPTKsparseDnMatDescr_t dnMatDescr,
                 int64_t*             rows,
                 int64_t*             cols,
                 int64_t*             ld,
                 void**               values,
                 UPTKDataType*        type,
                 UPTKsparseOrder_t*     order);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDnMatGetValues(UPTKsparseDnMatDescr_t dnMatDescr,
                       void**               values);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDnMatSetValues(UPTKsparseDnMatDescr_t dnMatDescr,
                       void*                values);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDnMatSetStridedBatch(UPTKsparseDnMatDescr_t dnMatDescr,
                             int                  batchCount,
                             int64_t              batchStride);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDnMatGetStridedBatch(UPTKsparseDnMatDescr_t dnMatDescr,
                             int*                 batchCount,
                             int64_t*             batchStride);

// #############################################################################
// # VECTOR-VECTOR OPERATIONS
// #############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseAxpby(UPTKsparseHandle_t     handle,
              const void*          alpha,
              UPTKsparseSpVecDescr_t vecX,
              const void*          beta,
              UPTKsparseDnVecDescr_t vecY);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseGather(UPTKsparseHandle_t     handle,
               UPTKsparseDnVecDescr_t vecY,
               UPTKsparseSpVecDescr_t vecX);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScatter(UPTKsparseHandle_t     handle,
                UPTKsparseSpVecDescr_t vecX,
                UPTKsparseDnVecDescr_t vecY);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseRot(UPTKsparseHandle_t     handle,
            const void*          c_coeff,
            const void*          s_coeff,
            UPTKsparseSpVecDescr_t vecX,
            UPTKsparseDnVecDescr_t vecY);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpVV_bufferSize(UPTKsparseHandle_t     handle,
                        UPTKsparseOperation_t  opX,
                        UPTKsparseSpVecDescr_t vecX,
                        UPTKsparseDnVecDescr_t vecY,
                        const void*          result,
                        UPTKDataType         computeType,
                        size_t*              bufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpVV(UPTKsparseHandle_t     handle,
             UPTKsparseOperation_t  opX,
             UPTKsparseSpVecDescr_t vecX,
             UPTKsparseDnVecDescr_t vecY,
             void*                result,
             UPTKDataType         computeType,
             void*                externalBuffer);

// #############################################################################
// # SPARSE TO DENSE
// #############################################################################

typedef enum {
    UPTKSPARSE_SPARSETODENSE_ALG_DEFAULT = 0
} UPTKsparseSparseToDenseAlg_t;

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSparseToDense_bufferSize(UPTKsparseHandle_t           handle,
                                 UPTKsparseSpMatDescr_t       matA,
                                 UPTKsparseDnMatDescr_t       matB,
                                 UPTKsparseSparseToDenseAlg_t alg,
                                 size_t*                    bufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSparseToDense(UPTKsparseHandle_t           handle,
                      UPTKsparseSpMatDescr_t       matA,
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
                                 UPTKsparseDnMatDescr_t       matA,
                                 UPTKsparseSpMatDescr_t       matB,
                                 UPTKsparseDenseToSparseAlg_t alg,
                                 size_t*                    bufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDenseToSparse_analysis(UPTKsparseHandle_t           handle,
                               UPTKsparseDnMatDescr_t       matA,
                               UPTKsparseSpMatDescr_t       matB,
                               UPTKsparseDenseToSparseAlg_t alg,
                               void*                      externalBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDenseToSparse_convert(UPTKsparseHandle_t           handle,
                              UPTKsparseDnMatDescr_t       matA,
                              UPTKsparseSpMatDescr_t       matB,
                              UPTKsparseDenseToSparseAlg_t alg,
                              void*                      externalBuffer);

// #############################################################################
// # SPARSE MATRIX-VECTOR MULTIPLICATION
// #############################################################################

typedef enum {
    UPTKSPARSE_MV_ALG_DEFAULT
                        /*UPTKSPARSE_DEPRECATED_ENUM(UPTKSPARSE_SPMV_ALG_DEFAULT)*/ = 0,
    UPTKSPARSE_COOMV_ALG  UPTKSPARSE_DEPRECATED_ENUM(UPTKSPARSE_SPMV_COO_ALG1)    = 1,
    UPTKSPARSE_CSRMV_ALG1 UPTKSPARSE_DEPRECATED_ENUM(UPTKSPARSE_SPMV_CSR_ALG1)    = 2,
    UPTKSPARSE_CSRMV_ALG2 UPTKSPARSE_DEPRECATED_ENUM(UPTKSPARSE_SPMV_CSR_ALG2)    = 3,
    UPTKSPARSE_SPMV_ALG_DEFAULT = 0,
    UPTKSPARSE_SPMV_CSR_ALG1    = 2,
    UPTKSPARSE_SPMV_CSR_ALG2    = 3,
    UPTKSPARSE_SPMV_COO_ALG1    = 1,
    UPTKSPARSE_SPMV_COO_ALG2    = 4
} UPTKsparseSpMVAlg_t;

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMV(UPTKsparseHandle_t     handle,
             UPTKsparseOperation_t  opA,
             const void*          alpha,
             UPTKsparseSpMatDescr_t matA,
             UPTKsparseDnVecDescr_t vecX,
             const void*          beta,
             UPTKsparseDnVecDescr_t vecY,
             UPTKDataType         computeType,
             UPTKsparseSpMVAlg_t    alg,
             void*                externalBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMV_bufferSize(UPTKsparseHandle_t    handle,
                        UPTKsparseOperation_t opA,
                        const void*         alpha,
                        UPTKsparseSpMatDescr_t matA,
                        UPTKsparseDnVecDescr_t vecX,
                        const void*          beta,
                        UPTKsparseDnVecDescr_t vecY,
                        UPTKDataType         computeType,
                        UPTKsparseSpMVAlg_t    alg,
                        size_t*              bufferSize);

// #############################################################################
// # SPARSE TRIANGULAR VECTOR SOLVE
// #############################################################################

typedef enum {
    UPTKSPARSE_SPSV_ALG_DEFAULT = 0,
} UPTKsparseSpSVAlg_t;

struct UPTKsparseSpSVDescr;
typedef struct UPTKsparseSpSVDescr* UPTKsparseSpSVDescr_t;

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSV_createDescr(UPTKsparseSpSVDescr_t* descr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSV_destroyDescr(UPTKsparseSpSVDescr_t descr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSV_bufferSize(UPTKsparseHandle_t     handle,
                        UPTKsparseOperation_t  opA,
                        const void*          alpha,
                        UPTKsparseSpMatDescr_t matA,
                        UPTKsparseDnVecDescr_t vecX,
                        UPTKsparseDnVecDescr_t vecY,
                        UPTKDataType         computeType,
                        UPTKsparseSpSVAlg_t    alg,
                        UPTKsparseSpSVDescr_t  spsvDescr,
                        size_t*              bufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSV_analysis(UPTKsparseHandle_t     handle,
                      UPTKsparseOperation_t  opA,
                      const void*          alpha,
                      UPTKsparseSpMatDescr_t matA,
                      UPTKsparseDnVecDescr_t vecX,
                      UPTKsparseDnVecDescr_t vecY,
                      UPTKDataType         computeType,
                      UPTKsparseSpSVAlg_t    alg,
                      UPTKsparseSpSVDescr_t  spsvDescr,
                      void*                externalBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSV_solve(UPTKsparseHandle_t     handle,
                   UPTKsparseOperation_t  opA,
                   const void*          alpha,
                   UPTKsparseSpMatDescr_t matA,
                   UPTKsparseDnVecDescr_t vecX,
                   UPTKsparseDnVecDescr_t vecY,
                   UPTKDataType         computeType,
                   UPTKsparseSpSVAlg_t    alg,
                   UPTKsparseSpSVDescr_t  spsvDescr);

// #############################################################################
// # SPARSE TRIANGULAR MATRIX SOLVE
// #############################################################################

typedef enum {
    UPTKSPARSE_SPSM_ALG_DEFAULT = 0,
} UPTKsparseSpSMAlg_t;

struct UPTKsparseSpSMDescr;
typedef struct UPTKsparseSpSMDescr* UPTKsparseSpSMDescr_t;

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSM_createDescr(UPTKsparseSpSMDescr_t* descr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSM_destroyDescr(UPTKsparseSpSMDescr_t descr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSM_bufferSize(UPTKsparseHandle_t     handle,
                        UPTKsparseOperation_t  opA,
                        UPTKsparseOperation_t  opB,
                        const void*          alpha,
                        UPTKsparseSpMatDescr_t matA,
                        UPTKsparseDnMatDescr_t matB,
                        UPTKsparseDnMatDescr_t matC,
                        UPTKDataType         computeType,
                        UPTKsparseSpSMAlg_t    alg,
                        UPTKsparseSpSMDescr_t  spsmDescr,
                        size_t*              bufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSM_analysis(UPTKsparseHandle_t     handle,
                        UPTKsparseOperation_t  opA,
                        UPTKsparseOperation_t  opB,
                        const void*          alpha,
                        UPTKsparseSpMatDescr_t matA,
                        UPTKsparseDnMatDescr_t matB,
                        UPTKsparseDnMatDescr_t matC,
                        UPTKDataType         computeType,
                        UPTKsparseSpSMAlg_t    alg,
                        UPTKsparseSpSMDescr_t  spsmDescr,
                        void*                externalBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSM_solve(UPTKsparseHandle_t     handle,
                    UPTKsparseOperation_t  opA,
                    UPTKsparseOperation_t  opB,
                    const void*          alpha,
                    UPTKsparseSpMatDescr_t matA,
                    UPTKsparseDnMatDescr_t matB,
                    UPTKsparseDnMatDescr_t matC,
                    UPTKDataType         computeType,
                    UPTKsparseSpSMAlg_t    alg,
                    UPTKsparseSpSMDescr_t  spsmDescr);

// #############################################################################
// # SPARSE MATRIX-MATRIX MULTIPLICATION
// #############################################################################

typedef enum {
    UPTKSPARSE_MM_ALG_DEFAULT
                        UPTKSPARSE_DEPRECATED_ENUM(UPTKSPARSE_SPMM_ALG_DEFAULT) = 0,
    UPTKSPARSE_COOMM_ALG1 UPTKSPARSE_DEPRECATED_ENUM(UPTKSPARSE_SPMM_COO_ALG1) = 1,
    UPTKSPARSE_COOMM_ALG2 UPTKSPARSE_DEPRECATED_ENUM(UPTKSPARSE_SPMM_COO_ALG2) = 2,
    UPTKSPARSE_COOMM_ALG3 UPTKSPARSE_DEPRECATED_ENUM(UPTKSPARSE_SPMM_COO_ALG3) = 3,
    UPTKSPARSE_CSRMM_ALG1 UPTKSPARSE_DEPRECATED_ENUM(UPTKSPARSE_SPMM_CSR_ALG1) = 4,
    UPTKSPARSE_SPMM_ALG_DEFAULT      = 0,
    UPTKSPARSE_SPMM_COO_ALG1         = 1,
    UPTKSPARSE_SPMM_COO_ALG2         = 2,
    UPTKSPARSE_SPMM_COO_ALG3         = 3,
    UPTKSPARSE_SPMM_COO_ALG4         = 5,
    UPTKSPARSE_SPMM_CSR_ALG1         = 4,
    UPTKSPARSE_SPMM_CSR_ALG2         = 6,
    UPTKSPARSE_SPMM_CSR_ALG3         = 12,
    UPTKSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13
} UPTKsparseSpMMAlg_t;

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMM_bufferSize(UPTKsparseHandle_t     handle,
                        UPTKsparseOperation_t  opA,
                        UPTKsparseOperation_t  opB,
                        const void*          alpha,
                        UPTKsparseSpMatDescr_t matA,
                        UPTKsparseDnMatDescr_t matB,
                        const void*          beta,
                        UPTKsparseDnMatDescr_t matC,
                        UPTKDataType         computeType,
                        UPTKsparseSpMMAlg_t    alg,
                        size_t*              bufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMM_preprocess(UPTKsparseHandle_t      handle,
                        UPTKsparseOperation_t   opA,
                        UPTKsparseOperation_t   opB,
                        const void*           alpha,
                        UPTKsparseSpMatDescr_t  matA,
                        UPTKsparseDnMatDescr_t  matB,
                        const void*           beta,
                        UPTKsparseDnMatDescr_t  matC,
                        UPTKDataType          computeType,
                        UPTKsparseSpMMAlg_t     alg,
                        void*                 externalBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMM(UPTKsparseHandle_t     handle,
             UPTKsparseOperation_t  opA,
             UPTKsparseOperation_t  opB,
             const void*          alpha,
             UPTKsparseSpMatDescr_t matA,
             UPTKsparseDnMatDescr_t matB,
             const void*          beta,
             UPTKsparseDnMatDescr_t matC,
             UPTKDataType         computeType,
             UPTKsparseSpMMAlg_t    alg,
             void*                externalBuffer);

// #############################################################################
// # SPARSE MATRIX - SPARSE MATRIX MULTIPLICATION (SpGEMM)
// #############################################################################

typedef enum {
    UPTKSPARSE_SPGEMM_DEFAULT                 = 0,
    UPTKSPARSE_SPGEMM_CSR_ALG_DETERMINITIC    = 1,
    UPTKSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC = 2
} UPTKsparseSpGEMMAlg_t;

struct UPTKsparseSpGEMMDescr;
typedef struct UPTKsparseSpGEMMDescr* UPTKsparseSpGEMMDescr_t;

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMM_createDescr(UPTKsparseSpGEMMDescr_t* descr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMM_destroyDescr(UPTKsparseSpGEMMDescr_t descr);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMM_workEstimation(UPTKsparseHandle_t      handle,
                              UPTKsparseOperation_t   opA,
                              UPTKsparseOperation_t   opB,
                              const void*           alpha,
                              UPTKsparseSpMatDescr_t  matA,
                              UPTKsparseSpMatDescr_t  matB,
                              const void*           beta,
                              UPTKsparseSpMatDescr_t  matC,
                              UPTKDataType          computeType,
                              UPTKsparseSpGEMMAlg_t   alg,
                              UPTKsparseSpGEMMDescr_t spgemmDescr,
                              size_t*               bufferSize1,
                              void*                 externalBuffer1);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMM_compute(UPTKsparseHandle_t      handle,
                       UPTKsparseOperation_t   opA,
                       UPTKsparseOperation_t   opB,
                       const void*           alpha,
                       UPTKsparseSpMatDescr_t  matA,
                       UPTKsparseSpMatDescr_t  matB,
                       const void*           beta,
                       UPTKsparseSpMatDescr_t  matC,
                       UPTKDataType          computeType,
                       UPTKsparseSpGEMMAlg_t   alg,
                       UPTKsparseSpGEMMDescr_t spgemmDescr,
                       size_t*               bufferSize2,
                       void*                 externalBuffer2);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMM_copy(UPTKsparseHandle_t      handle,
                    UPTKsparseOperation_t   opA,
                    UPTKsparseOperation_t   opB,
                    const void*           alpha,
                    UPTKsparseSpMatDescr_t  matA,
                    UPTKsparseSpMatDescr_t  matB,
                    const void*           beta,
                    UPTKsparseSpMatDescr_t  matC,
                    UPTKDataType          computeType,
                    UPTKsparseSpGEMMAlg_t   alg,
                    UPTKsparseSpGEMMDescr_t spgemmDescr);

// #############################################################################
// # SPARSE MATRIX - SPARSE MATRIX MULTIPLICATION (SpGEMM) STRUCTURE REUSE
// #############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMMreuse_workEstimation(UPTKsparseHandle_t      handle,
                                   UPTKsparseOperation_t   opA,
                                   UPTKsparseOperation_t   opB,
                                   UPTKsparseSpMatDescr_t  matA,
                                   UPTKsparseSpMatDescr_t  matB,
                                   UPTKsparseSpMatDescr_t  matC,
                                   UPTKsparseSpGEMMAlg_t   alg,
                                   UPTKsparseSpGEMMDescr_t spgemmDescr,
                                   size_t*               bufferSize1,
                                   void*                 externalBuffer1);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMMreuse_nnz(UPTKsparseHandle_t      handle,
                        UPTKsparseOperation_t   opA,
                        UPTKsparseOperation_t   opB,
                        UPTKsparseSpMatDescr_t  matA,
                        UPTKsparseSpMatDescr_t  matB,
                        UPTKsparseSpMatDescr_t  matC,
                        UPTKsparseSpGEMMAlg_t   alg,
                        UPTKsparseSpGEMMDescr_t spgemmDescr,
                        size_t*               bufferSize2,
                        void*                 externalBuffer2,
                        size_t*               bufferSize3,
                        void*                 externalBuffer3,
                        size_t*               bufferSize4,
                        void*                 externalBuffer4);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMMreuse_copy(UPTKsparseHandle_t      handle,
                         UPTKsparseOperation_t   opA,
                         UPTKsparseOperation_t   opB,
                         UPTKsparseSpMatDescr_t  matA,
                         UPTKsparseSpMatDescr_t  matB,
                         UPTKsparseSpMatDescr_t  matC,
                         UPTKsparseSpGEMMAlg_t   alg,
                         UPTKsparseSpGEMMDescr_t spgemmDescr,
                         size_t*               bufferSize5,
                         void*                 externalBuffer5);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMMreuse_compute(UPTKsparseHandle_t      handle,
                            UPTKsparseOperation_t   opA,
                            UPTKsparseOperation_t   opB,
                            const void*           alpha,
                            UPTKsparseSpMatDescr_t  matA,
                            UPTKsparseSpMatDescr_t  matB,
                            const void*           beta,
                            UPTKsparseSpMatDescr_t  matC,
                            UPTKDataType          computeType,
                            UPTKsparseSpGEMMAlg_t   alg,
                            UPTKsparseSpGEMMDescr_t spgemmDescr);

// #############################################################################
// # SAMPLED DENSE-DENSE MATRIX MULTIPLICATION
// #############################################################################

UPTKSPARSE_DEPRECATED(UPTKsparseSDDMM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstrainedGeMM(UPTKsparseHandle_t     handle,
                        UPTKsparseOperation_t  opA,
                        UPTKsparseOperation_t  opB,
                        const void*          alpha,
                        UPTKsparseDnMatDescr_t matA,
                        UPTKsparseDnMatDescr_t matB,
                        const void*          beta,
                        UPTKsparseSpMatDescr_t matC,
                        UPTKDataType         computeType,
                        void*                externalBuffer);

UPTKSPARSE_DEPRECATED(UPTKsparseSDDMM)
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstrainedGeMM_bufferSize(UPTKsparseHandle_t     handle,
                                   UPTKsparseOperation_t  opA,
                                   UPTKsparseOperation_t  opB,
                                   const void*          alpha,
                                   UPTKsparseDnMatDescr_t matA,
                                   UPTKsparseDnMatDescr_t matB,
                                   const void*          beta,
                                   UPTKsparseSpMatDescr_t matC,
                                   UPTKDataType         computeType,
                                   size_t*              bufferSize);

typedef enum {
    UPTKSPARSE_SDDMM_ALG_DEFAULT = 0
} UPTKsparseSDDMMAlg_t;

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSDDMM_bufferSize(UPTKsparseHandle_t     handle,
                         UPTKsparseOperation_t  opA,
                         UPTKsparseOperation_t  opB,
                         const void*          alpha,
                         UPTKsparseDnMatDescr_t matA,
                         UPTKsparseDnMatDescr_t matB,
                         const void*          beta,
                         UPTKsparseSpMatDescr_t matC,
                         UPTKDataType         computeType,
                         UPTKsparseSDDMMAlg_t   alg,
                         size_t*              bufferSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSDDMM_preprocess(UPTKsparseHandle_t     handle,
                         UPTKsparseOperation_t  opA,
                         UPTKsparseOperation_t  opB,
                         const void*          alpha,
                         UPTKsparseDnMatDescr_t matA,
                         UPTKsparseDnMatDescr_t matB,
                         const void*          beta,
                         UPTKsparseSpMatDescr_t matC,
                         UPTKDataType         computeType,
                         UPTKsparseSDDMMAlg_t   alg,
                         void*                externalBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSDDMM(UPTKsparseHandle_t     handle,
              UPTKsparseOperation_t  opA,
              UPTKsparseOperation_t  opB,
              const void*          alpha,
              UPTKsparseDnMatDescr_t matA,
              UPTKsparseDnMatDescr_t matB,
              const void*          beta,
              UPTKsparseSpMatDescr_t matC,
              UPTKDataType         computeType,
              UPTKsparseSDDMMAlg_t   alg,
              void*                externalBuffer);

// #############################################################################
// # GENERIC APIs WITH CUSTOM OPERATORS (PREVIEW)
// #############################################################################

struct UPTKsparseSpMMOpPlan;
typedef struct UPTKsparseSpMMOpPlan* UPTKsparseSpMMOpPlan_t;

typedef enum {
    UPTKSPARSE_SPMM_OP_ALG_DEFAULT
} UPTKsparseSpMMOpAlg_t;

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMMOp_createPlan(UPTKsparseHandle_t      handle,
                          UPTKsparseSpMMOpPlan_t* plan,
                          UPTKsparseOperation_t   opA,
                          UPTKsparseOperation_t   opB,
                          UPTKsparseSpMatDescr_t  matA,
                          UPTKsparseDnMatDescr_t  matB,
                          UPTKsparseDnMatDescr_t  matC,
                          UPTKDataType          computeType,
                          UPTKsparseSpMMOpAlg_t   alg,
                          const void*           addOperationNvvmBuffer,
                          size_t                addOperationBufferSize,
                          const void*           mulOperationNvvmBuffer,
                          size_t                mulOperationBufferSize,
                          const void*           epilogueNvvmBuffer,
                          size_t                epilogueBufferSize,
                          size_t*               SpMMWorkspaceSize);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMMOp(UPTKsparseSpMMOpPlan_t plan,
               void*                externalBuffer);

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMMOp_destroyPlan(UPTKsparseSpMMOpPlan_t plan);

//------------------------------------------------------------------------------

#if defined(__cplusplus)
} // extern "C"
#endif // defined(__cplusplus)

#undef UPTKSPARSE_DEPRECATED
#undef UPTKSPARSE_PREVIEW

#endif // !defined(UPTKSPARSE_H_)

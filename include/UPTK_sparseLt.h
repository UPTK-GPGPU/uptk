#if !defined(UPTKSPARSELT_HEADER_)
#define UPTKSPARSELT_HEADER_

#include "UPTKsparse.h"      // UPTKsparseStatus_t

#include <cstddef>         // size_t
#include <driver_types.h>  // UPTKStream_t
#include <library_types.h> // UPTKDataType
#include <stdint.h>        // uint8_t

//##############################################################################
//# UPTKSPARSELT VERSION INFORMATION
//##############################################################################

#define UPTKSPARSELT_VER_MAJOR 0
#define UPTKSPARSELT_VER_MINOR 5
#define UPTKSPARSELT_VER_PATCH 0
#define UPTKSPARSELT_VER_BUILD 1
#define UPTKSPARSELT_VERSION (UPTKSPARSELT_VER_MAJOR * 1000 + \
                            UPTKSPARSELT_VER_MINOR *  100 + \
                            UPTKSPARSELT_VER_PATCH)

// #############################################################################
// # MACRO
// #############################################################################

#if !defined(UPTKSPARSELT_API)
#    if defined(_WIN32)
#        define UPTKSPARSELT_API __stdcall
#    else
#        define UPTKSPARSELT_API
#    endif
#endif

//------------------------------------------------------------------------------

#if defined(__cplusplus)
extern "C" {
#endif // defined(__cplusplus)

//##############################################################################
//# OPAQUE DATA STRUCTURES
//##############################################################################

typedef struct { uint8_t data[13072]; } UPTKsparseLtHandle_t;

typedef struct { uint8_t data[13072]; } UPTKsparseLtMatDescriptor_t;

typedef struct { uint8_t data[13072]; } UPTKsparseLtMatmulDescriptor_t;

typedef struct { uint8_t data[13072]; } UPTKsparseLtMatmulAlgSelection_t;

typedef struct { uint8_t data[13072]; } UPTKsparseLtMatmulPlan_t;

//##############################################################################
//# INITIALIZATION, DESTROY
//##############################################################################

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtInit(UPTKsparseLtHandle_t* handle);

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtDestroy(const UPTKsparseLtHandle_t* handle);

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtGetVersion(const UPTKsparseLtHandle_t* handle,
                     int*                      version);

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtGetProperty(libraryPropertyType propertyType,
                      int*                value);

//##############################################################################
//# MATRIX DESCRIPTOR
//##############################################################################
// Dense Matrix

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtDenseDescriptorInit(const UPTKsparseLtHandle_t*  handle,
                              UPTKsparseLtMatDescriptor_t* matDescr,
                              int64_t                    rows,
                              int64_t                    cols,
                              int64_t                    ld,
                              uint32_t                   alignment,
                              UPTKDataType               valueType,
                              UPTKsparseOrder_t            order);

//------------------------------------------------------------------------------
// Structured Matrix

typedef enum {
    UPTKSPARSELT_SPARSITY_50_PERCENT
} UPTKsparseLtSparsity_t;

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtStructuredDescriptorInit(const UPTKsparseLtHandle_t*  handle,
                                   UPTKsparseLtMatDescriptor_t* matDescr,
                                   int64_t                    rows,
                                   int64_t                    cols,
                                   int64_t                    ld,
                                   uint32_t                   alignment,
                                   UPTKDataType               valueType,
                                   UPTKsparseOrder_t            order,
                                   UPTKsparseLtSparsity_t       sparsity);

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtMatDescriptorDestroy(const UPTKsparseLtMatDescriptor_t* matDescr);

//------------------------------------------------------------------------------

typedef enum {
    UPTKSPARSELT_MAT_NUM_BATCHES,  // READ/WRITE
    UPTKSPARSELT_MAT_BATCH_STRIDE  // READ/WRITE
} UPTKsparseLtMatDescAttribute_t;

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtMatDescSetAttribute(const UPTKsparseLtHandle_t*    handle,
                              UPTKsparseLtMatDescriptor_t*   matmulDescr,
                              UPTKsparseLtMatDescAttribute_t matAttribute,
                              const void*                  data,
                              size_t                       dataSize);

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtMatDescGetAttribute(const UPTKsparseLtHandle_t*        handle,
                              const UPTKsparseLtMatDescriptor_t* matmulDescr,
                              UPTKsparseLtMatDescAttribute_t     matAttribute,
                              void*                            data,
                              size_t                           dataSize);

//##############################################################################
//# MATMUL DESCRIPTOR
//##############################################################################

typedef enum {
    UPTKSPARSE_COMPUTE_16F,
    UPTKSPARSE_COMPUTE_8I,
    UPTKSPARSE_COMPUTE_32I,
    UPTKSPARSE_COMPUTE_TF32,
    UPTKSPARSE_COMPUTE_TF32_FAST
} UPTKsparseComputeType;

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtMatmulDescriptorInit(const UPTKsparseLtHandle_t*        handle,
                               UPTKsparseLtMatmulDescriptor_t*    matmulDescr,
                               UPTKsparseOperation_t              opA,
                               UPTKsparseOperation_t              opB,
                               const UPTKsparseLtMatDescriptor_t* matA,
                               const UPTKsparseLtMatDescriptor_t* matB,
                               const UPTKsparseLtMatDescriptor_t* matC,
                               const UPTKsparseLtMatDescriptor_t* matD,
                               UPTKsparseComputeType              computeType);

//------------------------------------------------------------------------------

typedef enum {
    UPTKSPARSELT_MATMUL_ACTIVATION_RELU,            // READ/WRITE
    UPTKSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND, // READ/WRITE
    UPTKSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD,  // READ/WRITE
    UPTKSPARSELT_MATMUL_ACTIVATION_GELU,            // READ/WRITE
    UPTKSPARSELT_MATMUL_ACTIVATION_GELU_SCALING,    // READ/WRITE
    UPTKSPARSELT_MATMUL_ALPHA_VECTOR_SCALING,       // READ/WRITE
    UPTKSPARSELT_MATMUL_BETA_VECTOR_SCALING,        // READ/WRITE
    UPTKSPARSELT_MATMUL_BIAS_STRIDE,                // READ/WRITE
    UPTKSPARSELT_MATMUL_BIAS_POINTER                // READ/WRITE
} UPTKsparseLtMatmulDescAttribute_t;

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtMatmulDescSetAttribute(const UPTKsparseLtHandle_t*      handle,
                                UPTKsparseLtMatmulDescriptor_t*   matmulDescr,
                                UPTKsparseLtMatmulDescAttribute_t matmulAttribute,
                                const void*                     data,
                                size_t                          dataSize);

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtMatmulDescGetAttribute(
                            const UPTKsparseLtHandle_t*           handle,
                            const UPTKsparseLtMatmulDescriptor_t* matmulDescr,
                            UPTKsparseLtMatmulDescAttribute_t     matmulAttribute,
                            void*                               data,
                            size_t                              dataSize);

//##############################################################################
//# ALGORITHM SELECTION
//##############################################################################

typedef enum {
    UPTKSPARSELT_MATMUL_ALG_DEFAULT
} UPTKsparseLtMatmulAlg_t;

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtMatmulAlgSelectionInit(
                            const UPTKsparseLtHandle_t*           handle,
                            UPTKsparseLtMatmulAlgSelection_t*     algSelection,
                            const UPTKsparseLtMatmulDescriptor_t* matmulDescr,
                            UPTKsparseLtMatmulAlg_t               alg);

typedef enum {
    UPTKSPARSELT_MATMUL_ALG_CONFIG_ID,     // READ/WRITE
    UPTKSPARSELT_MATMUL_ALG_CONFIG_MAX_ID, // READ-ONLY
    UPTKSPARSELT_MATMUL_SEARCH_ITERATIONS, // READ/WRITE
    UPTKSPARSELT_MATMUL_SPLIT_K,           // READ/WRITE
    UPTKSPARSELT_MATMUL_SPLIT_K_MODE,      // READ/WRITE
    UPTKSPARSELT_MATMUL_SPLIT_K_BUFFERS    // READ/WRITE
} UPTKsparseLtMatmulAlgAttribute_t;

typedef enum {
    UPTKSPARSELT_INVALID_MODE             = 0,
    UPTKSPARSELT_SPLIT_K_MODE_ONE_KERNEL  = 1,
    UPTKSPARSELT_SPLIT_K_MODE_TWO_KERNELS = 2
} UPTKsparseLtSplitKMode_t;

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtMatmulAlgSetAttribute(const UPTKsparseLtHandle_t*       handle,
                                UPTKsparseLtMatmulAlgSelection_t* algSelection,
                                UPTKsparseLtMatmulAlgAttribute_t  attribute,
                                const void*                     data,
                                size_t                          dataSize);

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtMatmulAlgGetAttribute(
                            const UPTKsparseLtHandle_t*             handle,
                            const UPTKsparseLtMatmulAlgSelection_t* algSelection,
                            UPTKsparseLtMatmulAlgAttribute_t        attribute,
                            void*                                 data,
                            size_t                                dataSize);

//##############################################################################
//# MATMUL PLAN
//##############################################################################

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtMatmulGetWorkspace(
                        const UPTKsparseLtHandle_t*     handle,
                        const UPTKsparseLtMatmulPlan_t* plan,
                        size_t*                       workspaceSize);

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtMatmulPlanInit(const UPTKsparseLtHandle_t*             handle,
                         UPTKsparseLtMatmulPlan_t*               plan,
                         const UPTKsparseLtMatmulDescriptor_t*   matmulDescr,
                         const UPTKsparseLtMatmulAlgSelection_t* algSelection);

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtMatmulPlanDestroy(const UPTKsparseLtMatmulPlan_t* plan);

//##############################################################################
//# MATMUL EXECUTION
//##############################################################################

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtMatmul(const UPTKsparseLtHandle_t*     handle,
                 const UPTKsparseLtMatmulPlan_t* plan,
                 const void*                   alpha,
                 const void*                   d_A,
                 const void*                   d_B,
                 const void*                   beta,
                 const void*                   d_C,
                 void*                         d_D,
                 void*                         workspace,
                 UPTKStream_t*                 streams,
                 int32_t                       numStreams);

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtMatmulSearch(const UPTKsparseLtHandle_t* handle,
                       UPTKsparseLtMatmulPlan_t*   plan,
                       const void*               alpha,
                       const void*               d_A,
                       const void*               d_B,
                       const void*               beta,
                       const void*               d_C,
                       void*                     d_D,
                       void*                     workspace,
                       // void*                     device_buf,
                       UPTKStream_t*             streams,
                       int32_t                   numStreams);

//##############################################################################
//# HELPER ROUTINES
//##############################################################################
// PRUNING

typedef enum {
    UPTKSPARSELT_PRUNE_SPMMA_TILE  = 0,
    UPTKSPARSELT_PRUNE_SPMMA_STRIP = 1
} UPTKsparseLtPruneAlg_t;

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtSpMMAPrune(const UPTKsparseLtHandle_t*           handle,
                     const UPTKsparseLtMatmulDescriptor_t* matmulDescr,
                     const void*                         d_in,
                     void*                               d_out,
                     UPTKsparseLtPruneAlg_t                pruneAlg,
                     UPTKStream_t                        stream);

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtSpMMAPruneCheck(const UPTKsparseLtHandle_t*           handle,
                          const UPTKsparseLtMatmulDescriptor_t* matmulDescr,
                          const void*                         d_in,
                          int*                                valid,
                          UPTKStream_t                        stream);

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtSpMMAPrune2(const UPTKsparseLtHandle_t*        handle,
                      const UPTKsparseLtMatDescriptor_t* sparseMatDescr,
                      int                              isSparseA,
                      UPTKsparseOperation_t              op,
                      const void*                      d_in,
                      void*                            d_out,
                      UPTKsparseLtPruneAlg_t             pruneAlg,
                      UPTKStream_t                     stream);

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtSpMMAPruneCheck2(const UPTKsparseLtHandle_t*        handle,
                           const UPTKsparseLtMatDescriptor_t* sparseMatDescr,
                           int                              isSparseA,
                           UPTKsparseOperation_t              op,
                           const void*                      d_in,
                           int*                             d_valid,
                           UPTKStream_t                     stream);

//------------------------------------------------------------------------------
// COMPRESSION

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtSpMMACompressedSize(
                        const UPTKsparseLtHandle_t*     handle,
                        const UPTKsparseLtMatmulPlan_t* plan,
                        size_t*                       compressedSize,
                        size_t*                       compressedBufferSize);

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtSpMMACompress(const UPTKsparseLtHandle_t*     handle,
                        const UPTKsparseLtMatmulPlan_t* plan,
                        const void*                   d_dense,
                        void*                         d_compressed,
                        void*                         d_compressed_buffer,
                        UPTKStream_t                  stream);

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtSpMMACompressedSize2(
                        const UPTKsparseLtHandle_t*        handle,
                        const UPTKsparseLtMatDescriptor_t* sparseMatDescr,
                        size_t*                          compressedSize,
                        size_t*                          compressedBufferSize);

UPTKsparseStatus_t UPTKSPARSELT_API
UPTKsparseLtSpMMACompress2(const UPTKsparseLtHandle_t*        handle,
                         const UPTKsparseLtMatDescriptor_t* sparseMatDescr,
                         int                              isSparseA,
                         UPTKsparseOperation_t              op,
                         const void*                      d_dense,
                         void*                            d_compressed,
                         void*                            d_compressed_buffer,
                         UPTKStream_t                     stream);

//==============================================================================
//==============================================================================

#if defined(__cplusplus)
}
#endif // defined(__cplusplus)

#endif // !defined(UPTKSPARSELT_HEADER_)

// for comment out UPTK code
#ifndef __DRIVER1_HPP__
#define __DRIVER1_HPP__

#define __UPTK_CUDA_PLATFORM_NVIDIA__

#include <nvrtc.h>
#include <cuda.h>
#include <cudaProfiler.h>
#include <cudaGL.h>
#include <stdio.h>
#include <algorithm>
#include <string.h>

#include <UPTK.h>
#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#define ERROR_INVALID_ENUM() do{printf("Error invalid enum. Fun: %s, para: %d\n", __FUNCTION__, para); abort(); }while(0)

/*
 * UPTK type aliases for the driver wrapper layer.
 *
 * These types are only consumed by the internal driver_fun_convert.cpp
 * wrappers. We expose them as plain typedef aliases of the underlying
 * CUDA driver types so that the wrapper signatures can read in UPTK
 * naming style while still calling cu* entry points without
 * incompatible-type errors.
 */
/* ---- Types available on all supported CUDA versions ---- */
typedef CUaddress_mode                              UPTKaddress_mode;
typedef CUarrayMapInfo                              UPTKarrayMapInfo;
typedef CUdevice_P2PAttribute                       UPTKdevice_P2PAttribute;
typedef CUdevprop                                   UPTKdevprop;
typedef CUexecAffinityParam                         UPTKexecAffinityParam;
typedef CUexecAffinityType                          UPTKexecAffinityType;
typedef CUfilter_mode                               UPTKfilter_mode;
typedef CUfunc_cache                                UPTKfunc_cache;
typedef CUGLDeviceList                              UPTKGLDeviceList;
typedef CUgraphMem_attribute                        UPTKgraphMem_attribute;
typedef CUhostFn                                    UPTKhostFn;
typedef CUmem_advise                                UPTKmem_advise;
typedef CUmem_range_attribute                       UPTKmem_range_attribute;
typedef CUmemAccess_flags                           UPTKmemAccess_flags;
typedef CUmemAllocationGranularity_flags            UPTKmemAllocationGranularity_flags;
typedef CUmemAllocationProp                         UPTKmemAllocationProp;
typedef CUmemGenericAllocationHandle                UPTKmemGenericAllocationHandle;
typedef CUmemPool_attribute                         UPTKmemPool_attribute;
typedef CUmemRangeHandleType                        UPTKmemRangeHandleType;
typedef CUmoduleLoadingMode                         UPTKmoduleLoadingMode;
typedef CUoccupancyB2DSize                          UPTKoccupancyB2DSize;
typedef CUoutput_mode                               UPTKoutput_mode;
typedef CUpointer_attribute                         UPTKpointer_attribute;
typedef CUsharedconfig                              UPTKsharedconfig;
typedef CUstreamBatchMemOpParams                    UPTKstreamBatchMemOpParams;
typedef CUstreamCallback                            UPTKstreamCallback;
typedef CUsurfref                                   UPTKsurfref;
typedef CUsurfObject                                UPTKSurfaceObject_t;
typedef CUtexObject                                 UPTKTextureObject_t;
typedef CUtexref                                    UPTKtexref;

typedef CUDA_ARRAY_MEMORY_REQUIREMENTS              UPTK_ARRAY_MEMORY_REQUIREMENTS;
typedef CUDA_ARRAY_SPARSE_PROPERTIES                UPTK_ARRAY_SPARSE_PROPERTIES;
typedef CUDA_ARRAY3D_DESCRIPTOR                     UPTK_ARRAY3D_DESCRIPTOR;
typedef CUDA_BATCH_MEM_OP_NODE_PARAMS               UPTK_BATCH_MEM_OP_NODE_PARAMS;
typedef CUDA_EXT_SEM_SIGNAL_NODE_PARAMS             UPTK_EXT_SEM_SIGNAL_NODE_PARAMS;
typedef CUDA_EXT_SEM_WAIT_NODE_PARAMS               UPTK_EXT_SEM_WAIT_NODE_PARAMS;
typedef CUDA_EXTERNAL_MEMORY_BUFFER_DESC            UPTK_EXTERNAL_MEMORY_BUFFER_DESC;
typedef CUDA_EXTERNAL_MEMORY_HANDLE_DESC            UPTK_EXTERNAL_MEMORY_HANDLE_DESC;
typedef CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC   UPTK_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC;
typedef CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC         UPTK_EXTERNAL_SEMAPHORE_HANDLE_DESC;
typedef CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS       UPTK_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS;
typedef CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS         UPTK_EXTERNAL_SEMAPHORE_WAIT_PARAMS;
typedef CUDA_HOST_NODE_PARAMS                       UPTK_HOST_NODE_PARAMS;
typedef CUDA_KERNEL_NODE_PARAMS                     UPTK_KERNEL_NODE_PARAMS;
typedef CUDA_LAUNCH_PARAMS                          UPTK_LAUNCH_PARAMS;
typedef CUDA_MEM_ALLOC_NODE_PARAMS                  UPTK_MEM_ALLOC_NODE_PARAMS;
typedef CUDA_MEMCPY2D                               UPTK_MEMCPY2D;
typedef CUDA_MEMCPY3D                               UPTK_MEMCPY3D;
typedef CUDA_MEMCPY3D_PEER                          UPTK_MEMCPY3D_PEER;
typedef CUDA_MEMSET_NODE_PARAMS                     UPTK_MEMSET_NODE_PARAMS;
typedef CUDA_RESOURCE_DESC                          UPTK_RESOURCE_DESC;
typedef CUDA_RESOURCE_VIEW_DESC                     UPTK_RESOURCE_VIEW_DESC;
typedef CUDA_TEXTURE_DESC                           UPTK_TEXTURE_DESC;

/* ---- Types that only exist starting CUDA 12.x ---- *
 *
 * DTK 25.04.1 ships CUDA 11 compatibility headers and does not declare any
 * of these. We compile the corresponding wrapper functions out of the
 * translation unit as well (search for the same CUDA_VERSION guard in
 * driver_fun_convert.cpp).
 */
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
typedef CUDA_GRAPH_INSTANTIATE_PARAMS               UPTK_GRAPH_INSTANTIATE_PARAMS;
typedef CUasyncCallback                             UPTKAsyncCallback;
typedef CUasyncCallbackHandle                       UPTKasyncCallbackHandle;
typedef CUcoredumpSettings                          UPTKcoredumpSettings;
typedef CUctxCreateParams                           UPTKctxCreateParams;
typedef CUdevResource                               UPTKdevResource;
typedef CUdevResourceDesc                           UPTKdevResourceDesc;
typedef CUdevResourceType                           UPTKdevResourceType;
typedef CUdriverProcAddressQueryResult              UPTKdriverProcAddressQueryResult;
typedef CUfunctionLoadingState                      UPTKfunctionLoadingState;
typedef CUgraphEdgeData                             UPTKGraphEdgeData;
typedef CUgraphExecUpdateResultInfo                 UPTKGraphExecUpdateResultInfo;
typedef CUgreenCtx                                  UPTKgreenCtx;
typedef CUkernel                                    UPTKKernel_t;
typedef CUlibrary                                   UPTKlibrary;
typedef CUlibraryOption                             UPTKlibraryOption;
typedef CUmulticastGranularity_flags                UPTKmulticastGranularity_flags;
typedef CUmulticastObjectProp                       UPTKmulticastObjectProp;
typedef CUtensorMap                                 UPTKtensorMap;
typedef CUtensorMapDataType                         UPTKtensorMapDataType;
typedef CUtensorMapFloatOOBfill                     UPTKtensorMapFloatOOBfill;
typedef CUtensorMapInterleave                       UPTKtensorMapInterleave;
typedef CUtensorMapL2promotion                      UPTKtensorMapL2promotion;
typedef CUtensorMapSwizzle                          UPTKtensorMapSwizzle;
#endif /* CUDA_VERSION >= 12000 */

enum UPTKError CUresultToUPTKError(CUresult para);
#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif // __DRIVER1_HPP__

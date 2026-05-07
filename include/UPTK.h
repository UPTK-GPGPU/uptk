#ifndef __UPTK_RUNTIME_API_H
#define __UPTK_RUNTIME_API_H
//#include <stddef.h>
#include "cuda_runtime.h"
#include "cuda.h"

#include <UPTK_runtime_api.h>
#if defined(__cplusplus)
extern "C"
{
#endif /* __cplusplus */

      // 数据类型定义
       typedef int UPTKdevice_v1;                                     /**< UPTK device */
       typedef UPTKdevice_v1 UPTKdevice;                                /**< UPTK device */
       typedef  unsigned long long UPTKGraphConditionalHandle;
       typedef struct UPTKctx_st *UPTKcontext;                          /**< A regular context handle */

   #ifdef __QNX__
   #if (__GNUC__ == 4 && __GNUC_MINOR__ >= 7)
       typedef unsigned size_t;
   #endif
   #endif

    /** \cond impl_private */
#if !defined(__dv)

#if defined(__cplusplus)

#define __dv(v) \
        = v

#else /* __cplusplus */

#define __dv(v)

#endif /* __cplusplus */

#endif /* !__dv */
/** \endcond impl_private */

#if defined(__CUDACC__)
#define UPTKStream_t cudaStream_t
#elif defined(__HIPCC__)
#define UPTKStream_t hipStream_t
#else
// #define UPTKStream_t void* 
#define UPTKStream_t cudaStream_t
#endif


#define UPTKGetDeviceProperties UPTKGetDeviceProperties_v2

 /** \cond impl_private */
 #if defined(__DOXYGEN_ONLY__) || defined(UPTK_ENABLE_DEPRECATED)
 #define __UPTK_DEPRECATED
 #elif defined(_MSC_VER)
 #define __UPTK_DEPRECATED __declspec(deprecated)
 #elif defined(__GNUC__)
 #define __UPTK_DEPRECATED __attribute__((deprecated))
 #else
 #define __UPTK_DEPRECATED
 #endif
 /** \endcond impl_private */
 
 /*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
#ifdef _WIN32
#define UPTKRT_CB __stdcall
#else
#define UPTKRT_CB
#endif

 #ifndef UPTK_UUID_HAS_BEEN_DEFINED
 #define UPTK_UUID_HAS_BEEN_DEFINED
 struct  UPTKuuid_st {
     char bytes[16];
 };
 typedef  struct UPTKuuid_st UPTKuuid;
 #endif
 
 #define UPTKDevicePropDontCare                                 \
         {                                                      \
           {'\0'},    /* char         name[256];               */ \
           {{0}},     /* UPTKUUID_t   uuid;                    */ \
           {'\0'},    /* char         luid[8];                 */ \
           0,         /* unsigned int luidDeviceNodeMask       */ \
           0,         /* size_t       totalGlobalMem;          */ \
           0,         /* size_t       sharedMemPerBlock;       */ \
           0,         /* int          regsPerBlock;            */ \
           0,         /* int          warpSize;                */ \
           0,         /* size_t       memPitch;                */ \
           0,         /* int          maxThreadsPerBlock;      */ \
           {0, 0, 0}, /* int          maxThreadsDim[3];        */ \
           {0, 0, 0}, /* int          maxGridSize[3];          */ \
           0,         /* int          clockRate;               */ \
           0,         /* size_t       totalConstMem;           */ \
           -1,        /* int          major;                   */ \
           -1,        /* int          minor;                   */ \
           0,         /* size_t       textureAlignment;        */ \
           0,         /* size_t       texturePitchAlignment    */ \
           -1,        /* int          deviceOverlap;           */ \
           0,         /* int          multiProcessorCount;     */ \
           0,         /* int          kernelExecTimeoutEnabled */ \
           0,         /* int          integrated               */ \
           0,         /* int          canMapHostMemory         */ \
           0,         /* int          computeMode              */ \
           0,         /* int          maxTexture1D             */ \
           0,         /* int          maxTexture1DMipmap       */ \
           0,         /* int          maxTexture1DLinear       */ \
           {0, 0},    /* int          maxTexture2D[2]          */ \
           {0, 0},    /* int          maxTexture2DMipmap[2]    */ \
           {0, 0, 0}, /* int          maxTexture2DLinear[3]    */ \
           {0, 0},    /* int          maxTexture2DGather[2]    */ \
           {0, 0, 0}, /* int          maxTexture3D[3]          */ \
           {0, 0, 0}, /* int          maxTexture3DAlt[3]       */ \
           0,         /* int          maxTextureCubemap        */ \
           {0, 0},    /* int          maxTexture1DLayered[2]   */ \
           {0, 0, 0}, /* int          maxTexture2DLayered[3]   */ \
           {0, 0},    /* int          maxTextureCubemapLayered[2] */ \
           0,         /* int          maxSurface1D             */ \
           {0, 0},    /* int          maxSurface2D[2]          */ \
           {0, 0, 0}, /* int          maxSurface3D[3]          */ \
           {0, 0},    /* int          maxSurface1DLayered[2]   */ \
           {0, 0, 0}, /* int          maxSurface2DLayered[3]   */ \
           0,         /* int          maxSurfaceCubemap        */ \
           {0, 0},    /* int          maxSurfaceCubemapLayered[2] */ \
           0,         /* size_t       surfaceAlignment         */ \
           0,         /* int          concurrentKernels        */ \
           0,         /* int          ECCEnabled               */ \
           0,         /* int          pciBusID                 */ \
           0,         /* int          pciDeviceID              */ \
           0,         /* int          pciDomainID              */ \
           0,         /* int          tccDriver                */ \
           0,         /* int          asyncEngineCount         */ \
           0,         /* int          unifiedAddressing        */ \
           0,         /* int          memoryClockRate          */ \
           0,         /* int          memoryBusWidth           */ \
           0,         /* int          l2CacheSize              */ \
           0,         /* int          persistingL2CacheMaxSize   */ \
           0,         /* int          maxThreadsPerMultiProcessor */ \
           0,         /* int          streamPrioritiesSupported */ \
           0,         /* int          globalL1CacheSupported   */ \
           0,         /* int          localL1CacheSupported    */ \
           0,         /* size_t       sharedMemPerMultiprocessor; */ \
           0,         /* int          regsPerMultiprocessor;   */ \
           0,         /* int          managedMemory            */ \
           0,         /* int          isMultiGpuBoard          */ \
           0,         /* int          multiGpuBoardGroupID     */ \
           0,         /* int          hostNativeAtomicSupported */ \
           0,         /* int          singleToDoublePrecisionPerfRatio */ \
           0,         /* int          pageableMemoryAccess     */ \
           0,         /* int          concurrentManagedAccess  */ \
           0,         /* int          computePreemptionSupported */ \
           0,         /* int          canUseHostPointerForRegisteredMem */ \
           0,         /* int          cooperativeLaunch */ \
           0,         /* int          cooperativeMultiDeviceLaunch */ \
           0,         /* size_t       sharedMemPerBlockOptin */ \
           0,         /* int          pageableMemoryAccessUsesHostPageTables */ \
           0,         /* int          directManagedMemAccessFromHost */ \
           0,         /* int          accessPolicyMaxWindowSize */ \
           0,         /* size_t       reservedSharedMemPerBlock */ \
         } /**< Empty device properties */
 
 /**
  * UPTK IPC Handle Size
  */
 #define UPTK_IPC_HANDLE_SIZE 64
};

    typedef enum UPTKError UPTKError_t;
    extern __host__ UPTKError_t  UPTKDriverGetVersionD(int * driverVersion);
    extern __host__  UPTKError_t  UPTKFuncSetAttribute(const void * func,enum UPTKFuncAttribute attr,int value);
    extern __host__ UPTKError_t UPTKMalloc(void **devPtr, size_t size);
    extern __host__ UPTKError_t UPTKMemcpy(void *dst, const void *src, size_t count, enum UPTKMemcpyKind kind);
    extern __host__ UPTKError_t UPTKDeviceSynchronize(void);
    extern __host__ UPTKError_t UPTKFree(void *devPtr);
    extern UPTKError_t __UPTKPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, UPTKStream_t stream);
    extern UPTKError_t __UPTKPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, UPTKStream_t *stream);
    extern __host__ UPTKError_t  UPTKStreamCreate(UPTKStream_t * pStream);
    extern __host__ UPTKError_t UPTKStreamCreateWithFlags(UPTKStream_t * pStream,unsigned int flags);
    extern __host__ UPTKError_t  UPTKStreamDestroy(UPTKStream_t stream);
    extern __host__ UPTKError_t UPTKStreamSynchronize(UPTKStream_t stream);
    extern __host__ UPTKError_t UPTKStreamWaitEvent(UPTKStream_t stream, UPTKEvent_t event, unsigned int flags);
    extern __host__ UPTKError_t UPTKStreamBeginCapture(UPTKStream_t stream, enum UPTKStreamCaptureMode mode);
    extern __host__ UPTKError_t UPTKStreamEndCapture(UPTKStream_t stream, UPTKGraph_t *pGraph);
    extern __host__ UPTKError_t UPTKStreamIsCapturing(UPTKStream_t stream, enum UPTKStreamCaptureStatus *pCaptureStatus);
    extern __host__ UPTKError_t UPTKEventCreate(UPTKEvent_t *event);
    extern __host__ UPTKError_t UPTKEventCreateWithFlags(UPTKEvent_t *event, unsigned int flags);
    extern __host__ UPTKError_t UPTKEventDestroy(UPTKEvent_t event);
    extern __host__ UPTKError_t UPTKEventSynchronize(UPTKEvent_t event);
    extern __host__ UPTKError_t UPTKEventQuery(UPTKEvent_t event);
    extern __host__ UPTKError_t UPTKIpcGetMemHandle(UPTKIpcMemHandle_t *handle, void *devPtr);
    extern __host__ UPTKError_t UPTKIpcOpenMemHandleD(CUdeviceptr * devPtr, UPTKIpcMemHandle_t handle, unsigned int flags);
    extern __host__ UPTKError_t UPTKIpcCloseMemHandle(void *devPtr);
    extern __host__ UPTKError_t UPTKChooseDevice(int *device, const struct UPTKDeviceProp *prop);
    extern __host__ UPTKError_t UPTKSetDevice(int device);
    extern __host__ UPTKError_t UPTKGetDevice(int *device);
    extern __host__ UPTKError_t UPTKDeviceReset(void);
    extern __host__ UPTKError_t UPTKGraphCreate(UPTKGraph_t *pGraph, unsigned int flags);
    extern __host__ UPTKError_t UPTKGraphClone(UPTKGraph_t *pGraphClone, UPTKGraph_t originalGraph);
    extern __host__ UPTKError_t UPTKGraphDestroy(UPTKGraph_t graph);
    extern __host__ UPTKError_t UPTKGraphAddDependencies(UPTKGraph_t graph, const UPTKGraphNode_t *from, const UPTKGraphNode_t *to, size_t numDependencies);
    extern __host__ UPTKError_t UPTKGraphRemoveDependencies(UPTKGraph_t graph, const UPTKGraphNode_t *from, const UPTKGraphNode_t *to, size_t numDependencies);
    extern __host__ UPTKError_t UPTKGraphInstantiate(UPTKGraphExec_t *pGraphExec, UPTKGraph_t graph, UPTKGraphNode_t *pErrorNode, char *pLogBuffer, size_t bufferSize);
    extern __host__ UPTKError_t UPTKGraphLaunch(UPTKGraphExec_t graphExec, UPTKStream_t stream);
    extern __host__ UPTKError_t UPTKGraphExecUpdate(UPTKGraphExec_t hGraphExec, UPTKGraph_t hGraph, UPTKGraphNode_t *hErrorNode_out, enum UPTKGraphExecUpdateResult *updateResult_out);
    extern __host__ UPTKError_t UPTKLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, UPTKStream_t stream);
    extern __host__ UPTKError_t UPTKFuncGetAttributes(struct UPTKFuncAttributes *attr, const void *func);
    extern __host__ UPTKError_t UPTKStreamGetDevice(UPTKStream_t stream, UPTKdevice *device);
    extern __host__ UPTKError_t UPTKStreamCreateWithPriority(UPTKStream_t *pStream, unsigned int flags, int priority);
    extern __host__ UPTKError_t UPTKMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count);
    extern __host__ UPTKError_t UPTKMallocAsync(void **devPtr, size_t size, UPTKStream_t hStream);
    extern __host__ UPTKError_t UPTKGraphNodeGetType(UPTKGraphNode_t node, enum UPTKGraphNodeType *pType);
    extern __host__ UPTKError_t UPTKGraphGetNodes(UPTKGraph_t graph, UPTKGraphNode_t *nodes, size_t *numNodes);
    extern __host__ UPTKError_t UPTKGraphExecDestroy(UPTKGraphExec_t graphExec);
    extern __host__ UPTKError_t UPTKGraphDestroyNode(UPTKGraphNode_t node);
    extern __host__ UPTKError_t UPTKGraphAddNode(UPTKGraphNode_t *pGraphNode, UPTKGraph_t graph, const UPTKGraphNode_t *pDependencies, size_t numDependencies, struct UPTKGraphNodeParams *nodeParams);
    extern __host__ UPTKError_t UPTKGetLastError(void);
    extern CUresult *UPTKGetErrorStringD(UPTKError_t error);
    extern __host__ const char *UPTKGetErrorName(UPTKError_t error);
    extern __host__ UPTKError_t UPTKGetDeviceCount(int *count);
    extern __host__ UPTKError_t UPTKFreeAsync(void *devPtr, UPTKStream_t hStream);
    extern __host__ UPTKError_t UPTKEventElapsedTime(float *ms, UPTKEvent_t start, UPTKEvent_t end);
    extern __host__ UPTKError_t UPTKDeviceGetAttribute(int *value, enum UPTKDeviceAttr attr, int device);
    extern __host__ UPTKError_t UPTKDeviceEnablePeerAccess(int peerDevice, unsigned int flags);

// Based on the newly added application interface
    extern __host__ UPTKError_t  UPTKMemset(void *devPtr, int value, size_t count);
    extern __host__ UPTKError_t  UPTKMallocHost(void **ptr, size_t size);
    extern __host__ UPTKError_t  UPTKFreeHost(void *ptr);
    extern __host__ UPTKError_t  UPTKOccupancyMaxPotentialBlockSize_internal(int* gridSize, int* blockSize, const void* f, size_t dynSharedMemPerBlk, int blockSizeLimit);
    extern __host__ UPTKError_t  UPTKHostAlloc(void **pHost, size_t size, unsigned int flags);
    extern __host__ UPTKError_t  UPTKGetDeviceProperties(struct UPTKDeviceProp *prop, int device);


/**
 * UPTK device pointer
 * UPTKdeviceptr is defined as an unsigned integer type whose size matches the size of a pointer on the target platform.
 */
#if defined(_WIN64) || defined(__LP64__)
typedef unsigned long long UPTKdeviceptr;
#else
typedef unsigned int UPTKdeviceptr;
#endif

/**
 * Array formats
 */
typedef enum UPTKarray_format_enum {
    UPTK_AD_FORMAT_UNSIGNED_INT8  = 0x01, /**< Unsigned 8-bit integers */
    UPTK_AD_FORMAT_UNSIGNED_INT16 = 0x02, /**< Unsigned 16-bit integers */
    UPTK_AD_FORMAT_UNSIGNED_INT32 = 0x03, /**< Unsigned 32-bit integers */
    UPTK_AD_FORMAT_SIGNED_INT8    = 0x08, /**< Signed 8-bit integers */
    UPTK_AD_FORMAT_SIGNED_INT16   = 0x09, /**< Signed 16-bit integers */
    UPTK_AD_FORMAT_SIGNED_INT32   = 0x0a, /**< Signed 32-bit integers */
    UPTK_AD_FORMAT_HALF           = 0x10, /**< 16-bit floating point */
    UPTK_AD_FORMAT_FLOAT          = 0x20, /**< 32-bit floating point */
    UPTK_AD_FORMAT_NV12           = 0xb0, /**< 8-bit YUV planar format, with 4:2:0 sampling */
    UPTK_AD_FORMAT_UNORM_INT8X1   = 0xc0, /**< 1 channel unsigned 8-bit normalized integer */
    UPTK_AD_FORMAT_UNORM_INT8X2   = 0xc1, /**< 2 channel unsigned 8-bit normalized integer */
    UPTK_AD_FORMAT_UNORM_INT8X4   = 0xc2, /**< 4 channel unsigned 8-bit normalized integer */
    UPTK_AD_FORMAT_UNORM_INT16X1  = 0xc3, /**< 1 channel unsigned 16-bit normalized integer */
    UPTK_AD_FORMAT_UNORM_INT16X2  = 0xc4, /**< 2 channel unsigned 16-bit normalized integer */
    UPTK_AD_FORMAT_UNORM_INT16X4  = 0xc5, /**< 4 channel unsigned 16-bit normalized integer */
    UPTK_AD_FORMAT_SNORM_INT8X1   = 0xc6, /**< 1 channel signed 8-bit normalized integer */
    UPTK_AD_FORMAT_SNORM_INT8X2   = 0xc7, /**< 2 channel signed 8-bit normalized integer */
    UPTK_AD_FORMAT_SNORM_INT8X4   = 0xc8, /**< 4 channel signed 8-bit normalized integer */
    UPTK_AD_FORMAT_SNORM_INT16X1  = 0xc9, /**< 1 channel signed 16-bit normalized integer */
    UPTK_AD_FORMAT_SNORM_INT16X2  = 0xca, /**< 2 channel signed 16-bit normalized integer */
    UPTK_AD_FORMAT_SNORM_INT16X4  = 0xcb, /**< 4 channel signed 16-bit normalized integer */
    UPTK_AD_FORMAT_BC1_UNORM      = 0x91, /**< 4 channel unsigned normalized block-compressed (BC1 compression) format */
    UPTK_AD_FORMAT_BC1_UNORM_SRGB = 0x92, /**< 4 channel unsigned normalized block-compressed (BC1 compression) format with sRGB encoding*/
    UPTK_AD_FORMAT_BC2_UNORM      = 0x93, /**< 4 channel unsigned normalized block-compressed (BC2 compression) format */
    UPTK_AD_FORMAT_BC2_UNORM_SRGB = 0x94, /**< 4 channel unsigned normalized block-compressed (BC2 compression) format with sRGB encoding*/
    UPTK_AD_FORMAT_BC3_UNORM      = 0x95, /**< 4 channel unsigned normalized block-compressed (BC3 compression) format */
    UPTK_AD_FORMAT_BC3_UNORM_SRGB = 0x96, /**< 4 channel unsigned normalized block-compressed (BC3 compression) format with sRGB encoding*/
    UPTK_AD_FORMAT_BC4_UNORM      = 0x97, /**< 1 channel unsigned normalized block-compressed (BC4 compression) format */
    UPTK_AD_FORMAT_BC4_SNORM      = 0x98, /**< 1 channel signed normalized block-compressed (BC4 compression) format */
    UPTK_AD_FORMAT_BC5_UNORM      = 0x99, /**< 2 channel unsigned normalized block-compressed (BC5 compression) format */
    UPTK_AD_FORMAT_BC5_SNORM      = 0x9a, /**< 2 channel signed normalized block-compressed (BC5 compression) format */
    UPTK_AD_FORMAT_BC6H_UF16      = 0x9b, /**< 3 channel unsigned half-float block-compressed (BC6H compression) format */
    UPTK_AD_FORMAT_BC6H_SF16      = 0x9c, /**< 3 channel signed half-float block-compressed (BC6H compression) format */
    UPTK_AD_FORMAT_BC7_UNORM      = 0x9d, /**< 4 channel unsigned normalized block-compressed (BC7 compression) format */
    UPTK_AD_FORMAT_BC7_UNORM_SRGB = 0x9e, /**< 4 channel unsigned normalized block-compressed (BC7 compression) format with sRGB encoding */
    UPTK_AD_FORMAT_P010           = 0x9f, /**< 10-bit YUV planar format, with 4:2:0 sampling */
    UPTK_AD_FORMAT_P016           = 0xa1, /**< 16-bit YUV planar format, with 4:2:0 sampling */
    UPTK_AD_FORMAT_NV16           = 0xa2, /**< 8-bit YUV planar format, with 4:2:2 sampling */
    UPTK_AD_FORMAT_P210           = 0xa3, /**< 10-bit YUV planar format, with 4:2:2 sampling */
    UPTK_AD_FORMAT_P216           = 0xa4, /**< 16-bit YUV planar format, with 4:2:2 sampling */
    UPTK_AD_FORMAT_YUY2           = 0xa5, /**< 2 channel, 8-bit YUV packed planar format, with 4:2:2 sampling */
    UPTK_AD_FORMAT_Y210           = 0xa6, /**< 2 channel, 10-bit YUV packed planar format, with 4:2:2 sampling */
    UPTK_AD_FORMAT_Y216           = 0xa7, /**< 2 channel, 16-bit YUV packed planar format, with 4:2:2 sampling */
    UPTK_AD_FORMAT_AYUV           = 0xa8, /**< 4 channel, 8-bit YUV packed planar format, with 4:4:4 sampling */
    UPTK_AD_FORMAT_Y410           = 0xa9, /**< 10-bit YUV packed planar format, with 4:4:4 sampling */
    UPTK_AD_FORMAT_Y416           = 0xb1, /**< 4 channel, 12-bit YUV packed planar format, with 4:4:4 sampling */
    UPTK_AD_FORMAT_Y444_PLANAR8   = 0xb2, /**< 3 channel 8-bit YUV planar format, with 4:4:4 sampling */
    UPTK_AD_FORMAT_Y444_PLANAR10  = 0xb3, /**< 3 channel 10-bit YUV planar format, with 4:4:4 sampling */
    UPTK_AD_FORMAT_MAX            = 0x7FFFFFFF
} UPTKarray_format;

// typedef struct UPTKstream_st *UPTKstream;                        /**< UPTK stream */
typedef struct UPTKarray_st *UPTKarray;                          /**< UPTK array */
typedef struct UPTKmod_st *UPTKmodule;                           /**< UPTK module */
typedef struct UPTKfunc_st *UPTKfunction;                        /**< UPTK function */
typedef enum UPTKjit_option_enum
{
    /**
     * Max number of registers that a thread may use.\n
     * Option type: unsigned int\n
     * Applies to: compiler only
     */
    UPTK_JIT_MAX_REGISTERS = 0,

    /**
     * IN: Specifies minimum number of threads per block to target compilation
     * for\n
     * OUT: Returns the number of threads the compiler actually targeted.
     * This restricts the resource utilization of the compiler (e.g. max
     * registers) such that a block with the given number of threads should be
     * able to launch based on register limitations. Note, this option does not
     * currently take into account any other resource limitations, such as
     * shared memory utilization.\n
     * Cannot be combined with ::UPTK_JIT_TARGET.\n
     * Option type: unsigned int\n
     * Applies to: compiler only
     */
    UPTK_JIT_THREADS_PER_BLOCK = 1,

    /**
     * Overwrites the option value with the total wall clock time, in
     * milliseconds, spent in the compiler and linker\n
     * Option type: float\n
     * Applies to: compiler and linker
     */
    UPTK_JIT_WALL_TIME = 2,

    /**
     * Pointer to a buffer in which to print any log messages
     * that are informational in nature (the buffer size is specified via
     * option ::UPTK_JIT_INFO_LOG_BUFFER_SIZE_BYTES)\n
     * Option type: char *\n
     * Applies to: compiler and linker
     */
    UPTK_JIT_INFO_LOG_BUFFER = 3,

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)\n
     * OUT: Amount of log buffer filled with messages\n
     * Option type: unsigned int\n
     * Applies to: compiler and linker
     */
    UPTK_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,

    /**
     * Pointer to a buffer in which to print any log messages that
     * reflect errors (the buffer size is specified via option
     * ::UPTK_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)\n
     * Option type: char *\n
     * Applies to: compiler and linker
     */
    UPTK_JIT_ERROR_LOG_BUFFER = 5,

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)\n
     * OUT: Amount of log buffer filled with messages\n
     * Option type: unsigned int\n
     * Applies to: compiler and linker
     */
    UPTK_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,

    /**
     * Level of optimizations to apply to generated code (0 - 4), with 4
     * being the default and highest level of optimizations.\n
     * Option type: unsigned int\n
     * Applies to: compiler only
     */
    UPTK_JIT_OPTIMIZATION_LEVEL = 7,

    /**
     * No option value required. Determines the target based on the current
     * attached context (default)\n
     * Option type: No option value needed\n
     * Applies to: compiler and linker
     */
    UPTK_JIT_TARGET_FROM_UPTKCONTEXT = 8,

    /**
     * Target is chosen based on supplied ::UPTKjit_target.  Cannot be
     * combined with ::UPTK_JIT_THREADS_PER_BLOCK.\n
     * Option type: unsigned int for enumerated type ::UPTKjit_target\n
     * Applies to: compiler and linker
     */
    UPTK_JIT_TARGET = 9,

    /**
     * Specifies choice of fallback strategy if matching cubin is not found.
     * Choice is based on supplied ::UPTKjit_fallback.  This option cannot be
     * used with cuLink* APIs as the linker requires exact matches.\n
     * Option type: unsigned int for enumerated type ::UPTKjit_fallback\n
     * Applies to: compiler only
     */
    UPTK_JIT_FALLBACK_STRATEGY = 10,

    /**
     * Specifies whether to create debug information in output (-g)
     * (0: false, default)\n
     * Option type: int\n
     * Applies to: compiler and linker
     */
    UPTK_JIT_GENERATE_DEBUG_INFO = 11,

    /**
     * Generate verbose log messages (0: false, default)\n
     * Option type: int\n
     * Applies to: compiler and linker
     */
    UPTK_JIT_LOG_VERBOSE = 12,

    /**
     * Generate line number information (-lineinfo) (0: false, default)\n
     * Option type: int\n
     * Applies to: compiler only
     */
    UPTK_JIT_GENERATE_LINE_INFO = 13,

    /**
     * Specifies whether to enable caching explicitly (-dlcm) \n
     * Choice is based on supplied ::UPTKjit_cacheMode_enum.\n
     * Option type: unsigned int for enumerated type ::UPTKjit_cacheMode_enum\n
     * Applies to: compiler only
     */
    UPTK_JIT_CACHE_MODE = 14,

    /**
     * \deprecated
     * This jit option is deprecated and should not be used.
     */
    UPTK_JIT_NEW_SM3X_OPT = 15,

    /**
     * This jit option is used for internal purpose only.
     */
    UPTK_JIT_FAST_COMPILE = 16,

    /**
     * Array of device symbol names that will be relocated to the corresponding
     * host addresses stored in ::UPTK_JIT_GLOBAL_SYMBOL_ADDRESSES.\n
     * Must contain ::UPTK_JIT_GLOBAL_SYMBOL_COUNT entries.\n
     * When loading a device module, driver will relocate all encountered
     * unresolved symbols to the host addresses.\n
     * It is only allowed to register symbols that correspond to unresolved
     * global variables.\n
     * It is illegal to register the same device symbol at multiple addresses.\n
     * Option type: const char **\n
     * Applies to: dynamic linker only
     */
    UPTK_JIT_GLOBAL_SYMBOL_NAMES = 17,

    /**
     * Array of host addresses that will be used to relocate corresponding
     * device symbols stored in ::UPTK_JIT_GLOBAL_SYMBOL_NAMES.\n
     * Must contain ::UPTK_JIT_GLOBAL_SYMBOL_COUNT entries.\n
     * Option type: void **\n
     * Applies to: dynamic linker only
     */
    UPTK_JIT_GLOBAL_SYMBOL_ADDRESSES = 18,

    /**
     * Number of entries in ::UPTK_JIT_GLOBAL_SYMBOL_NAMES and
     * ::UPTK_JIT_GLOBAL_SYMBOL_ADDRESSES arrays.\n
     * Option type: unsigned int\n
     * Applies to: dynamic linker only
     */
    UPTK_JIT_GLOBAL_SYMBOL_COUNT = 19,

    /**
     * \deprecated
     * Enable link-time optimization (-dlto) for device code (Disabled by default).\n
     * This option is not supported on 32-bit platforms.\n
     * Option type: int\n
     * Applies to: compiler and linker
     *
     * Only valid with LTO-IR compiled with toolkits prior to UPTK 12.0
     */
    UPTK_JIT_LTO = 20,

    /**
     * \deprecated
     * Control single-precision denormals (-ftz) support (0: false, default).
     * 1 : flushes denormal values to zero
     * 0 : preserves denormal values
     * Option type: int\n
     * Applies to: link-time optimization specified with UPTK_JIT_LTO
     *
     * Only valid with LTO-IR compiled with toolkits prior to UPTK 12.0
     */
    UPTK_JIT_FTZ = 21,

    /**
     * \deprecated
     * Control single-precision floating-point division and reciprocals
     * (-prec-div) support (1: true, default).
     * 1 : Enables the IEEE round-to-nearest mode
     * 0 : Enables the fast approximation mode
     * Option type: int\n
     * Applies to: link-time optimization specified with UPTK_JIT_LTO
     *
     * Only valid with LTO-IR compiled with toolkits prior to UPTK 12.0
     */
    UPTK_JIT_PREC_DIV = 22,

    /**
     * \deprecated
     * Control single-precision floating-point square root
     * (-prec-sqrt) support (1: true, default).
     * 1 : Enables the IEEE round-to-nearest mode
     * 0 : Enables the fast approximation mode
     * Option type: int\n
     * Applies to: link-time optimization specified with UPTK_JIT_LTO
     *
     * Only valid with LTO-IR compiled with toolkits prior to UPTK 12.0
     */
    UPTK_JIT_PREC_SQRT = 23,

    /**
     * \deprecated
     * Enable/Disable the contraction of floating-point multiplies
     * and adds/subtracts into floating-point multiply-add (-fma)
     * operations (1: Enable, default; 0: Disable).
     * Option type: int\n
     * Applies to: link-time optimization specified with UPTK_JIT_LTO
     *
     * Only valid with LTO-IR compiled with toolkits prior to UPTK 12.0
     */
    UPTK_JIT_FMA = 24,

    /**
     * \deprecated
     * Array of kernel names that should be preserved at link time while others
     * can be removed.\n
     * Must contain ::UPTK_JIT_REFERENCED_KERNEL_COUNT entries.\n
     * Note that kernel names can be mangled by the compiler in which case the
     * mangled name needs to be specified.\n
     * Wildcard "*" can be used to represent zero or more characters instead of
     * specifying the full or mangled name.\n
     * It is important to note that the wildcard "*" is also added implicitly.
     * For example, specifying "foo" will match "foobaz", "barfoo", "barfoobaz" and
     * thus preserve all kernels with those names. This can be avoided by providing
     * a more specific name like "barfoobaz".\n
     * Option type: const char **\n
     * Applies to: dynamic linker only
     *
     * Only valid with LTO-IR compiled with toolkits prior to UPTK 12.0
     */
    UPTK_JIT_REFERENCED_KERNEL_NAMES = 25,

    /**
     * \deprecated
     * Number of entries in ::UPTK_JIT_REFERENCED_KERNEL_NAMES array.\n
     * Option type: unsigned int\n
     * Applies to: dynamic linker only
     *
     * Only valid with LTO-IR compiled with toolkits prior to UPTK 12.0
     */
    UPTK_JIT_REFERENCED_KERNEL_COUNT = 26,

    /**
     * \deprecated
     * Array of variable names (__device__ and/or __constant__) that should be
     * preserved at link time while others can be removed.\n
     * Must contain ::UPTK_JIT_REFERENCED_VARIABLE_COUNT entries.\n
     * Note that variable names can be mangled by the compiler in which case the
     * mangled name needs to be specified.\n
     * Wildcard "*" can be used to represent zero or more characters instead of
     * specifying the full or mangled name.\n
     * It is important to note that the wildcard "*" is also added implicitly.
     * For example, specifying "foo" will match "foobaz", "barfoo", "barfoobaz" and
     * thus preserve all variables with those names. This can be avoided by providing
     * a more specific name like "barfoobaz".\n
     * Option type: const char **\n
     * Applies to: link-time optimization specified with UPTK_JIT_LTO
     *
     * Only valid with LTO-IR compiled with toolkits prior to UPTK 12.0
     */
    UPTK_JIT_REFERENCED_VARIABLE_NAMES = 27,

    /**
     * \deprecated
     * Number of entries in ::UPTK_JIT_REFERENCED_VARIABLE_NAMES array.\n
     * Option type: unsigned int\n
     * Applies to: link-time optimization specified with UPTK_JIT_LTO
     *
     * Only valid with LTO-IR compiled with toolkits prior to UPTK 12.0
     */
    UPTK_JIT_REFERENCED_VARIABLE_COUNT = 28,

    /**
     * \deprecated
     * This option serves as a hint to enable the JIT compiler/linker
     * to remove constant (__constant__) and device (__device__) variables
     * unreferenced in device code (Disabled by default).\n
     * Note that host references to constant and device variables using APIs like
     * ::cuModuleGetGlobal() with this option specified may result in undefined behavior unless
     * the variables are explicitly specified using ::UPTK_JIT_REFERENCED_VARIABLE_NAMES.\n
     * Option type: int\n
     * Applies to: link-time optimization specified with UPTK_JIT_LTO
     *
     * Only valid with LTO-IR compiled with toolkits prior to UPTK 12.0
     */
    UPTK_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES = 29,

    /**
     * Generate position independent code (0: false)\n
     * Option type: int\n
     * Applies to: compiler only
     */
    UPTK_JIT_POSITION_INDEPENDENT_CODE = 30,

    /**
     * This option hints to the JIT compiler the minimum number of CTAs from the
     * kernel’s grid to be mapped to a SM. This option is ignored when used together
     * with ::UPTK_JIT_MAX_REGISTERS or ::UPTK_JIT_THREADS_PER_BLOCK.
     * Optimizations based on this option need ::UPTK_JIT_MAX_THREADS_PER_BLOCK to
     * be specified as well. For kernels already using PTX directive .minnctapersm,
     * this option will be ignored by default. Use ::UPTK_JIT_OVERRIDE_DIRECTIVE_VALUES
     * to let this option take precedence over the PTX directive.
     * Option type: unsigned int\n
     * Applies to: compiler only
    */
    UPTK_JIT_MIN_CTA_PER_SM = 31,

     /**
     * Maximum number threads in a thread block, computed as the product of
     * the maximum extent specifed for each dimension of the block. This limit
     * is guaranteed not to be exeeded in any invocation of the kernel. Exceeding
     * the the maximum number of threads results in runtime error or kernel launch
     * failure. For kernels already using PTX directive .maxntid, this option will
     * be ignored by default. Use ::UPTK_JIT_OVERRIDE_DIRECTIVE_VALUES to let this
     * option take precedence over the PTX directive.
     * Option type: int\n
     * Applies to: compiler only
    */
    UPTK_JIT_MAX_THREADS_PER_BLOCK = 32,

    /**
     * This option lets the values specified using ::UPTK_JIT_MAX_REGISTERS,
     * ::UPTK_JIT_THREADS_PER_BLOCK, ::UPTK_JIT_MAX_THREADS_PER_BLOCK and
     * ::UPTK_JIT_MIN_CTA_PER_SM take precedence over any PTX directives.
     * (0: Disable, default; 1: Enable)
     * Option type: int\n
     * Applies to: compiler only
    */
    UPTK_JIT_OVERRIDE_DIRECTIVE_VALUES = 33,
    UPTK_JIT_NUM_OPTIONS

} UPTKjit_option;

typedef enum UPTKjitInputType_enum
{
    /**
     * Compiled device-class-specific device code\n
     * Applicable options: none
     */
    UPTK_JIT_INPUT_UPTKBIN = 0,

    /**
     * PTX source code\n
     * Applicable options: PTX compiler options
     */
    UPTK_JIT_INPUT_PTX = 1,

    /**
     * Bundle of multiple cubins and/or PTX of some device code\n
     * Applicable options: PTX compiler options, ::UPTK_JIT_FALLBACK_STRATEGY
     */
    UPTK_JIT_INPUT_FATBINARY = 2,

    /**
     * Host object with embedded device code\n
     * Applicable options: PTX compiler options, ::UPTK_JIT_FALLBACK_STRATEGY
     */
    UPTK_JIT_INPUT_OBJECT = 3,

    /**
     * Archive of host objects with embedded device code\n
     * Applicable options: PTX compiler options, ::UPTK_JIT_FALLBACK_STRATEGY
     */
    UPTK_JIT_INPUT_LIBRARY = 4,

    /**
     * \deprecated
     * High-level intermediate code for link-time optimization\n
     * Applicable options: NVVM compiler options, PTX compiler options
     *
     * Only valid with LTO-IR compiled with toolkits prior to UPTK 12.0
     */
    UPTK_JIT_INPUT_NVVM = 5,

    UPTK_JIT_NUM_INPUT_TYPES = 6
} UPTKjitInputType;

typedef struct UPTKlinkState_st *UPTKlinkState;
/**
 * Array descriptor
 */
typedef struct UPTK_ARRAY_DESCRIPTOR_st
{
    size_t Width;             /**< Width of array */
    size_t Height;            /**< Height of array */

    UPTKarray_format Format;    /**< Array format */
    unsigned int NumChannels; /**< Channels per array element */
} UPTK_ARRAY_DESCRIPTOR_v2;
typedef UPTK_ARRAY_DESCRIPTOR_v2 UPTK_ARRAY_DESCRIPTOR;

typedef enum UPTKfunction_attribute_enum {
    /**
     * The maximum number of threads per block, beyond which a launch of the
     * function would fail. This number depends on both the function and the
     * device on which the function is UPTKrrently loaded.
     */
    UPTK_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,

    /**
     * The size in bytes of statically-allocated shared memory required by
     * this function. This does not include dynamically-allocated shared
     * memory requested by the user at runtime.
     */
    UPTK_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,

    /**
     * The size in bytes of user-allocated constant memory required by this
     * function.
     */
    UPTK_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,

    /**
     * The size in bytes of local memory used by each thread of this function.
     */
    UPTK_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,

    /**
     * The number of registers used by each thread of this function.
     */
    UPTK_FUNC_ATTRIBUTE_NUM_REGS = 4,

    /**
     * The PTX virtual architecture version for which the function was
     * compiled. This value is the major PTX version * 10 + the minor PTX
     * version, so a PTX version 1.3 function would return the value 13.
     * Note that this may return the undefined value of 0 for UPTKbins
     * compiled prior to UPTKDA 3.0.
     */
    UPTK_FUNC_ATTRIBUTE_PTX_VERSION = 5,

    /**
     * The binary architecture version for which the function was compiled.
     * This value is the major binary version * 10 + the minor binary version,
     * so a binary version 1.3 function would return the value 13. Note that
     * this will return a value of 10 for legacy UPTKbins that do not have a
     * properly-encoded binary architecture version.
     */
    UPTK_FUNC_ATTRIBUTE_BINARY_VERSION = 6,

    /**
     * The attribute to indicate whether the function has been compiled with
     * user specified option "-Xptxas --dlcm=ca" set .
     */
    UPTK_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,

    /**
     * The maximum size in bytes of dynamically-allocated shared memory that can be used by
     * this function. If the user-specified dynamic shared memory size is larger than this
     * value, the launch will fail.
     * See ::UPTKFuncSetAttribute, ::UPTKKernelSetAttribute
     */
    UPTK_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,

    /**
     * On devices where the L1 cache and shared memory use the same hardware resources, 
     * this sets the shared memory carveout preference, in percent of the total shared memory.
     * Refer to ::UPTK_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR.
     * This is only a hint, and the driver can choose a different ratio if required to exeUPTKte the function.
     * See ::UPTKFuncSetAttribute, ::UPTKKernelSetAttribute
     */
    UPTK_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,

    /**
     * If this attribute is set, the kernel must launch with a valid cluster
     * size specified.
     * See ::UPTKFuncSetAttribute, ::UPTKKernelSetAttribute
     */
    UPTK_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET = 10,

    /**
     * The required cluster width in blocks. The values must either all be 0 or
     * all be positive. The validity of the cluster dimensions is otherwise
     * checked at launch time.
     *
     * If the value is set during compile time, it cannot be set at runtime.
     * Setting it at runtime will return UPTKDA_ERROR_NOT_PERMITTED.
     * See ::UPTKFuncSetAttribute, ::UPTKKernelSetAttribute
     */
    UPTK_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH = 11,

    /**
     * The required cluster height in blocks. The values must either all be 0 or
     * all be positive. The validity of the cluster dimensions is otherwise
     * checked at launch time.
     *
     * If the value is set during compile time, it cannot be set at runtime.
     * Setting it at runtime should return UPTKDA_ERROR_NOT_PERMITTED.
     * See ::UPTKFuncSetAttribute, ::UPTKKernelSetAttribute
     */
    UPTK_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT = 12,

    /**
     * The required cluster depth in blocks. The values must either all be 0 or
     * all be positive. The validity of the cluster dimensions is otherwise
     * checked at launch time.
     *
     * If the value is set during compile time, it cannot be set at runtime.
     * Setting it at runtime should return UPTKDA_ERROR_NOT_PERMITTED.
     * See ::UPTKFuncSetAttribute, ::UPTKKernelSetAttribute
     */
    UPTK_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH = 13,

    /**
     * Whether the function can be launched with non-portable cluster size. 1 is
     * allowed, 0 is disallowed. A non-portable cluster size may only function
     * on the specific SKUs the program is tested on. The launch might fail if
     * the program is run on a different hardware platform.
     *
     * UPTKDA API provides UPTKdaOcUPTKpancyMaxActiveClusters to assist with checking
     * whether the desired size can be launched on the UPTKrrent device.
     *
     * Portable Cluster Size
     *
     * A portable cluster size is guaranteed to be functional on all compute
     * capabilities higher than the target compute capability. The portable
     * cluster size for sm_90 is 8 blocks per cluster. This value may increase
     * for future compute capabilities.
     *
     * The specific hardware unit may support higher cluster sizes that’s not
     * guaranteed to be portable.
     * See ::UPTKFuncSetAttribute, ::UPTKKernelSetAttribute
     */
    UPTK_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 14,

    /**
     * The block scheduling policy of a function. The value type is
     * UPTKclusterSchedulingPolicy / UPTKdaClusterSchedulingPolicy.
     * See ::UPTKFuncSetAttribute, ::UPTKKernelSetAttribute
     */
    UPTK_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 15,

    UPTK_FUNC_ATTRIBUTE_MAX
} UPTKfunction_attribute;

typedef enum UPTKlimit_enum {
    UPTK_LIMIT_STACK_SIZE                       = 0x00, /**< GPU thread stack size */
    UPTK_LIMIT_PRINTF_FIFO_SIZE                 = 0x01, /**< GPU printf FIFO size */
    UPTK_LIMIT_MALLOC_HEAP_SIZE                 = 0x02, /**< GPU malloc heap size */
    UPTK_LIMIT_DEV_RUNTIME_SYNC_DEPTH           = 0x03, /**< GPU device runtime launch synchronize depth */
    UPTK_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04, /**< GPU device runtime pending launch count */
    UPTK_LIMIT_MAX_L2_FETCH_GRANULARITY         = 0x05, /**< A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint */
    UPTK_LIMIT_PERSISTING_L2_CACHE_SIZE         = 0x06, /**< A size in bytes for L2 persisting lines cache size */
    UPTK_LIMIT_SHMEM_SIZE                       = 0x07, /**< A maximum size in bytes of shared memory available to UPTKDA kernels on a CIG context. Can only be queried, cannot be set */
    UPTK_LIMIT_CIG_ENABLED                      = 0x08, /**< A non-zero value indicates this UPTKDA context is a CIG-enabled context. Can only be queried, cannot be set */
    UPTK_LIMIT_CIG_SHMEM_FALLBACK_ENABLED       = 0x09, /**< When set to a non-zero value, UPTKDA will fail to launch a kernel on a CIG context, instead of using the fallback path, if the kernel uses more shared memory than available */
    UPTK_LIMIT_MAX
} UPTKlimit;

typedef struct UPTKlinkState_st *UPTKlinkState;
template<class T>
static __inline__ __host__ UPTKError_t UPTKMalloc(
  T      **devPtr,
  size_t   size
)
{
  return ::UPTKMalloc((void**)(void*)devPtr, size);
}

template<class T>
static __inline__ __host__ UPTKError_t UPTKMallocManaged(
  T            **devPtr,
  size_t         size,
  unsigned int   flags = UPTKMemAttachGlobal
)
{
  return ::UPTKMallocManaged((void**)(void*)devPtr, size, flags);
}

template<class T>
static __inline__ __host__ UPTKError_t UPTKMallocAsync(
  T           **ptr,
  size_t        size,
  UPTKStream_t  stream
)
{
  return ::UPTKMallocAsync((void**)(void*)ptr, size, stream);
}

template<class T>
static __inline__ __host__  UPTKError_t UPTKOccupancyMaxPotentialBlockSize(
    int    *minGridSize,
    int    *blockSize,
    T       func,
    size_t  dynamicSMemSize = 0,
    int     blockSizeLimit = 0)
{
  return UPTKOccupancyMaxPotentialBlockSize_internal(minGridSize, blockSize, reinterpret_cast<const void*>(func), dynamicSMemSize, blockSizeLimit);
}

template<class T>
static __inline__ __host__ UPTKError_t UPTKMallocHost(
  T            **ptr,
  size_t         size,
  unsigned int   flags = 0
)
{
  return UPTKMallocHost((void**)(void*)ptr, size, flags);
}

static __inline__ __host__ UPTKError_t UPTKMallocHost(
  void         **ptr,
  size_t         size,
  unsigned int   flags
)
{
  return ::UPTKHostAlloc(ptr, size, flags);
}

template<class T>
static __inline__ __host__ UPTKError_t UPTKHostAlloc(
  T            **ptr,
  size_t         size,
  unsigned int   flags
)
{
  return ::UPTKHostAlloc((void**)(void*)ptr, size, flags);
}

template<class T>
static __inline__ __host__ UPTKError_t UPTKFuncGetAttributes(
  struct UPTKFuncAttributes *attr,
  T                         *entry
)
{
  return ::UPTKFuncGetAttributes(attr, (const void*)entry);
}



#endif

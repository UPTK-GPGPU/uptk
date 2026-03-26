/*
 * Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
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

#if !defined(__UPTK_RUNTIME_API_H__)
#define __UPTK_RUNTIME_API_H__

#if !defined(__UPTK_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __UPTK_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_UPTK_INCLUDE_COMPILER_INTERNAL_HEADERS_UPTK_RUNTIME_API_H__
#endif

/**
 * \latexonly
 * \page sync_async API synchronization behavior
 *
 * \section memcpy_sync_async_behavior Memcpy
 * The API provides memcpy/memset functions in both synchronous and asynchronous forms,
 * the latter having an \e "Async" suffix. This is a misnomer as each function
 * may exhibit synchronous or asynchronous behavior depending on the arguments
 * passed to the function. In the reference documentation, each memcpy function is
 * categorized as \e synchronous or \e asynchronous, corresponding to the definitions
 * below.
 * 
 * \subsection MemcpySynchronousBehavior Synchronous
 * 
 * <ol>
 * <li> For transfers from pageable host memory to device memory, a stream sync is performed
 * before the copy is initiated. The function will return once the pageable
 * buffer has been copied to the staging memory for DMA transfer to device memory,
 * but the DMA to final destination may not have completed.
 * 
 * <li> For transfers from pinned host memory to device memory, the function is synchronous
 * with respect to the host.
 *
 * <li> For transfers from device to either pageable or pinned host memory, the function returns
 * only once the copy has completed.
 * 
 * <li> For transfers from device memory to device memory, no host-side synchronization is
 * performed.
 *
 * <li> For transfers from any host memory to any host memory, the function is fully
 * synchronous with respect to the host.
 * </ol>
 * 
 * \subsection MemcpyAsynchronousBehavior Asynchronous
 *
 * <ol>
 * <li> For transfers between device memory and pageable host memory, the function might 
 * be synchronous with respect to host.
 *
 * <li> For transfers from any host memory to any host memory, the function is fully
 * synchronous with respect to the host.
 * 
 * <li> If pageable memory must first be staged to pinned memory, the driver may
 * synchronize with the stream and stage the copy into pinned memory.
 *
 * <li> For all other transfers, the function should be fully asynchronous.
 * </ol>
 *
 * \section memset_sync_async_behavior Memset
 * The UPTKMemset functions are asynchronous with respect to the host
 * except when the target memory is pinned host memory. The \e Async
 * versions are always asynchronous with respect to the host.
 *
 * \section kernel_launch_details Kernel Launches
 * Kernel launches are asynchronous with respect to the host. Details of
 * concurrent kernel execution and data transfers can be found in the UPTK
 * Programmers Guide.
 *
 * \endlatexonly
 */

/**
 * There are two levels for the runtime API.
 *
 * The C API (<i>UPTK_runtime_api.h</i>) is
 * a C-style interface that does not require compiling with \p nvcc.
 *
 * The \ref UPTKRT_HIGHLEVEL "C++ API" (<i>UPTK_runtime.h</i>) is a
 * C++-style interface built on top of the C API. It wraps some of the
 * C API routines, using overloading, references and default arguments.
 * These wrappers can be used from C++ code and can be compiled with any C++
 * compiler. The C++ API also has some UPTK-specific wrappers that wrap
 * C API routines that deal with symbols, textures, and device functions.
 * These wrappers require the use of \p nvcc because they depend on code being
 * generated by the compiler. For example, the execution configuration syntax
 * to invoke kernels is only available in source code compiled with \p nvcc.
 */

/** UPTK Runtime API Version */
#define UPTKRT_VERSION  12060

#if defined(__UPTK_API_VER_MAJOR__) && defined(__UPTK_API_VER_MINOR__)
# define __UPTKRT_API_VERSION ((__UPTK_API_VER_MAJOR__ * 1000) + (__UPTK_API_VER_MINOR__ * 10))
#else
# define __UPTKRT_API_VERSION UPTKRT_VERSION
#endif

#include "UPTK_driver_types.h"
#include "cuda_gl_interop.h"

#ifndef __UPTKCC_RTC_MINIMAL__
#if defined(UPTK_API_PER_THREAD_DEFAULT_STREAM) || defined(__UPTK_API_VERSION_INTERNAL)
    #define __UPTKRT_API_PER_THREAD_DEFAULT_STREAM
    #define __UPTKRT_API_PTDS(api) api ## _ptds
    #define __UPTKRT_API_PTSZ(api) api ## _ptsz
#else
    #define __UPTKRT_API_PTDS(api) api
    #define __UPTKRT_API_PTSZ(api) api
#endif

#define UPTKSignalExternalSemaphoresAsync  __UPTKRT_API_PTSZ(UPTKSignalExternalSemaphoresAsync_v2)
#define UPTKWaitExternalSemaphoresAsync    __UPTKRT_API_PTSZ(UPTKWaitExternalSemaphoresAsync_v2)

    #define UPTKStreamGetCaptureInfo       __UPTKRT_API_PTSZ(UPTKStreamGetCaptureInfo_v2)

#define UPTKGetDeviceProperties UPTKGetDeviceProperties_v2

#if defined(__UPTKRT_API_PER_THREAD_DEFAULT_STREAM)
    #define UPTKMemcpy                     __UPTKRT_API_PTDS(UPTKMemcpy)
    #define UPTKMemcpyToSymbol             __UPTKRT_API_PTDS(UPTKMemcpyToSymbol)
    #define UPTKMemcpyFromSymbol           __UPTKRT_API_PTDS(UPTKMemcpyFromSymbol)
    #define UPTKMemcpy2D                   __UPTKRT_API_PTDS(UPTKMemcpy2D)
    #define UPTKMemcpyToArray              __UPTKRT_API_PTDS(UPTKMemcpyToArray)
    #define UPTKMemcpy2DToArray            __UPTKRT_API_PTDS(UPTKMemcpy2DToArray)
    #define UPTKMemcpyFromArray            __UPTKRT_API_PTDS(UPTKMemcpyFromArray)
    #define UPTKMemcpy2DFromArray          __UPTKRT_API_PTDS(UPTKMemcpy2DFromArray)
    #define UPTKMemcpyArrayToArray         __UPTKRT_API_PTDS(UPTKMemcpyArrayToArray)
    #define UPTKMemcpy2DArrayToArray       __UPTKRT_API_PTDS(UPTKMemcpy2DArrayToArray)
    #define UPTKMemcpy3D                   __UPTKRT_API_PTDS(UPTKMemcpy3D)
    #define UPTKMemcpy3DPeer               __UPTKRT_API_PTDS(UPTKMemcpy3DPeer)
    #define UPTKMemset                     __UPTKRT_API_PTDS(UPTKMemset)
    #define UPTKMemset2D                   __UPTKRT_API_PTDS(UPTKMemset2D)
    #define UPTKMemset3D                   __UPTKRT_API_PTDS(UPTKMemset3D)
    #define UPTKGraphInstantiateWithParams __UPTKRT_API_PTSZ(UPTKGraphInstantiateWithParams)
    #define UPTKGraphUpload                __UPTKRT_API_PTSZ(UPTKGraphUpload)
    #define UPTKGraphLaunch                __UPTKRT_API_PTSZ(UPTKGraphLaunch)
    #define UPTKStreamBeginCapture         __UPTKRT_API_PTSZ(UPTKStreamBeginCapture)
    #define UPTKStreamBeginCaptureToGraph  __UPTKRT_API_PTSZ(UPTKStreamBeginCaptureToGraph)
    #define UPTKStreamEndCapture           __UPTKRT_API_PTSZ(UPTKStreamEndCapture)
    #define UPTKStreamGetCaptureInfo_v3    __UPTKRT_API_PTSZ(UPTKStreamGetCaptureInfo_v3)
    #define UPTKStreamUpdateCaptureDependencies  __UPTKRT_API_PTSZ(UPTKStreamUpdateCaptureDependencies)
    #define UPTKStreamUpdateCaptureDependencies_v2  __UPTKRT_API_PTSZ(UPTKStreamUpdateCaptureDependencies_v2)
    #define UPTKStreamIsCapturing          __UPTKRT_API_PTSZ(UPTKStreamIsCapturing)
    #define UPTKMemcpyAsync                __UPTKRT_API_PTSZ(UPTKMemcpyAsync)
    #define UPTKMemcpyToSymbolAsync        __UPTKRT_API_PTSZ(UPTKMemcpyToSymbolAsync)
    #define UPTKMemcpyFromSymbolAsync      __UPTKRT_API_PTSZ(UPTKMemcpyFromSymbolAsync)
    #define UPTKMemcpy2DAsync              __UPTKRT_API_PTSZ(UPTKMemcpy2DAsync)
    #define UPTKMemcpyToArrayAsync         __UPTKRT_API_PTSZ(UPTKMemcpyToArrayAsync)
    #define UPTKMemcpy2DToArrayAsync       __UPTKRT_API_PTSZ(UPTKMemcpy2DToArrayAsync)
    #define UPTKMemcpyFromArrayAsync       __UPTKRT_API_PTSZ(UPTKMemcpyFromArrayAsync)
    #define UPTKMemcpy2DFromArrayAsync     __UPTKRT_API_PTSZ(UPTKMemcpy2DFromArrayAsync)
    #define UPTKMemcpy3DAsync              __UPTKRT_API_PTSZ(UPTKMemcpy3DAsync)
    #define UPTKMemcpy3DPeerAsync          __UPTKRT_API_PTSZ(UPTKMemcpy3DPeerAsync)
    #define UPTKMemsetAsync                __UPTKRT_API_PTSZ(UPTKMemsetAsync)
    #define UPTKMemset2DAsync              __UPTKRT_API_PTSZ(UPTKMemset2DAsync)
    #define UPTKMemset3DAsync              __UPTKRT_API_PTSZ(UPTKMemset3DAsync)
    #define UPTKStreamQuery                __UPTKRT_API_PTSZ(UPTKStreamQuery)
    #define UPTKStreamGetFlags             __UPTKRT_API_PTSZ(UPTKStreamGetFlags)
    #define UPTKStreamGetId                __UPTKRT_API_PTSZ(UPTKStreamGetId)
    #define UPTKStreamGetPriority          __UPTKRT_API_PTSZ(UPTKStreamGetPriority)
    #define UPTKEventRecord                __UPTKRT_API_PTSZ(UPTKEventRecord)
    #define UPTKEventRecordWithFlags       __UPTKRT_API_PTSZ(UPTKEventRecordWithFlags)
    #define UPTKStreamWaitEvent            __UPTKRT_API_PTSZ(UPTKStreamWaitEvent)
    #define UPTKStreamAddCallback          __UPTKRT_API_PTSZ(UPTKStreamAddCallback)
    #define UPTKStreamAttachMemAsync       __UPTKRT_API_PTSZ(UPTKStreamAttachMemAsync)
    #define UPTKStreamSynchronize          __UPTKRT_API_PTSZ(UPTKStreamSynchronize)
    #define UPTKLaunchKernel               __UPTKRT_API_PTSZ(UPTKLaunchKernel)
    #define UPTKLaunchKernelExC            __UPTKRT_API_PTSZ(UPTKLaunchKernelExC)
    #define UPTKLaunchHostFunc             __UPTKRT_API_PTSZ(UPTKLaunchHostFunc)
    #define UPTKMemPrefetchAsync           __UPTKRT_API_PTSZ(UPTKMemPrefetchAsync)
    #define UPTKMemPrefetchAsync_v2        __UPTKRT_API_PTSZ(UPTKMemPrefetchAsync_v2)
    #define UPTKLaunchCooperativeKernel    __UPTKRT_API_PTSZ(UPTKLaunchCooperativeKernel)
    #define UPTKStreamCopyAttributes       __UPTKRT_API_PTSZ(UPTKStreamCopyAttributes)
    #define UPTKStreamGetAttribute         __UPTKRT_API_PTSZ(UPTKStreamGetAttribute)
    #define UPTKStreamSetAttribute         __UPTKRT_API_PTSZ(UPTKStreamSetAttribute)
    #define UPTKMallocAsync                __UPTKRT_API_PTSZ(UPTKMallocAsync)
    #define UPTKFreeAsync                  __UPTKRT_API_PTSZ(UPTKFreeAsync)
    #define UPTKMallocFromPoolAsync        __UPTKRT_API_PTSZ(UPTKMallocFromPoolAsync)
    #define UPTKGetDriverEntryPoint        __UPTKRT_API_PTSZ(UPTKGetDriverEntryPoint)
    #define UPTKGetDriverEntryPointByVersion  __UPTKRT_API_PTSZ(UPTKGetDriverEntryPointByVersion)
#endif

#endif  /* __UPTKCC_RTC_MINIMAL__ */

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

#if (defined(_NVHPC_UPTK) || !defined(__UPTK_ARCH__) || (__UPTK_ARCH__ >= 350))   /** Visible to SM>=3.5 and "__host__ __device__" only **/

#define UPTKRT_DEVICE __device__ 

#else

#define UPTKRT_DEVICE

#endif /** UPTKRT_DEVICE */

#if !defined(__UPTKCC_RTC__)
#define UPTK_EXCLUDE_FROM_RTC

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#define UPTKTextureType1D              0x01
#define UPTKTextureType2D              0x02
#define UPTKTextureType3D              0x03
#define UPTKTextureTypeCubemap         0x0C
#define UPTKTextureType1DLayered       0xF1
#define UPTKTextureType2DLayered       0xF2
#define UPTKTextureTypeCubemapLayered  0xFC

/**
 *  * UPTK texture address modes
 *   */
enum UPTKTextureAddressMode
{
    UPTKAddressModeWrap   = 0,    /**< Wrapping address mode */
    UPTKAddressModeClamp  = 1,    /**< Clamp to edge address mode */
    UPTKAddressModeMirror = 2,    /**< Mirror address mode */
    UPTKAddressModeBorder = 3     /**< Border address mode */
};

/**
 *  * UPTK texture filter modes
 *   */
enum UPTKTextureFilterMode
{
    UPTKFilterModePoint  = 0,     /**< Point filter mode */
    UPTKFilterModeLinear = 1      /**< Linear filter mode */
};

/**
 *  * UPTK texture read modes
 *   */
enum UPTKTextureReadMode
{
    UPTKReadModeElementType     = 0,  /**< Read texture as specified element type */
    UPTKReadModeNormalizedFloat = 1   /**< Read texture as normalized float */
};

/**
 *  * UPTK texture descriptor
 *   */
struct UPTKTextureDesc
{
    /**
 *      * Texture address mode for up to 3 dimensions
 *           */
    enum UPTKTextureAddressMode addressMode[3];
    /**
 *      * Texture filter mode
 *           */
    enum UPTKTextureFilterMode  filterMode;
    /**
 *      * Texture read mode
 *           */
    enum UPTKTextureReadMode    readMode;
    /**
 *      * Perform sRGB->linear conversion during texture read
 *           */
    int                         sRGB;
    /**
 *      * Texture Border Color
 *           */
    float                       borderColor[4];
    /**
 *      * Indicates whether texture reads are normalized or not
 *           */
    int                         normalizedCoords;
    /**
 *      * Limit to the anisotropy ratio
 *           */
    unsigned int                maxAnisotropy;
    /**
 *      * Mipmap filter mode
 *           */
    enum UPTKTextureFilterMode  mipmapFilterMode;
    /**
 *      * Offset applied to the supplied mipmap level
 *           */
    float                       mipmapLevelBias;
    /**
 *      * Lower end of the mipmap level range to clamp access to
 *           */
    float                       minMipmapLevelClamp;
    /**
 *      * Upper end of the mipmap level range to clamp access to
 *           */
    float                       maxMipmapLevelClamp;
    /**
 *      * Disable any trilinear filtering optimizations.
 *           */
    int                         disableTrilinearOptimization;
    /**
 *      * Enable seamless cube map filtering.
 *           */
    int                         seamlessCubemap;
};

/**
 *  * An opaque value that represents a UPTK texture object
 *   */
typedef unsigned long long UPTKTextureObject_t;


enum UPTKGLDeviceList
{
  UPTKGLDeviceListAll           = 1, /**< The UPTK devices for all GPUs used by the current OpenGL context */
  UPTKGLDeviceListCurrentFrame  = 2, /**< The UPTK devices for the GPUs used by the current OpenGL context in its currently rendering frame */
  UPTKGLDeviceListNextFrame     = 3  /**< The UPTK devices for the GPUs to be used by the current OpenGL context in the next frame  */
};

extern __host__ UPTKError_t UPTKBindSurfaceToArray(const struct surfaceReference *surfref, UPTKArray_const_t array, const struct UPTKChannelFormatDesc *desc);
extern __host__ UPTKError_t UPTKBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct UPTKChannelFormatDesc *desc, size_t size __dv(UINT_MAX));
extern __host__ UPTKError_t UPTKBindTexture2D(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct UPTKChannelFormatDesc *desc, size_t width, size_t height, size_t pitch);
extern __host__ UPTKError_t UPTKBindTextureToArray(const struct textureReference *texref, UPTKArray_const_t array, const struct UPTKChannelFormatDesc *desc);
extern __host__ UPTKError_t UPTKBindTextureToMipmappedArray(const struct textureReference *texref, UPTKMipmappedArray_const_t mipmappedArray, const struct UPTKChannelFormatDesc *desc);
extern __host__ UPTKError_t UPTKCreateTextureObject_v2(UPTKTextureObject_t *pTexObject, const struct UPTKResourceDesc *pResDesc, const struct UPTKTextureDesc_v2 *pTexDesc, const struct UPTKResourceViewDesc *pResViewDesc);
extern __host__ UPTKError_t UPTKGetDeviceProperties(struct UPTKDeviceProp *prop, int device);
extern __host__ const char* UPTKGetErrorName(UPTKError_t error);
extern __host__ UPTKError_t UPTKGetSurfaceReference(const struct surfaceReference **surfref, const void *symbol);
extern __host__ UPTKError_t UPTKGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref);
extern __host__ UPTKError_t UPTKGetTextureObjectTextureDesc_v2(struct UPTKTextureDesc_v2 *pTexDesc, UPTKTextureObject_t texObject);
extern __host__ UPTKError_t UPTKGetTextureReference(const struct textureReference **texref, const void *symbol);
extern __host__ UPTKError_t UPTKProfilerInitialize(const char * configFile,const char * outputFile,UPTKOutputMode_t outputMode);
extern __host__ UPTKError_t UPTKUnbindTexture(const struct textureReference *texref);
extern __host__ UPTKError_t UPTKGLGetDevices(unsigned int *pUPTKDeviceCount, int *pUPTKDevices, unsigned int UPTKDeviceCount, enum UPTKGLDeviceList deviceList);
extern __host__ UPTKError_t UPTKGraphicsGLRegisterImage(struct UPTKGraphicsResource **resource, GLuint image, GLenum target, unsigned int flags);
extern __host__ UPTKError_t UPTKGraphicsGLRegisterBuffer(struct UPTKGraphicsResource **resource, GLuint buffer, unsigned int flags);
enum UPTKGLMapFlags
{
  UPTKGLMapFlagsNone         = 0,  /**< Default; Assume resource can be read/written */
  UPTKGLMapFlagsReadOnly     = 1,  /**< UPTK kernels will not write to this resource */
  UPTKGLMapFlagsWriteDiscard = 2   /**< UPTK kernels will only write to and will not read from this resource */
};

extern __host__ UPTKError_t UPTKGLSetGLDevice(int device);

extern __host__ UPTKError_t UPTKGLRegisterBufferObject(GLuint bufObj);

extern __host__ UPTKError_t UPTKGLMapBufferObject(void **devPtr, GLuint bufObj);

extern __host__ UPTKError_t UPTKGLUnmapBufferObject(GLuint bufObj);

extern __host__ UPTKError_t UPTKGLUnregisterBufferObject(GLuint bufObj);

extern __host__ UPTKError_t UPTKGLSetBufferObjectMapFlags(GLuint bufObj, unsigned int flags);

extern __host__ UPTKError_t UPTKGLMapBufferObjectAsync(void **devPtr, GLuint bufObj, UPTKStream_t stream);

extern __host__ UPTKError_t UPTKGLUnmapBufferObjectAsync(GLuint bufObj, UPTKStream_t stream);

enum UPTKSurfaceBoundaryMode
{
    UPTKBoundaryModeZero  = 0,    
    UPTKBoundaryModeClamp = 1,    
    UPTKBoundaryModeTrap  = 2
};
enum UPTKSurfaceFormatMode
{
    UPTKFormatModeForced = 0,     
    UPTKFormatModeAuto = 1        
};

typedef unsigned long long UPTKSurfaceObject_t;

/**
 * \brief Destroy all allocations and reset all state on the current device
 * in the current process.
 *
 * Explicitly destroys and cleans up all resources associated with the current
 * device in the current process. It is the caller's responsibility to ensure
 * that the resources are not accessed or passed in subsequent API calls and
 * doing so will result in undefined behavior. These resources include UPTK types
 * ::UPTKStream_t, ::UPTKEvent_t, ::UPTKArray_t, ::UPTKMipmappedArray_t, ::UPTKPitchedPtr,
 * ::UPTKTextureObject_t, ::UPTKSurfaceObject_t, ::textureReference, ::surfaceReference,
 * ::UPTKExternalMemory_t, ::UPTKExternalSemaphore_t and ::UPTKGraphicsResource_t.
 * These resources also include memory allocations by ::UPTKMalloc, ::UPTKMallocHost,
 * ::UPTKMallocManaged and ::UPTKMallocPitch.
 * Any subsequent API call to this device will reinitialize the device.
 *
 * Note that this function will reset the device immediately.  It is the caller's
 * responsibility to ensure that the device is not being accessed by any 
 * other host threads from the process when this function is called.
 *
 * \note ::UPTKDeviceReset() will not destroy memory allocations by ::UPTKMallocAsync() and
 * ::UPTKMallocFromPoolAsync(). These memory allocations need to be destroyed explicitly.
 * \note If a non-primary ::CUcontext is current to the thread, ::UPTKDeviceReset()
 * will destroy only the internal UPTK RT state for that ::CUcontext.
 *
 * \return
 * ::UPTKSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKDeviceSynchronize
 */
extern __host__ UPTKError_t UPTKDeviceReset(void);

/**
 * \brief Wait for compute device to finish
 *
 * Blocks until the device has completed all preceding requested tasks.
 * ::UPTKDeviceSynchronize() returns an error if one of the preceding tasks
 * has failed. If the ::UPTKDeviceScheduleBlockingSync flag was set for 
 * this device, the host thread will block until the device has finished 
 * its work.
 *
 * \return
 * ::UPTKSuccess
 * \note_device_sync_deprecated
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKDeviceReset,
 * ::cuCtxSynchronize
 */
extern __host__ UPTKError_t UPTKDeviceSynchronize(void);

/**
 * \brief Set resource limits
 *
 * Setting \p limit to \p value is a request by the application to update
 * the current limit maintained by the device.  The driver is free to
 * modify the requested value to meet h/w requirements (this could be
 * clamping to minimum or maximum values, rounding up to nearest element
 * size, etc).  The application can use ::UPTKDeviceGetLimit() to find out
 * exactly what the limit has been set to.
 *
 * Setting each ::UPTKLimit has its own specific restrictions, so each is
 * discussed here.
 *
 * - ::UPTKLimitStackSize controls the stack size in bytes of each GPU thread.
 *
 * - ::UPTKLimitPrintfFifoSize controls the size in bytes of the shared FIFO
 *   used by the ::printf() device system call. Setting
 *   ::UPTKLimitPrintfFifoSize must not be performed after launching any kernel
 *   that uses the ::printf() device system call - in such case
 *   ::UPTKErrorInvalidValue will be returned.
 *
 * - ::UPTKLimitMallocHeapSize controls the size in bytes of the heap used by
 *   the ::malloc() and ::free() device system calls. Setting
 *   ::UPTKLimitMallocHeapSize must not be performed after launching any kernel
 *   that uses the ::malloc() or ::free() device system calls - in such case
 *   ::UPTKErrorInvalidValue will be returned.
 *
 * - ::UPTKLimitDevRuntimeSyncDepth controls the maximum nesting depth of a
 *   grid at which a thread can safely call ::UPTKDeviceSynchronize(). Setting
 *   this limit must be performed before any launch of a kernel that uses the
 *   device runtime and calls ::UPTKDeviceSynchronize() above the default sync
 *   depth, two levels of grids. Calls to ::UPTKDeviceSynchronize() will fail
 *   with error code ::UPTKErrorSyncDepthExceeded if the limitation is
 *   violated. This limit can be set smaller than the default or up the maximum
 *   launch depth of 24. When setting this limit, keep in mind that additional
 *   levels of sync depth require the runtime to reserve large amounts of
 *   device memory which can no longer be used for user allocations. If these
 *   reservations of device memory fail, ::UPTKDeviceSetLimit will return
 *   ::UPTKErrorMemoryAllocation, and the limit can be reset to a lower value.
 *   This limit is only applicable to devices of compute capability < 9.0.
 *   Attempting to set this limit on devices of other compute capability will
 *   results in error ::UPTKErrorUnsupportedLimit being returned.
 *
 * - ::UPTKLimitDevRuntimePendingLaunchCount controls the maximum number of
 *   outstanding device runtime launches that can be made from the current
 *   device. A grid is outstanding from the point of launch up until the grid
 *   is known to have been completed. Device runtime launches which violate
 *   this limitation fail and return ::UPTKErrorLaunchPendingCountExceeded when
 *   ::UPTKGetLastError() is called after launch. If more pending launches than
 *   the default (2048 launches) are needed for a module using the device
 *   runtime, this limit can be increased. Keep in mind that being able to
 *   sustain additional pending launches will require the runtime to reserve
 *   larger amounts of device memory upfront which can no longer be used for
 *   allocations. If these reservations fail, ::UPTKDeviceSetLimit will return
 *   ::UPTKErrorMemoryAllocation, and the limit can be reset to a lower value.
 *   This limit is only applicable to devices of compute capability 3.5 and
 *   higher. Attempting to set this limit on devices of compute capability less
 *   than 3.5 will result in the error ::UPTKErrorUnsupportedLimit being
 *   returned.
 *
 * - ::UPTKLimitMaxL2FetchGranularity controls the L2 cache fetch granularity.
 *   Values can range from 0B to 128B. This is purely a performance hint and
 *   it can be ignored or clamped depending on the platform.
 *
 * - ::UPTKLimitPersistingL2CacheSize controls size in bytes available
 *   for persisting L2 cache. This is purely a performance hint and it
 *   can be ignored or clamped depending on the platform.
 *
 * \param limit - Limit to set
 * \param value - Size of limit
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorUnsupportedLimit,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKDeviceGetLimit,
 * ::cuCtxSetLimit
 */
extern __host__ UPTKError_t UPTKDeviceSetLimit(enum UPTKLimit limit, size_t value);

/**
 * \brief Return resource limits
 *
 * Returns in \p *pValue the current size of \p limit. The following ::UPTKLimit values are supported.
 * - ::UPTKLimitStackSize is the stack size in bytes of each GPU thread.
 * - ::UPTKLimitPrintfFifoSize is the size in bytes of the shared FIFO used by the
 *   ::printf() device system call.
 * - ::UPTKLimitMallocHeapSize is the size in bytes of the heap used by the
 *   ::malloc() and ::free() device system calls.
 * - ::UPTKLimitDevRuntimeSyncDepth is the maximum grid depth at which a
 *   thread can isssue the device runtime call ::UPTKDeviceSynchronize()
 *   to wait on child grid launches to complete. This functionality is removed
 *   for devices of compute capability >= 9.0, and hence will return error
 *   ::UPTKErrorUnsupportedLimit on such devices.
 * - ::UPTKLimitDevRuntimePendingLaunchCount is the maximum number of outstanding
 *   device runtime launches.
 * - ::UPTKLimitMaxL2FetchGranularity is the L2 cache fetch granularity.
 * - ::UPTKLimitPersistingL2CacheSize is the persisting L2 cache size in bytes.
 *
 * \param limit  - Limit to query
 * \param pValue - Returned size of the limit
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorUnsupportedLimit,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKDeviceSetLimit,
 * ::cuCtxGetLimit
 */
extern __host__ UPTKError_t UPTKDeviceGetLimit(size_t *pValue, enum UPTKLimit limit);

/**
 * \brief Returns the maximum number of elements allocatable in a 1D linear texture for a given element size.
 *
 * Returns in \p maxWidthInElements the maximum number of elements allocatable in a 1D linear texture
 * for given format descriptor \p fmtDesc.
 *
 * \param maxWidthInElements    - Returns maximum number of texture elements allocatable for given \p fmtDesc.
 * \param fmtDesc               - Texture format description.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorUnsupportedLimit,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cuDeviceGetTexture1DLinearMaxWidth
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, const struct UPTKChannelFormatDesc *fmtDesc, int device);
#endif

/**
 * \brief Returns the preferred cache configuration for the current device.
 *
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this returns through \p pCacheConfig the preferred cache
 * configuration for the current device. This is only a preference. The
 * runtime will use the requested configuration if possible, but it is free to
 * choose a different configuration if required to execute functions.
 *
 * This will return a \p pCacheConfig of ::UPTKFuncCachePreferNone on devices
 * where the size of the L1 cache and shared memory are fixed.
 *
 * The supported cache configurations are:
 * - ::UPTKFuncCachePreferNone: no preference for shared memory or L1 (default)
 * - ::UPTKFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
 * - ::UPTKFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
 * - ::UPTKFuncCachePreferEqual: prefer equal size L1 cache and shared memory
 *
 * \param pCacheConfig - Returned cache configuration
 *
 * \return
 * ::UPTKSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKDeviceSetCacheConfig,
 * \ref ::UPTKFuncSetCacheConfig(const void*, enum UPTKFuncCache) "UPTKFuncSetCacheConfig (C API)",
 * \ref ::UPTKFuncSetCacheConfig(T*, enum UPTKFuncCache) "UPTKFuncSetCacheConfig (C++ API)",
 * ::cuCtxGetCacheConfig
 */
extern __host__ UPTKError_t UPTKDeviceGetCacheConfig(enum UPTKFuncCache *pCacheConfig);

/**
 * \brief Returns numerical values that correspond to the least and
 * greatest stream priorities.
 *
 * Returns in \p *leastPriority and \p *greatestPriority the numerical values that correspond
 * to the least and greatest stream priorities respectively. Stream priorities
 * follow a convention where lower numbers imply greater priorities. The range of
 * meaningful stream priorities is given by [\p *greatestPriority, \p *leastPriority].
 * If the user attempts to create a stream with a priority value that is
 * outside the the meaningful range as specified by this API, the priority is
 * automatically clamped down or up to either \p *leastPriority or \p *greatestPriority
 * respectively. See ::UPTKStreamCreateWithPriority for details on creating a
 * priority stream.
 * A NULL may be passed in for \p *leastPriority or \p *greatestPriority if the value
 * is not desired.
 *
 * This function will return '0' in both \p *leastPriority and \p *greatestPriority if
 * the current context's device does not support stream priorities
 * (see ::UPTKDeviceGetAttribute).
 *
 * \param leastPriority    - Pointer to an int in which the numerical value for least
 *                           stream priority is returned
 * \param greatestPriority - Pointer to an int in which the numerical value for greatest
 *                           stream priority is returned
 *
 * \return
 * ::UPTKSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKStreamCreateWithPriority,
 * ::UPTKStreamGetPriority,
 * ::cuCtxGetStreamPriorityRange
 */
extern __host__ UPTKError_t UPTKDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority);

/**
 * \brief Sets the preferred cache configuration for the current device.
 *
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this sets through \p cacheConfig the preferred cache
 * configuration for the current device. This is only a preference. The
 * runtime will use the requested configuration if possible, but it is free to
 * choose a different configuration if required to execute the function. Any
 * function preference set via
 * \ref ::UPTKFuncSetCacheConfig(const void*, enum UPTKFuncCache) "UPTKFuncSetCacheConfig (C API)"
 * or
 * \ref ::UPTKFuncSetCacheConfig(T*, enum UPTKFuncCache) "UPTKFuncSetCacheConfig (C++ API)"
 * will be preferred over this device-wide setting. Setting the device-wide
 * cache configuration to ::UPTKFuncCachePreferNone will cause subsequent
 * kernel launches to prefer to not change the cache configuration unless
 * required to launch the kernel.
 *
 * This setting does nothing on devices where the size of the L1 cache and
 * shared memory are fixed.
 *
 * Launching a kernel with a different preference than the most recent
 * preference setting may insert a device-side synchronization point.
 *
 * The supported cache configurations are:
 * - ::UPTKFuncCachePreferNone: no preference for shared memory or L1 (default)
 * - ::UPTKFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
 * - ::UPTKFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
 * - ::UPTKFuncCachePreferEqual: prefer equal size L1 cache and shared memory
 *
 * \param cacheConfig - Requested cache configuration
 *
 * \return
 * ::UPTKSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKDeviceGetCacheConfig,
 * \ref ::UPTKFuncSetCacheConfig(const void*, enum UPTKFuncCache) "UPTKFuncSetCacheConfig (C API)",
 * \ref ::UPTKFuncSetCacheConfig(T*, enum UPTKFuncCache) "UPTKFuncSetCacheConfig (C++ API)",
 * ::cuCtxSetCacheConfig
 */
extern __host__ UPTKError_t UPTKDeviceSetCacheConfig(enum UPTKFuncCache cacheConfig);

/**
 * \brief Returns a handle to a compute device
 *
 * Returns in \p *device a device ordinal given a PCI bus ID string.
 *
 * \param device   - Returned device ordinal
 *
 * \param pciBusId - String in one of the following forms: 
 * [domain]:[bus]:[device].[function]
 * [domain]:[bus]:[device]
 * [bus]:[device].[function]
 * where \p domain, \p bus, \p device, and \p function are all hexadecimal values
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidDevice
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKDeviceGetPCIBusId,
 * ::cuDeviceGetByPCIBusId
 */
extern __host__ UPTKError_t UPTKDeviceGetByPCIBusId(int *device, const char *pciBusId);

/**
 * \brief Returns a PCI Bus Id string for the device
 *
 * Returns an ASCII string identifying the device \p dev in the NULL-terminated
 * string pointed to by \p pciBusId. \p len specifies the maximum length of the
 * string that may be returned.
 *
 * \param pciBusId - Returned identifier string for the device in the following format
 * [domain]:[bus]:[device].[function]
 * where \p domain, \p bus, \p device, and \p function are all hexadecimal values.
 * pciBusId should be large enough to store 13 characters including the NULL-terminator.
 *
 * \param len      - Maximum length of string to store in \p name
 *
 * \param device   - Device to get identifier string for
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidDevice
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKDeviceGetByPCIBusId,
 * ::cuDeviceGetPCIBusId
 */
extern __host__ UPTKError_t UPTKDeviceGetPCIBusId(char *pciBusId, int len, int device);

/**
 * \brief Gets an interprocess handle for a previously allocated event
 *
 * Takes as input a previously allocated event. This event must have been 
 * created with the ::UPTKEventInterprocess and ::UPTKEventDisableTiming
 * flags set. This opaque handle may be copied into other processes and
 * opened with ::UPTKIpcOpenEventHandle to allow efficient hardware
 * synchronization between GPU work in different processes.
 *
 * After the event has been been opened in the importing process, 
 * ::UPTKEventRecord, ::UPTKEventSynchronize, ::UPTKStreamWaitEvent and 
 * ::UPTKEventQuery may be used in either process. Performing operations 
 * on the imported event after the exported event has been freed 
 * with ::UPTKEventDestroy will result in undefined behavior.
 *
 * IPC functionality is restricted to devices with support for unified
 * addressing on Linux and Windows operating systems.
 * IPC functionality on Windows is supported for compatibility purposes
 * but not recommended as it comes with performance cost.
 * Users can test their device for IPC functionality by calling
 * ::UPTKDeviceGetAttribute with ::UPTKDevAttrIpcEventSupport
 *
 * \param handle - Pointer to a user allocated UPTKIpcEventHandle
 *                    in which to return the opaque event handle
 * \param event   - Event allocated with ::UPTKEventInterprocess and 
 *                    ::UPTKEventDisableTiming flags.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorMemoryAllocation,
 * ::UPTKErrorMapBufferObjectFailed,
 * ::UPTKErrorNotSupported,
 * ::UPTKErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKEventCreate,
 * ::UPTKEventDestroy,
 * ::UPTKEventSynchronize,
 * ::UPTKEventQuery,
 * ::UPTKStreamWaitEvent,
 * ::UPTKIpcOpenEventHandle,
 * ::UPTKIpcGetMemHandle,
 * ::UPTKIpcOpenMemHandle,
 * ::UPTKIpcCloseMemHandle,
 * ::cuIpcGetEventHandle
 */
extern __host__ UPTKError_t UPTKIpcGetEventHandle(UPTKIpcEventHandle_t *handle, UPTKEvent_t event);

/**
 * \brief Opens an interprocess event handle for use in the current process
 *
 * Opens an interprocess event handle exported from another process with 
 * ::UPTKIpcGetEventHandle. This function returns a ::UPTKEvent_t that behaves like 
 * a locally created event with the ::UPTKEventDisableTiming flag specified. 
 * This event must be freed with ::UPTKEventDestroy.
 *
 * Performing operations on the imported event after the exported event has 
 * been freed with ::UPTKEventDestroy will result in undefined behavior.
 *
 * IPC functionality is restricted to devices with support for unified
 * addressing on Linux and Windows operating systems.
 * IPC functionality on Windows is supported for compatibility purposes
 * but not recommended as it comes with performance cost.
 * Users can test their device for IPC functionality by calling
 * ::UPTKDeviceGetAttribute with ::UPTKDevAttrIpcEventSupport
 *
 * \param event - Returns the imported event
 * \param handle  - Interprocess handle to open
 *
 * \returns
 * ::UPTKSuccess,
 * ::UPTKErrorMapBufferObjectFailed,
 * ::UPTKErrorNotSupported,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorDeviceUninitialized
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKEventCreate,
 * ::UPTKEventDestroy,
 * ::UPTKEventSynchronize,
 * ::UPTKEventQuery,
 * ::UPTKStreamWaitEvent,
 * ::UPTKIpcGetEventHandle,
 * ::UPTKIpcGetMemHandle,
 * ::UPTKIpcOpenMemHandle,
 * ::UPTKIpcCloseMemHandle,
 * ::cuIpcOpenEventHandle
 */
extern __host__ UPTKError_t UPTKIpcOpenEventHandle(UPTKEvent_t *event, UPTKIpcEventHandle_t handle);

/**
 * \brief Gets an interprocess memory handle for an existing device memory
 *          allocation
 *
 * Takes a pointer to the base of an existing device memory allocation created 
 * with ::UPTKMalloc and exports it for use in another process. This is a 
 * lightweight operation and may be called multiple times on an allocation
 * without adverse effects. 
 *
 * If a region of memory is freed with ::UPTKFree and a subsequent call
 * to ::UPTKMalloc returns memory with the same device address,
 * ::UPTKIpcGetMemHandle will return a unique handle for the
 * new memory. 
 *
 * IPC functionality is restricted to devices with support for unified
 * addressing on Linux and Windows operating systems.
 * IPC functionality on Windows is supported for compatibility purposes
 * but not recommended as it comes with performance cost.
 * Users can test their device for IPC functionality by calling
 * ::UPTKDeviceGetAttribute with ::UPTKDevAttrIpcEventSupport
 *
 * \param handle - Pointer to user allocated ::UPTKIpcMemHandle to return
 *                    the handle in.
 * \param devPtr - Base pointer to previously allocated device memory 
 *
 * \returns
 * ::UPTKSuccess,
 * ::UPTKErrorMemoryAllocation,
 * ::UPTKErrorMapBufferObjectFailed,
 * ::UPTKErrorNotSupported,
 * ::UPTKErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKMalloc,
 * ::UPTKFree,
 * ::UPTKIpcGetEventHandle,
 * ::UPTKIpcOpenEventHandle,
 * ::UPTKIpcOpenMemHandle,
 * ::UPTKIpcCloseMemHandle,
 * ::cuIpcGetMemHandle
 */
extern __host__ UPTKError_t UPTKIpcGetMemHandle(UPTKIpcMemHandle_t *handle, void *devPtr);

/**
 * \brief Opens an interprocess memory handle exported from another process
 *          and returns a device pointer usable in the local process.
 *
 * Maps memory exported from another process with ::UPTKIpcGetMemHandle into
 * the current device address space. For contexts on different devices 
 * ::UPTKIpcOpenMemHandle can attempt to enable peer access between the
 * devices as if the user called ::UPTKDeviceEnablePeerAccess. This behavior is 
 * controlled by the ::UPTKIpcMemLazyEnablePeerAccess flag. 
 * ::UPTKDeviceCanAccessPeer can determine if a mapping is possible.
 *
 * ::UPTKIpcOpenMemHandle can open handles to devices that may not be visible
 * in the process calling the API.
 *
 * Contexts that may open ::UPTKIpcMemHandles are restricted in the following way.
 * ::UPTKIpcMemHandles from each device in a given process may only be opened 
 * by one context per device per other process.
 *
 * If the memory handle has already been opened by the current context, the
 * reference count on the handle is incremented by 1 and the existing device pointer
 * is returned.
 *
 * Memory returned from ::UPTKIpcOpenMemHandle must be freed with
 * ::UPTKIpcCloseMemHandle.
 *
 * Calling ::UPTKFree on an exported memory region before calling
 * ::UPTKIpcCloseMemHandle in the importing context will result in undefined
 * behavior.
 * 
 * IPC functionality is restricted to devices with support for unified
 * addressing on Linux and Windows operating systems.
 * IPC functionality on Windows is supported for compatibility purposes
 * but not recommended as it comes with performance cost.
 * Users can test their device for IPC functionality by calling
 * ::UPTKDeviceGetAttribute with ::UPTKDevAttrIpcEventSupport
 *
 * \param devPtr - Returned device pointer
 * \param handle - ::UPTKIpcMemHandle to open
 * \param flags  - Flags for this operation. Must be specified as ::UPTKIpcMemLazyEnablePeerAccess
 *
 * \returns
 * ::UPTKSuccess,
 * ::UPTKErrorMapBufferObjectFailed,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorDeviceUninitialized,
 * ::UPTKErrorTooManyPeers,
 * ::UPTKErrorNotSupported,
 * ::UPTKErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \note No guarantees are made about the address returned in \p *devPtr.  
 * In particular, multiple processes may not receive the same address for the same \p handle.
 *
 * \sa
 * ::UPTKMalloc,
 * ::UPTKFree,
 * ::UPTKIpcGetEventHandle,
 * ::UPTKIpcOpenEventHandle,
 * ::UPTKIpcGetMemHandle,
 * ::UPTKIpcCloseMemHandle,
 * ::UPTKDeviceEnablePeerAccess,
 * ::UPTKDeviceCanAccessPeer,
 * ::cuIpcOpenMemHandle
 */
extern __host__ UPTKError_t UPTKIpcOpenMemHandle(void **devPtr, UPTKIpcMemHandle_t handle, unsigned int flags);

/**
 * \brief Attempts to close memory mapped with UPTKIpcOpenMemHandle
 * 
 * Decrements the reference count of the memory returnd by ::UPTKIpcOpenMemHandle by 1.
 * When the reference count reaches 0, this API unmaps the memory. The original allocation
 * in the exporting process as well as imported mappings in other processes
 * will be unaffected.
 *
 * Any resources used to enable peer access will be freed if this is the
 * last mapping using them.
 *
 * IPC functionality is restricted to devices with support for unified
 * addressing on Linux and Windows operating systems.
 * IPC functionality on Windows is supported for compatibility purposes
 * but not recommended as it comes with performance cost.
 * Users can test their device for IPC functionality by calling
 * ::UPTKDeviceGetAttribute with ::UPTKDevAttrIpcEventSupport
 *
 * \param devPtr - Device pointer returned by ::UPTKIpcOpenMemHandle
 * 
 * \returns
 * ::UPTKSuccess,
 * ::UPTKErrorMapBufferObjectFailed,
 * ::UPTKErrorNotSupported,
 * ::UPTKErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKMalloc,
 * ::UPTKFree,
 * ::UPTKIpcGetEventHandle,
 * ::UPTKIpcOpenEventHandle,
 * ::UPTKIpcGetMemHandle,
 * ::UPTKIpcOpenMemHandle,
 * ::cuIpcCloseMemHandle
 */
extern __host__ UPTKError_t UPTKIpcCloseMemHandle(void *devPtr);

/**
 * \brief Blocks until remote writes are visible to the specified scope
 *
 * Blocks until remote writes to the target context via mappings created
 * through GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see
 * https://docs.nvidia.com/UPTK/gpudirect-rdma for more information), are
 * visible to the specified scope.
 *
 * If the scope equals or lies within the scope indicated by
 * ::UPTKDevAttrGPUDirectRDMAWritesOrdering, the call will be a no-op and
 * can be safely omitted for performance. This can be determined by
 * comparing the numerical values between the two enums, with smaller
 * scopes having smaller values.
 *
 * Users may query support for this API via ::UPTKDevAttrGPUDirectRDMAFlushWritesOptions.
 *
 * \param target - The target of the operation, see UPTKFlushGPUDirectRDMAWritesTarget
 * \param scope  - The scope of the operation, see UPTKFlushGPUDirectRDMAWritesScope
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorNotSupported,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cuFlushGPUDirectRDMAWrites
 */
#if __UPTKRT_API_VERSION >= 11030
extern __host__ UPTKError_t UPTKDeviceFlushGPUDirectRDMAWrites(enum UPTKFlushGPUDirectRDMAWritesTarget target, enum UPTKFlushGPUDirectRDMAWritesScope scope);
#endif

/**
* \brief Registers a callback function to receive async notifications
*
* Registers \p callbackFunc to receive async notifications.
*
* The \p userData parameter is passed to the callback function at async notification time.
* Likewise, \p callback is also passed to the callback function to distinguish between
* multiple registered callbacks.
*
* The callback function being registered should be designed to return quickly (~10ms).
* Any long running tasks should be queued for execution on an application thread.
*
* Callbacks may not call UPTKDeviceRegisterAsyncNotification or UPTKDeviceUnregisterAsyncNotification.
* Doing so will result in ::UPTKErrorNotPermitted. Async notification callbacks execute
* in an undefined order and may be serialized.
*
* Returns in \p *callback a handle representing the registered callback instance.
*
* \param device - The device on which to register the callback
* \param callbackFunc - The function to register as a callback
* \param userData - A generic pointer to user data. This is passed into the callback function.
* \param callback - A handle representing the registered callback instance
*
* \return
* ::UPTKSuccess
* ::UPTKErrorNotSupported
* ::UPTKErrorInvalidDevice
* ::UPTKErrorInvalidValue
* ::UPTKErrorNotPermitted
* ::UPTKErrorUnknown
* \notefnerr
*
* \sa
* ::UPTKDeviceUnregisterAsyncNotification
*/
extern __host__ UPTKError_t UPTKDeviceRegisterAsyncNotification(int device, UPTKAsyncCallback callbackFunc, void* userData, UPTKAsyncCallbackHandle_t* callback);

/**
* \brief Unregisters an async notification callback
*
* Unregisters \p callback so that the corresponding callback function will stop receiving
* async notifications.
*
* \param device - The device from which to remove \p callback.
* \param callback - The callback instance to unregister from receiving async notifications.
*
* \return
* ::UPTKSuccess
* ::UPTKErrorNotSupported
* ::UPTKErrorInvalidDevice
* ::UPTKErrorInvalidValue
* ::UPTKErrorNotPermitted
* ::UPTKErrorUnknown
* \notefnerr
*
* \sa
* ::UPTKDeviceRegisterAsyncNotification
*/
extern __host__ UPTKError_t UPTKDeviceUnregisterAsyncNotification(int device, UPTKAsyncCallbackHandle_t callback);

/** @} */ /* END UPTKRT_DEVICE */

/**
 * \defgroup UPTKRT_DEVICE_DEPRECATED Device Management [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated device management functions of the UPTK runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the deprecated device management functions of the UPTK runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Returns the shared memory configuration for the current device.
 *
 * \deprecated
 *
 * This function will return in \p pConfig the current size of shared memory banks
 * on the current device. On devices with configurable shared memory banks, 
 * ::UPTKDeviceSetSharedMemConfig can be used to change this setting, so that all 
 * subsequent kernel launches will by default use the new bank size. When 
 * ::UPTKDeviceGetSharedMemConfig is called on devices without configurable shared 
 * memory, it will return the fixed bank size of the hardware.
 *
 * The returned bank configurations can be either:
 * - ::UPTKSharedMemBankSizeFourByte - shared memory bank width is four bytes.
 * - ::UPTKSharedMemBankSizeEightByte - shared memory bank width is eight bytes.
 *
 * \param pConfig - Returned cache configuration
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKDeviceSetCacheConfig,
 * ::UPTKDeviceGetCacheConfig,
 * ::UPTKDeviceSetSharedMemConfig,
 * ::UPTKFuncSetCacheConfig,
 * ::cuCtxGetSharedMemConfig
 */
extern __host__ UPTKError_t UPTKDeviceGetSharedMemConfig(enum UPTKSharedMemConfig *pConfig);

/**
 * \brief Sets the shared memory configuration for the current device.
 *
 * \deprecated
 *
 * On devices with configurable shared memory banks, this function will set
 * the shared memory bank size which is used for all subsequent kernel launches.
 * Any per-function setting of shared memory set via ::UPTKFuncSetSharedMemConfig
 * will override the device wide setting.
 *
 * Changing the shared memory configuration between launches may introduce
 * a device side synchronization point.
 *
 * Changing the shared memory bank size will not increase shared memory usage
 * or affect occupancy of kernels, but may have major effects on performance. 
 * Larger bank sizes will allow for greater potential bandwidth to shared memory,
 * but will change what kinds of accesses to shared memory will result in bank 
 * conflicts.
 *
 * This function will do nothing on devices with fixed shared memory bank size.
 *
 * The supported bank configurations are:
 * - ::UPTKSharedMemBankSizeDefault: set bank width the device default (currently,
 *   four bytes)
 * - ::UPTKSharedMemBankSizeFourByte: set shared memory bank width to be four bytes
 *   natively.
 * - ::UPTKSharedMemBankSizeEightByte: set shared memory bank width to be eight 
 *   bytes natively.
 *
 * \param config - Requested cache configuration
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKDeviceSetCacheConfig,
 * ::UPTKDeviceGetCacheConfig,
 * ::UPTKDeviceGetSharedMemConfig,
 * ::UPTKFuncSetCacheConfig,
 * ::cuCtxSetSharedMemConfig
 */
extern __host__ UPTKError_t UPTKDeviceSetSharedMemConfig(enum UPTKSharedMemConfig config);
/** @} */ /* END UPTKRT_DEVICE_DEPRECATED */

/**
 * \defgroup UPTKRT_THREAD_DEPRECATED Thread Management [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated thread management functions of the UPTK runtime
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes deprecated thread management functions of the UPTK runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Exit and clean up from UPTK launches
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is identical to the 
 * non-deprecated function ::UPTKDeviceReset(), which should be used
 * instead.
 *
 * Explicitly destroys all cleans up all resources associated with the current
 * device in the current process.  Any subsequent API call to this device will 
 * reinitialize the device.  
 *
 * Note that this function will reset the device immediately.  It is the caller's
 * responsibility to ensure that the device is not being accessed by any 
 * other host threads from the process when this function is called.
 *
 * \return
 * ::UPTKSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKDeviceReset
 */
extern __host__ UPTKError_t UPTKThreadExit(void);

/**
 * \brief Wait for compute device to finish
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is similar to the 
 * non-deprecated function ::UPTKDeviceSynchronize(), which should be used
 * instead.
 *
 * Blocks until the device has completed all preceding requested tasks.
 * ::UPTKThreadSynchronize() returns an error if one of the preceding tasks
 * has failed. If the ::UPTKDeviceScheduleBlockingSync flag was set for 
 * this device, the host thread will block until the device has finished 
 * its work.
 *
 * \return
 * ::UPTKSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKDeviceSynchronize
 */
extern __host__ UPTKError_t UPTKThreadSynchronize(void);

/**
 * \brief Set resource limits
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is identical to the 
 * non-deprecated function ::UPTKDeviceSetLimit(), which should be used
 * instead.
 *
 * Setting \p limit to \p value is a request by the application to update
 * the current limit maintained by the device.  The driver is free to
 * modify the requested value to meet h/w requirements (this could be
 * clamping to minimum or maximum values, rounding up to nearest element
 * size, etc).  The application can use ::UPTKThreadGetLimit() to find out
 * exactly what the limit has been set to.
 *
 * Setting each ::UPTKLimit has its own specific restrictions, so each is
 * discussed here.
 *
 * - ::UPTKLimitStackSize controls the stack size of each GPU thread.
 *
 * - ::UPTKLimitPrintfFifoSize controls the size of the shared FIFO
 *   used by the ::printf() device system call.
 *   Setting ::UPTKLimitPrintfFifoSize must be performed before
 *   launching any kernel that uses the ::printf() device
 *   system call, otherwise ::UPTKErrorInvalidValue will be returned.
 *
 * - ::UPTKLimitMallocHeapSize controls the size of the heap used
 *   by the ::malloc() and ::free() device system calls.  Setting
 *   ::UPTKLimitMallocHeapSize must be performed before launching
 *   any kernel that uses the ::malloc() or ::free() device system calls,
 *   otherwise ::UPTKErrorInvalidValue will be returned.
 *
 * \param limit - Limit to set
 * \param value - Size in bytes of limit
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorUnsupportedLimit,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKDeviceSetLimit
 */
extern __host__ UPTKError_t UPTKThreadSetLimit(enum UPTKLimit limit, size_t value);

/**
 * \brief Returns resource limits
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is identical to the 
 * non-deprecated function ::UPTKDeviceGetLimit(), which should be used
 * instead.
 *
 * Returns in \p *pValue the current size of \p limit.  The supported
 * ::UPTKLimit values are:
 * - ::UPTKLimitStackSize: stack size of each GPU thread;
 * - ::UPTKLimitPrintfFifoSize: size of the shared FIFO used by the
 *   ::printf() device system call.
 * - ::UPTKLimitMallocHeapSize: size of the heap used by the
 *   ::malloc() and ::free() device system calls;
 *
 * \param limit  - Limit to query
 * \param pValue - Returned size in bytes of limit
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorUnsupportedLimit,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKDeviceGetLimit
 */
extern __host__ UPTKError_t UPTKThreadGetLimit(size_t *pValue, enum UPTKLimit limit);

extern __host__ const char * UPTKGetErrorString(UPTKError error);

/**
 * \brief Returns the preferred cache configuration for the current device.
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is identical to the 
 * non-deprecated function ::UPTKDeviceGetCacheConfig(), which should be 
 * used instead.
 * 
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this returns through \p pCacheConfig the preferred cache
 * configuration for the current device. This is only a preference. The
 * runtime will use the requested configuration if possible, but it is free to
 * choose a different configuration if required to execute functions.
 *
 * This will return a \p pCacheConfig of ::UPTKFuncCachePreferNone on devices
 * where the size of the L1 cache and shared memory are fixed.
 *
 * The supported cache configurations are:
 * - ::UPTKFuncCachePreferNone: no preference for shared memory or L1 (default)
 * - ::UPTKFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
 * - ::UPTKFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
 *
 * \param pCacheConfig - Returned cache configuration
 *
 * \return
 * ::UPTKSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKDeviceGetCacheConfig
 */
extern __host__ UPTKError_t UPTKThreadGetCacheConfig(enum UPTKFuncCache *pCacheConfig);

/**
 * \brief Sets the preferred cache configuration for the current device.
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is identical to the 
 * non-deprecated function ::UPTKDeviceSetCacheConfig(), which should be 
 * used instead.
 * 
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this sets through \p cacheConfig the preferred cache
 * configuration for the current device. This is only a preference. The
 * runtime will use the requested configuration if possible, but it is free to
 * choose a different configuration if required to execute the function. Any
 * function preference set via
 * \ref ::UPTKFuncSetCacheConfig(const void*, enum UPTKFuncCache) "UPTKFuncSetCacheConfig (C API)"
 * or
 * \ref ::UPTKFuncSetCacheConfig(T*, enum UPTKFuncCache) "UPTKFuncSetCacheConfig (C++ API)"
 * will be preferred over this device-wide setting. Setting the device-wide
 * cache configuration to ::UPTKFuncCachePreferNone will cause subsequent
 * kernel launches to prefer to not change the cache configuration unless
 * required to launch the kernel.
 *
 * This setting does nothing on devices where the size of the L1 cache and
 * shared memory are fixed.
 *
 * Launching a kernel with a different preference than the most recent
 * preference setting may insert a device-side synchronization point.
 *
 * The supported cache configurations are:
 * - ::UPTKFuncCachePreferNone: no preference for shared memory or L1 (default)
 * - ::UPTKFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
 * - ::UPTKFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
 *
 * \param cacheConfig - Requested cache configuration
 *
 * \return
 * ::UPTKSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKDeviceSetCacheConfig
 */
extern __host__ UPTKError_t UPTKThreadSetCacheConfig(enum UPTKFuncCache cacheConfig);

/** @} */ /* END UPTKRT_THREAD_DEPRECATED */

/**
 * \defgroup UPTKRT_ERROR Error Handling
 *
 * ___MANBRIEF___ error handling functions of the UPTK runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the error handling functions of the UPTK runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Returns the last error from a runtime call
 *
 * Returns the last error that has been produced by any of the runtime calls
 * in the same instance of the UPTK Runtime library in the host thread and
 * resets it to ::UPTKSuccess.
 *
 * Note: Multiple instances of the UPTK Runtime library can be present in an
 * application when using a library that statically links the UPTK Runtime.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorMissingConfiguration,
 * ::UPTKErrorMemoryAllocation,
 * ::UPTKErrorInitializationError,
 * ::UPTKErrorLaunchFailure,
 * ::UPTKErrorLaunchTimeout,
 * ::UPTKErrorLaunchOutOfResources,
 * ::UPTKErrorInvalidDeviceFunction,
 * ::UPTKErrorInvalidConfiguration,
 * ::UPTKErrorInvalidDevice,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidPitchValue,
 * ::UPTKErrorInvalidSymbol,
 * ::UPTKErrorUnmapBufferObjectFailed,
 * ::UPTKErrorInvalidDevicePointer,
 * ::UPTKErrorInvalidTexture,
 * ::UPTKErrorInvalidTextureBinding,
 * ::UPTKErrorInvalidChannelDescriptor,
 * ::UPTKErrorInvalidMemcpyDirection,
 * ::UPTKErrorInvalidFilterSetting,
 * ::UPTKErrorInvalidNormSetting,
 * ::UPTKErrorUnknown,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorInsufficientDriver,
 * ::UPTKErrorNoDevice,
 * ::UPTKErrorSetOnActiveProcess,
 * ::UPTKErrorStartupFailure,
 * ::UPTKErrorInvalidPtx,
 * ::UPTKErrorUnsupportedPtxVersion,
 * ::UPTKErrorNoKernelImageForDevice,
 * ::UPTKErrorJitCompilerNotFound,
 * ::UPTKErrorJitCompilationDisabled
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKPeekAtLastError, ::UPTKGetErrorName, ::UPTKGetErrorString, ::UPTKError
 */
extern __host__ UPTKError_t UPTKGetLastError(void);

/**
 * \brief Returns the last error from a runtime call
 *
 * Returns the last error that has been produced by any of the runtime calls
 * in the same instance of the UPTK Runtime library in the host thread. This
 * call does not reset the error to ::UPTKSuccess like ::UPTKGetLastError().
 *
 * Note: Multiple instances of the UPTK Runtime library can be present in an
 * application when using a library that statically links the UPTK Runtime.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorMissingConfiguration,
 * ::UPTKErrorMemoryAllocation,
 * ::UPTKErrorInitializationError,
 * ::UPTKErrorLaunchFailure,
 * ::UPTKErrorLaunchTimeout,
 * ::UPTKErrorLaunchOutOfResources,
 * ::UPTKErrorInvalidDeviceFunction,
 * ::UPTKErrorInvalidConfiguration,
 * ::UPTKErrorInvalidDevice,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidPitchValue,
 * ::UPTKErrorInvalidSymbol,
 * ::UPTKErrorUnmapBufferObjectFailed,
 * ::UPTKErrorInvalidDevicePointer,
 * ::UPTKErrorInvalidTexture,
 * ::UPTKErrorInvalidTextureBinding,
 * ::UPTKErrorInvalidChannelDescriptor,
 * ::UPTKErrorInvalidMemcpyDirection,
 * ::UPTKErrorInvalidFilterSetting,
 * ::UPTKErrorInvalidNormSetting,
 * ::UPTKErrorUnknown,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorInsufficientDriver,
 * ::UPTKErrorNoDevice,
 * ::UPTKErrorSetOnActiveProcess,
 * ::UPTKErrorStartupFailure,
 * ::UPTKErrorInvalidPtx,
 * ::UPTKErrorUnsupportedPtxVersion,
 * ::UPTKErrorNoKernelImageForDevice,
 * ::UPTKErrorJitCompilerNotFound,
 * ::UPTKErrorJitCompilationDisabled
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKGetLastError, ::UPTKGetErrorName, ::UPTKGetErrorString, ::UPTKError
 */
extern __host__ UPTKError_t UPTKPeekAtLastError(void);

/**
 * \brief Returns the string representation of an error code enum name
 *
 * Returns a string containing the name of an error code in the enum.  If the error
 * code is not recognized, "unrecognized error code" is returned.
 *
 * \param error - Error code to convert to string
 *
 * \return
 * \p char* pointer to a NULL-terminated string
 *
 * \sa ::UPTKGetErrorString, ::UPTKGetLastError, ::UPTKPeekAtLastError, ::UPTKError,
 * ::cuGetErrorName
 */
extern __host__ const char* UPTKGetErrorName(UPTKError_t error);

/**
 * \addtogroup UPTKRT_DEVICE 
 *
 * @{
 */

/**
 * \brief Returns the number of compute-capable devices
 *
 * Returns in \p *count the number of devices with compute capability greater
 * or equal to 2.0 that are available for execution.
 *
 * \param count - Returns the number of devices with compute capability
 * greater or equal to 2.0
 *
 * \return
 * ::UPTKSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKGetDevice, ::UPTKSetDevice, ::UPTKGetDeviceProperties,
 * ::UPTKChooseDevice, 
 * ::UPTKInitDevice,
 * ::cuDeviceGetCount
 */
extern __host__ UPTKError_t UPTKGetDeviceCount(int *count);

/**
 * \brief Returns information about the compute-device
 *
 * Returns in \p *prop the properties of device \p dev. The ::UPTKDeviceProp
 * structure is defined as:
 * \code
    struct UPTKDeviceProp {
        char name[256];
        UPTKUUID_t uuid;
        size_t totalGlobalMem;
        size_t sharedMemPerBlock;
        int regsPerBlock;
        int warpSize;
        size_t memPitch;
        int maxThreadsPerBlock;
        int maxThreadsDim[3];
        int maxGridSize[3];
        int clockRate;
        size_t totalConstMem;
        int major;
        int minor;
        size_t textureAlignment;
        size_t texturePitchAlignment;
        int deviceOverlap;
        int multiProcessorCount;
        int kernelExecTimeoutEnabled;
        int integrated;
        int canMapHostMemory;
        int computeMode;
        int maxTexture1D;
        int maxTexture1DMipmap;
        int maxTexture1DLinear;
        int maxTexture2D[2];
        int maxTexture2DMipmap[2];
        int maxTexture2DLinear[3];
        int maxTexture2DGather[2];
        int maxTexture3D[3];
        int maxTexture3DAlt[3];
        int maxTextureCubemap;
        int maxTexture1DLayered[2];
        int maxTexture2DLayered[3];
        int maxTextureCubemapLayered[2];
        int maxSurface1D;
        int maxSurface2D[2];
        int maxSurface3D[3];
        int maxSurface1DLayered[2];
        int maxSurface2DLayered[3];
        int maxSurfaceCubemap;
        int maxSurfaceCubemapLayered[2];
        size_t surfaceAlignment;
        int concurrentKernels;
        int ECCEnabled;
        int pciBusID;
        int pciDeviceID;
        int pciDomainID;
        int tccDriver;
        int asyncEngineCount;
        int unifiedAddressing;
        int memoryClockRate;
        int memoryBusWidth;
        int l2CacheSize;
        int persistingL2CacheMaxSize;
        int maxThreadsPerMultiProcessor;
        int streamPrioritiesSupported;
        int globalL1CacheSupported;
        int localL1CacheSupported;
        size_t sharedMemPerMultiprocessor;
        int regsPerMultiprocessor;
        int managedMemory;
        int isMultiGpuBoard;
        int multiGpuBoardGroupID;
        int singleToDoublePrecisionPerfRatio;
        int pageableMemoryAccess;
        int concurrentManagedAccess;
        int computePreemptionSupported;
        int canUseHostPointerForRegisteredMem;
        int cooperativeLaunch;
        int cooperativeMultiDeviceLaunch;
        int pageableMemoryAccessUsesHostPageTables;
        int directManagedMemAccessFromHost;
        int accessPolicyMaxWindowSize;
    }
 \endcode
 * where:
 * - \ref ::UPTKDeviceProp::name "name[256]" is an ASCII string identifying
 *   the device.
 * - \ref ::UPTKDeviceProp::uuid "uuid" is a 16-byte unique identifier.
 * - \ref ::UPTKDeviceProp::totalGlobalMem "totalGlobalMem" is the total
 *   amount of global memory available on the device in bytes.
 * - \ref ::UPTKDeviceProp::sharedMemPerBlock "sharedMemPerBlock" is the
 *   maximum amount of shared memory available to a thread block in bytes.
 * - \ref ::UPTKDeviceProp::regsPerBlock "regsPerBlock" is the maximum number
 *   of 32-bit registers available to a thread block.
 * - \ref ::UPTKDeviceProp::warpSize "warpSize" is the warp size in threads.
 * - \ref ::UPTKDeviceProp::memPitch "memPitch" is the maximum pitch in
 *   bytes allowed by the memory copy functions that involve memory regions
 *   allocated through ::UPTKMallocPitch().
 * - \ref ::UPTKDeviceProp::maxThreadsPerBlock "maxThreadsPerBlock" is the
 *   maximum number of threads per block.
 * - \ref ::UPTKDeviceProp::maxThreadsDim "maxThreadsDim[3]" contains the
 *   maximum size of each dimension of a block.
 * - \ref ::UPTKDeviceProp::maxGridSize "maxGridSize[3]" contains the
 *   maximum size of each dimension of a grid.
 * - \ref ::UPTKDeviceProp::clockRate "clockRate" is the clock frequency in
 *   kilohertz.
 * - \ref ::UPTKDeviceProp::totalConstMem "totalConstMem" is the total amount
 *   of constant memory available on the device in bytes.
 * - \ref ::UPTKDeviceProp::major "major",
 *   \ref ::UPTKDeviceProp::minor "minor" are the major and minor revision
 *   numbers defining the device's compute capability.
 * - \ref ::UPTKDeviceProp::textureAlignment "textureAlignment" is the
 *   alignment requirement; texture base addresses that are aligned to
 *   \ref ::UPTKDeviceProp::textureAlignment "textureAlignment" bytes do not
 *   need an offset applied to texture fetches.
 * - \ref ::UPTKDeviceProp::texturePitchAlignment "texturePitchAlignment" is the
 *   pitch alignment requirement for 2D texture references that are bound to 
 *   pitched memory.
 * - \ref ::UPTKDeviceProp::deviceOverlap "deviceOverlap" is 1 if the device
 *   can concurrently copy memory between host and device while executing a
 *   kernel, or 0 if not.  Deprecated, use instead asyncEngineCount.
 * - \ref ::UPTKDeviceProp::multiProcessorCount "multiProcessorCount" is the
 *   number of multiprocessors on the device.
 * - \ref ::UPTKDeviceProp::kernelExecTimeoutEnabled "kernelExecTimeoutEnabled"
 *   is 1 if there is a run time limit for kernels executed on the device, or
 *   0 if not.
 * - \ref ::UPTKDeviceProp::integrated "integrated" is 1 if the device is an
 *   integrated (motherboard) GPU and 0 if it is a discrete (card) component.
 * - \ref ::UPTKDeviceProp::canMapHostMemory "canMapHostMemory" is 1 if the
 *   device can map host memory into the UPTK address space for use with
 *   ::UPTKHostAlloc()/::UPTKHostGetDevicePointer(), or 0 if not.
 * - \ref ::UPTKDeviceProp::computeMode "computeMode" is the compute mode
 *   that the device is currently in. Available modes are as follows:
 *   - UPTKComputeModeDefault: Default mode - Device is not restricted and
 *     multiple threads can use ::UPTKSetDevice() with this device.
 *   - UPTKComputeModeProhibited: Compute-prohibited mode - No threads can use
 *     ::UPTKSetDevice() with this device.
 *   - UPTKComputeModeExclusiveProcess: Compute-exclusive-process mode - Many 
 *     threads in one process will be able to use ::UPTKSetDevice() with this device.
 *   <br> When an occupied exclusive mode device is chosen with ::UPTKSetDevice,
 *   all subsequent non-device management runtime functions will return
 *   ::UPTKErrorDevicesUnavailable.
 * - \ref ::UPTKDeviceProp::maxTexture1D "maxTexture1D" is the maximum 1D
 *   texture size.
 * - \ref ::UPTKDeviceProp::maxTexture1DMipmap "maxTexture1DMipmap" is the maximum
 *   1D mipmapped texture texture size.
 * - \ref ::UPTKDeviceProp::maxTexture1DLinear "maxTexture1DLinear" is the maximum
 *   1D texture size for textures bound to linear memory.
 * - \ref ::UPTKDeviceProp::maxTexture2D "maxTexture2D[2]" contains the maximum
 *   2D texture dimensions.
 * - \ref ::UPTKDeviceProp::maxTexture2DMipmap "maxTexture2DMipmap[2]" contains the
 *   maximum 2D mipmapped texture dimensions.
 * - \ref ::UPTKDeviceProp::maxTexture2DLinear "maxTexture2DLinear[3]" contains the 
 *   maximum 2D texture dimensions for 2D textures bound to pitch linear memory.
 * - \ref ::UPTKDeviceProp::maxTexture2DGather "maxTexture2DGather[2]" contains the 
 *   maximum 2D texture dimensions if texture gather operations have to be performed.
 * - \ref ::UPTKDeviceProp::maxTexture3D "maxTexture3D[3]" contains the maximum
 *   3D texture dimensions.
 * - \ref ::UPTKDeviceProp::maxTexture3DAlt "maxTexture3DAlt[3]"
 *   contains the maximum alternate 3D texture dimensions.
 * - \ref ::UPTKDeviceProp::maxTextureCubemap "maxTextureCubemap" is the 
 *   maximum cubemap texture width or height.
 * - \ref ::UPTKDeviceProp::maxTexture1DLayered "maxTexture1DLayered[2]" contains
 *   the maximum 1D layered texture dimensions.
 * - \ref ::UPTKDeviceProp::maxTexture2DLayered "maxTexture2DLayered[3]" contains
 *   the maximum 2D layered texture dimensions.
 * - \ref ::UPTKDeviceProp::maxTextureCubemapLayered "maxTextureCubemapLayered[2]"
 *   contains the maximum cubemap layered texture dimensions.
 * - \ref ::UPTKDeviceProp::maxSurface1D "maxSurface1D" is the maximum 1D
 *   surface size.
 * - \ref ::UPTKDeviceProp::maxSurface2D "maxSurface2D[2]" contains the maximum
 *   2D surface dimensions.
 * - \ref ::UPTKDeviceProp::maxSurface3D "maxSurface3D[3]" contains the maximum
 *   3D surface dimensions.
 * - \ref ::UPTKDeviceProp::maxSurface1DLayered "maxSurface1DLayered[2]" contains
 *   the maximum 1D layered surface dimensions.
 * - \ref ::UPTKDeviceProp::maxSurface2DLayered "maxSurface2DLayered[3]" contains
 *   the maximum 2D layered surface dimensions.
 * - \ref ::UPTKDeviceProp::maxSurfaceCubemap "maxSurfaceCubemap" is the maximum 
 *   cubemap surface width or height.
 * - \ref ::UPTKDeviceProp::maxSurfaceCubemapLayered "maxSurfaceCubemapLayered[2]"
 *   contains the maximum cubemap layered surface dimensions.
 * - \ref ::UPTKDeviceProp::surfaceAlignment "surfaceAlignment" specifies the
 *   alignment requirements for surfaces.
 * - \ref ::UPTKDeviceProp::concurrentKernels "concurrentKernels" is 1 if the
 *   device supports executing multiple kernels within the same context
 *   simultaneously, or 0 if not. It is not guaranteed that multiple kernels
 *   will be resident on the device concurrently so this feature should not be
 *   relied upon for correctness.
 * - \ref ::UPTKDeviceProp::ECCEnabled "ECCEnabled" is 1 if the device has ECC
 *   support turned on, or 0 if not.
 * - \ref ::UPTKDeviceProp::pciBusID "pciBusID" is the PCI bus identifier of
 *   the device.
 * - \ref ::UPTKDeviceProp::pciDeviceID "pciDeviceID" is the PCI device
 *   (sometimes called slot) identifier of the device.
 * - \ref ::UPTKDeviceProp::pciDomainID "pciDomainID" is the PCI domain identifier
 *   of the device.
 * - \ref ::UPTKDeviceProp::tccDriver "tccDriver" is 1 if the device is using a
 *   TCC driver or 0 if not.
 * - \ref ::UPTKDeviceProp::asyncEngineCount "asyncEngineCount" is 1 when the
 *   device can concurrently copy memory between host and device while executing
 *   a kernel. It is 2 when the device can concurrently copy memory between host
 *   and device in both directions and execute a kernel at the same time. It is
 *   0 if neither of these is supported.
 * - \ref ::UPTKDeviceProp::unifiedAddressing "unifiedAddressing" is 1 if the device 
 *   shares a unified address space with the host and 0 otherwise.
 * - \ref ::UPTKDeviceProp::memoryClockRate "memoryClockRate" is the peak memory 
 *   clock frequency in kilohertz.
 * - \ref ::UPTKDeviceProp::memoryBusWidth "memoryBusWidth" is the memory bus width  
 *   in bits.
 * - \ref ::UPTKDeviceProp::l2CacheSize "l2CacheSize" is L2 cache size in bytes. 
 * - \ref ::UPTKDeviceProp::persistingL2CacheMaxSize "persistingL2CacheMaxSize" is L2 cache's maximum persisting lines size in bytes.
 * - \ref ::UPTKDeviceProp::maxThreadsPerMultiProcessor "maxThreadsPerMultiProcessor"  
 *   is the number of maximum resident threads per multiprocessor.
 * - \ref ::UPTKDeviceProp::streamPrioritiesSupported "streamPrioritiesSupported"
 *   is 1 if the device supports stream priorities, or 0 if it is not supported.
 * - \ref ::UPTKDeviceProp::globalL1CacheSupported "globalL1CacheSupported"
 *   is 1 if the device supports caching of globals in L1 cache, or 0 if it is not supported.
 * - \ref ::UPTKDeviceProp::localL1CacheSupported "localL1CacheSupported"
 *   is 1 if the device supports caching of locals in L1 cache, or 0 if it is not supported.
 * - \ref ::UPTKDeviceProp::sharedMemPerMultiprocessor "sharedMemPerMultiprocessor" is the
 *   maximum amount of shared memory available to a multiprocessor in bytes; this amount is
 *   shared by all thread blocks simultaneously resident on a multiprocessor.
 * - \ref ::UPTKDeviceProp::regsPerMultiprocessor "regsPerMultiprocessor" is the maximum number
 *   of 32-bit registers available to a multiprocessor; this number is shared
 *   by all thread blocks simultaneously resident on a multiprocessor.
 * - \ref ::UPTKDeviceProp::managedMemory "managedMemory"
 *   is 1 if the device supports allocating managed memory on this system, or 0 if it is not supported.
 * - \ref ::UPTKDeviceProp::isMultiGpuBoard "isMultiGpuBoard"
 *   is 1 if the device is on a multi-GPU board (e.g. Gemini cards), and 0 if not;
 * - \ref ::UPTKDeviceProp::multiGpuBoardGroupID "multiGpuBoardGroupID" is a unique identifier
 *   for a group of devices associated with the same board.
 *   Devices on the same multi-GPU board will share the same identifier.
 * - \ref ::UPTKDeviceProp::hostNativeAtomicSupported "hostNativeAtomicSupported"
 *   is 1 if the link between the device and the host supports native atomic operations, or 0 if it is not supported.
 * - \ref ::UPTKDeviceProp::singleToDoublePrecisionPerfRatio "singleToDoublePrecisionPerfRatio"  
 *   is the ratio of single precision performance (in floating-point operations per second)
 *   to double precision performance.
 * - \ref ::UPTKDeviceProp::pageableMemoryAccess "pageableMemoryAccess" is 1 if the device supports
 *   coherently accessing pageable memory without calling UPTKHostRegister on it, and 0 otherwise.
 * - \ref ::UPTKDeviceProp::concurrentManagedAccess "concurrentManagedAccess" is 1 if the device can
 *   coherently access managed memory concurrently with the CPU, and 0 otherwise.
 * - \ref ::UPTKDeviceProp::computePreemptionSupported "computePreemptionSupported" is 1 if the device
 *   supports Compute Preemption, and 0 otherwise.
 * - \ref ::UPTKDeviceProp::canUseHostPointerForRegisteredMem "canUseHostPointerForRegisteredMem" is 1 if
 *   the device can access host registered memory at the same virtual address as the CPU, and 0 otherwise.
 * - \ref ::UPTKDeviceProp::cooperativeLaunch "cooperativeLaunch" is 1 if the device supports launching
 *   cooperative kernels via ::UPTKLaunchCooperativeKernel, and 0 otherwise.
 * - \ref ::UPTKDeviceProp::cooperativeMultiDeviceLaunch "cooperativeMultiDeviceLaunch" is 1 if the device
 *   supports launching cooperative kernels via ::UPTKLaunchCooperativeKernelMultiDevice, and 0 otherwise.
 * - \ref ::UPTKDeviceProp::sharedMemPerBlockOptin "sharedMemPerBlockOptin"
 *   is the per device maximum shared memory per block usable by special opt in
 * - \ref ::UPTKDeviceProp::pageableMemoryAccessUsesHostPageTables "pageableMemoryAccessUsesHostPageTables" is 1 if the device accesses
 *   pageable memory via the host's page tables, and 0 otherwise.
 * - \ref ::UPTKDeviceProp::directManagedMemAccessFromHost "directManagedMemAccessFromHost" is 1 if the host can directly access managed
 *   memory on the device without migration, and 0 otherwise.
 * - \ref ::UPTKDeviceProp::maxBlocksPerMultiProcessor "maxBlocksPerMultiProcessor" is the maximum number of thread blocks
 *   that can reside on a multiprocessor.
 * - \ref ::UPTKDeviceProp::accessPolicyMaxWindowSize "accessPolicyMaxWindowSize" is
 *   the maximum value of ::UPTKAccessPolicyWindow::num_bytes.
 * - \ref ::UPTKDeviceProp::reservedSharedMemPerBlock "reservedSharedMemPerBlock"
 *   is the shared memory reserved by UPTK driver per block in bytes
 * - \ref ::UPTKDeviceProp::hostRegisterSupported "hostRegisterSupported"
 *  is 1 if the device supports host memory registration via ::UPTKHostRegister, and 0 otherwise.
 * - \ref ::UPTKDeviceProp::sparseUPTKArraySupported "sparseUPTKArraySupported"
 *  is 1 if the device supports sparse UPTK arrays and sparse UPTK mipmapped arrays, 0 otherwise
 * - \ref ::UPTKDeviceProp::hostRegisterReadOnlySupported "hostRegisterReadOnlySupported"
 *  is 1 if the device supports using the ::UPTKHostRegister flag UPTKHostRegisterReadOnly to register memory that must be mapped as
 *  read-only to the GPU
 * - \ref ::UPTKDeviceProp::timelineSemaphoreInteropSupported "timelineSemaphoreInteropSupported"
 *  is 1 if external timeline semaphore interop is supported on the device, 0 otherwise
 * - \ref ::UPTKDeviceProp::memoryPoolsSupported "memoryPoolsSupported"
 *  is 1 if the device supports using the UPTKMallocAsync and UPTKMemPool family of APIs, 0 otherwise
 * - \ref ::UPTKDeviceProp::gpuDirectRDMASupported "gpuDirectRDMASupported"
 *  is 1 if the device supports GPUDirect RDMA APIs, 0 otherwise
 * - \ref ::UPTKDeviceProp::gpuDirectRDMAFlushWritesOptions "gpuDirectRDMAFlushWritesOptions"
 *  is a bitmask to be interpreted according to the ::UPTKFlushGPUDirectRDMAWritesOptions enum
 * - \ref ::UPTKDeviceProp::gpuDirectRDMAWritesOrdering "gpuDirectRDMAWritesOrdering"
 *  See the ::UPTKGPUDirectRDMAWritesOrdering enum for numerical values
 * - \ref ::UPTKDeviceProp::memoryPoolSupportedHandleTypes "memoryPoolSupportedHandleTypes"
 *  is a bitmask of handle types supported with mempool-based IPC
 * - \ref ::UPTKDeviceProp::deferredMappingUPTKArraySupported "deferredMappingUPTKArraySupported"
 *  is 1 if the device supports deferred mapping UPTK arrays and UPTK mipmapped arrays
 * - \ref ::UPTKDeviceProp::ipcEventSupported "ipcEventSupported"
 *  is 1 if the device supports IPC Events, and 0 otherwise
 * - \ref ::UPTKDeviceProp::unifiedFunctionPointers "unifiedFunctionPointers"
 *  is 1 if the device support unified pointers, and 0 otherwise
 *
 * \param prop   - Properties for the specified device
 * \param device - Device number to get properties for
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDevice
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKGetDeviceCount, ::UPTKGetDevice, ::UPTKSetDevice, ::UPTKChooseDevice,
 * ::UPTKDeviceGetAttribute, 
 * ::UPTKInitDevice,
 * ::cuDeviceGetAttribute,
 * ::cuDeviceGetName
 */
extern __host__ UPTKError_t UPTKGetDeviceProperties(struct UPTKDeviceProp *prop, int device);

/**
 * \brief Returns information about the device
 *
 * Returns in \p *value the integer value of the attribute \p attr on device
 * \p device. The supported attributes are:
 * - ::UPTKDevAttrMaxThreadsPerBlock: Maximum number of threads per block
 * - ::UPTKDevAttrMaxBlockDimX: Maximum x-dimension of a block
 * - ::UPTKDevAttrMaxBlockDimY: Maximum y-dimension of a block
 * - ::UPTKDevAttrMaxBlockDimZ: Maximum z-dimension of a block
 * - ::UPTKDevAttrMaxGridDimX: Maximum x-dimension of a grid
 * - ::UPTKDevAttrMaxGridDimY: Maximum y-dimension of a grid
 * - ::UPTKDevAttrMaxGridDimZ: Maximum z-dimension of a grid
 * - ::UPTKDevAttrMaxSharedMemoryPerBlock: Maximum amount of shared memory
 *   available to a thread block in bytes
 * - ::UPTKDevAttrTotalConstantMemory: Memory available on device for
 *   __constant__ variables in a UPTK C kernel in bytes
 * - ::UPTKDevAttrWarpSize: Warp size in threads
 * - ::UPTKDevAttrMaxPitch: Maximum pitch in bytes allowed by the memory copy
 *   functions that involve memory regions allocated through ::UPTKMallocPitch()
 * - ::UPTKDevAttrMaxTexture1DWidth: Maximum 1D texture width
 * - ::UPTKDevAttrMaxTexture1DLinearWidth: Maximum width for a 1D texture bound
 *   to linear memory
 * - ::UPTKDevAttrMaxTexture1DMipmappedWidth: Maximum mipmapped 1D texture width
 * - ::UPTKDevAttrMaxTexture2DWidth: Maximum 2D texture width
 * - ::UPTKDevAttrMaxTexture2DHeight: Maximum 2D texture height
 * - ::UPTKDevAttrMaxTexture2DLinearWidth: Maximum width for a 2D texture
 *   bound to linear memory
 * - ::UPTKDevAttrMaxTexture2DLinearHeight: Maximum height for a 2D texture
 *   bound to linear memory
 * - ::UPTKDevAttrMaxTexture2DLinearPitch: Maximum pitch in bytes for a 2D
 *   texture bound to linear memory
 * - ::UPTKDevAttrMaxTexture2DMipmappedWidth: Maximum mipmapped 2D texture
 *   width
 * - ::UPTKDevAttrMaxTexture2DMipmappedHeight: Maximum mipmapped 2D texture
 *   height
 * - ::UPTKDevAttrMaxTexture3DWidth: Maximum 3D texture width
 * - ::UPTKDevAttrMaxTexture3DHeight: Maximum 3D texture height
 * - ::UPTKDevAttrMaxTexture3DDepth: Maximum 3D texture depth
 * - ::UPTKDevAttrMaxTexture3DWidthAlt: Alternate maximum 3D texture width,
 *   0 if no alternate maximum 3D texture size is supported
 * - ::UPTKDevAttrMaxTexture3DHeightAlt: Alternate maximum 3D texture height,
 *   0 if no alternate maximum 3D texture size is supported
 * - ::UPTKDevAttrMaxTexture3DDepthAlt: Alternate maximum 3D texture depth,
 *   0 if no alternate maximum 3D texture size is supported
 * - ::UPTKDevAttrMaxTextureCubemapWidth: Maximum cubemap texture width or
 *   height
 * - ::UPTKDevAttrMaxTexture1DLayeredWidth: Maximum 1D layered texture width
 * - ::UPTKDevAttrMaxTexture1DLayeredLayers: Maximum layers in a 1D layered
 *   texture
 * - ::UPTKDevAttrMaxTexture2DLayeredWidth: Maximum 2D layered texture width
 * - ::UPTKDevAttrMaxTexture2DLayeredHeight: Maximum 2D layered texture height
 * - ::UPTKDevAttrMaxTexture2DLayeredLayers: Maximum layers in a 2D layered
 *   texture
 * - ::UPTKDevAttrMaxTextureCubemapLayeredWidth: Maximum cubemap layered
 *   texture width or height
 * - ::UPTKDevAttrMaxTextureCubemapLayeredLayers: Maximum layers in a cubemap
 *   layered texture
 * - ::UPTKDevAttrMaxSurface1DWidth: Maximum 1D surface width
 * - ::UPTKDevAttrMaxSurface2DWidth: Maximum 2D surface width
 * - ::UPTKDevAttrMaxSurface2DHeight: Maximum 2D surface height
 * - ::UPTKDevAttrMaxSurface3DWidth: Maximum 3D surface width
 * - ::UPTKDevAttrMaxSurface3DHeight: Maximum 3D surface height
 * - ::UPTKDevAttrMaxSurface3DDepth: Maximum 3D surface depth
 * - ::UPTKDevAttrMaxSurface1DLayeredWidth: Maximum 1D layered surface width
 * - ::UPTKDevAttrMaxSurface1DLayeredLayers: Maximum layers in a 1D layered
 *   surface
 * - ::UPTKDevAttrMaxSurface2DLayeredWidth: Maximum 2D layered surface width
 * - ::UPTKDevAttrMaxSurface2DLayeredHeight: Maximum 2D layered surface height
 * - ::UPTKDevAttrMaxSurface2DLayeredLayers: Maximum layers in a 2D layered
 *   surface
 * - ::UPTKDevAttrMaxSurfaceCubemapWidth: Maximum cubemap surface width
 * - ::UPTKDevAttrMaxSurfaceCubemapLayeredWidth: Maximum cubemap layered
 *   surface width
 * - ::UPTKDevAttrMaxSurfaceCubemapLayeredLayers: Maximum layers in a cubemap
 *   layered surface
 * - ::UPTKDevAttrMaxRegistersPerBlock: Maximum number of 32-bit registers 
 *   available to a thread block
 * - ::UPTKDevAttrClockRate: Peak clock frequency in kilohertz
 * - ::UPTKDevAttrTextureAlignment: Alignment requirement; texture base
 *   addresses aligned to ::textureAlign bytes do not need an offset applied
 *   to texture fetches
 * - ::UPTKDevAttrTexturePitchAlignment: Pitch alignment requirement for 2D
 *   texture references bound to pitched memory
 * - ::UPTKDevAttrGpuOverlap: 1 if the device can concurrently copy memory
 *   between host and device while executing a kernel, or 0 if not
 * - ::UPTKDevAttrMultiProcessorCount: Number of multiprocessors on the device
 * - ::UPTKDevAttrKernelExecTimeout: 1 if there is a run time limit for kernels
 *   executed on the device, or 0 if not
 * - ::UPTKDevAttrIntegrated: 1 if the device is integrated with the memory
 *   subsystem, or 0 if not
 * - ::UPTKDevAttrCanMapHostMemory: 1 if the device can map host memory into
 *   the UPTK address space, or 0 if not
 * - ::UPTKDevAttrComputeMode: Compute mode is the compute mode that the device
 *   is currently in. Available modes are as follows:
 *   - ::UPTKComputeModeDefault: Default mode - Device is not restricted and
 *     multiple threads can use ::UPTKSetDevice() with this device.
 *   - ::UPTKComputeModeProhibited: Compute-prohibited mode - No threads can use
 *     ::UPTKSetDevice() with this device.
 *   - ::UPTKComputeModeExclusiveProcess: Compute-exclusive-process mode - Many 
 *     threads in one process will be able to use ::UPTKSetDevice() with this
 *     device.
 * - ::UPTKDevAttrConcurrentKernels: 1 if the device supports executing
 *   multiple kernels within the same context simultaneously, or 0 if
 *   not. It is not guaranteed that multiple kernels will be resident on the
 *   device concurrently so this feature should not be relied upon for
 *   correctness.
 * - ::UPTKDevAttrEccEnabled: 1 if error correction is enabled on the device,
 *   0 if error correction is disabled or not supported by the device
 * - ::UPTKDevAttrPciBusId: PCI bus identifier of the device
 * - ::UPTKDevAttrPciDeviceId: PCI device (also known as slot) identifier of
 *   the device
 * - ::UPTKDevAttrTccDriver: 1 if the device is using a TCC driver. TCC is only
 *   available on Tesla hardware running Windows Vista or later.
 * - ::UPTKDevAttrMemoryClockRate: Peak memory clock frequency in kilohertz
 * - ::UPTKDevAttrGlobalMemoryBusWidth: Global memory bus width in bits
 * - ::UPTKDevAttrL2CacheSize: Size of L2 cache in bytes. 0 if the device
 *   doesn't have L2 cache.
 * - ::UPTKDevAttrMaxThreadsPerMultiProcessor: Maximum resident threads per 
 *   multiprocessor
 * - ::UPTKDevAttrUnifiedAddressing: 1 if the device shares a unified address
 *   space with the host, or 0 if not
 * - ::UPTKDevAttrComputeCapabilityMajor: Major compute capability version
 *   number
 * - ::UPTKDevAttrComputeCapabilityMinor: Minor compute capability version
 *   number
 * - ::UPTKDevAttrStreamPrioritiesSupported: 1 if the device supports stream
 *   priorities, or 0 if not
 * - ::UPTKDevAttrGlobalL1CacheSupported: 1 if device supports caching globals 
 *    in L1 cache, 0 if not
 * - ::UPTKDevAttrLocalL1CacheSupported: 1 if device supports caching locals 
 *    in L1 cache, 0 if not
 * - ::UPTKDevAttrMaxSharedMemoryPerMultiprocessor: Maximum amount of shared memory
 *   available to a multiprocessor in bytes; this amount is shared by all 
 *   thread blocks simultaneously resident on a multiprocessor
 * - ::UPTKDevAttrMaxRegistersPerMultiprocessor: Maximum number of 32-bit registers 
 *   available to a multiprocessor; this number is shared by all thread blocks
 *   simultaneously resident on a multiprocessor
 * - ::UPTKDevAttrManagedMemory: 1 if device supports allocating
 *   managed memory, 0 if not
 * - ::UPTKDevAttrIsMultiGpuBoard: 1 if device is on a multi-GPU board, 0 if not
 * - ::UPTKDevAttrMultiGpuBoardGroupID: Unique identifier for a group of devices on the
 *   same multi-GPU board
 * - ::UPTKDevAttrHostNativeAtomicSupported: 1 if the link between the device and the
 *   host supports native atomic operations
 * - ::UPTKDevAttrSingleToDoublePrecisionPerfRatio: Ratio of single precision performance
 *   (in floating-point operations per second) to double precision performance
 * - ::UPTKDevAttrPageableMemoryAccess: 1 if the device supports coherently accessing
 *   pageable memory without calling UPTKHostRegister on it, and 0 otherwise
 * - ::UPTKDevAttrConcurrentManagedAccess: 1 if the device can coherently access managed
 *   memory concurrently with the CPU, and 0 otherwise
 * - ::UPTKDevAttrComputePreemptionSupported: 1 if the device supports
 *   Compute Preemption, 0 if not
 * - ::UPTKDevAttrCanUseHostPointerForRegisteredMem: 1 if the device can access host
 *   registered memory at the same virtual address as the CPU, and 0 otherwise
 * - ::UPTKDevAttrCooperativeLaunch: 1 if the device supports launching cooperative kernels
 *   via ::UPTKLaunchCooperativeKernel, and 0 otherwise
 * - ::UPTKDevAttrCooperativeMultiDeviceLaunch: 1 if the device supports launching cooperative
 *   kernels via ::UPTKLaunchCooperativeKernelMultiDevice, and 0 otherwise
 * - ::UPTKDevAttrCanFlushRemoteWrites: 1 if the device supports flushing of outstanding 
 *   remote writes, and 0 otherwise
 * - ::UPTKDevAttrHostRegisterSupported: 1 if the device supports host memory registration
 *   via ::UPTKHostRegister, and 0 otherwise
 * - ::UPTKDevAttrPageableMemoryAccessUsesHostPageTables: 1 if the device accesses pageable memory via the
 *   host's page tables, and 0 otherwise
 * - ::UPTKDevAttrDirectManagedMemAccessFromHost: 1 if the host can directly access managed memory on the device
 *   without migration, and 0 otherwise
 * - ::UPTKDevAttrMaxSharedMemoryPerBlockOptin: Maximum per block shared memory size on the device. This value can
 *   be opted into when using ::UPTKFuncSetAttribute
 * - ::UPTKDevAttrMaxBlocksPerMultiprocessor: Maximum number of thread blocks that can reside on a multiprocessor
 * - ::UPTKDevAttrMaxPersistingL2CacheSize: Maximum L2 persisting lines capacity setting in bytes
 * - ::UPTKDevAttrMaxAccessPolicyWindowSize: Maximum value of UPTKAccessPolicyWindow::num_bytes
 * - ::UPTKDevAttrReservedSharedMemoryPerBlock: Shared memory reserved by UPTK driver per block in bytes
 * - ::UPTKDevAttrSparseUPTKArraySupported: 1 if the device supports sparse UPTK arrays and sparse UPTK mipmapped arrays.
 * - ::UPTKDevAttrHostRegisterReadOnlySupported: Device supports using the ::UPTKHostRegister flag UPTKHostRegisterReadOnly
 *   to register memory that must be mapped as read-only to the GPU
 * - ::UPTKDevAttrMemoryPoolsSupported: 1 if the device supports using the UPTKMallocAsync and UPTKMemPool family of APIs, and 0 otherwise
 * - ::UPTKDevAttrGPUDirectRDMASupported: 1 if the device supports GPUDirect RDMA APIs, and 0 otherwise
 * - ::UPTKDevAttrGPUDirectRDMAFlushWritesOptions: bitmask to be interpreted according to the ::UPTKFlushGPUDirectRDMAWritesOptions enum 
 * - ::UPTKDevAttrGPUDirectRDMAWritesOrdering: see the ::UPTKGPUDirectRDMAWritesOrdering enum for numerical values
 * - ::UPTKDevAttrMemoryPoolSupportedHandleTypes: Bitmask of handle types supported with mempool based IPC
 * - ::UPTKDevAttrDeferredMappingUPTKArraySupported : 1 if the device supports deferred mapping UPTK arrays and UPTK mipmapped arrays.
 * - ::UPTKDevAttrIpcEventSupport: 1 if the device supports IPC Events.
 * - ::UPTKDevAttrNumaConfig: NUMA configuration of a device: value is of type ::UPTKDeviceNumaConfig enum
 * - ::UPTKDevAttrNumaId: NUMA node ID of the GPU memory
 *
 * \param value  - Returned device attribute value
 * \param attr   - Device attribute to query
 * \param device - Device number to query 
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDevice,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKGetDeviceCount, ::UPTKGetDevice, ::UPTKSetDevice, ::UPTKChooseDevice,
 * ::UPTKGetDeviceProperties, 
 * ::UPTKInitDevice,
 * ::cuDeviceGetAttribute
 */
extern __host__ UPTKError_t UPTKDeviceGetAttribute(int *value, enum UPTKDeviceAttr attr, int device);

/**
 * \brief Returns the default mempool of a device
 *
 * The default mempool of a device contains device memory from that device.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDevice,
 * ::UPTKErrorInvalidValue
 * ::UPTKErrorNotSupported
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cuDeviceGetDefaultMemPool, ::UPTKMallocAsync, ::UPTKMemPoolTrimTo, ::UPTKMemPoolGetAttribute, ::UPTKDeviceSetMemPool, ::UPTKMemPoolSetAttribute, ::UPTKMemPoolSetAccess
 */
extern __host__ UPTKError_t UPTKDeviceGetDefaultMemPool(UPTKMemPool_t *memPool, int device);


/**
 * \brief Sets the current memory pool of a device
 *
 * The memory pool must be local to the specified device.
 * Unless a mempool is specified in the ::UPTKMallocAsync call,
 * ::UPTKMallocAsync allocates from the current mempool of the provided stream's device.
 * By default, a device's current memory pool is its default memory pool.
 *
 * \note Use ::UPTKMallocFromPoolAsync to specify asynchronous allocations from a device different
 * than the one the stream runs on.
 *
 * \returns
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * ::UPTKErrorInvalidDevice
 * ::UPTKErrorNotSupported
 * \notefnerr
 * \note_callback
 *
 * \sa ::cuDeviceSetMemPool, ::UPTKDeviceGetMemPool, ::UPTKDeviceGetDefaultMemPool, ::UPTKMemPoolCreate, ::UPTKMemPoolDestroy, ::UPTKMallocFromPoolAsync
 */
extern __host__ UPTKError_t UPTKDeviceSetMemPool(int device, UPTKMemPool_t memPool);

/**
 * \brief Gets the current mempool for a device
 *
 * Returns the last pool provided to ::UPTKDeviceSetMemPool for this device
 * or the device's default memory pool if ::UPTKDeviceSetMemPool has never been called.
 * By default the current mempool is the default mempool for a device,
 * otherwise the returned pool must have been set with ::cuDeviceSetMemPool or ::UPTKDeviceSetMemPool.
 *
 * \returns
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * ::UPTKErrorNotSupported
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cuDeviceGetMemPool, ::UPTKDeviceGetDefaultMemPool, ::UPTKDeviceSetMemPool
 */
extern __host__ UPTKError_t UPTKDeviceGetMemPool(UPTKMemPool_t *memPool, int device);

/**
 * \brief Return NvSciSync attributes that this device can support.
 *
 * Returns in \p nvSciSyncAttrList, the properties of NvSciSync that
 * this UPTK device, \p dev can support. The returned \p nvSciSyncAttrList
 * can be used to create an NvSciSync that matches this device's capabilities.
 * 
 * If NvSciSyncAttrKey_RequiredPerm field in \p nvSciSyncAttrList is
 * already set this API will return ::UPTKErrorInvalidValue.
 * 
 * The applications should set \p nvSciSyncAttrList to a valid 
 * NvSciSyncAttrList failing which this API will return
 * ::UPTKErrorInvalidHandle.
 * 
 * The \p flags controls how applications intends to use
 * the NvSciSync created from the \p nvSciSyncAttrList. The valid flags are:
 * - ::UPTKNvSciSyncAttrSignal, specifies that the applications intends to 
 * signal an NvSciSync on this UPTK device.
 * - ::UPTKNvSciSyncAttrWait, specifies that the applications intends to 
 * wait on an NvSciSync on this UPTK device.
 *
 * At least one of these flags must be set, failing which the API
 * returns ::UPTKErrorInvalidValue. Both the flags are orthogonal
 * to one another: a developer may set both these flags that allows to
 * set both wait and signal specific attributes in the same \p nvSciSyncAttrList.
 *
 * Note that this API updates the input \p nvSciSyncAttrList with values equivalent
 * to the following public attribute key-values:
 * NvSciSyncAttrKey_RequiredPerm is set to
 * - NvSciSyncAccessPerm_SignalOnly if ::UPTKNvSciSyncAttrSignal is set in \p flags.
 * - NvSciSyncAccessPerm_WaitOnly if ::UPTKNvSciSyncAttrWait is set in \p flags.
 * - NvSciSyncAccessPerm_WaitSignal if both ::UPTKNvSciSyncAttrWait and
 * ::UPTKNvSciSyncAttrSignal are set in \p flags.
 * NvSciSyncAttrKey_PrimitiveInfo is set to
 * - NvSciSyncAttrValPrimitiveType_SysmemSemaphore on any valid \p device.
 * - NvSciSyncAttrValPrimitiveType_Syncpoint if \p device is a Tegra device.
 * - NvSciSyncAttrValPrimitiveType_SysmemSemaphorePayload64b if \p device is GA10X+.
 * NvSciSyncAttrKey_GpuId is set to the same UUID that is returned in 
 * \p UPTKDeviceProp.uuid from ::UPTKDeviceGetProperties for this \p device.
 *
 * \param nvSciSyncAttrList     - Return NvSciSync attributes supported.
 * \param device                - Valid UPTK Device to get NvSciSync attributes for.
 * \param flags                 - flags describing NvSciSync usage.
 *
 * \return
 *
 * ::UPTKSuccess,
 * ::UPTKErrorDeviceUninitialized,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidHandle,
 * ::UPTKErrorInvalidDevice,
 * ::UPTKErrorNotSupported,
 * ::UPTKErrorMemoryAllocation
 *
 * \sa
 * ::UPTKImportExternalSemaphore,
 * ::UPTKDestroyExternalSemaphore,
 * ::UPTKSignalExternalSemaphoresAsync,
 * ::UPTKWaitExternalSemaphoresAsync
 */
extern __host__ UPTKError_t UPTKDeviceGetNvSciSyncAttributes(void *nvSciSyncAttrList, int device, int flags);

/**
 * \brief Queries attributes of the link between two devices.
 *
 * Returns in \p *value the value of the requested attribute \p attrib of the
 * link between \p srcDevice and \p dstDevice. The supported attributes are:
 * - ::UPTKDevP2PAttrPerformanceRank: A relative value indicating the
 *   performance of the link between two devices. Lower value means better
 *   performance (0 being the value used for most performant link).
 * - ::UPTKDevP2PAttrAccessSupported: 1 if peer access is enabled.
 * - ::UPTKDevP2PAttrNativeAtomicSupported: 1 if native atomic operations over
 *   the link are supported.
 * - ::UPTKDevP2PAttrUPTKArrayAccessSupported: 1 if accessing UPTK arrays over
 *   the link is supported.
 *
 * Returns ::UPTKErrorInvalidDevice if \p srcDevice or \p dstDevice are not valid
 * or if they represent the same device.
 *
 * Returns ::UPTKErrorInvalidValue if \p attrib is not valid or if \p value is
 * a null pointer.
 *
 * \param value         - Returned value of the requested attribute
 * \param attrib        - The requested attribute of the link between \p srcDevice and \p dstDevice.
 * \param srcDevice     - The source device of the target link.
 * \param dstDevice     - The destination device of the target link.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDevice,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKDeviceEnablePeerAccess,
 * ::UPTKDeviceDisablePeerAccess,
 * ::UPTKDeviceCanAccessPeer,
 * ::cuDeviceGetP2PAttribute
 */
extern __host__ UPTKError_t UPTKDeviceGetP2PAttribute(int *value, enum UPTKDeviceP2PAttr attr, int srcDevice, int dstDevice);

/**
 * \brief Select compute-device which best matches criteria
 *
 * Returns in \p *device the device which has properties that best match
 * \p *prop.
 *
 * \param device - Device with best match
 * \param prop   - Desired device properties
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKGetDeviceCount, ::UPTKGetDevice, ::UPTKSetDevice,
 * ::UPTKGetDeviceProperties, 
 * ::UPTKInitDevice
 */
extern __host__ UPTKError_t UPTKChooseDevice(int *device, const struct UPTKDeviceProp *prop);
/**
 * \brief Initialize device to be used for GPU executions
 *
 * This function will initialize the UPTK Runtime structures and primary context on \p device when called,
 * but the context will not be made current to \p device.
 *
 * When ::UPTKInitDeviceFlagsAreValid is set in \p flags, deviceFlags are applied to the requested device.
 * The values of deviceFlags match those of the flags parameters in ::UPTKSetDeviceFlags. 
 * The effect may be verified by ::UPTKGetDeviceFlags.
 *
 * This function will return an error if the device is in ::UPTKComputeModeExclusiveProcess
 * and is occupied by another process or if the device is in ::UPTKComputeModeProhibited.
 *
 * \param device - Device on which the runtime will initialize itself.
 * \param deviceFlags - Parameters for device operation.
 * \param flags - Flags for controlling the device initialization.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDevice,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKGetDeviceCount, ::UPTKGetDevice, ::UPTKGetDeviceProperties,
 * ::UPTKChooseDevice, ::UPTKSetDevice
 * ::cuCtxSetCurrent
 */
extern __host__ UPTKError_t UPTKInitDevice(int device, unsigned int deviceFlags, unsigned int flags);
/**
 * \brief Set device to be used for GPU executions
 *
 * Sets \p device as the current device for the calling host thread.
 * Valid device id's are 0 to (::UPTKGetDeviceCount() - 1).
 *
 * Any device memory subsequently allocated from this host thread
 * using ::UPTKMalloc(), ::UPTKMallocPitch() or ::UPTKMallocArray()
 * will be physically resident on \p device.  Any host memory allocated
 * from this host thread using ::UPTKMallocHost() or ::UPTKHostAlloc() 
 * or ::UPTKHostRegister() will have its lifetime associated  with
 * \p device.  Any streams or events created from this host thread will 
 * be associated with \p device.  Any kernels launched from this host
 * thread using the <<<>>> operator or ::UPTKLaunchKernel() will be executed
 * on \p device.
 *
 * This call may be made from any host thread, to any device, and at 
 * any time.  This function will do no synchronization with the previous 
 * or new device, 
 * and should only take significant time when it initializes the runtime's context state.
 * This call will bind the primary context of the specified device to the calling thread and all the
 * subsequent memory allocations, stream and event creations, and kernel launches
 * will be associated with the primary context. 
 * This function will also immediately initialize the runtime state on the primary context, 
 * and the context will be current on \p device immediately. This function will return an 
 * error if the device is in ::UPTKComputeModeExclusiveProcess and is occupied by another 
 * process or if the device is in ::UPTKComputeModeProhibited.
 * 
 * It is not required to call ::UPTKInitDevice before using this function.
 * \param device - Device on which the active host thread should execute the
 * device code.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDevice,
 * ::UPTKErrorDeviceUnavailable,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKGetDeviceCount, ::UPTKGetDevice, ::UPTKGetDeviceProperties,
 * ::UPTKChooseDevice,
 * ::UPTKInitDevice,
 * ::cuCtxSetCurrent
 */
extern __host__ UPTKError_t UPTKSetDevice(int device);

/**
 * \brief Returns which device is currently being used
 *
 * Returns in \p *device the current device for the calling host thread.
 *
 * \param device - Returns the device on which the active host thread
 * executes the device code.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorDeviceUnavailable,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKGetDeviceCount, ::UPTKSetDevice, ::UPTKGetDeviceProperties,
 * ::UPTKChooseDevice,
 * ::cuCtxGetCurrent
 */
extern __host__ UPTKError_t UPTKGetDevice(int *device);

/**
 * \brief Set a list of devices that can be used for UPTK
 *
 * Sets a list of devices for UPTK execution in priority order using
 * \p device_arr. The parameter \p len specifies the number of elements in the
 * list.  UPTK will try devices from the list sequentially until it finds one
 * that works.  If this function is not called, or if it is called with a \p len
 * of 0, then UPTK will go back to its default behavior of trying devices
 * sequentially from a default list containing all of the available UPTK
 * devices in the system. If a specified device ID in the list does not exist,
 * this function will return ::UPTKErrorInvalidDevice. If \p len is not 0 and
 * \p device_arr is NULL or if \p len exceeds the number of devices in
 * the system, then ::UPTKErrorInvalidValue is returned.
 *
 * \param device_arr - List of devices to try
 * \param len        - Number of devices in specified list
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidDevice
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKGetDeviceCount, ::UPTKSetDevice, ::UPTKGetDeviceProperties,
 * ::UPTKSetDeviceFlags,
 * ::UPTKChooseDevice
 */
extern __host__ UPTKError_t UPTKSetValidDevices(int *device_arr, int len);

/**
 * \brief Sets flags to be used for device executions
 * 
 * Records \p flags as the flags for the current device. If the current device
 * has been set and that device has already been initialized, the previous flags
 * are overwritten. If the current device has not been initialized, it is
 * initialized with the provided flags. If no device has been made current to
 * the calling thread, a default device is selected and initialized with the
 * provided flags.
 * 
 * The three LSBs of the \p flags parameter can be used to control how the CPU
 * thread interacts with the OS scheduler when waiting for results from the
 * device.
 *
 * - ::UPTKDeviceScheduleAuto: The default value if the \p flags parameter is
 * zero, uses a heuristic based on the number of active UPTK contexts in the
 * process \p C and the number of logical processors in the system \p P. If
 * \p C \> \p P, then UPTK will yield to other OS threads when waiting for the
 * device, otherwise UPTK will not yield while waiting for results and
 * actively spin on the processor. Additionally, on Tegra devices,
 * ::UPTKDeviceScheduleAuto uses a heuristic based on the power profile of
 * the platform and may choose ::UPTKDeviceScheduleBlockingSync for low-powered
 * devices.
 * - ::UPTKDeviceScheduleSpin: Instruct UPTK to actively spin when waiting for
 * results from the device. This can decrease latency when waiting for the
 * device, but may lower the performance of CPU threads if they are performing
 * work in parallel with the UPTK thread.
 * - ::UPTKDeviceScheduleYield: Instruct UPTK to yield its thread when waiting
 * for results from the device. This can increase latency when waiting for the
 * device, but can increase the performance of CPU threads performing work in
 * parallel with the device.
 * - ::UPTKDeviceScheduleBlockingSync: Instruct UPTK to block the CPU thread 
 * on a synchronization primitive when waiting for the device to finish work.
 * - ::UPTKDeviceBlockingSync: Instruct UPTK to block the CPU thread on a 
 * synchronization primitive when waiting for the device to finish work. <br>
 * \ref deprecated "Deprecated:" This flag was deprecated as of UPTK 4.0 and
 * replaced with ::UPTKDeviceScheduleBlockingSync.
 * - ::UPTKDeviceMapHost: This flag enables allocating pinned
 * host memory that is accessible to the device. It is implicit for the
 * runtime but may be absent if a context is created using the driver API.
 * If this flag is not set, ::UPTKHostGetDevicePointer() will always return
 * a failure code.
 * - ::UPTKDeviceLmemResizeToMax: Instruct UPTK to not reduce local memory
 * after resizing local memory for a kernel. This can prevent thrashing by
 * local memory allocations when launching many kernels with high local
 * memory usage at the cost of potentially increased memory usage. <br>
 * \ref deprecated "Deprecated:" This flag is deprecated and the behavior enabled
 * by this flag is now the default and cannot be disabled.
 * - ::UPTKDeviceSyncMemops: Ensures that synchronous memory operations initiated
 * on this context will always synchronize. See further documentation in the
 * section titled "API Synchronization behavior" to learn more about cases when
 * synchronous memory operations can exhibit asynchronous behavior.
 *
 * \param flags - Parameters for device operation
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKGetDeviceFlags, ::UPTKGetDeviceCount, ::UPTKGetDevice, ::UPTKGetDeviceProperties,
 * ::UPTKSetDevice, ::UPTKSetValidDevices,
 * ::UPTKInitDevice,
 * ::UPTKChooseDevice,
 * ::cuDevicePrimaryCtxSetFlags
 */
extern __host__ UPTKError_t UPTKSetDeviceFlags( unsigned int flags );

/**
 * \brief Gets the flags for the current device
 *
 * 
 * Returns in \p flags the flags for the current device. If there is a current
 * device for the calling thread, the flags for the device are returned. If
 * there is no current device, the flags for the first device are returned,
 * which may be the default flags.  Compare to the behavior of
 * ::UPTKSetDeviceFlags.
 *
 * Typically, the flags returned should match the behavior that will be seen
 * if the calling thread uses a device after this call, without any change to
 * the flags or current device inbetween by this or another thread.  Note that
 * if the device is not initialized, it is possible for another thread to
 * change the flags for the current device before it is initialized.
 * Additionally, when using exclusive mode, if this thread has not requested a
 * specific device, it may use a device other than the first device, contrary
 * to the assumption made by this function.
 *
 * If a context has been created via the driver API and is current to the
 * calling thread, the flags for that context are always returned.
 *
 * Flags returned by this function may specifically include ::UPTKDeviceMapHost
 * even though it is not accepted by ::UPTKSetDeviceFlags because it is
 * implicit in runtime API flags.  The reason for this is that the current
 * context may have been created via the driver API in which case the flag is
 * not implicit and may be unset.
 *
 * \param flags - Pointer to store the device flags
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDevice
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKGetDevice, ::UPTKGetDeviceProperties,
 * ::UPTKSetDevice, ::UPTKSetDeviceFlags,
 * ::UPTKInitDevice,
 * ::cuCtxGetFlags,
 * ::cuDevicePrimaryCtxGetState
 */
extern __host__ UPTKError_t UPTKGetDeviceFlags( unsigned int *flags );
/** @} */ /* END UPTKRT_DEVICE */

/**
 * \defgroup UPTKRT_STREAM Stream Management
 *
 * ___MANBRIEF___ stream management functions of the UPTK runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the stream management functions of the UPTK runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Create an asynchronous stream
 *
 * Creates a new asynchronous stream on the context that is current to the calling host thread.
 * If no context is current to the calling host thread, then the primary context for a device
 * is selected, made current to the calling thread, and initialized before creating a stream on it.
 *
 * \param pStream - Pointer to new stream identifier
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKStreamCreateWithPriority,
 * ::UPTKStreamCreateWithFlags,
 * ::UPTKStreamGetPriority,
 * ::UPTKStreamGetFlags,
 * ::UPTKStreamQuery,
 * ::UPTKStreamSynchronize,
 * ::UPTKStreamWaitEvent,
 * ::UPTKStreamAddCallback,
 * ::UPTKSetDevice,
 * ::UPTKStreamDestroy,
 * ::cuStreamCreate
 */
extern __host__ UPTKError_t UPTKStreamCreate(UPTKStream_t *pStream);

/**
 * \brief Create an asynchronous stream
 *
 * Creates a new asynchronous stream on the context that is current to the calling host thread.
 * If no context is current to the calling host thread, then the primary context for a device
 * is selected, made current to the calling thread, and initialized before creating a stream on it.
 * The \p flags argument determines the behaviors of the stream.  Valid values for \p flags are
 * - ::UPTKStreamDefault: Default stream creation flag.
 * - ::UPTKStreamNonBlocking: Specifies that work running in the created 
 *   stream may run concurrently with work in stream 0 (the NULL stream), and that
 *   the created stream should perform no implicit synchronization with stream 0.
 *
 * \param pStream - Pointer to new stream identifier
 * \param flags   - Parameters for stream creation
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKStreamCreate,
 * ::UPTKStreamCreateWithPriority,
 * ::UPTKStreamGetFlags,
 * ::UPTKStreamQuery,
 * ::UPTKStreamSynchronize,
 * ::UPTKStreamWaitEvent,
 * ::UPTKStreamAddCallback,
 * ::UPTKSetDevice,
 * ::UPTKStreamDestroy,
 * ::cuStreamCreate
 */
extern __host__ UPTKError_t UPTKStreamCreateWithFlags(UPTKStream_t *pStream, unsigned int flags);

/**
 * \brief Create an asynchronous stream with the specified priority
 *
 * Creates a stream with the specified priority and returns a handle in \p pStream.
 * The stream is created on the context that is current to the calling host thread.
 * If no context is current to the calling host thread, then the primary context for a device
 * is selected, made current to the calling thread, and initialized before creating a stream on it.
 * This affects the scheduling priority of work in the stream. Priorities provide a
 * hint to preferentially run work with higher priority when possible, but do
 * not preempt already-running work or provide any other functional guarantee on
 * execution order.
 *
 * \p priority follows a convention where lower numbers represent higher priorities.
 * '0' represents default priority. The range of meaningful numerical priorities can
 * be queried using ::UPTKDeviceGetStreamPriorityRange. If the specified priority is
 * outside the numerical range returned by ::UPTKDeviceGetStreamPriorityRange,
 * it will automatically be clamped to the lowest or the highest number in the range.
 *
 * \param pStream  - Pointer to new stream identifier
 * \param flags    - Flags for stream creation. See ::UPTKStreamCreateWithFlags for a list of valid flags that can be passed
 * \param priority - Priority of the stream. Lower numbers represent higher priorities.
 *                   See ::UPTKDeviceGetStreamPriorityRange for more information about
 *                   the meaningful stream priorities that can be passed.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \note Stream priorities are supported only on GPUs
 * with compute capability 3.5 or higher.
 *
 * \note In the current implementation, only compute kernels launched in
 * priority streams are affected by the stream's priority. Stream priorities have
 * no effect on host-to-device and device-to-host memory operations.
 *
 * \sa ::UPTKStreamCreate,
 * ::UPTKStreamCreateWithFlags,
 * ::UPTKDeviceGetStreamPriorityRange,
 * ::UPTKStreamGetPriority,
 * ::UPTKStreamQuery,
 * ::UPTKStreamWaitEvent,
 * ::UPTKStreamAddCallback,
 * ::UPTKStreamSynchronize,
 * ::UPTKSetDevice,
 * ::UPTKStreamDestroy,
 * ::cuStreamCreateWithPriority
 */
extern __host__ UPTKError_t UPTKStreamCreateWithPriority(UPTKStream_t *pStream, unsigned int flags, int priority);

/**
 * \brief Query the priority of a stream
 *
 * Query the priority of a stream. The priority is returned in in \p priority.
 * Note that if the stream was created with a priority outside the meaningful
 * numerical range returned by ::UPTKDeviceGetStreamPriorityRange,
 * this function returns the clamped priority.
 * See ::UPTKStreamCreateWithPriority for details about priority clamping.
 *
 * \param hStream    - Handle to the stream to be queried
 * \param priority   - Pointer to a signed integer in which the stream's priority is returned
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKStreamCreateWithPriority,
 * ::UPTKDeviceGetStreamPriorityRange,
 * ::UPTKStreamGetFlags,
 * ::cuStreamGetPriority
 */
extern __host__ UPTKError_t UPTKStreamGetPriority(UPTKStream_t hStream, int *priority);

/**
 * \brief Query the flags of a stream
 *
 * Query the flags of a stream. The flags are returned in \p flags.
 * See ::UPTKStreamCreateWithFlags for a list of valid flags.
 *
 * \param hStream - Handle to the stream to be queried
 * \param flags   - Pointer to an unsigned integer in which the stream's flags are returned
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKStreamCreateWithPriority,
 * ::UPTKStreamCreateWithFlags,
 * ::UPTKStreamGetPriority,
 * ::cuStreamGetFlags
 */
extern __host__ UPTKError_t UPTKStreamGetFlags(UPTKStream_t hStream, unsigned int *flags);

/**
 * \brief Query the Id of a stream
 *
 * Query the Id of a stream. The Id is returned in \p streamId.
 * The Id is unique for the life of the program.
 *
 * The stream handle \p hStream can refer to any of the following:
 * <ul>
 *   <li>a stream created via any of the UPTK runtime APIs such as ::UPTKStreamCreate, 
 *   ::UPTKStreamCreateWithFlags and ::UPTKStreamCreateWithPriority, or their driver 
 *   API equivalents such as ::cuStreamCreate or ::cuStreamCreateWithPriority.
 *   Passing an invalid handle will result in undefined behavior.</li>
 *   <li>any of the special streams such as the NULL stream, ::UPTKStreamLegacy 
 *   and ::UPTKStreamPerThread respectively.  The driver API equivalents of these 
 *   are also accepted which are NULL, ::CU_STREAM_LEGACY and ::CU_STREAM_PER_THREAD.</li>
 * </ul>
 * 
 * \param hStream    - Handle to the stream to be queried
 * \param streamId   - Pointer to an unsigned long long in which the stream Id is returned
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKStreamCreateWithPriority,
 * ::UPTKStreamCreateWithFlags,
 * ::UPTKStreamGetPriority,
 * ::UPTKStreamGetFlags,
 * ::cuStreamGetId
 */
extern __host__ UPTKError_t UPTKStreamGetId(UPTKStream_t hStream, unsigned long long *streamId);

/**
 * \brief Resets all persisting lines in cache to normal status.
 *
 * Resets all persisting lines in cache to normal status.
 * Takes effect on function return.
 *
 * \return
 * ::UPTKSuccess,
 * \notefnerr
 *
 * \sa
 * ::UPTKAccessPolicyWindow
 */
extern __host__ UPTKError_t UPTKCtxResetPersistingL2Cache(void);

/**
 * \brief Copies attributes from source stream to destination stream.
 *
 * Copies attributes from source stream \p src to destination stream \p dst.
 * Both streams must have the same context.
 *
 * \param[out] dst Destination stream
 * \param[in] src Source stream
 * For attributes see ::UPTKStreamAttrID
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorNotSupported
 * \notefnerr
 *
 * \sa
 * ::UPTKAccessPolicyWindow
 */
extern __host__ UPTKError_t UPTKStreamCopyAttributes(UPTKStream_t dst, UPTKStream_t src);

 /**
 * \brief Queries stream attribute.
 *
 * Queries attribute \p attr from \p hStream and stores it in corresponding
 * member of \p value_out.
 *
 * \param[in] hStream
 * \param[in] attr
 * \param[out] value_out
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle
 * \notefnerr
 *
 * \sa
 * ::UPTKAccessPolicyWindow
 */
extern __host__ UPTKError_t UPTKStreamGetAttribute(
        UPTKStream_t hStream, UPTKStreamAttrID attr,
        UPTKStreamAttrValue *value_out);

 /**
 * \brief Sets stream attribute.
 *
 * Sets attribute \p attr on \p hStream from corresponding attribute of
 * \p value. The updated attribute will be applied to subsequent work
 * submitted to the stream. It will not affect previously submitted work.
 *
 * \param[out] hStream
 * \param[in] attr
 * \param[in] value
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle
 * \notefnerr
 *
 * \sa
 * ::UPTKAccessPolicyWindow
 */
extern __host__ UPTKError_t UPTKStreamSetAttribute(
        UPTKStream_t hStream, UPTKStreamAttrID attr,
        const UPTKStreamAttrValue *value);

 /**
 * \brief Destroys and cleans up an asynchronous stream
 *
 * Destroys and cleans up the asynchronous stream specified by \p stream.
 *
 * In case the device is still doing work in the stream \p stream
 * when ::UPTKStreamDestroy() is called, the function will return immediately 
 * and the resources associated with \p stream will be released automatically 
 * once the device has completed all work in \p stream.
 *
 * \param stream - Stream identifier
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_destroy_ub
 *
 * \sa ::UPTKStreamCreate,
 * ::UPTKStreamCreateWithFlags,
 * ::UPTKStreamQuery,
 * ::UPTKStreamWaitEvent,
 * ::UPTKStreamSynchronize,
 * ::UPTKStreamAddCallback,
 * ::cuStreamDestroy
 */
extern __host__ UPTKError_t UPTKStreamDestroy(UPTKStream_t stream);

/**
 * \brief Make a compute stream wait on an event
 *
 * Makes all future work submitted to \p stream wait for all work captured in
 * \p event.  See ::UPTKEventRecord() for details on what is captured by an event.
 * The synchronization will be performed efficiently on the device when applicable.
 * \p event may be from a different device than \p stream.
 *
 * flags include:
 * - ::UPTKEventWaitDefault: Default event creation flag.
 * - ::UPTKEventWaitExternal: Event is captured in the graph as an external
 *   event node when performing stream capture.
 *
 * \param stream - Stream to wait
 * \param event  - Event to wait on
 * \param flags  - Parameters for the operation(See above)
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKStreamCreate, ::UPTKStreamCreateWithFlags, ::UPTKStreamQuery, ::UPTKStreamSynchronize, ::UPTKStreamAddCallback, ::UPTKStreamDestroy,
 * ::cuStreamWaitEvent
 */
extern __host__ UPTKError_t UPTKStreamWaitEvent(UPTKStream_t stream, UPTKEvent_t event, unsigned int flags __dv(0));

/**
 * Type of stream callback functions.
 * \param stream The stream as passed to ::UPTKStreamAddCallback, may be NULL.
 * \param status ::UPTKSuccess or any persistent error on the stream.
 * \param userData User parameter provided at registration.
 */
typedef void (UPTKRT_CB *UPTKStreamCallback_t)(UPTKStream_t stream, UPTKError_t status, void *userData);

/**
 * \brief Add a callback to a compute stream
 *
 * \note This function is slated for eventual deprecation and removal. If
 * you do not require the callback to execute in case of a device error,
 * consider using ::UPTKLaunchHostFunc. Additionally, this function is not
 * supported with ::UPTKStreamBeginCapture and ::UPTKStreamEndCapture, unlike
 * ::UPTKLaunchHostFunc.
 *
 * Adds a callback to be called on the host after all currently enqueued
 * items in the stream have completed.  For each 
 * UPTKStreamAddCallback call, a callback will be executed exactly once.
 * The callback will block later work in the stream until it is finished.
 *
 * The callback may be passed ::UPTKSuccess or an error code.  In the event
 * of a device error, all subsequently executed callbacks will receive an
 * appropriate ::UPTKError_t.
 *
 * Callbacks must not make any UPTK API calls.  Attempting to use UPTK APIs
 * may result in ::UPTKErrorNotPermitted.  Callbacks must not perform any
 * synchronization that may depend on outstanding device work or other callbacks
 * that are not mandated to run earlier.  Callbacks without a mandated order
 * (in independent streams) execute in undefined order and may be serialized.
 *
 * For the purposes of Unified Memory, callback execution makes a number of
 * guarantees:
 * <ul>
 *   <li>The callback stream is considered idle for the duration of the
 *   callback.  Thus, for example, a callback may always use memory attached
 *   to the callback stream.</li>
 *   <li>The start of execution of a callback has the same effect as
 *   synchronizing an event recorded in the same stream immediately prior to
 *   the callback.  It thus synchronizes streams which have been "joined"
 *   prior to the callback.</li>
 *   <li>Adding device work to any stream does not have the effect of making
 *   the stream active until all preceding callbacks have executed.  Thus, for
 *   example, a callback might use global attached memory even if work has
 *   been added to another stream, if it has been properly ordered with an
 *   event.</li>
 *   <li>Completion of a callback does not cause a stream to become
 *   active except as described above.  The callback stream will remain idle
 *   if no device work follows the callback, and will remain idle across
 *   consecutive callbacks without device work in between.  Thus, for example,
 *   stream synchronization can be done by signaling from a callback at the
 *   end of the stream.</li>
 * </ul>
 *
 * \param stream   - Stream to add callback to
 * \param callback - The function to call once preceding stream operations are complete
 * \param userData - User specified data to be passed to the callback function
 * \param flags    - Reserved for future use, must be 0
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorNotSupported
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKStreamCreate, ::UPTKStreamCreateWithFlags, ::UPTKStreamQuery, ::UPTKStreamSynchronize, ::UPTKStreamWaitEvent, ::UPTKStreamDestroy, ::UPTKMallocManaged, ::UPTKStreamAttachMemAsync,
 * ::UPTKLaunchHostFunc, ::cuStreamAddCallback
 */
extern __host__ UPTKError_t UPTKStreamAddCallback(UPTKStream_t stream,
        UPTKStreamCallback_t callback, void *userData, unsigned int flags);

/**
 * \brief Waits for stream tasks to complete
 *
 * Blocks until \p stream has completed all operations. If the
 * ::UPTKDeviceScheduleBlockingSync flag was set for this device, 
 * the host thread will block until the stream is finished with 
 * all of its tasks.
 *
 * \param stream - Stream identifier
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidResourceHandle
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKStreamCreate, ::UPTKStreamCreateWithFlags, ::UPTKStreamQuery, ::UPTKStreamWaitEvent, ::UPTKStreamAddCallback, ::UPTKStreamDestroy,
 * ::cuStreamSynchronize
 */
extern __host__ UPTKError_t UPTKStreamSynchronize(UPTKStream_t stream);

/**
 * \brief Queries an asynchronous stream for completion status
 *
 * Returns ::UPTKSuccess if all operations in \p stream have
 * completed, or ::UPTKErrorNotReady if not.
 *
 * For the purposes of Unified Memory, a return value of ::UPTKSuccess
 * is equivalent to having called ::UPTKStreamSynchronize().
 *
 * \param stream - Stream identifier
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorNotReady,
 * ::UPTKErrorInvalidResourceHandle
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKStreamCreate, ::UPTKStreamCreateWithFlags, ::UPTKStreamWaitEvent, ::UPTKStreamSynchronize, ::UPTKStreamAddCallback, ::UPTKStreamDestroy,
 * ::cuStreamQuery
 */
extern __host__ UPTKError_t UPTKStreamQuery(UPTKStream_t stream);

/**
 * \brief Attach memory to a stream asynchronously
 *
 * Enqueues an operation in \p stream to specify stream association of
 * \p length bytes of memory starting from \p devPtr. This function is a
 * stream-ordered operation, meaning that it is dependent on, and will
 * only take effect when, previous work in stream has completed. Any
 * previous association is automatically replaced.
 *
 * \p devPtr must point to an one of the following types of memories:
 * - managed memory declared using the __managed__ keyword or allocated with
 *   ::UPTKMallocManaged.
 * - a valid host-accessible region of system-allocated pageable memory. This
 *   type of memory may only be specified if the device associated with the
 *   stream reports a non-zero value for the device attribute
 *   ::UPTKDevAttrPageableMemoryAccess.
 *
 * For managed allocations, \p length must be either zero or the entire
 * allocation's size. Both indicate that the entire allocation's stream
 * association is being changed. Currently, it is not possible to change stream
 * association for a portion of a managed allocation.
 *
 * For pageable allocations, \p length must be non-zero.
 *
 * The stream association is specified using \p flags which must be
 * one of ::UPTKMemAttachGlobal, ::UPTKMemAttachHost or ::UPTKMemAttachSingle.
 * The default value for \p flags is ::UPTKMemAttachSingle
 * If the ::UPTKMemAttachGlobal flag is specified, the memory can be accessed
 * by any stream on any device.
 * If the ::UPTKMemAttachHost flag is specified, the program makes a guarantee
 * that it won't access the memory on the device from any stream on a device that
 * has a zero value for the device attribute ::UPTKDevAttrConcurrentManagedAccess.
 * If the ::UPTKMemAttachSingle flag is specified and \p stream is associated with
 * a device that has a zero value for the device attribute ::UPTKDevAttrConcurrentManagedAccess,
 * the program makes a guarantee that it will only access the memory on the device
 * from \p stream. It is illegal to attach singly to the NULL stream, because the
 * NULL stream is a virtual global stream and not a specific stream. An error will
 * be returned in this case.
 *
 * When memory is associated with a single stream, the Unified Memory system will
 * allow CPU access to this memory region so long as all operations in \p stream
 * have completed, regardless of whether other streams are active. In effect,
 * this constrains exclusive ownersUPTK of the managed memory region by
 * an active GPU to per-stream activity instead of whole-GPU activity.
 *
 * Accessing memory on the device from streams that are not associated with
 * it will produce undefined results. No error checking is performed by the
 * Unified Memory system to ensure that kernels launched into other streams
 * do not access this region. 
 *
 * It is a program's responsibility to order calls to ::UPTKStreamAttachMemAsync
 * via events, synchronization or other means to ensure legal access to memory
 * at all times. Data visibility and coherency will be changed appropriately
 * for all kernels which follow a stream-association change.
 *
 * If \p stream is destroyed while data is associated with it, the association is
 * removed and the association reverts to the default visibility of the allocation
 * as specified at ::UPTKMallocManaged. For __managed__ variables, the default
 * association is always ::UPTKMemAttachGlobal. Note that destroying a stream is an
 * asynchronous operation, and as a result, the change to default association won't
 * happen until all work in the stream has completed.
 *
 * \param stream  - Stream in which to enqueue the attach operation
 * \param devPtr  - Pointer to memory (must be a pointer to managed memory or
 *                  to a valid host-accessible region of system-allocated
 *                  memory)
 * \param length  - Length of memory (defaults to zero)
 * \param flags   - Must be one of ::UPTKMemAttachGlobal, ::UPTKMemAttachHost or ::UPTKMemAttachSingle (defaults to ::UPTKMemAttachSingle)
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorNotReady,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKStreamCreate, ::UPTKStreamCreateWithFlags, ::UPTKStreamWaitEvent, ::UPTKStreamSynchronize, ::UPTKStreamAddCallback, ::UPTKStreamDestroy, ::UPTKMallocManaged,
 * ::cuStreamAttachMemAsync
 */
#if defined(__cplusplus)
extern __host__ UPTKError_t UPTKStreamAttachMemAsync(UPTKStream_t stream, void *devPtr, size_t length __dv(0), unsigned int flags = UPTKMemAttachSingle);
#else
extern __host__ UPTKError_t UPTKStreamAttachMemAsync(UPTKStream_t stream, void *devPtr, size_t length __dv(0), unsigned int flags);
#endif

/**
 * \brief Begins graph capture on a stream
 *
 * Begin graph capture on \p stream. When a stream is in capture mode, all operations
 * pushed into the stream will not be executed, but will instead be captured into
 * a graph, which will be returned via ::UPTKStreamEndCapture. Capture may not be initiated
 * if \p stream is ::UPTKStreamLegacy. Capture must be ended on the same stream in which
 * it was initiated, and it may only be initiated if the stream is not already in capture
 * mode. The capture mode may be queried via ::UPTKStreamIsCapturing. A unique id
 * representing the capture sequence may be queried via ::UPTKStreamGetCaptureInfo.
 *
 * If \p mode is not ::UPTKStreamCaptureModeRelaxed, ::UPTKStreamEndCapture must be
 * called on this stream from the same thread.
 *
 * \note Kernels captured using this API must not use texture and surface references.
 *       Reading or writing through any texture or surface reference is undefined
 *       behavior. This restriction does not apply to texture and surface objects.
 *
 * \param stream - Stream in which to initiate capture
 * \param mode    - Controls the interaction of this capture sequence with other API
 *                  calls that are potentially unsafe. For more details see
 *                  ::UPTKThreadExchangeStreamCaptureMode.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 *
 * \sa
 * ::UPTKStreamCreate,
 * ::UPTKStreamIsCapturing,
 * ::UPTKStreamEndCapture,
 * ::UPTKThreadExchangeStreamCaptureMode
 */
extern __host__ UPTKError_t UPTKStreamBeginCapture(UPTKStream_t stream, enum UPTKStreamCaptureMode mode);

/**
 * \brief Begins graph capture on a stream to an existing graph
 *
 * Begin graph capture on \p stream. When a stream is in capture mode, all operations
 * pushed into the stream will not be executed, but will instead be captured into
 * \p graph, which will be returned via ::UPTKStreamEndCapture.
 *
 * Capture may not be initiated if \p stream is ::UPTKStreamLegacy. Capture must be ended on the
 * same stream in which it was initiated, and it may only be initiated if the stream is not
 * already in capture mode. The capture mode may be queried via ::UPTKStreamIsCapturing. A unique id
 * representing the capture sequence may be queried via ::UPTKStreamGetCaptureInfo.
 *
 * If \p mode is not ::UPTKStreamCaptureModeRelaxed, ::UPTKStreamEndCapture must be
 * called on this stream from the same thread.
 *
 * \note Kernels captured using this API must not use texture and surface references.
 *       Reading or writing through any texture or surface reference is undefined
 *       behavior. This restriction does not apply to texture and surface objects.
 *
 * \param stream          - Stream in which to initiate capture.
 * \param graph           - Graph to capture into.
 * \param dependencies    - Dependencies of the first node captured in the stream.  Can be NULL if numDependencies is 0.
 * \param dependencyData  - Optional array of data associated with each dependency.
 * \param numDependencies - Number of dependencies.
 * \param mode            - Controls the interaction of this capture sequence with other API
 *                          calls that are potentially unsafe. For more details see
 *                          ::UPTKThreadExchangeStreamCaptureMode.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 *
 * \sa
 * ::UPTKStreamCreate,
 * ::UPTKStreamIsCapturing,
 * ::UPTKStreamEndCapture,
 * ::UPTKThreadExchangeStreamCaptureMode
 */
extern __host__ UPTKError_t UPTKStreamBeginCaptureToGraph(UPTKStream_t stream, UPTKGraph_t graph, const UPTKGraphNode_t *dependencies, const UPTKGraphEdgeData *dependencyData, size_t numDependencies, enum UPTKStreamCaptureMode mode);

/**
 * \brief Swaps the stream capture interaction mode for a thread
 *
 * Sets the calling thread's stream capture interaction mode to the value contained
 * in \p *mode, and overwrites \p *mode with the previous mode for the thread. To
 * facilitate deterministic behavior across function or module boundaries, callers
 * are encouraged to use this API in a push-pop fashion: \code
     UPTKStreamCaptureMode mode = desiredMode;
     UPTKThreadExchangeStreamCaptureMode(&mode);
     ...
     UPTKThreadExchangeStreamCaptureMode(&mode); // restore previous mode
 * \endcode
 *
 * During stream capture (see ::UPTKStreamBeginCapture), some actions, such as a call
 * to ::UPTKMalloc, may be unsafe. In the case of ::UPTKMalloc, the operation is
 * not enqueued asynchronously to a stream, and is not observed by stream capture.
 * Therefore, if the sequence of operations captured via ::UPTKStreamBeginCapture
 * depended on the allocation being replayed whenever the graph is launched, the
 * captured graph would be invalid.
 *
 * Therefore, stream capture places restrictions on API calls that can be made within
 * or concurrently to a ::UPTKStreamBeginCapture-::UPTKStreamEndCapture sequence. This
 * behavior can be controlled via this API and flags to ::UPTKStreamBeginCapture.
 *
 * A thread's mode is one of the following:
 * - \p UPTKStreamCaptureModeGlobal: This is the default mode. If the local thread has
 *   an ongoing capture sequence that was not initiated with
 *   \p UPTKStreamCaptureModeRelaxed at \p cuStreamBeginCapture, or if any other thread
 *   has a concurrent capture sequence initiated with \p UPTKStreamCaptureModeGlobal,
 *   this thread is prohibited from potentially unsafe API calls.
 * - \p UPTKStreamCaptureModeThreadLocal: If the local thread has an ongoing capture
 *   sequence not initiated with \p UPTKStreamCaptureModeRelaxed, it is prohibited
 *   from potentially unsafe API calls. Concurrent capture sequences in other threads
 *   are ignored.
 * - \p UPTKStreamCaptureModeRelaxed: The local thread is not prohibited from potentially
 *   unsafe API calls. Note that the thread is still prohibited from API calls which
 *   necessarily conflict with stream capture, for example, attempting ::UPTKEventQuery
 *   on an event that was last recorded inside a capture sequence.
 *
 * \param mode - Pointer to mode value to swap with the current mode
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 *
 * \sa
 * ::UPTKStreamBeginCapture
 */
extern __host__ UPTKError_t UPTKThreadExchangeStreamCaptureMode(enum UPTKStreamCaptureMode *mode);

/**
 * \brief Ends capture on a stream, returning the captured graph
 *
 * End capture on \p stream, returning the captured graph via \p pGraph.
 * Capture must have been initiated on \p stream via a call to ::UPTKStreamBeginCapture.
 * If capture was invalidated, due to a violation of the rules of stream capture, then
 * a NULL graph will be returned.
 *
 * If the \p mode argument to ::UPTKStreamBeginCapture was not
 * ::UPTKStreamCaptureModeRelaxed, this call must be from the same thread as
 * ::UPTKStreamBeginCapture.
 *
 * \param stream - Stream to query
 * \param pGraph - The captured graph
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorStreamCaptureWrongThread
 * \notefnerr
 *
 * \sa
 * ::UPTKStreamCreate,
 * ::UPTKStreamBeginCapture,
 * ::UPTKStreamIsCapturing,
 * ::UPTKGraphDestroy
 */
extern __host__ UPTKError_t UPTKStreamEndCapture(UPTKStream_t stream, UPTKGraph_t *pGraph);

/**
 * \brief Returns a stream's capture status
 *
 * Return the capture status of \p stream via \p pCaptureStatus. After a successful
 * call, \p *pCaptureStatus will contain one of the following:
 * - ::UPTKStreamCaptureStatusNone: The stream is not capturing.
 * - ::UPTKStreamCaptureStatusActive: The stream is capturing.
 * - ::UPTKStreamCaptureStatusInvalidated: The stream was capturing but an error
 *   has invalidated the capture sequence. The capture sequence must be terminated
 *   with ::UPTKStreamEndCapture on the stream where it was initiated in order to
 *   continue using \p stream.
 *
 * Note that, if this is called on ::UPTKStreamLegacy (the "null stream") while
 * a blocking stream on the same device is capturing, it will return
 * ::UPTKErrorStreamCaptureImplicit and \p *pCaptureStatus is unspecified
 * after the call. The blocking stream capture is not invalidated.
 *
 * When a blocking stream is capturing, the legacy stream is in an
 * unusable state until the blocking stream capture is terminated. The legacy
 * stream is not supported for stream capture, but attempted use would have an
 * implicit dependency on the capturing stream(s).
 *
 * \param stream         - Stream to query
 * \param pCaptureStatus - Returns the stream's capture status
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorStreamCaptureImplicit
 * \notefnerr
 *
 * \sa
 * ::UPTKStreamCreate,
 * ::UPTKStreamBeginCapture,
 * ::UPTKStreamEndCapture
 */
extern __host__ UPTKError_t UPTKStreamIsCapturing(UPTKStream_t stream, enum UPTKStreamCaptureStatus *pCaptureStatus);


/**
 * \brief Query a stream's capture state
 *
 * Query stream state related to stream capture.
 *
 * If called on ::UPTKStreamLegacy (the "null stream") while a stream not created 
 * with ::UPTKStreamNonBlocking is capturing, returns ::UPTKErrorStreamCaptureImplicit.
 *
 * Valid data (other than capture status) is returned only if both of the following are true:
 * - the call returns UPTKSuccess
 * - the returned capture status is ::UPTKStreamCaptureStatusActive
 *
 * \param stream - The stream to query
 * \param captureStatus_out - Location to return the capture status of the stream; required
 * \param id_out - Optional location to return an id for the capture sequence, which is
 *           unique over the lifetime of the process
 * \param graph_out - Optional location to return the graph being captured into. All
 *           operations other than destroy and node removal are permitted on the graph
 *           while the capture sequence is in progress. This API does not transfer
 *           ownersUPTK of the graph, which is transferred or destroyed at
 *           ::UPTKStreamEndCapture. Note that the graph handle may be invalidated before
 *           end of capture for certain errors. Nodes that are or become
 *           unreachable from the original stream at ::UPTKStreamEndCapture due to direct
 *           actions on the graph do not trigger ::UPTKErrorStreamCaptureUnjoined.
 * \param dependencies_out - Optional location to store a pointer to an array of nodes.
 *           The next node to be captured in the stream will depend on this set of nodes,
 *           absent operations such as event wait which modify this set. The array pointer
 *           is valid until the next API call which operates on the stream or until the
 *           capture is terminated. The node handles may be copied out and are valid until
 *           they or the graph is destroyed. The driver-owned array may also be passed
 *           directly to APIs that operate on the graph (not the stream) without copying.
 * \param numDependencies_out - Optional location to store the size of the array
 *           returned in dependencies_out.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorStreamCaptureImplicit
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::UPTKStreamGetCaptureInfo_v3,
 * ::UPTKStreamBeginCapture,
 * ::UPTKStreamIsCapturing,
 * ::UPTKStreamUpdateCaptureDependencies
 */
extern __host__ UPTKError_t UPTKStreamGetCaptureInfo(UPTKStream_t stream, enum UPTKStreamCaptureStatus *captureStatus_out, unsigned long long *id_out __dv(0), UPTKGraph_t *graph_out __dv(0), const UPTKGraphNode_t **dependencies_out __dv(0), size_t *numDependencies_out __dv(0));

/**
 * \brief Query a stream's capture state (12.3+)
 *
 * Query stream state related to stream capture.
 *
 * If called on ::UPTKStreamLegacy (the "null stream") while a stream not created 
 * with ::UPTKStreamNonBlocking is capturing, returns ::UPTKErrorStreamCaptureImplicit.
 *
 * Valid data (other than capture status) is returned only if both of the following are true:
 * - the call returns UPTKSuccess
 * - the returned capture status is ::UPTKStreamCaptureStatusActive
 *
 * If \p edgeData_out is non-NULL then \p dependencies_out must be as well. If
 * \p dependencies_out is non-NULL and \p edgeData_out is NULL, but there is non-zero edge
 * data for one or more of the current stream dependencies, the call will return
 * ::UPTKErrorLossyQuery.
 *
 * \param stream - The stream to query
 * \param captureStatus_out - Location to return the capture status of the stream; required
 * \param id_out - Optional location to return an id for the capture sequence, which is
 *           unique over the lifetime of the process
 * \param graph_out - Optional location to return the graph being captured into. All
 *           operations other than destroy and node removal are permitted on the graph
 *           while the capture sequence is in progress. This API does not transfer
 *           ownersUPTK of the graph, which is transferred or destroyed at
 *           ::UPTKStreamEndCapture. Note that the graph handle may be invalidated before
 *           end of capture for certain errors. Nodes that are or become
 *           unreachable from the original stream at ::UPTKStreamEndCapture due to direct
 *           actions on the graph do not trigger ::UPTKErrorStreamCaptureUnjoined.
 * \param dependencies_out - Optional location to store a pointer to an array of nodes.
 *           The next node to be captured in the stream will depend on this set of nodes,
 *           absent operations such as event wait which modify this set. The array pointer
 *           is valid until the next API call which operates on the stream or until the
 *           capture is terminated. The node handles may be copied out and are valid until
 *           they or the graph is destroyed. The driver-owned array may also be passed
 *           directly to APIs that operate on the graph (not the stream) without copying.
 * \param edgeData_out - Optional location to store a pointer to an array of graph edge
 *           data. This array parallels \c dependencies_out; the next node to be added
 *           has an edge to \c dependencies_out[i] with annotation \c edgeData_out[i] for
 *           each \c i. The array pointer is valid until the next API call which operates
 *           on the stream or until the capture is terminated.
 * \param numDependencies_out - Optional location to store the size of the array
 *           returned in dependencies_out.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorStreamCaptureImplicit,
 * ::UPTKErrorLossyQuery
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::UPTKStreamBeginCapture,
 * ::UPTKStreamIsCapturing,
 * ::UPTKStreamUpdateCaptureDependencies
 */
extern __host__ UPTKError_t UPTKStreamGetCaptureInfo_v3(UPTKStream_t stream,
    enum UPTKStreamCaptureStatus *captureStatus_out, unsigned long long *id_out __dv(0),
    UPTKGraph_t *graph_out __dv(0), const UPTKGraphNode_t **dependencies_out __dv(0),
    const UPTKGraphEdgeData **edgeData_out __dv(0), size_t *numDependencies_out __dv(0));

/**
 * \brief Update the set of dependencies in a capturing stream (11.3+)
 *
 * Modifies the dependency set of a capturing stream. The dependency set is the set
 * of nodes that the next captured node in the stream will depend on.
 *
 * Valid flags are ::UPTKStreamAddCaptureDependencies and
 * ::UPTKStreamSetCaptureDependencies. These control whether the set passed to
 * the API is added to the existing set or replaces it. A flags value of 0 defaults
 * to ::UPTKStreamAddCaptureDependencies.
 *
 * Nodes that are removed from the dependency set via this API do not result in
 * ::UPTKErrorStreamCaptureUnjoined if they are unreachable from the stream at
 * ::UPTKStreamEndCapture.
 *
 * Returns ::UPTKErrorIllegalState if the stream is not capturing.
 *
 * This API is new in UPTK 11.3. Developers requiring compatibility across minor
 * versions of the UPTK driver to 11.0 should not use this API or provide a fallback.
 *
 * \param stream - The stream to update
 * \param dependencies - The set of dependencies to add
 * \param numDependencies - The size of the dependencies array
 * \param flags - See above
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorIllegalState
 * \notefnerr
 *
 * \sa
 * ::UPTKStreamBeginCapture,
 * ::UPTKStreamGetCaptureInfo,
 */
extern __host__ UPTKError_t UPTKStreamUpdateCaptureDependencies(UPTKStream_t stream, UPTKGraphNode_t *dependencies, size_t numDependencies, unsigned int flags __dv(0));

/**
 * \brief Update the set of dependencies in a capturing stream (12.3+)
 *
 * Modifies the dependency set of a capturing stream. The dependency set is the set
 * of nodes that the next captured node in the stream will depend on.
 *
 * Valid flags are ::UPTKStreamAddCaptureDependencies and
 * ::UPTKStreamSetCaptureDependencies. These control whether the set passed to
 * the API is added to the existing set or replaces it. A flags value of 0 defaults
 * to ::UPTKStreamAddCaptureDependencies.
 *
 * Nodes that are removed from the dependency set via this API do not result in
 * ::UPTKErrorStreamCaptureUnjoined if they are unreachable from the stream at
 * ::UPTKStreamEndCapture.
 *
 * Returns ::UPTKErrorIllegalState if the stream is not capturing.
 *
 * \param stream - The stream to update
 * \param dependencies - The set of dependencies to add
 * \param dependencyData - Optional array of data associated with each dependency.
 * \param numDependencies - The size of the dependencies array
 * \param flags - See above
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorIllegalState
 * \notefnerr
 *
 * \sa
 * ::UPTKStreamBeginCapture,
 * ::UPTKStreamGetCaptureInfo,
 */
extern __host__ UPTKError_t UPTKStreamUpdateCaptureDependencies_v2(UPTKStream_t stream, UPTKGraphNode_t *dependencies, const UPTKGraphEdgeData *dependencyData, size_t numDependencies, unsigned int flags __dv(0));
/** @} */ /* END UPTKRT_STREAM */

/**
 * \defgroup UPTKRT_EVENT Event Management
 *
 * ___MANBRIEF___ event management functions of the UPTK runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the event management functions of the UPTK runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Creates an event object
 *
 * Creates an event object for the current device using ::UPTKEventDefault.
 *
 * \param event - Newly created event
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorLaunchFailure,
 * ::UPTKErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::UPTKEventCreate(UPTKEvent_t*, unsigned int) "UPTKEventCreate (C++ API)",
 * ::UPTKEventCreateWithFlags, ::UPTKEventRecord, ::UPTKEventQuery,
 * ::UPTKEventSynchronize, ::UPTKEventDestroy, ::UPTKEventElapsedTime,
 * ::UPTKStreamWaitEvent,
 * ::cuEventCreate
 */
extern __host__ UPTKError_t UPTKEventCreate(UPTKEvent_t *event);

/**
 * \brief Creates an event object with the specified flags
 *
 * Creates an event object for the current device with the specified flags. Valid
 * flags include:
 * - ::UPTKEventDefault: Default event creation flag.
 * - ::UPTKEventBlockingSync: Specifies that event should use blocking
 *   synchronization. A host thread that uses ::UPTKEventSynchronize() to wait
 *   on an event created with this flag will block until the event actually
 *   completes.
 * - ::UPTKEventDisableTiming: Specifies that the created event does not need
 *   to record timing data.  Events created with this flag specified and
 *   the ::UPTKEventBlockingSync flag not specified will provide the best
 *   performance when used with ::UPTKStreamWaitEvent() and ::UPTKEventQuery().
 * - ::UPTKEventInterprocess: Specifies that the created event may be used as an
 *   interprocess event by ::UPTKIpcGetEventHandle(). ::UPTKEventInterprocess must
 *   be specified along with ::UPTKEventDisableTiming.
 *
 * \param event - Newly created event
 * \param flags - Flags for new event
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorLaunchFailure,
 * ::UPTKErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::UPTKEventCreate(UPTKEvent_t*) "UPTKEventCreate (C API)",
 * ::UPTKEventSynchronize, ::UPTKEventDestroy, ::UPTKEventElapsedTime,
 * ::UPTKStreamWaitEvent,
 * ::cuEventCreate
 */
extern __host__ UPTKError_t UPTKEventCreateWithFlags(UPTKEvent_t *event, unsigned int flags);

/**
 * \brief Records an event
 *
 * Captures in \p event the contents of \p stream at the time of this call.
 * \p event and \p stream must be on the same UPTK context.
 * Calls such as ::UPTKEventQuery() or ::UPTKStreamWaitEvent() will then
 * examine or wait for completion of the work that was captured. Uses of
 * \p stream after this call do not modify \p event. See note on default
 * stream behavior for what is captured in the default case.
 *
 * ::UPTKEventRecord() can be called multiple times on the same event and
 * will overwrite the previously captured state. Other APIs such as
 * ::UPTKStreamWaitEvent() use the most recently captured state at the time
 * of the API call, and are not affected by later calls to
 * ::UPTKEventRecord(). Before the first call to ::UPTKEventRecord(), an
 * event represents an empty set of work, so for example ::UPTKEventQuery()
 * would return ::UPTKSuccess.
 *
 * \param event  - Event to record
 * \param stream - Stream in which to record event
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorLaunchFailure
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_null_event
 *
 * \sa \ref ::UPTKEventCreate(UPTKEvent_t*) "UPTKEventCreate (C API)",
 * ::UPTKEventCreateWithFlags, ::UPTKEventQuery,
 * ::UPTKEventSynchronize, ::UPTKEventDestroy, ::UPTKEventElapsedTime,
 * ::UPTKStreamWaitEvent,
 * ::UPTKEventRecordWithFlags,
 * ::cuEventRecord
 */
extern __host__ UPTKError_t UPTKEventRecord(UPTKEvent_t event, UPTKStream_t stream __dv(0));

/**
 * \brief Records an event
 *
 * Captures in \p event the contents of \p stream at the time of this call.
 * \p event and \p stream must be on the same UPTK context.
 * Calls such as ::UPTKEventQuery() or ::UPTKStreamWaitEvent() will then
 * examine or wait for completion of the work that was captured. Uses of
 * \p stream after this call do not modify \p event. See note on default
 * stream behavior for what is captured in the default case.
 *
 * ::UPTKEventRecordWithFlags() can be called multiple times on the same event and
 * will overwrite the previously captured state. Other APIs such as
 * ::UPTKStreamWaitEvent() use the most recently captured state at the time
 * of the API call, and are not affected by later calls to
 * ::UPTKEventRecordWithFlags(). Before the first call to ::UPTKEventRecordWithFlags(), an
 * event represents an empty set of work, so for example ::UPTKEventQuery()
 * would return ::UPTKSuccess.
 *
 * flags include:
 * - ::UPTKEventRecordDefault: Default event creation flag.
 * - ::UPTKEventRecordExternal: Event is captured in the graph as an external
 *   event node when performing stream capture.
 *
 * \param event  - Event to record
 * \param stream - Stream in which to record event
 * \param flags  - Parameters for the operation(See above)
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorLaunchFailure
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_null_event
 *
 * \sa \ref ::UPTKEventCreate(UPTKEvent_t*) "UPTKEventCreate (C API)",
 * ::UPTKEventCreateWithFlags, ::UPTKEventQuery,
 * ::UPTKEventSynchronize, ::UPTKEventDestroy, ::UPTKEventElapsedTime,
 * ::UPTKStreamWaitEvent,
 * ::UPTKEventRecord,
 * ::cuEventRecord,
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKEventRecordWithFlags(UPTKEvent_t event, UPTKStream_t stream __dv(0), unsigned int flags __dv(0));
#endif

/**
 * \brief Queries an event's status
 *
 * Queries the status of all work currently captured by \p event. See
 * ::UPTKEventRecord() for details on what is captured by an event.
 *
 * Returns ::UPTKSuccess if all captured work has been completed, or
 * ::UPTKErrorNotReady if any captured work is incomplete.
 *
 * For the purposes of Unified Memory, a return value of ::UPTKSuccess
 * is equivalent to having called ::UPTKEventSynchronize().
 *
 * \param event - Event to query
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorNotReady,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorLaunchFailure
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_null_event
 *
 * \sa \ref ::UPTKEventCreate(UPTKEvent_t*) "UPTKEventCreate (C API)",
 * ::UPTKEventCreateWithFlags, ::UPTKEventRecord,
 * ::UPTKEventSynchronize, ::UPTKEventDestroy, ::UPTKEventElapsedTime,
 * ::cuEventQuery
 */
extern __host__ UPTKError_t UPTKEventQuery(UPTKEvent_t event);

/**
 * \brief Waits for an event to complete
 *
 * Waits until the completion of all work currently captured in \p event.
 * See ::UPTKEventRecord() for details on what is captured by an event.
 *
 * Waiting for an event that was created with the ::UPTKEventBlockingSync
 * flag will cause the calling CPU thread to block until the event has
 * been completed by the device.  If the ::UPTKEventBlockingSync flag has
 * not been set, then the CPU thread will busy-wait until the event has
 * been completed by the device.
 *
 * \param event - Event to wait for
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorLaunchFailure
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_null_event
 *
 * \sa \ref ::UPTKEventCreate(UPTKEvent_t*) "UPTKEventCreate (C API)",
 * ::UPTKEventCreateWithFlags, ::UPTKEventRecord,
 * ::UPTKEventQuery, ::UPTKEventDestroy, ::UPTKEventElapsedTime,
 * ::cuEventSynchronize
 */
extern __host__ UPTKError_t UPTKEventSynchronize(UPTKEvent_t event);

/**
 * \brief Destroys an event object
 *
 * Destroys the event specified by \p event.
 *
 * An event may be destroyed before it is complete (i.e., while
 * ::UPTKEventQuery() would return ::UPTKErrorNotReady). In this case, the
 * call does not block on completion of the event, and any associated
 * resources will automatically be released asynchronously at completion.
 *
 * \param event - Event to destroy
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorLaunchFailure
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_destroy_ub
 * \note_null_event
 *
 * \sa \ref ::UPTKEventCreate(UPTKEvent_t*) "UPTKEventCreate (C API)",
 * ::UPTKEventCreateWithFlags, ::UPTKEventQuery,
 * ::UPTKEventSynchronize, ::UPTKEventRecord, ::UPTKEventElapsedTime,
 * ::cuEventDestroy
 */
extern __host__ UPTKError_t UPTKEventDestroy(UPTKEvent_t event);

/**
 * \brief Computes the elapsed time between events
 *
 * Computes the elapsed time between two events (in milliseconds with a
 * resolution of around 0.5 microseconds).
 *
 * If either event was last recorded in a non-NULL stream, the resulting time
 * may be greater than expected (even if both used the same stream handle). This
 * happens because the ::UPTKEventRecord() operation takes place asynchronously
 * and there is no guarantee that the measured latency is actually just between
 * the two events. Any number of other different stream operations could execute
 * in between the two measured events, thus altering the timing in a significant
 * way.
 *
 * If ::UPTKEventRecord() has not been called on either event, then
 * ::UPTKErrorInvalidResourceHandle is returned. If ::UPTKEventRecord() has been
 * called on both events but one or both of them has not yet been completed
 * (that is, ::UPTKEventQuery() would return ::UPTKErrorNotReady on at least one
 * of the events), ::UPTKErrorNotReady is returned. If either event was created
 * with the ::UPTKEventDisableTiming flag, then this function will return
 * ::UPTKErrorInvalidResourceHandle.
 *
 * \param ms    - Time between \p start and \p end in ms
 * \param start - Starting event
 * \param end   - Ending event
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorNotReady,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorLaunchFailure,
 * ::UPTKErrorUnknown
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_null_event
 *
 * \sa \ref ::UPTKEventCreate(UPTKEvent_t*) "UPTKEventCreate (C API)",
 * ::UPTKEventCreateWithFlags, ::UPTKEventQuery,
 * ::UPTKEventSynchronize, ::UPTKEventDestroy, ::UPTKEventRecord,
 * ::cuEventElapsedTime
 */
extern __host__ UPTKError_t UPTKEventElapsedTime(float *ms, UPTKEvent_t start, UPTKEvent_t end);

/** @} */ /* END UPTKRT_EVENT */

/**
 * \defgroup UPTKRT_EXTRES_INTEROP External Resource Interoperability
 *
 * ___MANBRIEF___ External resource interoperability functions of the UPTK runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the external resource interoperability functions of the UPTK
 * runtime application programming interface.
 *
 * @{
 */

/**
 * \brief Imports an external memory object
 *
 * Imports an externally allocated memory object and returns
 * a handle to that in \p extMem_out.
 *
 * The properties of the handle being imported must be described in
 * \p memHandleDesc. The ::UPTKExternalMemoryHandleDesc structure
 * is defined as follows:
 *
 * \code
        typedef struct UPTKExternalMemoryHandleDesc_st {
            UPTKExternalMemoryHandleType type;
            union {
                int fd;
                struct {
                    void *handle;
                    const void *name;
                } win32;
                const void *nvSciBufObject;
            } handle;
            unsigned long long size;
            unsigned int flags;
        } UPTKExternalMemoryHandleDesc;
 * \endcode
 *
 * where ::UPTKExternalMemoryHandleDesc::type specifies the type
 * of handle being imported. ::UPTKExternalMemoryHandleType is
 * defined as:
 *
 * \code
        typedef enum UPTKExternalMemoryHandleType_enum {
            UPTKExternalMemoryHandleTypeOpaqueFd         = 1,
            UPTKExternalMemoryHandleTypeOpaqueWin32      = 2,
            UPTKExternalMemoryHandleTypeOpaqueWin32Kmt   = 3,
            UPTKExternalMemoryHandleTypeD3D12Heap        = 4,
            UPTKExternalMemoryHandleTypeD3D12Resource    = 5,
	        UPTKExternalMemoryHandleTypeD3D11Resource    = 6,
		    UPTKExternalMemoryHandleTypeD3D11ResourceKmt = 7,
            UPTKExternalMemoryHandleTypeNvSciBuf         = 8
        } UPTKExternalMemoryHandleType;
 * \endcode
 *
 * If ::UPTKExternalMemoryHandleDesc::type is
 * ::UPTKExternalMemoryHandleTypeOpaqueFd, then
 * ::UPTKExternalMemoryHandleDesc::handle::fd must be a valid
 * file descriptor referencing a memory object. OwnersUPTK of
 * the file descriptor is transferred to the UPTK driver when the
 * handle is imported successfully. Performing any operations on the
 * file descriptor after it is imported results in undefined behavior.
 *
 * If ::UPTKExternalMemoryHandleDesc::type is
 * ::UPTKExternalMemoryHandleTypeOpaqueWin32, then exactly one
 * of ::UPTKExternalMemoryHandleDesc::handle::win32::handle and
 * ::UPTKExternalMemoryHandleDesc::handle::win32::name must not be
 * NULL. If ::UPTKExternalMemoryHandleDesc::handle::win32::handle
 * is not NULL, then it must represent a valid shared NT handle that
 * references a memory object. OwnersUPTK of this handle is
 * not transferred to UPTK after the import operation, so the
 * application must release the handle using the appropriate system
 * call. If ::UPTKExternalMemoryHandleDesc::handle::win32::name
 * is not NULL, then it must point to a NULL-terminated array of
 * UTF-16 characters that refers to a memory object.
 *
 * If ::UPTKExternalMemoryHandleDesc::type is
 * ::UPTKExternalMemoryHandleTypeOpaqueWin32Kmt, then
 * ::UPTKExternalMemoryHandleDesc::handle::win32::handle must
 * be non-NULL and
 * ::UPTKExternalMemoryHandleDesc::handle::win32::name
 * must be NULL. The handle specified must be a globally shared KMT
 * handle. This handle does not hold a reference to the underlying
 * object, and thus will be invalid when all references to the
 * memory object are destroyed.
 *
 * If ::UPTKExternalMemoryHandleDesc::type is
 * ::UPTKExternalMemoryHandleTypeD3D12Heap, then exactly one
 * of ::UPTKExternalMemoryHandleDesc::handle::win32::handle and
 * ::UPTKExternalMemoryHandleDesc::handle::win32::name must not be
 * NULL. If ::UPTKExternalMemoryHandleDesc::handle::win32::handle
 * is not NULL, then it must represent a valid shared NT handle that
 * is returned by ID3D12Device::CreateSharedHandle when referring to a
 * ID3D12Heap object. This handle holds a reference to the underlying
 * object. If ::UPTKExternalMemoryHandleDesc::handle::win32::name
 * is not NULL, then it must point to a NULL-terminated array of
 * UTF-16 characters that refers to a ID3D12Heap object.
 *
 * If ::UPTKExternalMemoryHandleDesc::type is
 * ::UPTKExternalMemoryHandleTypeD3D12Resource, then exactly one
 * of ::UPTKExternalMemoryHandleDesc::handle::win32::handle and
 * ::UPTKExternalMemoryHandleDesc::handle::win32::name must not be
 * NULL. If ::UPTKExternalMemoryHandleDesc::handle::win32::handle
 * is not NULL, then it must represent a valid shared NT handle that
 * is returned by ID3D12Device::CreateSharedHandle when referring to a
 * ID3D12Resource object. This handle holds a reference to the
 * underlying object. If
 * ::UPTKExternalMemoryHandleDesc::handle::win32::name
 * is not NULL, then it must point to a NULL-terminated array of
 * UTF-16 characters that refers to a ID3D12Resource object.
 *
 * If ::UPTKExternalMemoryHandleDesc::type is
 * ::UPTKExternalMemoryHandleTypeD3D11Resource,then exactly one
 * of ::UPTKExternalMemoryHandleDesc::handle::win32::handle and
 * ::UPTKExternalMemoryHandleDesc::handle::win32::name must not be
 * NULL. If ::UPTKExternalMemoryHandleDesc::handle::win32::handle is    
 * not NULL, then it must represent a valid shared NT handle that is  
 * returned by  IDXGIResource1::CreateSharedHandle when referring to a 
 * ID3D11Resource object. If
 * ::UPTKExternalMemoryHandleDesc::handle::win32::name
 * is not NULL, then it must point to a NULL-terminated array of
 * UTF-16 characters that refers to a ID3D11Resource object.
 *
 * If ::UPTKExternalMemoryHandleDesc::type is
 * ::UPTKExternalMemoryHandleTypeD3D11ResourceKmt, then
 * ::UPTKExternalMemoryHandleDesc::handle::win32::handle must
 * be non-NULL and ::UPTKExternalMemoryHandleDesc::handle::win32::name
 * must be NULL. The handle specified must be a valid shared KMT
 * handle that is returned by IDXGIResource::GetSharedHandle when
 * referring to a ID3D11Resource object.
 *
 * If ::UPTKExternalMemoryHandleDesc::type is
 * ::UPTKExternalMemoryHandleTypeNvSciBuf, then
 * ::UPTKExternalMemoryHandleDesc::handle::nvSciBufObject must be NON-NULL
 * and reference a valid NvSciBuf object.
 * If the NvSciBuf object imported into UPTK is also mapped by other drivers, then the
 * application must use ::UPTKWaitExternalSemaphoresAsync or ::UPTKSignalExternalSemaphoresAsync
 * as approprriate barriers to maintain coherence between UPTK and the other drivers.
 * See ::UPTKExternalSemaphoreWaitSkipNvSciBufMemSync and ::UPTKExternalSemaphoreSignalSkipNvSciBufMemSync 
 * for memory synchronization.
 *
 * The size of the memory object must be specified in
 * ::UPTKExternalMemoryHandleDesc::size.
 *
 * Specifying the flag ::UPTKExternalMemoryDedicated in
 * ::UPTKExternalMemoryHandleDesc::flags indicates that the
 * resource is a dedicated resource. The definition of what a
 * dedicated resource is outside the scope of this extension.
 * This flag must be set if ::UPTKExternalMemoryHandleDesc::type
 * is one of the following:
 * ::UPTKExternalMemoryHandleTypeD3D12Resource
 * ::UPTKExternalMemoryHandleTypeD3D11Resource
 * ::UPTKExternalMemoryHandleTypeD3D11ResourceKmt
 *
 * \param extMem_out    - Returned handle to an external memory object
 * \param memHandleDesc - Memory import handle descriptor
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorOperatingSystem
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \note If the Vulkan memory imported into UPTK is mapped on the CPU then the
 * application must use vkInvalidateMappedMemoryRanges/vkFlushMappedMemoryRanges
 * as well as appropriate Vulkan pipeline barriers to maintain coherence between
 * CPU and GPU. For more information on these APIs, please refer to "Synchronization
 * and Cache Control" chapter from Vulkan specification.
 *
 *
 * \sa ::UPTKDestroyExternalMemory,
 * ::UPTKExternalMemoryGetMappedBuffer,
 * ::UPTKExternalMemoryGetMappedMipmappedArray
 */
extern __host__ UPTKError_t UPTKImportExternalMemory(UPTKExternalMemory_t *extMem_out, const struct UPTKExternalMemoryHandleDesc *memHandleDesc);

/**
 * \brief Maps a buffer onto an imported memory object
 *
 * Maps a buffer onto an imported memory object and returns a device
 * pointer in \p devPtr.
 *
 * The properties of the buffer being mapped must be described in
 * \p bufferDesc. The ::UPTKExternalMemoryBufferDesc structure is
 * defined as follows:
 *
 * \code
        typedef struct UPTKExternalMemoryBufferDesc_st {
            unsigned long long offset;
            unsigned long long size;
            unsigned int flags;
        } UPTKExternalMemoryBufferDesc;
 * \endcode
 *
 * where ::UPTKExternalMemoryBufferDesc::offset is the offset in
 * the memory object where the buffer's base address is.
 * ::UPTKExternalMemoryBufferDesc::size is the size of the buffer.
 * ::UPTKExternalMemoryBufferDesc::flags must be zero.
 *
 * The offset and size have to be suitably aligned to match the
 * requirements of the external API. Mapping two buffers whose ranges
 * overlap may or may not result in the same virtual address being
 * returned for the overlapped portion. In such cases, the application
 * must ensure that all accesses to that region from the GPU are
 * volatile. Otherwise writes made via one address are not guaranteed
 * to be visible via the other address, even if they're issued by the
 * same thread. It is recommended that applications map the combined
 * range instead of mapping separate buffers and then apply the
 * appropriate offsets to the returned pointer to derive the
 * individual buffers.
 *
 * The returned pointer \p devPtr must be freed using ::UPTKFree.
 *
 * \param devPtr     - Returned device pointer to buffer
 * \param extMem     - Handle to external memory object
 * \param bufferDesc - Buffer descriptor
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKImportExternalMemory,
 * ::UPTKDestroyExternalMemory,
 * ::UPTKExternalMemoryGetMappedMipmappedArray
 */
extern __host__ UPTKError_t UPTKExternalMemoryGetMappedBuffer(void **devPtr, UPTKExternalMemory_t extMem, const struct UPTKExternalMemoryBufferDesc *bufferDesc);

/**
 * \brief Maps a UPTK mipmapped array onto an external memory object
 *
 * Maps a UPTK mipmapped array onto an external object and returns a
 * handle to it in \p mipmap.
 *
 * The properties of the UPTK mipmapped array being mapped must be
 * described in \p mipmapDesc. The structure
 * ::UPTKExternalMemoryMipmappedArrayDesc is defined as follows:
 *
 * \code
        typedef struct UPTKExternalMemoryMipmappedArrayDesc_st {
            unsigned long long offset;
            UPTKChannelFormatDesc formatDesc;
            UPTKExtent extent;
            unsigned int flags;
            unsigned int numLevels;
        } UPTKExternalMemoryMipmappedArrayDesc;
 * \endcode
 *
 * where ::UPTKExternalMemoryMipmappedArrayDesc::offset is the
 * offset in the memory object where the base level of the mipmap
 * chain is.
 * ::UPTKExternalMemoryMipmappedArrayDesc::formatDesc describes the
 * format of the data.
 * ::UPTKExternalMemoryMipmappedArrayDesc::extent specifies the
 * dimensions of the base level of the mipmap chain.
 * ::UPTKExternalMemoryMipmappedArrayDesc::flags are flags associated
 * with UPTK mipmapped arrays. For further details, please refer to
 * the documentation for ::UPTKMalloc3DArray. Note that if the mipmapped
 * array is bound as a color target in the graphics API, then the flag
 * ::UPTKArrayColorAttachment must be specified in 
 * ::UPTKExternalMemoryMipmappedArrayDesc::flags.
 * ::UPTKExternalMemoryMipmappedArrayDesc::numLevels specifies
 * the total number of levels in the mipmap chain.
 *
 * The returned UPTK mipmapped array must be freed using ::UPTKFreeMipmappedArray.
 *
 * \param mipmap     - Returned UPTK mipmapped array
 * \param extMem     - Handle to external memory object
 * \param mipmapDesc - UPTK array descriptor
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKImportExternalMemory,
 * ::UPTKDestroyExternalMemory,
 * ::UPTKExternalMemoryGetMappedBuffer
 *
 * \note If ::UPTKExternalMemoryHandleDesc::type is
 * ::UPTKExternalMemoryHandleTypeNvSciBuf, then
 * ::UPTKExternalMemoryMipmappedArrayDesc::numLevels must not be greater than 1.
 */
extern __host__ UPTKError_t UPTKExternalMemoryGetMappedMipmappedArray(UPTKMipmappedArray_t *mipmap, UPTKExternalMemory_t extMem, const struct UPTKExternalMemoryMipmappedArrayDesc *mipmapDesc);

/**
 * \brief Destroys an external memory object.
 *
 * Destroys the specified external memory object. Any existing buffers
 * and UPTK mipmapped arrays mapped onto this object must no longer be
 * used and must be explicitly freed using ::UPTKFree and
 * ::UPTKFreeMipmappedArray respectively.
 *
 * \param extMem - External memory object to be destroyed
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidResourceHandle
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_destroy_ub
 *
 * \sa ::UPTKImportExternalMemory,
 * ::UPTKExternalMemoryGetMappedBuffer,
 * ::UPTKExternalMemoryGetMappedMipmappedArray
 */
extern __host__ UPTKError_t UPTKDestroyExternalMemory(UPTKExternalMemory_t extMem);

/**
 * \brief Imports an external semaphore
 *
 * Imports an externally allocated synchronization object and returns
 * a handle to that in \p extSem_out.
 *
 * The properties of the handle being imported must be described in
 * \p semHandleDesc. The ::UPTKExternalSemaphoreHandleDesc is defined
 * as follows:
 *
 * \code
        typedef struct UPTKExternalSemaphoreHandleDesc_st {
            UPTKExternalSemaphoreHandleType type;
            union {
                int fd;
                struct {
                    void *handle;
                    const void *name;
                } win32;
                const void* NvSciSyncObj;
            } handle;
            unsigned int flags;
        } UPTKExternalSemaphoreHandleDesc;
 * \endcode
 *
 * where ::UPTKExternalSemaphoreHandleDesc::type specifies the type of
 * handle being imported. ::UPTKExternalSemaphoreHandleType is defined
 * as:
 *
 * \code
        typedef enum UPTKExternalSemaphoreHandleType_enum {
            UPTKExternalSemaphoreHandleTypeOpaqueFd                = 1,
            UPTKExternalSemaphoreHandleTypeOpaqueWin32             = 2,
            UPTKExternalSemaphoreHandleTypeOpaqueWin32Kmt          = 3,
            UPTKExternalSemaphoreHandleTypeD3D12Fence              = 4,
            UPTKExternalSemaphoreHandleTypeD3D11Fence              = 5,
            UPTKExternalSemaphoreHandleTypeNvSciSync               = 6,
            UPTKExternalSemaphoreHandleTypeKeyedMutex              = 7,
            UPTKExternalSemaphoreHandleTypeKeyedMutexKmt           = 8,
            UPTKExternalSemaphoreHandleTypeTimelineSemaphoreFd     = 9,
            UPTKExternalSemaphoreHandleTypeTimelineSemaphoreWin32  = 10
        } UPTKExternalSemaphoreHandleType;
 * \endcode
 *
 * If ::UPTKExternalSemaphoreHandleDesc::type is
 * ::UPTKExternalSemaphoreHandleTypeOpaqueFd, then
 * ::UPTKExternalSemaphoreHandleDesc::handle::fd must be a valid file
 * descriptor referencing a synchronization object. OwnersUPTK of the
 * file descriptor is transferred to the UPTK driver when the handle
 * is imported successfully. Performing any operations on the file
 * descriptor after it is imported results in undefined behavior.
 *
 * If ::UPTKExternalSemaphoreHandleDesc::type is
 * ::UPTKExternalSemaphoreHandleTypeOpaqueWin32, then exactly one of
 * ::UPTKExternalSemaphoreHandleDesc::handle::win32::handle and
 * ::UPTKExternalSemaphoreHandleDesc::handle::win32::name must not be
 * NULL. If ::UPTKExternalSemaphoreHandleDesc::handle::win32::handle
 * is not NULL, then it must represent a valid shared NT handle that
 * references a synchronization object. OwnersUPTK of this handle is
 * not transferred to UPTK after the import operation, so the
 * application must release the handle using the appropriate system
 * call. If ::UPTKExternalSemaphoreHandleDesc::handle::win32::name is
 * not NULL, then it must name a valid synchronization object.
 *
 * If ::UPTKExternalSemaphoreHandleDesc::type is
 * ::UPTKExternalSemaphoreHandleTypeOpaqueWin32Kmt, then
 * ::UPTKExternalSemaphoreHandleDesc::handle::win32::handle must be
 * non-NULL and ::UPTKExternalSemaphoreHandleDesc::handle::win32::name
 * must be NULL. The handle specified must be a globally shared KMT
 * handle. This handle does not hold a reference to the underlying
 * object, and thus will be invalid when all references to the
 * synchronization object are destroyed.
 *
 * If ::UPTKExternalSemaphoreHandleDesc::type is
 * ::UPTKExternalSemaphoreHandleTypeD3D12Fence, then exactly one of
 * ::UPTKExternalSemaphoreHandleDesc::handle::win32::handle and
 * ::UPTKExternalSemaphoreHandleDesc::handle::win32::name must not be
 * NULL. If ::UPTKExternalSemaphoreHandleDesc::handle::win32::handle
 * is not NULL, then it must represent a valid shared NT handle that
 * is returned by ID3D12Device::CreateSharedHandle when referring to a
 * ID3D12Fence object. This handle holds a reference to the underlying
 * object. If ::UPTKExternalSemaphoreHandleDesc::handle::win32::name
 * is not NULL, then it must name a valid synchronization object that
 * refers to a valid ID3D12Fence object.
 *
 * If ::UPTKExternalSemaphoreHandleDesc::type is
 * ::UPTKExternalSemaphoreHandleTypeD3D11Fence, then exactly one of
 * ::UPTKExternalSemaphoreHandleDesc::handle::win32::handle and
 * ::UPTKExternalSemaphoreHandleDesc::handle::win32::name must not be
 * NULL. If ::UPTKExternalSemaphoreHandleDesc::handle::win32::handle
 * is not NULL, then it must represent a valid shared NT handle that
 * is returned by ID3D11Fence::CreateSharedHandle. If 
 * ::UPTKExternalSemaphoreHandleDesc::handle::win32::name
 * is not NULL, then it must name a valid synchronization object that
 * refers to a valid ID3D11Fence object.
 *
 * If ::UPTKExternalSemaphoreHandleDesc::type is
 * ::UPTKExternalSemaphoreHandleTypeNvSciSync, then
 * ::UPTKExternalSemaphoreHandleDesc::handle::nvSciSyncObj
 * represents a valid NvSciSyncObj.
 *
 * ::UPTKExternalSemaphoreHandleTypeKeyedMutex, then exactly one of
 * ::UPTKExternalSemaphoreHandleDesc::handle::win32::handle and
 * ::UPTKExternalSemaphoreHandleDesc::handle::win32::name must not be
 * NULL. If ::UPTKExternalSemaphoreHandleDesc::handle::win32::handle
 * is not NULL, then it represent a valid shared NT handle that
 * is returned by IDXGIResource1::CreateSharedHandle when referring to
 * a IDXGIKeyedMutex object.
 *
 * If ::UPTKExternalSemaphoreHandleDesc::type is
 * ::UPTKExternalSemaphoreHandleTypeKeyedMutexKmt, then
 * ::UPTKExternalSemaphoreHandleDesc::handle::win32::handle must be
 * non-NULL and ::UPTKExternalSemaphoreHandleDesc::handle::win32::name
 * must be NULL. The handle specified must represent a valid KMT
 * handle that is returned by IDXGIResource::GetSharedHandle when
 * referring to a IDXGIKeyedMutex object.
 *
 * If ::UPTKExternalSemaphoreHandleDesc::type is
 * ::UPTKExternalSemaphoreHandleTypeTimelineSemaphoreFd, then
 * ::UPTKExternalSemaphoreHandleDesc::handle::fd must be a valid file
 * descriptor referencing a synchronization object. OwnersUPTK of the
 * file descriptor is transferred to the UPTK driver when the handle
 * is imported successfully. Performing any operations on the file
 * descriptor after it is imported results in undefined behavior.
 *
 * If ::UPTKExternalSemaphoreHandleDesc::type is
 * ::UPTKExternalSemaphoreHandleTypeTimelineSemaphoreWin32, then exactly one of
 * ::UPTKExternalSemaphoreHandleDesc::handle::win32::handle and
 * ::UPTKExternalSemaphoreHandleDesc::handle::win32::name must not be
 * NULL. If ::UPTKExternalSemaphoreHandleDesc::handle::win32::handle
 * is not NULL, then it must represent a valid shared NT handle that
 * references a synchronization object. OwnersUPTK of this handle is
 * not transferred to UPTK after the import operation, so the
 * application must release the handle using the appropriate system
 * call. If ::UPTKExternalSemaphoreHandleDesc::handle::win32::name is
 * not NULL, then it must name a valid synchronization object.
 *
 * \param extSem_out    - Returned handle to an external semaphore
 * \param semHandleDesc - Semaphore import handle descriptor
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorOperatingSystem
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKDestroyExternalSemaphore,
 * ::UPTKSignalExternalSemaphoresAsync,
 * ::UPTKWaitExternalSemaphoresAsync
 */
extern __host__ UPTKError_t UPTKImportExternalSemaphore(UPTKExternalSemaphore_t *extSem_out, const struct UPTKExternalSemaphoreHandleDesc *semHandleDesc);

/**
 * \brief Signals a set of external semaphore objects
 *
 * Enqueues a signal operation on a set of externally allocated
 * semaphore object in the specified stream. The operations will be
 * executed when all prior operations in the stream complete.
 *
 * The exact semantics of signaling a semaphore depends on the type of
 * the object.
 *
 * If the semaphore object is any one of the following types:
 * ::UPTKExternalSemaphoreHandleTypeOpaqueFd,
 * ::UPTKExternalSemaphoreHandleTypeOpaqueWin32,
 * ::UPTKExternalSemaphoreHandleTypeOpaqueWin32Kmt
 * then signaling the semaphore will set it to the signaled state.
 *
 * If the semaphore object is any one of the following types:
 * ::UPTKExternalSemaphoreHandleTypeD3D12Fence,
 * ::UPTKExternalSemaphoreHandleTypeD3D11Fence,
 * ::UPTKExternalSemaphoreHandleTypeTimelineSemaphoreFd,
 * ::UPTKExternalSemaphoreHandleTypeTimelineSemaphoreWin32
 * then the semaphore will be set to the value specified in
 * ::UPTKExternalSemaphoreSignalParams::params::fence::value.
 *
 * If the semaphore object is of the type ::UPTKExternalSemaphoreHandleTypeNvSciSync
 * this API sets ::UPTKExternalSemaphoreSignalParams::params::nvSciSync::fence to a
 * value that can be used by subsequent waiters of the same NvSciSync object to
 * order operations with those currently submitted in \p stream. Such an update
 * will overwrite previous contents of
 * ::UPTKExternalSemaphoreSignalParams::params::nvSciSync::fence. By default,
 * signaling such an external semaphore object causes appropriate memory synchronization
 * operations to be performed over all the external memory objects that are imported as
 * ::UPTKExternalMemoryHandleTypeNvSciBuf. This ensures that any subsequent accesses
 * made by other importers of the same set of NvSciBuf memory object(s) are coherent.
 * These operations can be skipped by specifying the flag
 * ::UPTKExternalSemaphoreSignalSkipNvSciBufMemSync, which can be used as a
 * performance optimization when data coherency is not required. But specifying this
 * flag in scenarios where data coherency is required results in undefined behavior.
 * Also, for semaphore object of the type ::UPTKExternalSemaphoreHandleTypeNvSciSync,
 * if the NvSciSyncAttrList used to create the NvSciSyncObj had not set the flags in
 * ::UPTKDeviceGetNvSciSyncAttributes to UPTKNvSciSyncAttrSignal, this API will return
 * UPTKErrorNotSupported.
 * 
 * ::UPTKExternalSemaphoreSignalParams::params::nvSciSync::fence associated with 
 * semaphore object of the type ::UPTKExternalSemaphoreHandleTypeNvSciSync can be 
 * deterministic. For this the NvSciSyncAttrList used to create the semaphore object 
 * must have value of NvSciSyncAttrKey_RequireDeterministicFences key set to true. 
 * Deterministic fences allow users to enqueue a wait over the semaphore object even 
 * before corresponding signal is enqueued. For such a semaphore object, UPTK guarantees 
 * that each signal operation will increment the fence value by '1'. Users are expected 
 * to track count of signals enqueued on the semaphore object and insert waits accordingly. 
 * When such a semaphore object is signaled from multiple streams, due to concurrent 
 * stream execution, it is possible that the order in which the semaphore gets signaled 
 * is indeterministic. This could lead to waiters of the semaphore getting unblocked 
 * incorrectly. Users are expected to handle such situations, either by not using the 
 * same semaphore object with deterministic fence support enabled in different streams 
 * or by adding explicit dependency amongst such streams so that the semaphore is 
 * signaled in order.
 *
 * If the semaphore object is any one of the following types:
 * ::UPTKExternalSemaphoreHandleTypeKeyedMutex,
 * ::UPTKExternalSemaphoreHandleTypeKeyedMutexKmt,
 * then the keyed mutex will be released with the key specified in
 * ::UPTKExternalSemaphoreSignalParams::params::keyedmutex::key.
 *
 * \param extSemArray - Set of external semaphores to be signaled
 * \param paramsArray - Array of semaphore parameters
 * \param numExtSems  - Number of semaphores to signal
 * \param stream     - Stream to enqueue the signal operations in
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidResourceHandle
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKImportExternalSemaphore,
 * ::UPTKDestroyExternalSemaphore,
 * ::UPTKWaitExternalSemaphoresAsync
 */
extern __host__ UPTKError_t UPTKSignalExternalSemaphoresAsync(const UPTKExternalSemaphore_t *extSemArray, const struct UPTKExternalSemaphoreSignalParams *paramsArray, unsigned int numExtSems, UPTKStream_t stream __dv(0));

/**
 * \brief Waits on a set of external semaphore objects
 *
 * Enqueues a wait operation on a set of externally allocated
 * semaphore object in the specified stream. The operations will be
 * executed when all prior operations in the stream complete.
 *
 * The exact semantics of waiting on a semaphore depends on the type
 * of the object.
 *
 * If the semaphore object is any one of the following types:
 * ::UPTKExternalSemaphoreHandleTypeOpaqueFd,
 * ::UPTKExternalSemaphoreHandleTypeOpaqueWin32,
 * ::UPTKExternalSemaphoreHandleTypeOpaqueWin32Kmt
 * then waiting on the semaphore will wait until the semaphore reaches
 * the signaled state. The semaphore will then be reset to the
 * unsignaled state. Therefore for every signal operation, there can
 * only be one wait operation.
 *
 * If the semaphore object is any one of the following types:
 * ::UPTKExternalSemaphoreHandleTypeD3D12Fence,
 * ::UPTKExternalSemaphoreHandleTypeD3D11Fence,
 * ::UPTKExternalSemaphoreHandleTypeTimelineSemaphoreFd,
 * ::UPTKExternalSemaphoreHandleTypeTimelineSemaphoreWin32
 * then waiting on the semaphore will wait until the value of the
 * semaphore is greater than or equal to
 * ::UPTKExternalSemaphoreWaitParams::params::fence::value.
 *
 * If the semaphore object is of the type ::UPTKExternalSemaphoreHandleTypeNvSciSync
 * then, waiting on the semaphore will wait until the
 * ::UPTKExternalSemaphoreSignalParams::params::nvSciSync::fence is signaled by the
 * signaler of the NvSciSyncObj that was associated with this semaphore object.
 * By default, waiting on such an external semaphore object causes appropriate
 * memory synchronization operations to be performed over all external memory objects
 * that are imported as ::UPTKExternalMemoryHandleTypeNvSciBuf. This ensures that
 * any subsequent accesses made by other importers of the same set of NvSciBuf memory
 * object(s) are coherent. These operations can be skipped by specifying the flag
 * ::UPTKExternalSemaphoreWaitSkipNvSciBufMemSync, which can be used as a
 * performance optimization when data coherency is not required. But specifying this
 * flag in scenarios where data coherency is required results in undefined behavior.
 * Also, for semaphore object of the type ::UPTKExternalSemaphoreHandleTypeNvSciSync,
 * if the NvSciSyncAttrList used to create the NvSciSyncObj had not set the flags in
 * ::UPTKDeviceGetNvSciSyncAttributes to UPTKNvSciSyncAttrWait, this API will return
 * UPTKErrorNotSupported.
 *
 * If the semaphore object is any one of the following types:
 * ::UPTKExternalSemaphoreHandleTypeKeyedMutex,
 * ::UPTKExternalSemaphoreHandleTypeKeyedMutexKmt,
 * then the keyed mutex will be acquired when it is released with the key specified 
 * in ::UPTKExternalSemaphoreSignalParams::params::keyedmutex::key or
 * until the timeout specified by
 * ::UPTKExternalSemaphoreSignalParams::params::keyedmutex::timeoutMs
 * has lapsed. The timeout interval can either be a finite value
 * specified in milliseconds or an infinite value. In case an infinite
 * value is specified the timeout never elapses. The windows INFINITE
 * macro must be used to specify infinite timeout
 *
 * \param extSemArray - External semaphores to be waited on
 * \param paramsArray - Array of semaphore parameters
 * \param numExtSems  - Number of semaphores to wait on
 * \param stream      - Stream to enqueue the wait operations in
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidResourceHandle
 * ::UPTKErrorTimeout
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKImportExternalSemaphore,
 * ::UPTKDestroyExternalSemaphore,
 * ::UPTKSignalExternalSemaphoresAsync
 */
extern __host__ UPTKError_t UPTKWaitExternalSemaphoresAsync(const UPTKExternalSemaphore_t *extSemArray, const struct UPTKExternalSemaphoreWaitParams *paramsArray, unsigned int numExtSems, UPTKStream_t stream __dv(0));

/**
 * \brief Destroys an external semaphore
 *
 * Destroys an external semaphore object and releases any references
 * to the underlying resource. Any outstanding signals or waits must
 * have completed before the semaphore is destroyed.
 *
 * \param extSem - External semaphore to be destroyed
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidResourceHandle
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_destroy_ub
 *
 * \sa ::UPTKImportExternalSemaphore,
 * ::UPTKSignalExternalSemaphoresAsync,
 * ::UPTKWaitExternalSemaphoresAsync
 */
extern __host__ UPTKError_t UPTKDestroyExternalSemaphore(UPTKExternalSemaphore_t extSem);

/** @} */ /* END UPTKRT_EXTRES_INTEROP */

/**
 * \defgroup UPTKRT_EXECUTION Execution Control
 *
 * ___MANBRIEF___ execution control functions of the UPTK runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the execution control functions of the UPTK runtime
 * application programming interface.
 *
 * Some functions have overloaded C++ API template versions documented separately in the
 * \ref UPTKRT_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */

/**
 * \brief Launches a device function
 *
 * The function invokes kernel \p func on \p gridDim (\p gridDim.x &times; \p gridDim.y
 * &times; \p gridDim.z) grid of blocks. Each block contains \p blockDim (\p blockDim.x &times;
 * \p blockDim.y &times; \p blockDim.z) threads.
 *
 * If the kernel has N parameters the \p args should point to array of N pointers.
 * Each pointer, from <tt>args[0]</tt> to <tt>args[N - 1]</tt>, point to the region
 * of memory from which the actual parameter will be copied.
 *
 * For templated functions, pass the function symbol as follows:
 * func_name<template_arg_0,...,template_arg_N>
 *
 * \p sharedMem sets the amount of dynamic shared memory that will be available to
 * each thread block.
 *
 * \p stream specifies a stream the invocation is associated to.
 *
 * \param func        - Device function symbol
 * \param gridDim     - Grid dimentions
 * \param blockDim    - Block dimentions
 * \param args        - Arguments
 * \param sharedMem   - Shared memory
 * \param stream      - Stream identifier
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDeviceFunction,
 * ::UPTKErrorInvalidConfiguration,
 * ::UPTKErrorLaunchFailure,
 * ::UPTKErrorLaunchTimeout,
 * ::UPTKErrorLaunchOutOfResources,
 * ::UPTKErrorSharedObjectInitFailed,
 * ::UPTKErrorInvalidPtx,
 * ::UPTKErrorUnsupportedPtxVersion,
 * ::UPTKErrorNoKernelImageForDevice,
 * ::UPTKErrorJitCompilerNotFound,
 * ::UPTKErrorJitCompilationDisabled
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * \ref ::UPTKLaunchKernel(const T *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, UPTKStream_t stream) "UPTKLaunchKernel (C++ API)",
 * ::cuLaunchKernel
 */
extern __host__ UPTKError_t UPTKLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, UPTKStream_t stream);

/**
 * \brief Launches a UPTK function with launch-time configuration
 *
 * Note that the functionally equivalent variadic template ::UPTKLaunchKernelEx
 * is available for C++11 and newer.
 *
 * Invokes the kernel \p func on \p config->gridDim (\p config->gridDim.x
 * &times; \p config->gridDim.y &times; \p config->gridDim.z) grid of blocks.
 * Each block contains \p config->blockDim (\p config->blockDim.x &times;
 * \p config->blockDim.y &times; \p config->blockDim.z) threads.
 *
 * \p config->dynamicSmemBytes sets the amount of dynamic shared memory that
 * will be available to each thread block.
 *
 * \p config->stream specifies a stream the invocation is associated to.
 *
 * Configuration beyond grid and block dimensions, dynamic shared memory size,
 * and stream can be provided with the following two fields of \p config:
 *
 * \p config->attrs is an array of \p config->numAttrs contiguous
 * ::UPTKLaunchAttribute elements. The value of this pointer is not considered
 * if \p config->numAttrs is zero. However, in that case, it is recommended to
 * set the pointer to NULL.                                  
 * \p config->numAttrs is the number of attributes populating the first
 * \p config->numAttrs positions of the \p config->attrs array.
 *
 * If the kernel has N parameters the \p args should point to array of N
 * pointers. Each pointer, from <tt>args[0]</tt> to <tt>args[N - 1]</tt>, point
 * to the region of memory from which the actual parameter will be copied.
 *
 * N.B. This function is so named to avoid unintentionally invoking the
 *      templated version, \p UPTKLaunchKernelEx, for kernels taking a single
 *      void** or void* parameter.
 *
 * \param config - Launch configuration
 * \param func   - Kernel to launch
 * \param args   - Array of pointers to kernel parameters
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDeviceFunction,
 * ::UPTKErrorInvalidConfiguration,
 * ::UPTKErrorLaunchFailure,
 * ::UPTKErrorLaunchTimeout,
 * ::UPTKErrorLaunchOutOfResources,
 * ::UPTKErrorSharedObjectInitFailed,
 * ::UPTKErrorInvalidPtx,
 * ::UPTKErrorUnsupportedPtxVersion,
 * ::UPTKErrorNoKernelImageForDevice,
 * ::UPTKErrorJitCompilerNotFound,
 * ::UPTKErrorJitCompilationDisabled
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * \ref ::UPTKLaunchKernelEx(const UPTKLaunchConfig_t *config, void (*kernel)(ExpTypes...), ActTypes &&... args) "UPTKLaunchKernelEx (C++ API)",
 * ::cuLaunchKernelEx
 */
extern __host__ UPTKError_t UPTKLaunchKernelExC(const UPTKLaunchConfig_t *config, const void *func, void **args);

/**
 * \brief Launches a device function where thread blocks can cooperate and synchronize as they execute
 *
 * The function invokes kernel \p func on \p gridDim (\p gridDim.x &times; \p gridDim.y
 * &times; \p gridDim.z) grid of blocks. Each block contains \p blockDim (\p blockDim.x &times;
 * \p blockDim.y &times; \p blockDim.z) threads.
 *
 * The device on which this kernel is invoked must have a non-zero value for
 * the device attribute ::UPTKDevAttrCooperativeLaunch.
 *
 * The total number of blocks launched cannot exceed the maximum number of blocks per
 * multiprocessor as returned by ::UPTKOccupancyMaxActiveBlocksPerMultiprocessor (or
 * ::UPTKOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) times the number of multiprocessors
 * as specified by the device attribute ::UPTKDevAttrMultiProcessorCount.
 *
 * The kernel cannot make use of UPTK dynamic parallelism.
 *
 * If the kernel has N parameters the \p args should point to array of N pointers.
 * Each pointer, from <tt>args[0]</tt> to <tt>args[N - 1]</tt>, point to the region
 * of memory from which the actual parameter will be copied.
 *
 * For templated functions, pass the function symbol as follows:
 * func_name<template_arg_0,...,template_arg_N>
 *
 * \p sharedMem sets the amount of dynamic shared memory that will be available to
 * each thread block.
 *
 * \p stream specifies a stream the invocation is associated to.
 *
 * \param func        - Device function symbol
 * \param gridDim     - Grid dimentions
 * \param blockDim    - Block dimentions
 * \param args        - Arguments
 * \param sharedMem   - Shared memory
 * \param stream      - Stream identifier
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDeviceFunction,
 * ::UPTKErrorInvalidConfiguration,
 * ::UPTKErrorLaunchFailure,
 * ::UPTKErrorLaunchTimeout,
 * ::UPTKErrorLaunchOutOfResources,
 * ::UPTKErrorCooperativeLaunchTooLarge,
 * ::UPTKErrorSharedObjectInitFailed
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * \ref ::UPTKLaunchCooperativeKernel(const T *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, UPTKStream_t stream) "UPTKLaunchCooperativeKernel (C++ API)",
 * ::UPTKLaunchCooperativeKernelMultiDevice,
 * ::cuLaunchCooperativeKernel
 */
extern __host__ UPTKError_t UPTKLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, UPTKStream_t stream);

/**
 * \brief Launches device functions on multiple devices where thread blocks can cooperate and synchronize as they execute
 *
 * \deprecated This function is deprecated as of UPTK 11.3.
 *
 * Invokes kernels as specified in the \p launchParamsList array where each element
 * of the array specifies all the parameters required to perform a single kernel launch.
 * These kernels can cooperate and synchronize as they execute. The size of the array is
 * specified by \p numDevices.
 *
 * No two kernels can be launched on the same device. All the devices targeted by this
 * multi-device launch must be identical. All devices must have a non-zero value for the
 * device attribute ::UPTKDevAttrCooperativeMultiDeviceLaunch.
 *
 * The same kernel must be launched on all devices. Note that any __device__ or __constant__
 * variables are independently instantiated on every device. It is the application's
 * responsiblity to ensure these variables are initialized and used appropriately.
 *
 * The size of the grids as specified in blocks, the size of the blocks themselves and the
 * amount of shared memory used by each thread block must also match across all launched kernels.
 *
 * The streams used to launch these kernels must have been created via either ::UPTKStreamCreate
 * or ::UPTKStreamCreateWithPriority or ::UPTKStreamCreateWithPriority. The NULL stream or
 * ::UPTKStreamLegacy or ::UPTKStreamPerThread cannot be used.
 *
 * The total number of blocks launched per kernel cannot exceed the maximum number of blocks
 * per multiprocessor as returned by ::UPTKOccupancyMaxActiveBlocksPerMultiprocessor (or
 * ::UPTKOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) times the number of multiprocessors
 * as specified by the device attribute ::UPTKDevAttrMultiProcessorCount. Since the
 * total number of blocks launched per device has to match across all devices, the maximum
 * number of blocks that can be launched per device will be limited by the device with the
 * least number of multiprocessors.
 *
 * The kernel cannot make use of UPTK dynamic parallelism.
 *
 * The ::UPTKLaunchParams structure is defined as:
 * \code
        struct UPTKLaunchParams
        {
            void *func;
            dim3 gridDim;
            dim3 blockDim;
            void **args;
            size_t sharedMem;
            UPTKStream_t stream;
        };
 * \endcode
 * where:
 * - ::UPTKLaunchParams::func specifies the kernel to be launched. This same functions must
 *   be launched on all devices. For templated functions, pass the function symbol as follows:
 *   func_name<template_arg_0,...,template_arg_N>
 * - ::UPTKLaunchParams::gridDim specifies the width, height and depth of the grid in blocks.
 *   This must match across all kernels launched.
 * - ::UPTKLaunchParams::blockDim is the width, height and depth of each thread block. This
 *   must match across all kernels launched.
 * - ::UPTKLaunchParams::args specifies the arguments to the kernel. If the kernel has
 *   N parameters then ::UPTKLaunchParams::args should point to array of N pointers. Each
 *   pointer, from <tt>::UPTKLaunchParams::args[0]</tt> to <tt>::UPTKLaunchParams::args[N - 1]</tt>,
 *   point to the region of memory from which the actual parameter will be copied.
 * - ::UPTKLaunchParams::sharedMem is the dynamic shared-memory size per thread block in bytes.
 *   This must match across all kernels launched.
 * - ::UPTKLaunchParams::stream is the handle to the stream to perform the launch in. This cannot
 *   be the NULL stream or ::UPTKStreamLegacy or ::UPTKStreamPerThread.
 *
 * By default, the kernel won't begin execution on any GPU until all prior work in all the specified
 * streams has completed. This behavior can be overridden by specifying the flag
 * ::UPTKCooperativeLaunchMultiDeviceNoPreSync. When this flag is specified, each kernel
 * will only wait for prior work in the stream corresponding to that GPU to complete before it begins
 * execution.
 *
 * Similarly, by default, any subsequent work pushed in any of the specified streams will not begin
 * execution until the kernels on all GPUs have completed. This behavior can be overridden by specifying
 * the flag ::UPTKCooperativeLaunchMultiDeviceNoPostSync. When this flag is specified,
 * any subsequent work pushed in any of the specified streams will only wait for the kernel launched
 * on the GPU corresponding to that stream to complete before it begins execution.
 *
 * \param launchParamsList - List of launch parameters, one per device
 * \param numDevices       - Size of the \p launchParamsList array
 * \param flags            - Flags to control launch behavior
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDeviceFunction,
 * ::UPTKErrorInvalidConfiguration,
 * ::UPTKErrorLaunchFailure,
 * ::UPTKErrorLaunchTimeout,
 * ::UPTKErrorLaunchOutOfResources,
 * ::UPTKErrorCooperativeLaunchTooLarge,
 * ::UPTKErrorSharedObjectInitFailed
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * \ref ::UPTKLaunchCooperativeKernel(const T *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, UPTKStream_t stream) "UPTKLaunchCooperativeKernel (C++ API)",
 * ::UPTKLaunchCooperativeKernel,
 * ::cuLaunchCooperativeKernelMultiDevice
 */
extern __host__ UPTKError_t UPTKLaunchCooperativeKernelMultiDevice(struct UPTKLaunchParams *launchParamsList, unsigned int numDevices, unsigned int flags  __dv(0));

/**
 * \brief Sets the preferred cache configuration for a device function
 *
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this sets through \p cacheConfig the preferred cache configuration
 * for the function specified via \p func. This is only a preference. The
 * runtime will use the requested configuration if possible, but it is free to
 * choose a different configuration if required to execute \p func.
 *
 * \p func is a device function symbol and must be declared as a
 * \c __global__ function. If the specified function does not exist,
 * then ::UPTKErrorInvalidDeviceFunction is returned. For templated functions,
 * pass the function symbol as follows: func_name<template_arg_0,...,template_arg_N>
 *
 * This setting does nothing on devices where the size of the L1 cache and
 * shared memory are fixed.
 *
 * Launching a kernel with a different preference than the most recent
 * preference setting may insert a device-side synchronization point.
 *
 * The supported cache configurations are:
 * - ::UPTKFuncCachePreferNone: no preference for shared memory or L1 (default)
 * - ::UPTKFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
 * - ::UPTKFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
 * - ::UPTKFuncCachePreferEqual: prefer equal size L1 cache and shared memory
 *
 * \param func        - Device function symbol
 * \param cacheConfig - Requested cache configuration
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDeviceFunction
 * \notefnerr
 * \note_string_api_deprecation2
 * \note_init_rt
 * \note_callback
 *
 * \sa 
 * \ref ::UPTKFuncSetCacheConfig(T*, enum UPTKFuncCache) "UPTKFuncSetCacheConfig (C++ API)",
 * \ref ::UPTKFuncGetAttributes(struct UPTKFuncAttributes*, const void*) "UPTKFuncGetAttributes (C API)",
 * \ref ::UPTKLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, UPTKStream_t stream) "UPTKLaunchKernel (C API)",
 * ::cuFuncSetCacheConfig
 */
extern __host__ UPTKError_t UPTKFuncSetCacheConfig(const void *func, enum UPTKFuncCache cacheConfig);

/**
 * \brief Find out attributes for a given function
 *
 * This function obtains the attributes of a function specified via \p func.
 * \p func is a device function symbol and must be declared as a
 * \c __global__ function. The fetched attributes are placed in \p attr.
 * If the specified function does not exist, then 
 * ::UPTKErrorInvalidDeviceFunction is returned.
 * For templated functions, pass the function symbol as follows: 
 * func_name<template_arg_0,...,template_arg_N>
 *
 * Note that some function attributes such as
 * \ref ::UPTKFuncAttributes::maxThreadsPerBlock "maxThreadsPerBlock"
 * may vary based on the device that is currently being used.
 *
 * \param attr - Return pointer to function's attributes
 * \param func - Device function symbol
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDeviceFunction
 * \notefnerr
 * \note_string_api_deprecation2
 * \note_init_rt
 * \note_callback
 *
 * \sa 
 * \ref ::UPTKFuncSetCacheConfig(const void*, enum UPTKFuncCache) "UPTKFuncSetCacheConfig (C API)",
 * \ref ::UPTKFuncGetAttributes(struct UPTKFuncAttributes*, T*) "UPTKFuncGetAttributes (C++ API)",
 * \ref ::UPTKLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, UPTKStream_t stream) "UPTKLaunchKernel (C API)",
 * ::cuFuncGetAttribute
 */
extern __host__ UPTKError_t UPTKFuncGetAttributes(struct UPTKFuncAttributes *attr, const void *func);


/**
 * \brief Set attributes for a given function
 *
 * This function sets the attributes of a function specified via \p func.
 * The parameter \p func must be a pointer to a function that executes
 * on the device. The parameter specified by \p func must be declared as a \p __global__
 * function. The enumeration defined by \p attr is set to the value defined by \p value.
 * If the specified function does not exist, then 
 * ::UPTKErrorInvalidDeviceFunction is returned.
 * If the specified attribute cannot be written, or if the value is incorrect, 
 * then ::UPTKErrorInvalidValue is returned.
 *
 * Valid values for \p attr are:
 * - ::UPTKFuncAttributeMaxDynamicSharedMemorySize - The requested maximum size in bytes of dynamically-allocated shared memory. The sum of this value and the function attribute ::sharedSizeBytes
 *   cannot exceed the device attribute ::UPTKDevAttrMaxSharedMemoryPerBlockOptin. The maximal size of requestable dynamic shared memory may differ by GPU architecture.
 * - ::UPTKFuncAttributePreferredSharedMemoryCarveout - On devices where the L1 cache and shared memory use the same hardware resources, 
 *   this sets the shared memory carveout preference, in percent of the total shared memory. See ::UPTKDevAttrMaxSharedMemoryPerMultiprocessor.
 *   This is only a hint, and the driver can choose a different ratio if required to execute the function.
 * - ::UPTKFuncAttributeRequiredClusterWidth: The required cluster width in
 *   blocks. The width, height, and depth values must either all be 0 or all be
 *   positive. The validity of the cluster dimensions is checked at launch time.
 *   If the value is set during compile time, it cannot be set at runtime.
 *   Setting it at runtime will return UPTKErrorNotPermitted.
 * - ::UPTKFuncAttributeRequiredClusterHeight: The required cluster height in
 *   blocks. The width, height, and depth values must either all be 0 or all be
 *   positive. The validity of the cluster dimensions is checked at launch time.
 *   If the value is set during compile time, it cannot be set at runtime.
 *   Setting it at runtime will return UPTKErrorNotPermitted.
 * - ::UPTKFuncAttributeRequiredClusterDepth: The required cluster depth in
 *   blocks. The width, height, and depth values must either all be 0 or all be
 *   positive. The validity of the cluster dimensions is checked at launch time.
 *   If the value is set during compile time, it cannot be set at runtime.
 *   Setting it at runtime will return UPTKErrorNotPermitted.
 * - ::UPTKFuncAttributeNonPortableClusterSizeAllowed: Indicates whether the
 *   function can be launched with non-portable cluster size. 1 is allowed, 0 is
 *   disallowed.
 * - ::UPTKFuncAttributeClusterSchedulingPolicyPreference: The block
 *   scheduling policy of a function. The value type is UPTKClusterSchedulingPolicy.
 *
 * \param func  - Function to get attributes of
 * \param attr  - Attribute to set
 * \param value - Value to set
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDeviceFunction,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \ref ::UPTKLaunchKernel(const T *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, UPTKStream_t stream) "UPTKLaunchKernel (C++ API)",
 * \ref ::UPTKFuncSetCacheConfig(T*, enum UPTKFuncCache) "UPTKFuncSetCacheConfig (C++ API)",
 * \ref ::UPTKFuncGetAttributes(struct UPTKFuncAttributes*, const void*) "UPTKFuncGetAttributes (C API)",
 */
extern __host__ UPTKError_t UPTKFuncSetAttribute(const void *func, enum UPTKFuncAttribute attr, int value);

/**
 * \brief Returns the function name for a device entry function pointer.
 *
 * Returns in \p **name the function name associated with the symbol \p func .
 * The function name is returned as a null-terminated string. This API may
 * return a mangled name if the function is not declared as having C linkage.
 * If \p **name is NULL, ::UPTKErrorInvalidValue is returned.
 * If \p func is not a device entry function, ::UPTKErrorInvalidDeviceFunction is returned.
 *
 * \param name - The returned name of the function
 * \param func - The function pointer to retrieve name for
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidDeviceFunction
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \ref ::UPTKFuncGetName(const char **name, const T *func) "UPTKFuncGetName (C++ API)"
 */
extern __host__ UPTKError_t UPTKFuncGetName(const char **name, const void *func);

/**
 * \brief Returns the offset and size of a kernel parameter in the device-side parameter layout.
 *
 * Queries the kernel parameter at \p paramIndex in \p func's list of parameters and returns
 * parameter information via \p paramOffset and \p paramSize. \p paramOffset returns the
 * offset of the parameter in the device-side parameter layout. \p paramSize returns the size
 * in bytes of the parameter. This information can be used to update kernel node parameters
 * from the device via ::UPTKGraphKernelNodeSetParam() and ::UPTKGraphKernelNodeUpdatesApply().
 * \p paramIndex must be less than the number of parameters that \p func takes.
 *
 * \param func        - The function to query
 * \param paramIndex  - The parameter index to query
 * \param paramOffset - The offset into the device-side parameter layout at which the parameter resides
 * \param paramSize   - The size of the parameter in the device-side parameter layout
 *
 * \return
 * ::UPTK_SUCCESS,
 * ::UPTK_ERROR_INVALID_VALUE,
 * \notefnerr
 */
extern __host__ UPTKError_t UPTKFuncGetParamInfo(const void *func, size_t paramIndex, size_t *paramOffset, size_t *paramSize);

/**
 * \brief Converts a double argument to be executed on a device
 *
 * \param d - Double to convert
 *
 * \deprecated This function is deprecated as of UPTK 7.5
 *
 * Converts the double value of \p d to an internal float representation if
 * the device does not support double arithmetic. If the device does natively
 * support doubles, then this function does nothing.
 *
 * \return
 * ::UPTKSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * \ref ::UPTKFuncSetCacheConfig(const void*, enum UPTKFuncCache) "UPTKFuncSetCacheConfig (C API)",
 * \ref ::UPTKFuncGetAttributes(struct UPTKFuncAttributes*, const void*) "UPTKFuncGetAttributes (C API)",
 * ::UPTKSetDoubleForHost
 */
extern __host__ UPTKError_t UPTKSetDoubleForDevice(double *d);

/**
 * \brief Converts a double argument after execution on a device
 *
 * \deprecated This function is deprecated as of UPTK 7.5
 *
 * Converts the double value of \p d from a potentially internal float
 * representation if the device does not support double arithmetic. If the
 * device does natively support doubles, then this function does nothing.
 *
 * \param d - Double to convert
 *
 * \return
 * ::UPTKSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * \ref ::UPTKFuncSetCacheConfig(const void*, enum UPTKFuncCache) "UPTKFuncSetCacheConfig (C API)",
 * \ref ::UPTKFuncGetAttributes(struct UPTKFuncAttributes*, const void*) "UPTKFuncGetAttributes (C API)",
 * ::UPTKSetDoubleForDevice
 */
extern  __host__ UPTKError_t UPTKSetDoubleForHost(double *d);

/**
 * \brief Enqueues a host function call in a stream
 *
 * Enqueues a host function to run in a stream.  The function will be called
 * after currently enqueued work and will block work added after it.
 *
 * The host function must not make any UPTK API calls.  Attempting to use a
 * UPTK API may result in ::UPTKErrorNotPermitted, but this is not required.
 * The host function must not perform any synchronization that may depend on
 * outstanding UPTK work not mandated to run earlier.  Host functions without a
 * mandated order (such as in independent streams) execute in undefined order
 * and may be serialized.
 *
 * For the purposes of Unified Memory, execution makes a number of guarantees:
 * <ul>
 *   <li>The stream is considered idle for the duration of the function's
 *   execution.  Thus, for example, the function may always use memory attached
 *   to the stream it was enqueued in.</li>
 *   <li>The start of execution of the function has the same effect as
 *   synchronizing an event recorded in the same stream immediately prior to
 *   the function.  It thus synchronizes streams which have been "joined"
 *   prior to the function.</li>
 *   <li>Adding device work to any stream does not have the effect of making
 *   the stream active until all preceding host functions and stream callbacks
 *   have executed.  Thus, for
 *   example, a function might use global attached memory even if work has
 *   been added to another stream, if the work has been ordered behind the
 *   function call with an event.</li>
 *   <li>Completion of the function does not cause a stream to become
 *   active except as described above.  The stream will remain idle
 *   if no device work follows the function, and will remain idle across
 *   consecutive host functions or stream callbacks without device work in
 *   between.  Thus, for example,
 *   stream synchronization can be done by signaling from a host function at the
 *   end of the stream.</li>
 * </ul>
 *
 * Note that, in constrast to ::cuStreamAddCallback, the function will not be
 * called in the event of an error in the UPTK context.
 *
 * \param hStream  - Stream to enqueue function call in
 * \param fn       - The function to call once preceding stream operations are complete
 * \param userData - User-specified data to be passed to the function
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorNotSupported
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKStreamCreate,
 * ::UPTKStreamQuery,
 * ::UPTKStreamSynchronize,
 * ::UPTKStreamWaitEvent,
 * ::UPTKStreamDestroy,
 * ::UPTKMallocManaged,
 * ::UPTKStreamAttachMemAsync,
 * ::UPTKStreamAddCallback,
 * ::cuLaunchHostFunc
 */
extern __host__ UPTKError_t UPTKLaunchHostFunc(UPTKStream_t stream, UPTKHostFn_t fn, void *userData);

/** @} */ /* END UPTKRT_EXECUTION */

/**
 * \defgroup UPTKRT_EXECUTION_DEPRECATED Execution Control [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated execution control functions of the UPTK runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the deprecated execution control functions of the UPTK runtime
 * application programming interface.
 *
 * Some functions have overloaded C++ API template versions documented separately in the
 * \ref UPTKRT_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */

/**
 * \brief Sets the shared memory configuration for a device function
 *
 * \deprecated
 *
 * On devices with configurable shared memory banks, this function will 
 * force all subsequent launches of the specified device function to have
 * the given shared memory bank size configuration. On any given launch of the
 * function, the shared memory configuration of the device will be temporarily
 * changed if needed to suit the function's preferred configuration. Changes in
 * shared memory configuration between subsequent launches of functions, 
 * may introduce a device side synchronization point.
 *
 * Any per-function setting of shared memory bank size set via 
 * ::UPTKFuncSetSharedMemConfig will override the device wide setting set by
 * ::UPTKDeviceSetSharedMemConfig.
 *
 * Changing the shared memory bank size will not increase shared memory usage
 * or affect occupancy of kernels, but may have major effects on performance. 
 * Larger bank sizes will allow for greater potential bandwidth to shared memory,
 * but will change what kinds of accesses to shared memory will result in bank 
 * conflicts.
 *
 * This function will do nothing on devices with fixed shared memory bank size.
 *
 * For templated functions, pass the function symbol as follows:
 * func_name<template_arg_0,...,template_arg_N>
 *
 * The supported bank configurations are:
 * - ::UPTKSharedMemBankSizeDefault: use the device's shared memory configuration
 *   when launching this function.
 * - ::UPTKSharedMemBankSizeFourByte: set shared memory bank width to be 
 *   four bytes natively when launching this function.
 * - ::UPTKSharedMemBankSizeEightByte: set shared memory bank width to be eight 
 *   bytes natively when launching this function.
 *
 * \param func   - Device function symbol
 * \param config - Requested shared memory configuration
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDeviceFunction,
 * ::UPTKErrorInvalidValue,
 * \notefnerr
 * \note_string_api_deprecation2
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKDeviceSetSharedMemConfig,
 * ::UPTKDeviceGetSharedMemConfig,
 * ::UPTKDeviceSetCacheConfig,
 * ::UPTKDeviceGetCacheConfig,
 * ::UPTKFuncSetCacheConfig,
 * ::cuFuncSetSharedMemConfig
 */
extern __host__ UPTKError_t UPTKFuncSetSharedMemConfig(const void *func, enum UPTKSharedMemConfig config);
/** @} */ /* END UPTKRT_EXECUTION_DEPRECATED */

/**
 * \defgroup UPTKRT_OCCUPANCY Occupancy
 *
 * ___MANBRIEF___ occupancy calculation functions of the UPTK runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the occupancy calculation functions of the UPTK runtime
 * application programming interface.
 *
 * Besides the occupancy calculator functions
 * (\ref ::UPTKOccupancyMaxActiveBlocksPerMultiprocessor and \ref ::UPTKOccupancyMaxActiveBlocksPerMultiprocessorWithFlags),
 * there are also C++ only occupancy-based launch configuration functions documented in
 * \ref UPTKRT_HIGHLEVEL "C++ API Routines" module.
 *
 * See
 * \ref ::UPTKOccupancyMaxPotentialBlockSize(int*, int*, T, size_t, int) "UPTKOccupancyMaxPotentialBlockSize (C++ API)",
 * \ref ::UPTKOccupancyMaxPotentialBlockSizeWithFlags(int*, int*, T, size_t, int, unsigned int) "UPTKOccupancyMaxPotentialBlockSize (C++ API)",
 * \ref ::UPTKOccupancyMaxPotentialBlockSizeVariableSMem(int*, int*, T, UnaryFunction, int) "UPTKOccupancyMaxPotentialBlockSizeVariableSMem (C++ API)",
 * \ref ::UPTKOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int*, int*, T, UnaryFunction, int, unsigned int) "UPTKOccupancyMaxPotentialBlockSizeVariableSMem (C++ API)"
 * \ref ::UPTKOccupancyAvailableDynamicSMemPerBlock(size_t*, T, int, int) "UPTKOccupancyAvailableDynamicSMemPerBlock (C++ API)",
 *
 * @{
 */

/**
 * \brief Returns occupancy for a device function
 *
 * Returns in \p *numBlocks the maximum number of active blocks per
 * streaming multiprocessor for the device function.
 *
 * \param numBlocks       - Returned occupancy
 * \param func            - Kernel function for which occupancy is calculated
 * \param blockSize       - Block size the kernel is intended to be launched with
 * \param dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDevice,
 * ::UPTKErrorInvalidDeviceFunction,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorUnknown,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
 * \ref ::UPTKOccupancyMaxPotentialBlockSize(int*, int*, T, size_t, int) "UPTKOccupancyMaxPotentialBlockSize (C++ API)",
 * \ref ::UPTKOccupancyMaxPotentialBlockSizeWithFlags(int*, int*, T, size_t, int, unsigned int) "UPTKOccupancyMaxPotentialBlockSizeWithFlags (C++ API)",
 * \ref ::UPTKOccupancyMaxPotentialBlockSizeVariableSMem(int*, int*, T, UnaryFunction, int) "UPTKOccupancyMaxPotentialBlockSizeVariableSMem (C++ API)",
 * \ref ::UPTKOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int*, int*, T, UnaryFunction, int, unsigned int) "UPTKOccupancyMaxPotentialBlockSizeVariableSMemWithFlags (C++ API)",
 * \ref ::UPTKOccupancyAvailableDynamicSMemPerBlock(size_t*, T, int, int) "UPTKOccupancyAvailableDynamicSMemPerBlock (C++ API)",
 * ::cuOccupancyMaxActiveBlocksPerMultiprocessor
 */
extern __host__ UPTKError_t UPTKOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize);

/**
 * \brief Returns dynamic shared memory available per block when launching \p numBlocks blocks on SM.
 *
 * Returns in \p *dynamicSmemSize the maximum size of dynamic shared memory to allow \p numBlocks blocks per SM. 
 *
 * \param dynamicSmemSize - Returned maximum dynamic shared memory 
 * \param func            - Kernel function for which occupancy is calculated
 * \param numBlocks       - Number of blocks to fit on SM 
 * \param blockSize       - Size of the block
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDevice,
 * ::UPTKErrorInvalidDeviceFunction,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorUnknown,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
 * \ref ::UPTKOccupancyMaxPotentialBlockSize(int*, int*, T, size_t, int) "UPTKOccupancyMaxPotentialBlockSize (C++ API)",
 * \ref ::UPTKOccupancyMaxPotentialBlockSizeWithFlags(int*, int*, T, size_t, int, unsigned int) "UPTKOccupancyMaxPotentialBlockSizeWithFlags (C++ API)",
 * \ref ::UPTKOccupancyMaxPotentialBlockSizeVariableSMem(int*, int*, T, UnaryFunction, int) "UPTKOccupancyMaxPotentialBlockSizeVariableSMem (C++ API)",
 * \ref ::UPTKOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int*, int*, T, UnaryFunction, int, unsigned int) "UPTKOccupancyMaxPotentialBlockSizeVariableSMemWithFlags (C++ API)",
 * ::UPTKOccupancyAvailableDynamicSMemPerBlock
 */
extern __host__ UPTKError_t UPTKOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, const void *func, int numBlocks, int blockSize);

/**
 * \brief Returns occupancy for a device function with the specified flags
 *
 * Returns in \p *numBlocks the maximum number of active blocks per
 * streaming multiprocessor for the device function.
 *
 * The \p flags parameter controls how special cases are handled. Valid flags include:
 *
 * - ::UPTKOccupancyDefault: keeps the default behavior as
 *   ::UPTKOccupancyMaxActiveBlocksPerMultiprocessor
 *
 * - ::UPTKOccupancyDisableCachingOverride: This flag suppresses the default behavior
 *   on platform where global caching affects occupancy. On such platforms, if caching
 *   is enabled, but per-block SM resource usage would result in zero occupancy, the
 *   occupancy calculator will calculate the occupancy as if caching is disabled.
 *   Setting this flag makes the occupancy calculator to return 0 in such cases.
 *   More information can be found about this feature in the "Unified L1/Texture Cache"
 *   section of the Maxwell tuning guide.
 *
 * \param numBlocks       - Returned occupancy
 * \param func            - Kernel function for which occupancy is calculated
 * \param blockSize       - Block size the kernel is intended to be launched with
 * \param dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes
 * \param flags           - Requested behavior for the occupancy calculator
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDevice,
 * ::UPTKErrorInvalidDeviceFunction,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorUnknown,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKOccupancyMaxActiveBlocksPerMultiprocessor,
 * \ref ::UPTKOccupancyMaxPotentialBlockSize(int*, int*, T, size_t, int) "UPTKOccupancyMaxPotentialBlockSize (C++ API)",
 * \ref ::UPTKOccupancyMaxPotentialBlockSizeWithFlags(int*, int*, T, size_t, int, unsigned int) "UPTKOccupancyMaxPotentialBlockSizeWithFlags (C++ API)",
 * \ref ::UPTKOccupancyMaxPotentialBlockSizeVariableSMem(int*, int*, T, UnaryFunction, int) "UPTKOccupancyMaxPotentialBlockSizeVariableSMem (C++ API)",
 * \ref ::UPTKOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int*, int*, T, UnaryFunction, int, unsigned int) "UPTKOccupancyMaxPotentialBlockSizeVariableSMemWithFlags (C++ API)",
 * \ref ::UPTKOccupancyAvailableDynamicSMemPerBlock(size_t*, T, int, int) "UPTKOccupancyAvailableDynamicSMemPerBlock (C++ API)",
 * ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
 */
extern __host__ UPTKError_t UPTKOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags);

/**
 * \brief Given the kernel function (\p func) and launch configuration
 * (\p config), return the maximum cluster size in \p *clusterSize.
 *
 * The cluster dimensions in \p config are ignored. If func has a required
 * cluster size set (see ::UPTKFuncGetAttributes),\p *clusterSize will reflect 
 * the required cluster size.
 *
 * By default this function will always return a value that's portable on
 * future hardware. A higher value may be returned if the kernel function
 * allows non-portable cluster sizes.
 *
 * This function will respect the compile time launch bounds.
 *
 * \param clusterSize - Returned maximum cluster size that can be launched
 *                      for the given kernel function and launch configuration
 * \param func        - Kernel function for which maximum cluster
 *                      size is calculated
 * \param config      - Launch configuration for the given kernel function
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDeviceFunction,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorUnknown,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKFuncGetAttributes
 * \ref ::UPTKOccupancyMaxPotentialClusterSize(int*, T, const UPTKLaunchConfig_t*) "UPTKOccupancyMaxPotentialClusterSize (C++ API)",
 * ::cuOccupancyMaxPotentialClusterSize
 */
extern __host__ UPTKError_t UPTKOccupancyMaxPotentialClusterSize(int *clusterSize, const void *func, const UPTKLaunchConfig_t *launchConfig);


/**
 * \brief Given the kernel function (\p func) and launch configuration
 * (\p config), return the maximum number of clusters that could co-exist
 * on the target device in \p *numClusters.
 *
 * If the function has required cluster size already set (see
 * ::UPTKFuncGetAttributes), the cluster size from config must either be
 * unspecified or match the required size.
 * Without required sizes, the cluster size must be specified in config,
 * else the function will return an error.
 *
 * Note that various attributes of the kernel function may affect occupancy
 * calculation. Runtime environment may affect how the hardware schedules
 * the clusters, so the calculated occupancy is not guaranteed to be achievable.
 *
 * \param numClusters - Returned maximum number of clusters that
 *                      could co-exist on the target device
 * \param func        - Kernel function for which maximum number
 *                      of clusters are calculated
 * \param config      - Launch configuration for the given kernel function
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDeviceFunction,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidClusterSize,
 * ::UPTKErrorUnknown,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKFuncGetAttributes
 * \ref ::UPTKOccupancyMaxActiveClusters(int*, T, const UPTKLaunchConfig_t*) "UPTKOccupancyMaxActiveClusters (C++ API)",
 * ::cuOccupancyMaxActiveClusters
 */
extern __host__ UPTKError_t UPTKOccupancyMaxActiveClusters(int *numClusters, const void *func, const UPTKLaunchConfig_t *launchConfig);
/** @} */ /* END UPTK_OCCUPANCY */

/**
 * \defgroup UPTKRT_MEMORY Memory Management
 *
 * ___MANBRIEF___ memory management functions of the UPTK runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the memory management functions of the UPTK runtime
 * application programming interface.
 *
 * Some functions have overloaded C++ API template versions documented separately in the
 * \ref UPTKRT_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */

/**
 * \brief Allocates memory that will be automatically managed by the Unified Memory system
 *
 * Allocates \p size bytes of managed memory on the device and returns in
 * \p *devPtr a pointer to the allocated memory. If the device doesn't support
 * allocating managed memory, ::UPTKErrorNotSupported is returned. Support
 * for managed memory can be queried using the device attribute
 * ::UPTKDevAttrManagedMemory. The allocated memory is suitably
 * aligned for any kind of variable. The memory is not cleared. If \p size
 * is 0, ::UPTKMallocManaged returns ::UPTKErrorInvalidValue. The pointer
 * is valid on the CPU and on all GPUs in the system that support managed memory.
 * All accesses to this pointer must obey the Unified Memory programming model.
 *
 * \p flags specifies the default stream association for this allocation.
 * \p flags must be one of ::UPTKMemAttachGlobal or ::UPTKMemAttachHost. The
 * default value for \p flags is ::UPTKMemAttachGlobal.
 * If ::UPTKMemAttachGlobal is specified, then this memory is accessible from
 * any stream on any device. If ::UPTKMemAttachHost is specified, then the
 * allocation should not be accessed from devices that have a zero value for the
 * device attribute ::UPTKDevAttrConcurrentManagedAccess; an explicit call to
 * ::UPTKStreamAttachMemAsync will be required to enable access on such devices.
 *
 * If the association is later changed via ::UPTKStreamAttachMemAsync to
 * a single stream, the default association, as specifed during ::UPTKMallocManaged,
 * is restored when that stream is destroyed. For __managed__ variables, the
 * default association is always ::UPTKMemAttachGlobal. Note that destroying a
 * stream is an asynchronous operation, and as a result, the change to default
 * association won't happen until all work in the stream has completed.
 *
 * Memory allocated with ::UPTKMallocManaged should be released with ::UPTKFree.
 *
 * Device memory oversubscription is possible for GPUs that have a non-zero value for the
 * device attribute ::UPTKDevAttrConcurrentManagedAccess. Managed memory on
 * such GPUs may be evicted from device memory to host memory at any time by the Unified
 * Memory driver in order to make room for other allocations.
 *
 * In a system where all GPUs have a non-zero value for the device attribute
 * ::UPTKDevAttrConcurrentManagedAccess, managed memory may not be populated when this
 * API returns and instead may be populated on access. In such systems, managed memory can
 * migrate to any processor's memory at any time. The Unified Memory driver will employ heuristics to
 * maintain data locality and prevent excessive page faults to the extent possible. The application
 * can also guide the driver about memory usage patterns via ::UPTKMemAdvise. The application
 * can also explicitly migrate memory to a desired processor's memory via
 * ::UPTKMemPrefetchAsync.
 *
 * In a multi-GPU system where all of the GPUs have a zero value for the device attribute
 * ::UPTKDevAttrConcurrentManagedAccess and all the GPUs have peer-to-peer support
 * with each other, the physical storage for managed memory is created on the GPU which is active
 * at the time ::UPTKMallocManaged is called. All other GPUs will reference the data at reduced
 * bandwidth via peer mappings over the PCIe bus. The Unified Memory driver does not migrate
 * memory among such GPUs.
 *
 * In a multi-GPU system where not all GPUs have peer-to-peer support with each other and
 * where the value of the device attribute ::UPTKDevAttrConcurrentManagedAccess
 * is zero for at least one of those GPUs, the location chosen for physical storage of managed
 * memory is system-dependent.
 * - On Linux, the location chosen will be device memory as long as the current set of active
 * contexts are on devices that either have peer-to-peer support with each other or have a
 * non-zero value for the device attribute ::UPTKDevAttrConcurrentManagedAccess.
 * If there is an active context on a GPU that does not have a non-zero value for that device
 * attribute and it does not have peer-to-peer support with the other devices that have active
 * contexts on them, then the location for physical storage will be 'zero-copy' or host memory.
 * Note that this means that managed memory that is located in device memory is migrated to
 * host memory if a new context is created on a GPU that doesn't have a non-zero value for
 * the device attribute and does not support peer-to-peer with at least one of the other devices
 * that has an active context. This in turn implies that context creation may fail if there is
 * insufficient host memory to migrate all managed allocations.
 * - On Windows, the physical storage is always created in 'zero-copy' or host memory.
 * All GPUs will reference the data at reduced bandwidth over the PCIe bus. In these
 * circumstances, use of the environment variable UPTK_VISIBLE_DEVICES is recommended to
 * restrict UPTK to only use those GPUs that have peer-to-peer support.
 * Alternatively, users can also set UPTK_MANAGED_FORCE_DEVICE_ALLOC to a non-zero
 * value to force the driver to always use device memory for physical storage.
 * When this environment variable is set to a non-zero value, all devices used in
 * that process that support managed memory have to be peer-to-peer compatible
 * with each other. The error ::UPTKErrorInvalidDevice will be returned if a device
 * that supports managed memory is used and it is not peer-to-peer compatible with
 * any of the other managed memory supporting devices that were previously used in
 * that process, even if ::UPTKDeviceReset has been called on those devices. These
 * environment variables are described in the UPTK programming guide under the
 * "UPTK environment variables" section.
 *
 * \param devPtr - Pointer to allocated device memory
 * \param size   - Requested allocation size in bytes
 * \param flags  - Must be either ::UPTKMemAttachGlobal or ::UPTKMemAttachHost (defaults to ::UPTKMemAttachGlobal)
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorMemoryAllocation,
 * ::UPTKErrorNotSupported,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMallocPitch, ::UPTKFree, ::UPTKMallocArray, ::UPTKFreeArray,
 * ::UPTKMalloc3D, ::UPTKMalloc3DArray,
 * \ref ::UPTKMallocHost(void**, size_t) "UPTKMallocHost (C API)",
 * ::UPTKFreeHost, ::UPTKHostAlloc, ::UPTKDeviceGetAttribute, ::UPTKStreamAttachMemAsync,
 * ::cuMemAllocManaged
 */
#if defined(__cplusplus)
extern __host__ UPTKError_t UPTKMallocManaged(void **devPtr, size_t size, unsigned int flags = UPTKMemAttachGlobal);
#else
extern __host__ UPTKError_t UPTKMallocManaged(void **devPtr, size_t size, unsigned int flags);
#endif

/**
 * \brief Allocate memory on the device
 *
 * Allocates \p size bytes of linear memory on the device and returns in
 * \p *devPtr a pointer to the allocated memory. The allocated memory is
 * suitably aligned for any kind of variable. The memory is not cleared.
 * ::UPTKMalloc() returns ::UPTKErrorMemoryAllocation in case of failure.
 *
 * The device version of ::UPTKFree cannot be used with a \p *devPtr
 * allocated using the host API, and vice versa.
 *
 * \param devPtr - Pointer to allocated device memory
 * \param size   - Requested allocation size in bytes
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMallocPitch, ::UPTKFree, ::UPTKMallocArray, ::UPTKFreeArray,
 * ::UPTKMalloc3D, ::UPTKMalloc3DArray,
 * \ref ::UPTKMallocHost(void**, size_t) "UPTKMallocHost (C API)",
 * ::UPTKFreeHost, ::UPTKHostAlloc,
 * ::cuMemAlloc
 */
extern __host__ UPTKError_t UPTKMalloc(void **devPtr, size_t size);

/**
 * \brief Allocates page-locked memory on the host
 *
 * Allocates \p size bytes of host memory that is page-locked and accessible
 * to the device. The driver tracks the virtual memory ranges allocated with
 * this function and automatically accelerates calls to functions such as
 * ::UPTKMemcpy*(). Since the memory can be accessed directly by the device,
 * it can be read or written with much higher bandwidth than pageable memory
 * obtained with functions such as ::malloc(). 

 * On systems where ::pageableMemoryAccessUsesHostPageTables
 * is true, ::UPTKMallocHost may not page-lock the allocated memory.

 * Page-locking excessive amounts of memory with ::UPTKMallocHost() may degrade 
 * system performance, since it reduces the amount of memory available to the 
 * system for paging. As a result, this function is best used sparingly to allocate 
 * staging areas for data exchange between host and device.
 *
 * \param ptr  - Pointer to allocated host memory
 * \param size - Requested allocation size in bytes
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMalloc, ::UPTKMallocPitch, ::UPTKMallocArray, ::UPTKMalloc3D,
 * ::UPTKMalloc3DArray, ::UPTKHostAlloc, ::UPTKFree, ::UPTKFreeArray,
 * \ref ::UPTKMallocHost(void**, size_t, unsigned int) "UPTKMallocHost (C++ API)",
 * ::UPTKFreeHost, ::UPTKHostAlloc,
 * ::cuMemAllocHost
 */
extern __host__ UPTKError_t UPTKMallocHost(void **ptr, size_t size, unsigned int flags);

/**
 * \brief Allocates pitched memory on the device
 *
 * Allocates at least \p width (in bytes) * \p height bytes of linear memory
 * on the device and returns in \p *devPtr a pointer to the allocated memory.
 * The function may pad the allocation to ensure that corresponding pointers
 * in any given row will continue to meet the alignment requirements for
 * coalescing as the address is updated from row to row. The pitch returned in
 * \p *pitch by ::UPTKMallocPitch() is the width in bytes of the allocation.
 * The intended usage of \p pitch is as a separate parameter of the allocation,
 * used to compute addresses within the 2D array. Given the row and column of
 * an array element of type \p T, the address is computed as:
 * \code
    T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
   \endcode
 *
 * For allocations of 2D arrays, it is recommended that programmers consider
 * performing pitch allocations using ::UPTKMallocPitch(). Due to pitch
 * alignment restrictions in the hardware, this is especially true if the
 * application will be performing 2D memory copies between different regions
 * of device memory (whether linear memory or UPTK arrays).
 *
 * \param devPtr - Pointer to allocated pitched device memory
 * \param pitch  - Pitch for allocation
 * \param width  - Requested pitched allocation width (in bytes)
 * \param height - Requested pitched allocation height
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMalloc, ::UPTKFree, ::UPTKMallocArray, ::UPTKFreeArray,
 * \ref ::UPTKMallocHost(void**, size_t) "UPTKMallocHost (C API)",
 * ::UPTKFreeHost, ::UPTKMalloc3D, ::UPTKMalloc3DArray,
 * ::UPTKHostAlloc,
 * ::cuMemAllocPitch
 */
extern __host__ UPTKError_t UPTKMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);

/**
 * \brief Allocate an array on the device
 *
 * Allocates a UPTK array according to the ::UPTKChannelFormatDesc structure
 * \p desc and returns a handle to the new UPTK array in \p *array.
 *
 * The ::UPTKChannelFormatDesc is defined as:
 * \code
    struct UPTKChannelFormatDesc {
        int x, y, z, w;
    enum UPTKChannelFormatKind f;
    };
    \endcode
 * where ::UPTKChannelFormatKind is one of ::UPTKChannelFormatKindSigned,
 * ::UPTKChannelFormatKindUnsigned, or ::UPTKChannelFormatKindFloat.
 *
 * The \p flags parameter enables different options to be specified that affect
 * the allocation, as follows.
 * - ::UPTKArrayDefault: This flag's value is defined to be 0 and provides default array allocation
 * - ::UPTKArraySurfaceLoadStore: Allocates an array that can be read from or written to using a surface reference
 * - ::UPTKArrayTextureGather: This flag indicates that texture gather operations will be performed on the array.
 * - ::UPTKArraySparse: Allocates a UPTK array without physical backing memory. The subregions within this sparse array
 *   can later be mapped onto a physical memory allocation by calling ::cuMemMapArrayAsync. 
 *   The physical backing memory must be allocated via ::cuMemCreate.
 * - ::UPTKArrayDeferredMapping: Allocates a UPTK array without physical backing memory. The entire array can 
 *   later be mapped onto a physical memory allocation by calling ::cuMemMapArrayAsync. 
 *   The physical backing memory must be allocated via ::cuMemCreate.
 *
 * \p width and \p height must meet certain size requirements. See ::UPTKMalloc3DArray() for more details.
 *
 * \param array  - Pointer to allocated array in device memory
 * \param desc   - Requested channel format
 * \param width  - Requested array allocation width
 * \param height - Requested array allocation height
 * \param flags  - Requested properties of allocated array
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMalloc, ::UPTKMallocPitch, ::UPTKFree, ::UPTKFreeArray,
 * \ref ::UPTKMallocHost(void**, size_t) "UPTKMallocHost (C API)",
 * ::UPTKFreeHost, ::UPTKMalloc3D, ::UPTKMalloc3DArray,
 * ::UPTKHostAlloc,
 * ::cuArrayCreate
 */
extern __host__ UPTKError_t UPTKMallocArray(UPTKArray_t *array, const struct UPTKChannelFormatDesc *desc, size_t width, size_t height __dv(0), unsigned int flags __dv(0));

/**
 * \brief Frees memory on the device
 *
 * Frees the memory space pointed to by \p devPtr, which must have been
 * returned by a previous call to one of the following memory allocation APIs -
 * ::UPTKMalloc(), ::UPTKMallocPitch(), ::UPTKMallocManaged(), ::UPTKMallocAsync(),
 * ::UPTKMallocFromPoolAsync().
 * 
 * Note - This API will not perform any implicit synchronization when the pointer was
 * allocated with ::UPTKMallocAsync or ::UPTKMallocFromPoolAsync. Callers must ensure
 * that all accesses to these pointer have completed before invoking ::UPTKFree. For
 * best performance and memory reuse, users should use ::UPTKFreeAsync to free memory
 * allocated via the stream ordered memory allocator.
 * For all other pointers, this API may perform implicit synchronization.
 * 
 * If ::UPTKFree(\p devPtr) has already been called before,
 * an error is returned. If \p devPtr is 0, no operation is performed.
 * ::UPTKFree() returns ::UPTKErrorValue in case of failure.
 *
 * The device version of ::UPTKFree cannot be used with a \p *devPtr
 * allocated using the host API, and vice versa.
 *
 * \param devPtr - Device pointer to memory to free
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMalloc, ::UPTKMallocPitch, ::UPTKMallocManaged, ::UPTKMallocArray, ::UPTKFreeArray, ::UPTKMallocAsync, ::UPTKMallocFromPoolAsync
 * \ref ::UPTKMallocHost(void**, size_t) "UPTKMallocHost (C API)",
 * ::UPTKFreeHost, ::UPTKMalloc3D, ::UPTKMalloc3DArray, ::UPTKFreeAsync
 * ::UPTKHostAlloc,
 * ::cuMemFree
 */
extern __host__ UPTKError_t UPTKFree(void *devPtr);

/**
 * \brief Frees page-locked memory
 *
 * Frees the memory space pointed to by \p hostPtr, which must have been
 * returned by a previous call to ::UPTKMallocHost() or ::UPTKHostAlloc().
 *
 * \param ptr - Pointer to memory to free
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMalloc, ::UPTKMallocPitch, ::UPTKFree, ::UPTKMallocArray,
 * ::UPTKFreeArray,
 * \ref ::UPTKMallocHost(void**, size_t) "UPTKMallocHost (C API)",
 * ::UPTKMalloc3D, ::UPTKMalloc3DArray, ::UPTKHostAlloc,
 * ::cuMemFreeHost
 */
extern __host__ UPTKError_t UPTKFreeHost(void *ptr);

/**
 * \brief Frees an array on the device
 *
 * Frees the UPTK array \p array, which must have been returned by a
 * previous call to ::UPTKMallocArray(). If \p devPtr is 0,
 * no operation is performed.
 *
 * \param array - Pointer to array to free
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMalloc, ::UPTKMallocPitch, ::UPTKFree, ::UPTKMallocArray,
 * \ref ::UPTKMallocHost(void**, size_t) "UPTKMallocHost (C API)",
 * ::UPTKFreeHost, ::UPTKHostAlloc,
 * ::cuArrayDestroy
 */
extern __host__ UPTKError_t UPTKFreeArray(UPTKArray_t array);

/**
 * \brief Frees a mipmapped array on the device
 *
 * Frees the UPTK mipmapped array \p mipmappedArray, which must have been 
 * returned by a previous call to ::UPTKMallocMipmappedArray(). If \p devPtr
 * is 0, no operation is performed.
 *
 * \param mipmappedArray - Pointer to mipmapped array to free
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMalloc, ::UPTKMallocPitch, ::UPTKFree, ::UPTKMallocArray,
 * \ref ::UPTKMallocHost(void**, size_t) "UPTKMallocHost (C API)",
 * ::UPTKFreeHost, ::UPTKHostAlloc,
 * ::cuMipmappedArrayDestroy
 */
extern __host__ UPTKError_t UPTKFreeMipmappedArray(UPTKMipmappedArray_t mipmappedArray);


/**
 * \brief Allocates page-locked memory on the host
 *
 * Allocates \p size bytes of host memory that is page-locked and accessible
 * to the device. The driver tracks the virtual memory ranges allocated with
 * this function and automatically accelerates calls to functions such as
 * ::UPTKMemcpy(). Since the memory can be accessed directly by the device, it
 * can be read or written with much higher bandwidth than pageable memory
 * obtained with functions such as ::malloc(). Allocating excessive amounts of
 * pinned memory may degrade system performance, since it reduces the amount
 * of memory available to the system for paging. As a result, this function is
 * best used sparingly to allocate staging areas for data exchange between host
 * and device.
 *
 * The \p flags parameter enables different options to be specified that affect
 * the allocation, as follows.
 * - ::UPTKHostAllocDefault: This flag's value is defined to be 0 and causes
 * ::UPTKHostAlloc() to emulate ::UPTKMallocHost().
 * - ::UPTKHostAllocPortable: The memory returned by this call will be
 * considered as pinned memory by all UPTK contexts, not just the one that
 * performed the allocation.
 * - ::UPTKHostAllocMapped: Maps the allocation into the UPTK address space.
 * The device pointer to the memory may be obtained by calling
 * ::UPTKHostGetDevicePointer().
 * - ::UPTKHostAllocWriteCombined: Allocates the memory as write-combined (WC).
 * WC memory can be transferred across the PCI Express bus more quickly on some
 * system configurations, but cannot be read efficiently by most CPUs.  WC
 * memory is a good option for buffers that will be written by the CPU and read
 * by the device via mapped pinned memory or host->device transfers.
 *
 * All of these flags are orthogonal to one another: a developer may allocate
 * memory that is portable, mapped and/or write-combined with no restrictions.
 *
 * In order for the ::UPTKHostAllocMapped flag to have any effect, the UPTK context
 * must support the ::UPTKDeviceMapHost flag, which can be checked via
 * ::UPTKGetDeviceFlags(). The ::UPTKDeviceMapHost flag is implicitly set for
 * contexts created via the runtime API.
 *
 * The ::UPTKHostAllocMapped flag may be specified on UPTK contexts for devices
 * that do not support mapped pinned memory. The failure is deferred to
 * ::UPTKHostGetDevicePointer() because the memory may be mapped into other
 * UPTK contexts via the ::UPTKHostAllocPortable flag.
 *
 * Memory allocated by this function must be freed with ::UPTKFreeHost().
 *
 * \param pHost - Device pointer to allocated memory
 * \param size  - Requested allocation size in bytes
 * \param flags - Requested properties of allocated memory
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKSetDeviceFlags,
 * \ref ::UPTKMallocHost(void**, size_t) "UPTKMallocHost (C API)",
 * ::UPTKFreeHost,
 * ::UPTKGetDeviceFlags,
 * ::cuMemHostAlloc
 */
extern __host__ UPTKError_t UPTKHostAlloc(void **pHost, size_t size, unsigned int flags);

/**
 * \brief Registers an existing host memory range for use by UPTK
 *
 * Page-locks the memory range specified by \p ptr and \p size and maps it
 * for the device(s) as specified by \p flags. This memory range also is added
 * to the same tracking mechanism as ::UPTKHostAlloc() to automatically accelerate
 * calls to functions such as ::UPTKMemcpy(). Since the memory can be accessed 
 * directly by the device, it can be read or written with much higher bandwidth 
 * than pageable memory that has not been registered.  Page-locking excessive
 * amounts of memory may degrade system performance, since it reduces the amount
 * of memory available to the system for paging. As a result, this function is
 * best used sparingly to register staging areas for data exchange between
 * host and device.
 * 
 * On systems where ::pageableMemoryAccessUsesHostPageTables is true, ::UPTKHostRegister 
 * will not page-lock the memory range specified by \p ptr but only populate 
 * unpopulated pages.
 *
 * ::UPTKHostRegister is supported only on I/O coherent devices that have a non-zero
 * value for the device attribute ::UPTKDevAttrHostRegisterSupported.
 *
 * The \p flags parameter enables different options to be specified that
 * affect the allocation, as follows.
 *
 * - ::UPTKHostRegisterDefault: On a system with unified virtual addressing,
 *   the memory will be both mapped and portable.  On a system with no unified
 *   virtual addressing, the memory will be neither mapped nor portable.
 *
 * - ::UPTKHostRegisterPortable: The memory returned by this call will be
 *   considered as pinned memory by all UPTK contexts, not just the one that
 *   performed the allocation.
 *
 * - ::UPTKHostRegisterMapped: Maps the allocation into the UPTK address
 *   space. The device pointer to the memory may be obtained by calling
 *   ::UPTKHostGetDevicePointer().
 *
 * - ::UPTKHostRegisterIoMemory: The passed memory pointer is treated as
 *   pointing to some memory-mapped I/O space, e.g. belonging to a
 *   third-party PCIe device, and it will marked as non cache-coherent and
 *   contiguous.
 *
 * - ::UPTKHostRegisterReadOnly: The passed memory pointer is treated as
 *   pointing to memory that is considered read-only by the device.  On
 *   platforms without ::UPTKDevAttrPageableMemoryAccessUsesHostPageTables, this
 *   flag is required in order to register memory mapped to the CPU as
 *   read-only.  Support for the use of this flag can be queried from the device
 *   attribute UPTKDeviceAttrReadOnlyHostRegisterSupported.  Using this flag with
 *   a current context associated with a device that does not have this attribute
 *   set will cause ::UPTKHostRegister to error with UPTKErrorNotSupported.
 *
 * All of these flags are orthogonal to one another: a developer may page-lock
 * memory that is portable or mapped with no restrictions.
 *
 * The UPTK context must have been created with the ::UPTKMapHost flag in
 * order for the ::UPTKHostRegisterMapped flag to have any effect.
 *
 * The ::UPTKHostRegisterMapped flag may be specified on UPTK contexts for
 * devices that do not support mapped pinned memory. The failure is deferred
 * to ::UPTKHostGetDevicePointer() because the memory may be mapped into
 * other UPTK contexts via the ::UPTKHostRegisterPortable flag.
 *
 * For devices that have a non-zero value for the device attribute
 * ::UPTKDevAttrCanUseHostPointerForRegisteredMem, the memory
 * can also be accessed from the device using the host pointer \p ptr.
 * The device pointer returned by ::UPTKHostGetDevicePointer() may or may not
 * match the original host pointer \p ptr and depends on the devices visible to the
 * application. If all devices visible to the application have a non-zero value for the
 * device attribute, the device pointer returned by ::UPTKHostGetDevicePointer()
 * will match the original pointer \p ptr. If any device visible to the application
 * has a zero value for the device attribute, the device pointer returned by
 * ::UPTKHostGetDevicePointer() will not match the original host pointer \p ptr,
 * but it will be suitable for use on all devices provided Unified Virtual Addressing
 * is enabled. In such systems, it is valid to access the memory using either pointer
 * on devices that have a non-zero value for the device attribute. Note however that
 * such devices should access the memory using only of the two pointers and not both.
 *
 * The memory page-locked by this function must be unregistered with ::UPTKHostUnregister().
 *
 * \param ptr   - Host pointer to memory to page-lock
 * \param size  - Size in bytes of the address range to page-lock in bytes
 * \param flags - Flags for allocation request
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorMemoryAllocation,
 * ::UPTKErrorHostMemoryAlreadyRegistered,
 * ::UPTKErrorNotSupported
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKHostUnregister, ::UPTKHostGetFlags, ::UPTKHostGetDevicePointer,
 * ::cuMemHostRegister
 */
extern __host__ UPTKError_t UPTKHostRegister(void *ptr, size_t size, unsigned int flags);

/**
 * \brief Unregisters a memory range that was registered with UPTKHostRegister
 *
 * Unmaps the memory range whose base address is specified by \p ptr, and makes
 * it pageable again.
 *
 * The base address must be the same one specified to ::UPTKHostRegister().
 *
 * \param ptr - Host pointer to memory to unregister
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorHostMemoryNotRegistered
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKHostUnregister,
 * ::cuMemHostUnregister
 */
extern __host__ UPTKError_t UPTKHostUnregister(void *ptr);

/**
 * \brief Passes back device pointer of mapped host memory allocated by
 * UPTKHostAlloc or registered by UPTKHostRegister
 *
 * Passes back the device pointer corresponding to the mapped, pinned host
 * buffer allocated by ::UPTKHostAlloc() or registered by ::UPTKHostRegister().
 *
 * ::UPTKHostGetDevicePointer() will fail if the ::UPTKDeviceMapHost flag was
 * not specified before deferred context creation occurred, or if called on a
 * device that does not support mapped, pinned memory.
 *
 * For devices that have a non-zero value for the device attribute
 * ::UPTKDevAttrCanUseHostPointerForRegisteredMem, the memory
 * can also be accessed from the device using the host pointer \p pHost.
 * The device pointer returned by ::UPTKHostGetDevicePointer() may or may not
 * match the original host pointer \p pHost and depends on the devices visible to the
 * application. If all devices visible to the application have a non-zero value for the
 * device attribute, the device pointer returned by ::UPTKHostGetDevicePointer()
 * will match the original pointer \p pHost. If any device visible to the application
 * has a zero value for the device attribute, the device pointer returned by
 * ::UPTKHostGetDevicePointer() will not match the original host pointer \p pHost,
 * but it will be suitable for use on all devices provided Unified Virtual Addressing
 * is enabled. In such systems, it is valid to access the memory using either pointer
 * on devices that have a non-zero value for the device attribute. Note however that
 * such devices should access the memory using only of the two pointers and not both.
 *
 * \p flags provides for future releases.  For now, it must be set to 0.
 *
 * \param pDevice - Returned device pointer for mapped memory
 * \param pHost   - Requested host pointer mapping
 * \param flags   - Flags for extensions (must be 0 for now)
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKSetDeviceFlags, ::UPTKHostAlloc,
 * ::cuMemHostGetDevicePointer
 */
extern __host__ UPTKError_t UPTKHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);

/**
 * \brief Passes back flags used to allocate pinned host memory allocated by
 * UPTKHostAlloc
 *
 * ::UPTKHostGetFlags() will fail if the input pointer does not
 * reside in an address range allocated by ::UPTKHostAlloc().
 *
 * \param pFlags - Returned flags word
 * \param pHost - Host pointer
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKHostAlloc,
 * ::cuMemHostGetFlags
 */
extern __host__ UPTKError_t UPTKHostGetFlags(unsigned int *pFlags, void *pHost);

/**
 * \brief Allocates logical 1D, 2D, or 3D memory objects on the device
 *
 * Allocates at least \p width * \p height * \p depth bytes of linear memory
 * on the device and returns a ::UPTKPitchedPtr in which \p ptr is a pointer
 * to the allocated memory. The function may pad the allocation to ensure
 * hardware alignment requirements are met. The pitch returned in the \p pitch
 * field of \p pitchedDevPtr is the width in bytes of the allocation.
 *
 * The returned ::UPTKPitchedPtr contains additional fields \p xsize and
 * \p ysize, the logical width and height of the allocation, which are
 * equivalent to the \p width and \p height \p extent parameters provided by
 * the programmer during allocation.
 *
 * For allocations of 2D and 3D objects, it is highly recommended that
 * programmers perform allocations using ::UPTKMalloc3D() or
 * ::UPTKMallocPitch(). Due to alignment restrictions in the hardware, this is
 * especially true if the application will be performing memory copies
 * involving 2D or 3D objects (whether linear memory or UPTK arrays).
 *
 * \param pitchedDevPtr  - Pointer to allocated pitched device memory
 * \param extent         - Requested allocation size (\p width field in bytes)
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMallocPitch, ::UPTKFree, ::UPTKMemcpy3D, ::UPTKMemset3D,
 * ::UPTKMalloc3DArray, ::UPTKMallocArray, ::UPTKFreeArray,
 * \ref ::UPTKMallocHost(void**, size_t) "UPTKMallocHost (C API)",
 * ::UPTKFreeHost, ::UPTKHostAlloc, ::make_UPTKPitchedPtr, ::make_UPTKExtent,
 * ::cuMemAllocPitch
 */
extern __host__ UPTKError_t UPTKMalloc3D(struct UPTKPitchedPtr* pitchedDevPtr, struct UPTKExtent extent);

/**
 * \brief Allocate an array on the device
 *
 * Allocates a UPTK array according to the ::UPTKChannelFormatDesc structure
 * \p desc and returns a handle to the new UPTK array in \p *array.
 *
 * The ::UPTKChannelFormatDesc is defined as:
 * \code
    struct UPTKChannelFormatDesc {
        int x, y, z, w;
        enum UPTKChannelFormatKind f;
    };
    \endcode
 * where ::UPTKChannelFormatKind is one of ::UPTKChannelFormatKindSigned,
 * ::UPTKChannelFormatKindUnsigned, or ::UPTKChannelFormatKindFloat.
 *
 * ::UPTKMalloc3DArray() can allocate the following:
 *
 * - A 1D array is allocated if the height and depth extents are both zero.
 * - A 2D array is allocated if only the depth extent is zero.
 * - A 3D array is allocated if all three extents are non-zero.
 * - A 1D layered UPTK array is allocated if only the height extent is zero and
 * the UPTKArrayLayered flag is set. Each layer is a 1D array. The number of layers is 
 * determined by the depth extent.
 * - A 2D layered UPTK array is allocated if all three extents are non-zero and 
 * the UPTKArrayLayered flag is set. Each layer is a 2D array. The number of layers is 
 * determined by the depth extent.
 * - A cubemap UPTK array is allocated if all three extents are non-zero and the
 * UPTKArrayCubemap flag is set. Width must be equal to height, and depth must be six. A cubemap is
 * a special type of 2D layered UPTK array, where the six layers represent the six faces of a cube. 
 * The order of the six layers in memory is the same as that listed in ::UPTKGraphicsCubeFace.
 * - A cubemap layered UPTK array is allocated if all three extents are non-zero, and both,
 * UPTKArrayCubemap and UPTKArrayLayered flags are set. Width must be equal to height, and depth must be 
 * a multiple of six. A cubemap layered UPTK array is a special type of 2D layered UPTK array that consists 
 * of a collection of cubemaps. The first six layers represent the first cubemap, the next six layers form 
 * the second cubemap, and so on.
 *
 *
 * The \p flags parameter enables different options to be specified that affect
 * the allocation, as follows.
 * - ::UPTKArrayDefault: This flag's value is defined to be 0 and provides default array allocation
 * - ::UPTKArrayLayered: Allocates a layered UPTK array, with the depth extent indicating the number of layers
 * - ::UPTKArrayCubemap: Allocates a cubemap UPTK array. Width must be equal to height, and depth must be six.
 *   If the UPTKArrayLayered flag is also set, depth must be a multiple of six.
 * - ::UPTKArraySurfaceLoadStore: Allocates a UPTK array that could be read from or written to using a surface
 *   reference.
 * - ::UPTKArrayTextureGather: This flag indicates that texture gather operations will be performed on the UPTK 
 *   array. Texture gather can only be performed on 2D UPTK arrays.
 * - ::UPTKArraySparse: Allocates a UPTK array without physical backing memory. The subregions within this sparse array 
 *   can later be mapped onto a physical memory allocation by calling ::cuMemMapArrayAsync. This flag can only be used for 
 *   creating 2D, 3D or 2D layered sparse UPTK arrays. The physical backing memory must be allocated via ::cuMemCreate.
 * - ::UPTKArrayDeferredMapping: Allocates a UPTK array without physical backing memory. The entire array can
 *   later be mapped onto a physical memory allocation by calling ::cuMemMapArrayAsync. The physical backing memory must be allocated
 *   via ::cuMemCreate.
 *
 * The width, height and depth extents must meet certain size requirements as listed in the following table.
 * All values are specified in elements.
 *
 * Note that 2D UPTK arrays have different size requirements if the ::UPTKArrayTextureGather flag is set. In that
 * case, the valid range for (width, height, depth) is ((1,maxTexture2DGather[0]), (1,maxTexture2DGather[1]), 0).
 *
 * \xmlonly
 * <table outputclass="xmlonly">
 * <tgroup cols="3" colsep="1" rowsep="1">
 * <colspec colname="c1" colwidth="1.0*"/>
 * <colspec colname="c2" colwidth="3.0*"/>
 * <colspec colname="c3" colwidth="3.0*"/>
 * <thead>
 * <row>
 * <entry>UPTK array type</entry>
 * <entry>Valid extents that must always be met {(width range in elements),
 * (height range), (depth range)}</entry>
 * <entry>Valid extents with UPTKArraySurfaceLoadStore set {(width range in
 * elements), (height range), (depth range)}</entry>
 * </row>
 * </thead>
 * <tbody>
 * <row>
 * <entry>1D</entry>
 * <entry>{ (1,maxTexture1D), 0, 0 }</entry>
 * <entry>{ (1,maxSurface1D), 0, 0 }</entry>
 * </row>
 * <row>
 * <entry>2D</entry>
 * <entry>{ (1,maxTexture2D[0]), (1,maxTexture2D[1]), 0 }</entry>
 * <entry>{ (1,maxSurface2D[0]), (1,maxSurface2D[1]), 0 }</entry>
 * </row>
 * <row>
 * <entry>3D</entry>
 * <entry>{ (1,maxTexture3D[0]), (1,maxTexture3D[1]), (1,maxTexture3D[2]) }
 * OR { (1,maxTexture3DAlt[0]), (1,maxTexture3DAlt[1]),
 * (1,maxTexture3DAlt[2]) }</entry>
 * <entry>{ (1,maxSurface3D[0]), (1,maxSurface3D[1]), (1,maxSurface3D[2]) }</entry>
 * </row>
 * <row>
 * <entry>1D Layered</entry>
 * <entry>{ (1,maxTexture1DLayered[0]), 0, (1,maxTexture1DLayered[1]) }</entry>
 * <entry>{ (1,maxSurface1DLayered[0]), 0, (1,maxSurface1DLayered[1]) }</entry>
 * </row>
 * <row>
 * <entry>2D Layered</entry>
 * <entry>{ (1,maxTexture2DLayered[0]), (1,maxTexture2DLayered[1]),
 * (1,maxTexture2DLayered[2]) }</entry>
 * <entry>{ (1,maxSurface2DLayered[0]), (1,maxSurface2DLayered[1]),
 * (1,maxSurface2DLayered[2]) }</entry>
 * </row>
 * <row>
 * <entry>Cubemap</entry>
 * <entry>{ (1,maxTextureCubemap), (1,maxTextureCubemap), 6 }</entry>
 * <entry>{ (1,maxSurfaceCubemap), (1,maxSurfaceCubemap), 6 }</entry>
 * </row>
 * <row>
 * <entry>Cubemap Layered</entry>
 * <entry>{ (1,maxTextureCubemapLayered[0]), (1,maxTextureCubemapLayered[0]),
 * (1,maxTextureCubemapLayered[1]) }</entry>
 * <entry>{ (1,maxSurfaceCubemapLayered[0]), (1,maxSurfaceCubemapLayered[0]),
 * (1,maxSurfaceCubemapLayered[1]) }</entry>
 * </row>
 * </tbody>
 * </tgroup>
 * </table>
 * \endxmlonly
 *
 * \param array  - Pointer to allocated array in device memory
 * \param desc   - Requested channel format
 * \param extent - Requested allocation size (\p width field in elements)
 * \param flags  - Flags for extensions
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMalloc3D, ::UPTKMalloc, ::UPTKMallocPitch, ::UPTKFree,
 * ::UPTKFreeArray,
 * \ref ::UPTKMallocHost(void**, size_t) "UPTKMallocHost (C API)",
 * ::UPTKFreeHost, ::UPTKHostAlloc,
 * ::make_UPTKExtent,
 * ::cuArray3DCreate
 */
extern __host__ UPTKError_t UPTKMalloc3DArray(UPTKArray_t *array, const struct UPTKChannelFormatDesc* desc, struct UPTKExtent extent, unsigned int flags __dv(0));

/**
 * \brief Allocate a mipmapped array on the device
 *
 * Allocates a UPTK mipmapped array according to the ::UPTKChannelFormatDesc structure
 * \p desc and returns a handle to the new UPTK mipmapped array in \p *mipmappedArray.
 * \p numLevels specifies the number of mipmap levels to be allocated. This value is
 * clamped to the range [1, 1 + floor(log2(max(width, height, depth)))].
 *
 * The ::UPTKChannelFormatDesc is defined as:
 * \code
    struct UPTKChannelFormatDesc {
        int x, y, z, w;
        enum UPTKChannelFormatKind f;
    };
    \endcode
 * where ::UPTKChannelFormatKind is one of ::UPTKChannelFormatKindSigned,
 * ::UPTKChannelFormatKindUnsigned, or ::UPTKChannelFormatKindFloat.
 *
 * ::UPTKMallocMipmappedArray() can allocate the following:
 *
 * - A 1D mipmapped array is allocated if the height and depth extents are both zero.
 * - A 2D mipmapped array is allocated if only the depth extent is zero.
 * - A 3D mipmapped array is allocated if all three extents are non-zero.
 * - A 1D layered UPTK mipmapped array is allocated if only the height extent is zero and
 * the UPTKArrayLayered flag is set. Each layer is a 1D mipmapped array. The number of layers is 
 * determined by the depth extent.
 * - A 2D layered UPTK mipmapped array is allocated if all three extents are non-zero and 
 * the UPTKArrayLayered flag is set. Each layer is a 2D mipmapped array. The number of layers is 
 * determined by the depth extent.
 * - A cubemap UPTK mipmapped array is allocated if all three extents are non-zero and the
 * UPTKArrayCubemap flag is set. Width must be equal to height, and depth must be six.
 * The order of the six layers in memory is the same as that listed in ::UPTKGraphicsCubeFace.
 * - A cubemap layered UPTK mipmapped array is allocated if all three extents are non-zero, and both,
 * UPTKArrayCubemap and UPTKArrayLayered flags are set. Width must be equal to height, and depth must be 
 * a multiple of six. A cubemap layered UPTK mipmapped array is a special type of 2D layered UPTK mipmapped
 * array that consists of a collection of cubemap mipmapped arrays. The first six layers represent the 
 * first cubemap mipmapped array, the next six layers form the second cubemap mipmapped array, and so on.
 *
 *
 * The \p flags parameter enables different options to be specified that affect
 * the allocation, as follows.
 * - ::UPTKArrayDefault: This flag's value is defined to be 0 and provides default mipmapped array allocation
 * - ::UPTKArrayLayered: Allocates a layered UPTK mipmapped array, with the depth extent indicating the number of layers
 * - ::UPTKArrayCubemap: Allocates a cubemap UPTK mipmapped array. Width must be equal to height, and depth must be six.
 *   If the UPTKArrayLayered flag is also set, depth must be a multiple of six.
 * - ::UPTKArraySurfaceLoadStore: This flag indicates that individual mipmap levels of the UPTK mipmapped array 
 *   will be read from or written to using a surface reference.
 * - ::UPTKArrayTextureGather: This flag indicates that texture gather operations will be performed on the UPTK 
 *   array. Texture gather can only be performed on 2D UPTK mipmapped arrays, and the gather operations are
 *   performed only on the most detailed mipmap level.
 * - ::UPTKArraySparse: Allocates a UPTK mipmapped array without physical backing memory. The subregions within this sparse array
 *   can later be mapped onto a physical memory allocation by calling ::cuMemMapArrayAsync. This flag can only be used for creating 
 *   2D, 3D or 2D layered sparse UPTK mipmapped arrays. The physical backing memory must be allocated via ::cuMemCreate.
 * - ::UPTKArrayDeferredMapping: Allocates a UPTK mipmapped array without physical backing memory. The entire array can
 *   later be mapped onto a physical memory allocation by calling ::cuMemMapArrayAsync. The physical backing memory must be allocated
 *   via ::cuMemCreate.
 *
 * The width, height and depth extents must meet certain size requirements as listed in the following table.
 * All values are specified in elements.
 *
 * \xmlonly
 * <table outputclass="xmlonly">
 * <tgroup cols="3" colsep="1" rowsep="1">
 * <colspec colname="c1" colwidth="1.0*"/>
 * <colspec colname="c2" colwidth="3.0*"/>
 * <colspec colname="c3" colwidth="3.0*"/>
 * <thead>
 * <row>
 * <entry>UPTK array type</entry>
 * <entry>Valid extents that must always be met {(width range in elements),
 * (height range), (depth range)}</entry>
 * <entry>Valid extents with UPTKArraySurfaceLoadStore set {(width range in
 * elements), (height range), (depth range)}</entry>
 * </row>
 * </thead>
 * <tbody>
 * <row>
 * <entry>1D</entry>
 * <entry>{ (1,maxTexture1DMipmap), 0, 0 }</entry>
 * <entry>{ (1,maxSurface1D), 0, 0 }</entry>
 * </row>
 * <row>
 * <entry>2D</entry>
 * <entry>{ (1,maxTexture2DMipmap[0]), (1,maxTexture2DMipmap[1]), 0 }</entry>
 * <entry>{ (1,maxSurface2D[0]), (1,maxSurface2D[1]), 0 }</entry>
 * </row>
 * <row>
 * <entry>3D</entry>
 * <entry>{ (1,maxTexture3D[0]), (1,maxTexture3D[1]), (1,maxTexture3D[2]) }
 * OR { (1,maxTexture3DAlt[0]), (1,maxTexture3DAlt[1]),
 * (1,maxTexture3DAlt[2]) }</entry>
 * <entry>{ (1,maxSurface3D[0]), (1,maxSurface3D[1]), (1,maxSurface3D[2]) }</entry>
 * </row>
 * <row>
 * <entry>1D Layered</entry>
 * <entry>{ (1,maxTexture1DLayered[0]), 0, (1,maxTexture1DLayered[1]) }</entry>
 * <entry>{ (1,maxSurface1DLayered[0]), 0, (1,maxSurface1DLayered[1]) }</entry>
 * </row>
 * <row>
 * <entry>2D Layered</entry>
 * <entry>{ (1,maxTexture2DLayered[0]), (1,maxTexture2DLayered[1]),
 * (1,maxTexture2DLayered[2]) }</entry>
 * <entry>{ (1,maxSurface2DLayered[0]), (1,maxSurface2DLayered[1]),
 * (1,maxSurface2DLayered[2]) }</entry>
 * </row>
 * <row>
 * <entry>Cubemap</entry>
 * <entry>{ (1,maxTextureCubemap), (1,maxTextureCubemap), 6 }</entry>
 * <entry>{ (1,maxSurfaceCubemap), (1,maxSurfaceCubemap), 6 }</entry>
 * </row>
 * <row>
 * <entry>Cubemap Layered</entry>
 * <entry>{ (1,maxTextureCubemapLayered[0]), (1,maxTextureCubemapLayered[0]),
 * (1,maxTextureCubemapLayered[1]) }</entry>
 * <entry>{ (1,maxSurfaceCubemapLayered[0]), (1,maxSurfaceCubemapLayered[0]),
 * (1,maxSurfaceCubemapLayered[1]) }</entry>
 * </row>
 * </tbody>
 * </tgroup>
 * </table>
 * \endxmlonly
 *
 * \param mipmappedArray  - Pointer to allocated mipmapped array in device memory
 * \param desc            - Requested channel format
 * \param extent          - Requested allocation size (\p width field in elements)
 * \param numLevels       - Number of mipmap levels to allocate
 * \param flags           - Flags for extensions
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMalloc3D, ::UPTKMalloc, ::UPTKMallocPitch, ::UPTKFree,
 * ::UPTKFreeArray,
 * \ref ::UPTKMallocHost(void**, size_t) "UPTKMallocHost (C API)",
 * ::UPTKFreeHost, ::UPTKHostAlloc,
 * ::make_UPTKExtent,
 * ::cuMipmappedArrayCreate
 */
extern __host__ UPTKError_t UPTKMallocMipmappedArray(UPTKMipmappedArray_t *mipmappedArray, const struct UPTKChannelFormatDesc* desc, struct UPTKExtent extent, unsigned int numLevels, unsigned int flags __dv(0));

/**
 * \brief Gets a mipmap level of a UPTK mipmapped array
 *
 * Returns in \p *levelArray a UPTK array that represents a single mipmap level
 * of the UPTK mipmapped array \p mipmappedArray.
 *
 * If \p level is greater than the maximum number of levels in this mipmapped array,
 * ::UPTKErrorInvalidValue is returned.
 *
 * If \p mipmappedArray is NULL,
 * ::UPTKErrorInvalidResourceHandle is returned.
 *
 * \param levelArray     - Returned mipmap level UPTK array
 * \param mipmappedArray - UPTK mipmapped array
 * \param level          - Mipmap level
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * ::UPTKErrorInvalidResourceHandle
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMalloc3D, ::UPTKMalloc, ::UPTKMallocPitch, ::UPTKFree,
 * ::UPTKFreeArray,
 * \ref ::UPTKMallocHost(void**, size_t) "UPTKMallocHost (C API)",
 * ::UPTKFreeHost, ::UPTKHostAlloc,
 * ::make_UPTKExtent,
 * ::cuMipmappedArrayGetLevel
 */
extern __host__ UPTKError_t UPTKGetMipmappedArrayLevel(UPTKArray_t *levelArray, UPTKMipmappedArray_const_t mipmappedArray, unsigned int level);

/**
 * \brief Copies data between 3D objects
 *
\code
struct UPTKExtent {
  size_t width;
  size_t height;
  size_t depth;
};
struct UPTKExtent make_UPTKExtent(size_t w, size_t h, size_t d);

struct UPTKPos {
  size_t x;
  size_t y;
  size_t z;
};
struct UPTKPos make_UPTKPos(size_t x, size_t y, size_t z);

struct UPTKMemcpy3DParms {
  UPTKArray_t           srcArray;
  struct UPTKPos        srcPos;
  struct UPTKPitchedPtr srcPtr;
  UPTKArray_t           dstArray;
  struct UPTKPos        dstPos;
  struct UPTKPitchedPtr dstPtr;
  struct UPTKExtent     extent;
  enum UPTKMemcpyKind   kind;
};
\endcode
 *
 * ::UPTKMemcpy3D() copies data betwen two 3D objects. The source and
 * destination objects may be in either host memory, device memory, or a UPTK
 * array. The source, destination, extent, and kind of copy performed is
 * specified by the ::UPTKMemcpy3DParms struct which should be initialized to
 * zero before use:
\code
UPTKMemcpy3DParms myParms = {0};
\endcode
 *
 * The struct passed to ::UPTKMemcpy3D() must specify one of \p srcArray or
 * \p srcPtr and one of \p dstArray or \p dstPtr. Passing more than one
 * non-zero source or destination will cause ::UPTKMemcpy3D() to return an
 * error.
 *
 * The \p srcPos and \p dstPos fields are optional offsets into the source and
 * destination objects and are defined in units of each object's elements. The
 * element for a host or device pointer is assumed to be <b>unsigned char</b>.
 *
 * The \p extent field defines the dimensions of the transferred area in
 * elements. If a UPTK array is participating in the copy, the extent is
 * defined in terms of that array's elements. If no UPTK array is
 * participating in the copy then the extents are defined in elements of
 * <b>unsigned char</b>.
 *
 * The \p kind field defines the direction of the copy. It must be one of
 * ::UPTKMemcpyHostToHost, ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToHost,
 * ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault. Passing
 * ::UPTKMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * For ::UPTKMemcpyHostToHost or ::UPTKMemcpyHostToDevice or ::UPTKMemcpyDeviceToHost
 * passed as kind and UPTKArray type passed as source or destination, if the kind
 * implies UPTKArray type to be present on the host, ::UPTKMemcpy3D() will
 * disregard that implication and silently correct the kind based on the fact that
 * UPTKArray type can only be present on the device.
 *
 * If the source and destination are both arrays, ::UPTKMemcpy3D() will return
 * an error if they do not have the same element size.
 *
 * The source and destination object may not overlap. If overlapping source
 * and destination objects are specified, undefined behavior will result.
 *
 * The source object must entirely contain the region defined by \p srcPos
 * and \p extent. The destination object must entirely contain the region
 * defined by \p dstPos and \p extent.
 *
 * ::UPTKMemcpy3D() returns an error if the pitch of \p srcPtr or \p dstPtr
 * exceeds the maximum allowed. The pitch of a ::UPTKPitchedPtr allocated
 * with ::UPTKMalloc3D() will always be valid.
 *
 * \param p - 3D memory copy parameters
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidPitchValue,
 * ::UPTKErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_sync
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMalloc3D, ::UPTKMalloc3DArray, ::UPTKMemset3D, ::UPTKMemcpy3DAsync,
 * ::UPTKMemcpy, ::UPTKMemcpy2D,
 * ::UPTKMemcpy2DToArray, ::UPTKMemcpy2DFromArray,
 * ::UPTKMemcpy2DArrayToArray, ::UPTKMemcpyToSymbol,
 * ::UPTKMemcpyFromSymbol, ::UPTKMemcpyAsync, ::UPTKMemcpy2DAsync,
 * ::UPTKMemcpy2DToArrayAsync,
 * ::UPTKMemcpy2DFromArrayAsync,
 * ::UPTKMemcpyToSymbolAsync, ::UPTKMemcpyFromSymbolAsync,
 * ::make_UPTKExtent, ::make_UPTKPos,
 * ::cuMemcpy3D
 */
extern __host__ UPTKError_t UPTKMemcpy3D(const struct UPTKMemcpy3DParms *p);

/**
 * \brief Copies memory between devices
 *
 * Perform a 3D memory copy according to the parameters specified in
 * \p p.  See the definition of the ::UPTKMemcpy3DPeerParms structure
 * for documentation of its parameters.
 *
 * Note that this function is synchronous with respect to the host only if
 * the source or destination of the transfer is host memory.  Note also 
 * that this copy is serialized with respect to all pending and future 
 * asynchronous work in to the current device, the copy's source device,
 * and the copy's destination device (use ::UPTKMemcpy3DPeerAsync to avoid 
 * this synchronization).
 *
 * \param p - Parameters for the memory copy
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidDevice,
 * ::UPTKErrorInvalidPitchValue
 * \notefnerr
 * \note_sync
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpyPeer, ::UPTKMemcpyAsync, ::UPTKMemcpyPeerAsync,
 * ::UPTKMemcpy3DPeerAsync,
 * ::cuMemcpy3DPeer
 */
extern __host__ UPTKError_t UPTKMemcpy3DPeer(const struct UPTKMemcpy3DPeerParms *p);

/**
 * \brief Copies data between 3D objects
 *
\code
struct UPTKExtent {
  size_t width;
  size_t height;
  size_t depth;
};
struct UPTKExtent make_UPTKExtent(size_t w, size_t h, size_t d);

struct UPTKPos {
  size_t x;
  size_t y;
  size_t z;
};
struct UPTKPos make_UPTKPos(size_t x, size_t y, size_t z);

struct UPTKMemcpy3DParms {
  UPTKArray_t           srcArray;
  struct UPTKPos        srcPos;
  struct UPTKPitchedPtr srcPtr;
  UPTKArray_t           dstArray;
  struct UPTKPos        dstPos;
  struct UPTKPitchedPtr dstPtr;
  struct UPTKExtent     extent;
  enum UPTKMemcpyKind   kind;
};
\endcode
 *
 * ::UPTKMemcpy3DAsync() copies data betwen two 3D objects. The source and
 * destination objects may be in either host memory, device memory, or a UPTK
 * array. The source, destination, extent, and kind of copy performed is
 * specified by the ::UPTKMemcpy3DParms struct which should be initialized to
 * zero before use:
\code
UPTKMemcpy3DParms myParms = {0};
\endcode
 *
 * The struct passed to ::UPTKMemcpy3DAsync() must specify one of \p srcArray
 * or \p srcPtr and one of \p dstArray or \p dstPtr. Passing more than one
 * non-zero source or destination will cause ::UPTKMemcpy3DAsync() to return an
 * error.
 *
 * The \p srcPos and \p dstPos fields are optional offsets into the source and
 * destination objects and are defined in units of each object's elements. The
 * element for a host or device pointer is assumed to be <b>unsigned char</b>.
 * For UPTK arrays, positions must be in the range [0, 2048) for any
 * dimension.
 *
 * The \p extent field defines the dimensions of the transferred area in
 * elements. If a UPTK array is participating in the copy, the extent is
 * defined in terms of that array's elements. If no UPTK array is
 * participating in the copy then the extents are defined in elements of
 * <b>unsigned char</b>.
 *
 * The \p kind field defines the direction of the copy. It must be one of
 * ::UPTKMemcpyHostToHost, ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToHost,
 * ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault. Passing
 * ::UPTKMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * For ::UPTKMemcpyHostToHost or ::UPTKMemcpyHostToDevice or ::UPTKMemcpyDeviceToHost
 * passed as kind and UPTKArray type passed as source or destination, if the kind
 * implies UPTKArray type to be present on the host, ::UPTKMemcpy3DAsync() will
 * disregard that implication and silently correct the kind based on the fact that
 * UPTKArray type can only be present on the device.
 *
 * If the source and destination are both arrays, ::UPTKMemcpy3DAsync() will
 * return an error if they do not have the same element size.
 *
 * The source and destination object may not overlap. If overlapping source
 * and destination objects are specified, undefined behavior will result.
 *
 * The source object must lie entirely within the region defined by \p srcPos
 * and \p extent. The destination object must lie entirely within the region
 * defined by \p dstPos and \p extent.
 *
 * ::UPTKMemcpy3DAsync() returns an error if the pitch of \p srcPtr or
 * \p dstPtr exceeds the maximum allowed. The pitch of a
 * ::UPTKPitchedPtr allocated with ::UPTKMalloc3D() will always be valid.
 *
 * ::UPTKMemcpy3DAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If
 * \p kind is ::UPTKMemcpyHostToDevice or ::UPTKMemcpyDeviceToHost and \p stream
 * is non-zero, the copy may overlap with operations in other streams.
 *
 * The device version of this function only handles device to device copies and
 * cannot be given local or shared pointers.
 *
 * \param p      - 3D memory copy parameters
 * \param stream - Stream identifier
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidPitchValue,
 * ::UPTKErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMalloc3D, ::UPTKMalloc3DArray, ::UPTKMemset3D, ::UPTKMemcpy3D,
 * ::UPTKMemcpy, ::UPTKMemcpy2D,
 * ::UPTKMemcpy2DToArray, :::UPTKMemcpy2DFromArray,
 * ::UPTKMemcpy2DArrayToArray, ::UPTKMemcpyToSymbol,
 * ::UPTKMemcpyFromSymbol, ::UPTKMemcpyAsync, ::UPTKMemcpy2DAsync,
 * ::UPTKMemcpy2DToArrayAsync,
 * ::UPTKMemcpy2DFromArrayAsync,
 * ::UPTKMemcpyToSymbolAsync, ::UPTKMemcpyFromSymbolAsync,
 * ::make_UPTKExtent, ::make_UPTKPos,
 * ::cuMemcpy3DAsync
 */
extern __host__ UPTKError_t UPTKMemcpy3DAsync(const struct UPTKMemcpy3DParms *p, UPTKStream_t stream __dv(0));

/**
 * \brief Copies memory between devices asynchronously.
 *
 * Perform a 3D memory copy according to the parameters specified in
 * \p p.  See the definition of the ::UPTKMemcpy3DPeerParms structure
 * for documentation of its parameters.
 *
 * \param p      - Parameters for the memory copy
 * \param stream - Stream identifier
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidDevice,
 * ::UPTKErrorInvalidPitchValue
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpyPeer, ::UPTKMemcpyAsync, ::UPTKMemcpyPeerAsync,
 * ::UPTKMemcpy3DPeerAsync,
 * ::cuMemcpy3DPeerAsync
 */
extern __host__ UPTKError_t UPTKMemcpy3DPeerAsync(const struct UPTKMemcpy3DPeerParms *p, UPTKStream_t stream __dv(0));

/**
 * \brief Gets free and total device memory
 *
 * Returns in \p *total the total amount of memory available to the the current context.
 * Returns in \p *free the amount of memory on the device that is free according to the OS.
 * UPTK is not guaranteed to be able to allocate all of the memory that the OS reports as free.
 * In a multi-tenet situation, free estimate returned is prone to race condition where
 * a new allocation/free done by a different process or a different thread in the same
 * process between the time when free memory was estimated and reported, will result in
 * deviation in free value reported and actual free memory.
 *
 * The integrated GPU on Tegra shares memory with CPU and other component
 * of the SoC. The free and total values returned by the API excludes
 * the SWAP memory space maintained by the OS on some platforms.
 * The OS may move some of the memory pages into swap area as the GPU or
 * CPU allocate or access memory. See Tegra app note on how to calculate
 * total and free memory on Tegra.
 *
 * \param free  - Returned free memory in bytes
 * \param total - Returned total memory in bytes
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorLaunchFailure
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cuMemGetInfo
 */
extern __host__ UPTKError_t UPTKMemGetInfo(size_t *free, size_t *total);

/**
 * \brief Gets info about the specified UPTKArray
 * 
 * Returns in \p *desc, \p *extent and \p *flags respectively, the type, shape 
 * and flags of \p array.
 *
 * Any of \p *desc, \p *extent and \p *flags may be specified as NULL.
 *
 * \param desc   - Returned array type
 * \param extent - Returned array shape. 2D arrays will have depth of zero
 * \param flags  - Returned array flags
 * \param array  - The ::UPTKArray to get info for
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cuArrayGetDescriptor,
 * ::cuArray3DGetDescriptor
 */
extern __host__ UPTKError_t UPTKArrayGetInfo(struct UPTKChannelFormatDesc *desc, struct UPTKExtent *extent, unsigned int *flags, UPTKArray_t array);

/**
 * \brief Gets a UPTK array plane from a UPTK array
 *
 * Returns in \p pPlaneArray a UPTK array that represents a single format plane
 * of the UPTK array \p hArray.
 *
 * If \p planeIdx is greater than the maximum number of planes in this array or if the array does
 * not have a multi-planar format e.g: ::UPTKChannelFormatKindNV12, then ::UPTKErrorInvalidValue is returned.
 *
 * Note that if the \p hArray has format ::UPTKChannelFormatKindNV12, then passing in 0 for \p planeIdx returns
 * a UPTK array of the same size as \p hArray but with one 8-bit channel and ::UPTKChannelFormatKindUnsigned as its format kind.
 * If 1 is passed for \p planeIdx, then the returned UPTK array has half the height and width
 * of \p hArray with two 8-bit channels and ::UPTKChannelFormatKindUnsigned as its format kind.
 *
 * \param pPlaneArray   - Returned UPTK array referenced by the \p planeIdx
 * \param hArray        - UPTK array
 * \param planeIdx      - Plane index
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * ::UPTKErrorInvalidResourceHandle
 * \notefnerr
 *
 * \sa
 * ::cuArrayGetPlane
 */
extern __host__ UPTKError_t UPTKArrayGetPlane(UPTKArray_t *pPlaneArray, UPTKArray_t hArray, unsigned int planeIdx);

/**
 * \brief Returns the memory requirements of a UPTK array
 *
 * Returns the memory requirements of a UPTK array in \p memoryRequirements
 * If the UPTK array is not allocated with flag ::UPTKArrayDeferredMapping
 * ::UPTKErrorInvalidValue will be returned.
 *
 * The returned value in ::UPTKArrayMemoryRequirements::size
 * represents the total size of the UPTK array.
 * The returned value in ::UPTKArrayMemoryRequirements::alignment
 * represents the alignment necessary for mapping the UPTK array.
 *
 * \return
 * ::UPTKSuccess
 * ::UPTKErrorInvalidValue
 *
 * \param[out] memoryRequirements - Pointer to ::UPTKArrayMemoryRequirements
 * \param[in] array - UPTK array to get the memory requirements of
 * \param[in] device - Device to get the memory requirements for
 * \sa ::UPTKMipmappedArrayGetMemoryRequirements
 */
extern __host__ UPTKError_t UPTKArrayGetMemoryRequirements(struct UPTKArrayMemoryRequirements  *memoryRequirements, UPTKArray_t array, int device);

/**
 * \brief Returns the memory requirements of a UPTK mipmapped array
 *
 * Returns the memory requirements of a UPTK mipmapped array in \p memoryRequirements
 * If the UPTK mipmapped array is not allocated with flag ::UPTKArrayDeferredMapping
 * ::UPTKErrorInvalidValue will be returned.
 *
 * The returned value in ::UPTKArrayMemoryRequirements::size
 * represents the total size of the UPTK mipmapped array.
 * The returned value in ::UPTKArrayMemoryRequirements::alignment
 * represents the alignment necessary for mapping the UPTK mipmapped
 * array.
 *
 * \return
 * ::UPTKSuccess
 * ::UPTKErrorInvalidValue
 *
 * \param[out] memoryRequirements - Pointer to ::UPTKArrayMemoryRequirements
 * \param[in] mipmap - UPTK mipmapped array to get the memory requirements of
 * \param[in] device - Device to get the memory requirements for
 * \sa ::UPTKArrayGetMemoryRequirements
 */
extern __host__ UPTKError_t UPTKMipmappedArrayGetMemoryRequirements(struct UPTKArrayMemoryRequirements *memoryRequirements, UPTKMipmappedArray_t mipmap, int device);

/**
 * \brief Returns the layout properties of a sparse UPTK array
 *
 * Returns the layout properties of a sparse UPTK array in \p sparseProperties.
 * If the UPTK array is not allocated with flag ::UPTKArraySparse
 * ::UPTKErrorInvalidValue will be returned.
 *
 * If the returned value in ::UPTKArraySparseProperties::flags contains ::UPTKArraySparsePropertiesSingleMipTail,
 * then ::UPTKArraySparseProperties::miptailSize represents the total size of the array. Otherwise, it will be zero.
 * Also, the returned value in ::UPTKArraySparseProperties::miptailFirstLevel is always zero.
 * Note that the \p array must have been allocated using ::UPTKMallocArray or ::UPTKMalloc3DArray. For UPTK arrays obtained
 * using ::UPTKMipmappedArrayGetLevel, ::UPTKErrorInvalidValue will be returned. Instead, ::UPTKMipmappedArrayGetSparseProperties
 * must be used to obtain the sparse properties of the entire UPTK mipmapped array to which \p array belongs to.
 *
 * \return
 * ::UPTKSuccess
 * ::UPTKErrorInvalidValue
 *
 * \param[out] sparseProperties - Pointer to return the ::UPTKArraySparseProperties
 * \param[in] array             - The UPTK array to get the sparse properties of 
 *
 * \sa
 * ::UPTKMipmappedArrayGetSparseProperties,
 * ::cuMemMapArrayAsync
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKArrayGetSparseProperties(struct UPTKArraySparseProperties *sparseProperties, UPTKArray_t array);
#endif

/**
 * \brief Returns the layout properties of a sparse UPTK mipmapped array
 *
 * Returns the sparse array layout properties in \p sparseProperties.
 * If the UPTK mipmapped array is not allocated with flag ::UPTKArraySparse
 * ::UPTKErrorInvalidValue will be returned.
 *
 * For non-layered UPTK mipmapped arrays, ::UPTKArraySparseProperties::miptailSize returns the
 * size of the mip tail region. The mip tail region includes all mip levels whose width, height or depth
 * is less than that of the tile.
 * For layered UPTK mipmapped arrays, if ::UPTKArraySparseProperties::flags contains ::UPTKArraySparsePropertiesSingleMipTail,
 * then ::UPTKArraySparseProperties::miptailSize specifies the size of the mip tail of all layers combined.
 * Otherwise, ::UPTKArraySparseProperties::miptailSize specifies mip tail size per layer.
 * The returned value of ::UPTKArraySparseProperties::miptailFirstLevel is valid only if ::UPTKArraySparseProperties::miptailSize is non-zero.
 *
 * \return
 * ::UPTKSuccess
 * ::UPTKErrorInvalidValue
 *
 * \param[out] sparseProperties - Pointer to return ::UPTKArraySparseProperties
 * \param[in] mipmap            - The UPTK mipmapped array to get the sparse properties of
 *
 * \sa
 * ::UPTKArrayGetSparseProperties,
 * ::cuMemMapArrayAsync
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKMipmappedArrayGetSparseProperties(struct UPTKArraySparseProperties *sparseProperties, UPTKMipmappedArray_t mipmap);
#endif

/**
 * \brief Copies data between host and device
 *
 * Copies \p count bytes from the memory area pointed to by \p src to the
 * memory area pointed to by \p dst, where \p kind specifies the direction
 * of the copy, and must be one of ::UPTKMemcpyHostToHost,
 * ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToHost,
 * ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault. Passing
 * ::UPTKMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing. Calling
 * ::UPTKMemcpy() with dst and src pointers that do not match the direction of
 * the copy results in an undefined behavior.
 *
 * \param dst   - Destination memory address
 * \param src   - Source memory address
 * \param count - Size in bytes to copy
 * \param kind  - Type of transfer
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \note_sync
 * \note_memcpy
 *
 * \sa ::UPTKMemcpy2D,
 * ::UPTKMemcpy2DToArray, ::UPTKMemcpy2DFromArray,
 * ::UPTKMemcpy2DArrayToArray, ::UPTKMemcpyToSymbol,
 * ::UPTKMemcpyFromSymbol, ::UPTKMemcpyAsync, ::UPTKMemcpy2DAsync,
 * ::UPTKMemcpy2DToArrayAsync,
 * ::UPTKMemcpy2DFromArrayAsync,
 * ::UPTKMemcpyToSymbolAsync, ::UPTKMemcpyFromSymbolAsync,
 * ::cuMemcpyDtoH,
 * ::cuMemcpyHtoD,
 * ::cuMemcpyDtoD,
 * ::cuMemcpy
 */
extern __host__ UPTKError_t UPTKMemcpy(void *dst, const void *src, size_t count, enum UPTKMemcpyKind kind);

/**
 * \brief Copies memory between two devices
 *
 * Copies memory from one device to memory on another device.  \p dst is the 
 * base device pointer of the destination memory and \p dstDevice is the 
 * destination device.  \p src is the base device pointer of the source memory 
 * and \p srcDevice is the source device.  \p count specifies the number of bytes 
 * to copy.
 *
 * Note that this function is asynchronous with respect to the host, but 
 * serialized with respect all pending and future asynchronous work in to the 
 * current device, \p srcDevice, and \p dstDevice (use ::UPTKMemcpyPeerAsync 
 * to avoid this synchronization).
 *
 * \param dst       - Destination device pointer
 * \param dstDevice - Destination device
 * \param src       - Source device pointer
 * \param srcDevice - Source device
 * \param count     - Size of memory copy in bytes
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidDevice
 * \notefnerr
 * \note_sync
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpyAsync, ::UPTKMemcpyPeerAsync,
 * ::UPTKMemcpy3DPeerAsync,
 * ::cuMemcpyPeer
 */
extern __host__ UPTKError_t UPTKMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count);

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the memory
 * area pointed to by \p src to the memory area pointed to by \p dst, where
 * \p kind specifies the direction of the copy, and must be one of
 * ::UPTKMemcpyHostToHost, ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToHost,
 * ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault. Passing
 * ::UPTKMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing. \p dpitch and
 * \p spitch are the widths in memory in bytes of the 2D arrays pointed to by
 * \p dst and \p src, including any padding added to the end of each row. The
 * memory areas may not overlap. \p width must not exceed either \p dpitch or
 * \p spitch. Calling ::UPTKMemcpy2D() with \p dst and \p src pointers that do
 * not match the direction of the copy results in an undefined behavior.
 * ::UPTKMemcpy2D() returns an error if \p dpitch or \p spitch exceeds
 * the maximum allowed.
 *
 * \param dst    - Destination memory address
 * \param dpitch - Pitch of destination memory
 * \param src    - Source memory address
 * \param spitch - Pitch of source memory
 * \param width  - Width of matrix transfer (columns in bytes)
 * \param height - Height of matrix transfer (rows)
 * \param kind   - Type of transfer
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidPitchValue,
 * ::UPTKErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_memcpy
 *
 * \sa ::UPTKMemcpy,
 * ::UPTKMemcpy2DToArray, ::UPTKMemcpy2DFromArray,
 * ::UPTKMemcpy2DArrayToArray, ::UPTKMemcpyToSymbol,
 * ::UPTKMemcpyFromSymbol, ::UPTKMemcpyAsync, ::UPTKMemcpy2DAsync,
 * ::UPTKMemcpy2DToArrayAsync,
 * ::UPTKMemcpy2DFromArrayAsync,
 * ::UPTKMemcpyToSymbolAsync, ::UPTKMemcpyFromSymbolAsync,
 * ::cuMemcpy2D,
 * ::cuMemcpy2DUnaligned
 */
extern __host__ UPTKError_t UPTKMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum UPTKMemcpyKind kind);

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the memory
 * area pointed to by \p src to the UPTK array \p dst starting at
 * \p hOffset rows and \p wOffset bytes from the upper left corner,
 * where \p kind specifies the direction of the copy, and must be one
 * of ::UPTKMemcpyHostToHost, ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToHost,
 * ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault. Passing
 * ::UPTKMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * \p spitch is the width in memory in bytes of the 2D array pointed to by
 * \p src, including any padding added to the end of each row. \p wOffset +
 * \p width must not exceed the width of the UPTK array \p dst. \p width must
 * not exceed \p spitch. ::UPTKMemcpy2DToArray() returns an error if \p spitch
 * exceeds the maximum allowed.
 *
 * \param dst     - Destination memory address
 * \param wOffset - Destination starting X offset (columns in bytes)
 * \param hOffset - Destination starting Y offset (rows)
 * \param src     - Source memory address
 * \param spitch  - Pitch of source memory
 * \param width   - Width of matrix transfer (columns in bytes)
 * \param height  - Height of matrix transfer (rows)
 * \param kind    - Type of transfer
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidPitchValue,
 * ::UPTKErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_sync
 * \note_init_rt
 * \note_callback
 * \note_memcpy
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpy2D,
 * ::UPTKMemcpy2DFromArray,
 * ::UPTKMemcpy2DArrayToArray, ::UPTKMemcpyToSymbol,
 * ::UPTKMemcpyFromSymbol, ::UPTKMemcpyAsync, ::UPTKMemcpy2DAsync,
 * ::UPTKMemcpy2DToArrayAsync,
 * ::UPTKMemcpy2DFromArrayAsync,
 * ::UPTKMemcpyToSymbolAsync, ::UPTKMemcpyFromSymbolAsync,
 * ::cuMemcpy2D,
 * ::cuMemcpy2DUnaligned
 */
extern __host__ UPTKError_t UPTKMemcpy2DToArray(UPTKArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum UPTKMemcpyKind kind);

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the UPTK
 * array \p src starting at \p hOffset rows and \p wOffset bytes from the
 * upper left corner to the memory area pointed to by \p dst, where
 * \p kind specifies the direction of the copy, and must be one of
 * ::UPTKMemcpyHostToHost, ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToHost,
 * ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault. Passing
 * ::UPTKMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing. \p dpitch is the
 * width in memory in bytes of the 2D array pointed to by \p dst, including any
 * padding added to the end of each row. \p wOffset + \p width must not exceed
 * the width of the UPTK array \p src. \p width must not exceed \p dpitch.
 * ::UPTKMemcpy2DFromArray() returns an error if \p dpitch exceeds the maximum
 * allowed.
 *
 * \param dst     - Destination memory address
 * \param dpitch  - Pitch of destination memory
 * \param src     - Source memory address
 * \param wOffset - Source starting X offset (columns in bytes)
 * \param hOffset - Source starting Y offset (rows)
 * \param width   - Width of matrix transfer (columns in bytes)
 * \param height  - Height of matrix transfer (rows)
 * \param kind    - Type of transfer
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidPitchValue,
 * ::UPTKErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_sync
 * \note_init_rt
 * \note_callback
 * \note_memcpy
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpy2D,
 * ::UPTKMemcpy2DToArray,
 * ::UPTKMemcpy2DArrayToArray, ::UPTKMemcpyToSymbol,
 * ::UPTKMemcpyFromSymbol, ::UPTKMemcpyAsync, ::UPTKMemcpy2DAsync,
 * ::UPTKMemcpy2DToArrayAsync,
 * ::UPTKMemcpy2DFromArrayAsync,
 * ::UPTKMemcpyToSymbolAsync, ::UPTKMemcpyFromSymbolAsync,
 * ::cuMemcpy2D,
 * ::cuMemcpy2DUnaligned
 */
extern __host__ UPTKError_t UPTKMemcpy2DFromArray(void *dst, size_t dpitch, UPTKArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum UPTKMemcpyKind kind);

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the UPTK
 * array \p src starting at \p hOffsetSrc rows and \p wOffsetSrc bytes from the
 * upper left corner to the UPTK array \p dst starting at \p hOffsetDst rows
 * and \p wOffsetDst bytes from the upper left corner, where \p kind
 * specifies the direction of the copy, and must be one of
 * ::UPTKMemcpyHostToHost, ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToHost,
 * ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault. Passing
 * ::UPTKMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * \p wOffsetDst + \p width must not exceed the width of the UPTK array \p dst.
 * \p wOffsetSrc + \p width must not exceed the width of the UPTK array \p src.
 *
 * \param dst        - Destination memory address
 * \param wOffsetDst - Destination starting X offset (columns in bytes)
 * \param hOffsetDst - Destination starting Y offset (rows)
 * \param src        - Source memory address
 * \param wOffsetSrc - Source starting X offset (columns in bytes)
 * \param hOffsetSrc - Source starting Y offset (rows)
 * \param width      - Width of matrix transfer (columns in bytes)
 * \param height     - Height of matrix transfer (rows)
 * \param kind       - Type of transfer
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_sync
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpy2D,
 * ::UPTKMemcpy2DToArray, ::UPTKMemcpy2DFromArray,
 * ::UPTKMemcpyToSymbol,
 * ::UPTKMemcpyFromSymbol, ::UPTKMemcpyAsync, ::UPTKMemcpy2DAsync,
 * ::UPTKMemcpy2DToArrayAsync,
 * ::UPTKMemcpy2DFromArrayAsync,
 * ::UPTKMemcpyToSymbolAsync, ::UPTKMemcpyFromSymbolAsync,
 * ::cuMemcpy2D,
 * ::cuMemcpy2DUnaligned
 */
extern __host__ UPTKError_t UPTKMemcpy2DArrayToArray(UPTKArray_t dst, size_t wOffsetDst, size_t hOffsetDst, UPTKArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum UPTKMemcpyKind kind __dv(UPTKMemcpyDeviceToDevice));

/**
 * \brief Copies data to the given symbol on the device
 *
 * Copies \p count bytes from the memory area pointed to by \p src
 * to the memory area pointed to by \p offset bytes from the start of symbol
 * \p symbol. The memory areas may not overlap. \p symbol is a variable that
 * resides in global or constant memory space. \p kind can be either
 * ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault.
 * Passing ::UPTKMemcpyDefault is recommended, in which case the type of
 * transfer is inferred from the pointer values. However, ::UPTKMemcpyDefault
 * is only allowed on systems that support unified virtual addressing.
 *
 * \param symbol - Device symbol address
 * \param src    - Source memory address
 * \param count  - Size in bytes to copy
 * \param offset - Offset from start of symbol in bytes
 * \param kind   - Type of transfer
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidSymbol,
 * ::UPTKErrorInvalidMemcpyDirection,
 * ::UPTKErrorNoKernelImageForDevice
 * \notefnerr
 * \note_sync
 * \note_string_api_deprecation
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpy2D,
 * ::UPTKMemcpy2DToArray,  ::UPTKMemcpy2DFromArray,
 * ::UPTKMemcpy2DArrayToArray,
 * ::UPTKMemcpyFromSymbol, ::UPTKMemcpyAsync, ::UPTKMemcpy2DAsync,
 * ::UPTKMemcpy2DToArrayAsync,
 * ::UPTKMemcpy2DFromArrayAsync,
 * ::UPTKMemcpyToSymbolAsync, ::UPTKMemcpyFromSymbolAsync,
 * ::cuMemcpy,
 * ::cuMemcpyHtoD,
 * ::cuMemcpyDtoD
 */
extern __host__ UPTKError_t UPTKMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset __dv(0), enum UPTKMemcpyKind kind __dv(UPTKMemcpyHostToDevice));


/**
 * \brief Copies data from the given symbol on the device
 *
 * Copies \p count bytes from the memory area pointed to by \p offset bytes
 * from the start of symbol \p symbol to the memory area pointed to by \p dst.
 * The memory areas may not overlap. \p symbol is a variable that
 * resides in global or constant memory space. \p kind can be either
 * ::UPTKMemcpyDeviceToHost, ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault.
 * Passing ::UPTKMemcpyDefault is recommended, in which case the type of
 * transfer is inferred from the pointer values. However, ::UPTKMemcpyDefault
 * is only allowed on systems that support unified virtual addressing.
 *
 * \param dst    - Destination memory address
 * \param symbol - Device symbol address
 * \param count  - Size in bytes to copy
 * \param offset - Offset from start of symbol in bytes
 * \param kind   - Type of transfer
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidSymbol,
 * ::UPTKErrorInvalidMemcpyDirection,
 * ::UPTKErrorNoKernelImageForDevice
 * \notefnerr
 * \note_sync
 * \note_string_api_deprecation
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpy2D,
 * ::UPTKMemcpy2DToArray, ::UPTKMemcpy2DFromArray,
 * ::UPTKMemcpy2DArrayToArray, ::UPTKMemcpyToSymbol,
 * ::UPTKMemcpyAsync, ::UPTKMemcpy2DAsync,
 * ::UPTKMemcpy2DToArrayAsync,
 * ::UPTKMemcpy2DFromArrayAsync,
 * ::UPTKMemcpyToSymbolAsync, ::UPTKMemcpyFromSymbolAsync,
 * ::cuMemcpy,
 * ::cuMemcpyDtoH,
 * ::cuMemcpyDtoD
 */
extern __host__ UPTKError_t UPTKMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset __dv(0), enum UPTKMemcpyKind kind __dv(UPTKMemcpyDeviceToHost));


/**
 * \brief Copies data between host and device
 *
 * Copies \p count bytes from the memory area pointed to by \p src to the
 * memory area pointed to by \p dst, where \p kind specifies the
 * direction of the copy, and must be one of ::UPTKMemcpyHostToHost,
 * ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToHost,
 * ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault. Passing
 * ::UPTKMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * 
 * The memory areas may not overlap. Calling ::UPTKMemcpyAsync() with \p dst and
 * \p src pointers that do not match the direction of the copy results in an
 * undefined behavior.
 *
 * ::UPTKMemcpyAsync() is asynchronous with respect to the host, so the call
 * may return before the copy is complete. The copy can optionally be
 * associated to a stream by passing a non-zero \p stream argument. If \p kind
 * is ::UPTKMemcpyHostToDevice or ::UPTKMemcpyDeviceToHost and the \p stream is
 * non-zero, the copy may overlap with operations in other streams.
 *
 * The device version of this function only handles device to device copies and
 * cannot be given local or shared pointers.
 *
 * \param dst    - Destination memory address
 * \param src    - Source memory address
 * \param count  - Size in bytes to copy
 * \param kind   - Type of transfer
 * \param stream - Stream identifier
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 * \note_memcpy
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpy2D,
 * ::UPTKMemcpy2DToArray, ::UPTKMemcpy2DFromArray,
 * ::UPTKMemcpy2DArrayToArray, ::UPTKMemcpyToSymbol,
 * ::UPTKMemcpyFromSymbol, ::UPTKMemcpy2DAsync,
 * ::UPTKMemcpy2DToArrayAsync,
 * ::UPTKMemcpy2DFromArrayAsync,
 * ::UPTKMemcpyToSymbolAsync, ::UPTKMemcpyFromSymbolAsync,
 * ::cuMemcpyAsync,
 * ::cuMemcpyDtoHAsync,
 * ::cuMemcpyHtoDAsync,
 * ::cuMemcpyDtoDAsync
 */
extern __host__ UPTKError_t UPTKMemcpyAsync(void *dst, const void *src, size_t count, enum UPTKMemcpyKind kind, UPTKStream_t stream __dv(0));

/**
 * \brief Copies memory between two devices asynchronously.
 *
 * Copies memory from one device to memory on another device.  \p dst is the 
 * base device pointer of the destination memory and \p dstDevice is the 
 * destination device.  \p src is the base device pointer of the source memory 
 * and \p srcDevice is the source device.  \p count specifies the number of bytes 
 * to copy.
 *
 * Note that this function is asynchronous with respect to the host and all work
 * on other devices.
 *
 * \param dst       - Destination device pointer
 * \param dstDevice - Destination device
 * \param src       - Source device pointer
 * \param srcDevice - Source device
 * \param count     - Size of memory copy in bytes
 * \param stream    - Stream identifier
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidDevice
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpyPeer, ::UPTKMemcpyAsync,
 * ::UPTKMemcpy3DPeerAsync,
 * ::cuMemcpyPeerAsync
 */
extern __host__ UPTKError_t UPTKMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, UPTKStream_t stream __dv(0));

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the memory
 * area pointed to by \p src to the memory area pointed to by \p dst, where
 * \p kind specifies the direction of the copy, and must be one of
 * ::UPTKMemcpyHostToHost, ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToHost,
 * ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault. Passing
 * ::UPTKMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * \p dpitch and \p spitch are the widths in memory in bytes of the 2D arrays
 * pointed to by \p dst and \p src, including any padding added to the end of
 * each row. The memory areas may not overlap. \p width must not exceed either
 * \p dpitch or \p spitch.
 *
 * Calling ::UPTKMemcpy2DAsync() with \p dst and \p src pointers that do not
 * match the direction of the copy results in an undefined behavior.
 * ::UPTKMemcpy2DAsync() returns an error if \p dpitch or \p spitch is greater
 * than the maximum allowed.
 *
 * ::UPTKMemcpy2DAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If
 * \p kind is ::UPTKMemcpyHostToDevice or ::UPTKMemcpyDeviceToHost and
 * \p stream is non-zero, the copy may overlap with operations in other
 * streams.
 *
 * The device version of this function only handles device to device copies and
 * cannot be given local or shared pointers.
 *
 * \param dst    - Destination memory address
 * \param dpitch - Pitch of destination memory
 * \param src    - Source memory address
 * \param spitch - Pitch of source memory
 * \param width  - Width of matrix transfer (columns in bytes)
 * \param height - Height of matrix transfer (rows)
 * \param kind   - Type of transfer
 * \param stream - Stream identifier
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidPitchValue,
 * ::UPTKErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 * \note_memcpy
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpy2D,
 * ::UPTKMemcpy2DToArray, ::UPTKMemcpy2DFromArray,
 * ::UPTKMemcpy2DArrayToArray, ::UPTKMemcpyToSymbol,
 * ::UPTKMemcpyFromSymbol, ::UPTKMemcpyAsync,
 * ::UPTKMemcpy2DToArrayAsync,
 * ::UPTKMemcpy2DFromArrayAsync,
 * ::UPTKMemcpyToSymbolAsync, ::UPTKMemcpyFromSymbolAsync,
 * ::cuMemcpy2DAsync
 */
extern __host__ UPTKError_t UPTKMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum UPTKMemcpyKind kind, UPTKStream_t stream __dv(0));

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the memory
 * area pointed to by \p src to the UPTK array \p dst starting at \p hOffset
 * rows and \p wOffset bytes from the upper left corner, where \p kind specifies
 * the direction of the copy, and must be one of ::UPTKMemcpyHostToHost,
 * ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToHost,
 * ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault. Passing
 * ::UPTKMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * \p spitch is the width in memory in bytes of the 2D array pointed to by
 * \p src, including any padding added to the end of each row. \p wOffset +
 * \p width must not exceed the width of the UPTK array \p dst. \p width must
 * not exceed \p spitch. ::UPTKMemcpy2DToArrayAsync() returns an error if
 * \p spitch exceeds the maximum allowed.
 *
 * ::UPTKMemcpy2DToArrayAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If
 * \p kind is ::UPTKMemcpyHostToDevice or ::UPTKMemcpyDeviceToHost and
 * \p stream is non-zero, the copy may overlap with operations in other
 * streams.
 *
 * \param dst     - Destination memory address
 * \param wOffset - Destination starting X offset (columns in bytes)
 * \param hOffset - Destination starting Y offset (rows)
 * \param src     - Source memory address
 * \param spitch  - Pitch of source memory
 * \param width   - Width of matrix transfer (columns in bytes)
 * \param height  - Height of matrix transfer (rows)
 * \param kind    - Type of transfer
 * \param stream  - Stream identifier
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidPitchValue,
 * ::UPTKErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 * \note_memcpy
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpy2D,
 * ::UPTKMemcpy2DToArray, ::UPTKMemcpy2DFromArray,
 * ::UPTKMemcpy2DArrayToArray, ::UPTKMemcpyToSymbol,
 * ::UPTKMemcpyFromSymbol, ::UPTKMemcpyAsync, ::UPTKMemcpy2DAsync,
 *
 * ::UPTKMemcpy2DFromArrayAsync,
 * ::UPTKMemcpyToSymbolAsync, ::UPTKMemcpyFromSymbolAsync,
 * ::cuMemcpy2DAsync
 */
extern __host__ UPTKError_t UPTKMemcpy2DToArrayAsync(UPTKArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum UPTKMemcpyKind kind, UPTKStream_t stream __dv(0));

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the UPTK
 * array \p src starting at \p hOffset rows and \p wOffset bytes from the
 * upper left corner to the memory area pointed to by \p dst,
 * where \p kind specifies the direction of the copy, and must be one of
 * ::UPTKMemcpyHostToHost, ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToHost,
 * ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault. Passing
 * ::UPTKMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * \p dpitch is the width in memory in bytes of the 2D
 * array pointed to by \p dst, including any padding added to the end of each
 * row. \p wOffset + \p width must not exceed the width of the UPTK array
 * \p src. \p width must not exceed \p dpitch. ::UPTKMemcpy2DFromArrayAsync()
 * returns an error if \p dpitch exceeds the maximum allowed.
 *
 * ::UPTKMemcpy2DFromArrayAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally be
 * associated to a stream by passing a non-zero \p stream argument. If \p kind
 * is ::UPTKMemcpyHostToDevice or ::UPTKMemcpyDeviceToHost and \p stream is
 * non-zero, the copy may overlap with operations in other streams.
 *
 * \param dst     - Destination memory address
 * \param dpitch  - Pitch of destination memory
 * \param src     - Source memory address
 * \param wOffset - Source starting X offset (columns in bytes)
 * \param hOffset - Source starting Y offset (rows)
 * \param width   - Width of matrix transfer (columns in bytes)
 * \param height  - Height of matrix transfer (rows)
 * \param kind    - Type of transfer
 * \param stream  - Stream identifier
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidPitchValue,
 * ::UPTKErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 * \note_memcpy
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpy2D,
 * ::UPTKMemcpy2DToArray, ::UPTKMemcpy2DFromArray,
 * ::UPTKMemcpy2DArrayToArray, ::UPTKMemcpyToSymbol,
 * ::UPTKMemcpyFromSymbol, ::UPTKMemcpyAsync, ::UPTKMemcpy2DAsync,
 * ::UPTKMemcpy2DToArrayAsync,
 *
 * ::UPTKMemcpyToSymbolAsync, ::UPTKMemcpyFromSymbolAsync,
 * ::cuMemcpy2DAsync
 */
extern __host__ UPTKError_t UPTKMemcpy2DFromArrayAsync(void *dst, size_t dpitch, UPTKArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum UPTKMemcpyKind kind, UPTKStream_t stream __dv(0));

/**
 * \brief Copies data to the given symbol on the device
 *
 * Copies \p count bytes from the memory area pointed to by \p src
 * to the memory area pointed to by \p offset bytes from the start of symbol
 * \p symbol. The memory areas may not overlap. \p symbol is a variable that
 * resides in global or constant memory space. \p kind can be either
 * ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault.
 * Passing ::UPTKMemcpyDefault is recommended, in which case the type of transfer
 * is inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * ::UPTKMemcpyToSymbolAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If
 * \p kind is ::UPTKMemcpyHostToDevice and \p stream is non-zero, the copy
 * may overlap with operations in other streams.
 *
 * \param symbol - Device symbol address
 * \param src    - Source memory address
 * \param count  - Size in bytes to copy
 * \param offset - Offset from start of symbol in bytes
 * \param kind   - Type of transfer
 * \param stream - Stream identifier
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidSymbol,
 * ::UPTKErrorInvalidMemcpyDirection,
 * ::UPTKErrorNoKernelImageForDevice
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_string_api_deprecation
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpy2D,
 * ::UPTKMemcpy2DToArray, ::UPTKMemcpy2DFromArray,
 * ::UPTKMemcpy2DArrayToArray, ::UPTKMemcpyToSymbol,
 * ::UPTKMemcpyFromSymbol, ::UPTKMemcpyAsync, ::UPTKMemcpy2DAsync,
 * ::UPTKMemcpy2DToArrayAsync,
 * ::UPTKMemcpy2DFromArrayAsync,
 * ::UPTKMemcpyFromSymbolAsync,
 * ::cuMemcpyAsync,
 * ::cuMemcpyHtoDAsync,
 * ::cuMemcpyDtoDAsync
 */
extern __host__ UPTKError_t UPTKMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset, enum UPTKMemcpyKind kind, UPTKStream_t stream __dv(0));

/**
 * \brief Copies data from the given symbol on the device
 *
 * Copies \p count bytes from the memory area pointed to by \p offset bytes
 * from the start of symbol \p symbol to the memory area pointed to by \p dst.
 * The memory areas may not overlap. \p symbol is a variable that resides in
 * global or constant memory space. \p kind can be either
 * ::UPTKMemcpyDeviceToHost, ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault.
 * Passing ::UPTKMemcpyDefault is recommended, in which case the type of transfer
 * is inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * ::UPTKMemcpyFromSymbolAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally be
 * associated to a stream by passing a non-zero \p stream argument. If \p kind
 * is ::UPTKMemcpyDeviceToHost and \p stream is non-zero, the copy may overlap
 * with operations in other streams.
 *
 * \param dst    - Destination memory address
 * \param symbol - Device symbol address
 * \param count  - Size in bytes to copy
 * \param offset - Offset from start of symbol in bytes
 * \param kind   - Type of transfer
 * \param stream - Stream identifier
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidSymbol,
 * ::UPTKErrorInvalidMemcpyDirection,
 * ::UPTKErrorNoKernelImageForDevice
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_string_api_deprecation
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpy2D,
 * ::UPTKMemcpy2DToArray, ::UPTKMemcpy2DFromArray,
 * ::UPTKMemcpy2DArrayToArray, ::UPTKMemcpyToSymbol,
 * ::UPTKMemcpyFromSymbol, ::UPTKMemcpyAsync, ::UPTKMemcpy2DAsync,
 * ::UPTKMemcpy2DToArrayAsync,
 * ::UPTKMemcpy2DFromArrayAsync,
 * ::UPTKMemcpyToSymbolAsync,
 * ::cuMemcpyAsync,
 * ::cuMemcpyDtoHAsync,
 * ::cuMemcpyDtoDAsync
 */
extern __host__ UPTKError_t UPTKMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset, enum UPTKMemcpyKind kind, UPTKStream_t stream __dv(0));


/**
 * \brief Initializes or sets device memory to a value
 *
 * Fills the first \p count bytes of the memory area pointed to by \p devPtr
 * with the constant byte value \p value.
 *
 * Note that this function is asynchronous with respect to the host unless
 * \p devPtr refers to pinned host memory.
 *
 * \param devPtr - Pointer to device memory
 * \param value  - Value to set for each byte of specified memory
 * \param count  - Size in bytes to set
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * \notefnerr
 * \note_memset
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cuMemsetD8,
 * ::cuMemsetD16,
 * ::cuMemsetD32
 */
extern __host__ UPTKError_t UPTKMemset(void *devPtr, int value, size_t count);

/**
 * \brief Initializes or sets device memory to a value
 *
 * Sets to the specified value \p value a matrix (\p height rows of \p width
 * bytes each) pointed to by \p dstPtr. \p pitch is the width in bytes of the
 * 2D array pointed to by \p dstPtr, including any padding added to the end
 * of each row. This function performs fastest when the pitch is one that has
 * been passed back by ::UPTKMallocPitch().
 *
 * Note that this function is asynchronous with respect to the host unless
 * \p devPtr refers to pinned host memory.
 *
 * \param devPtr - Pointer to 2D device memory
 * \param pitch  - Pitch in bytes of 2D device memory(Unused if \p height is 1)
 * \param value  - Value to set for each byte of specified memory
 * \param width  - Width of matrix set (columns in bytes)
 * \param height - Height of matrix set (rows)
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * \notefnerr
 * \note_memset
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemset, ::UPTKMemset3D, ::UPTKMemsetAsync,
 * ::UPTKMemset2DAsync, ::UPTKMemset3DAsync,
 * ::cuMemsetD2D8,
 * ::cuMemsetD2D16,
 * ::cuMemsetD2D32
 */
extern __host__ UPTKError_t UPTKMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height);

/**
 * \brief Initializes or sets device memory to a value
 *
 * Initializes each element of a 3D array to the specified value \p value.
 * The object to initialize is defined by \p pitchedDevPtr. The \p pitch field
 * of \p pitchedDevPtr is the width in memory in bytes of the 3D array pointed
 * to by \p pitchedDevPtr, including any padding added to the end of each row.
 * The \p xsize field specifies the logical width of each row in bytes, while
 * the \p ysize field specifies the height of each 2D slice in rows.
 * The \p pitch field of \p pitchedDevPtr is ignored when \p height and \p depth 
 * are both equal to 1. 
 *
 * The extents of the initialized region are specified as a \p width in bytes,
 * a \p height in rows, and a \p depth in slices.
 *
 * Extents with \p width greater than or equal to the \p xsize of
 * \p pitchedDevPtr may perform significantly faster than extents narrower
 * than the \p xsize. Secondarily, extents with \p height equal to the
 * \p ysize of \p pitchedDevPtr will perform faster than when the \p height is
 * shorter than the \p ysize.
 *
 * This function performs fastest when the \p pitchedDevPtr has been allocated
 * by ::UPTKMalloc3D().
 *
 * Note that this function is asynchronous with respect to the host unless
 * \p pitchedDevPtr refers to pinned host memory.
 *
 * \param pitchedDevPtr - Pointer to pitched device memory
 * \param value         - Value to set for each byte of specified memory
 * \param extent        - Size parameters for where to set device memory (\p width field in bytes)
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * \notefnerr
 * \note_memset
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemset, ::UPTKMemset2D,
 * ::UPTKMemsetAsync, ::UPTKMemset2DAsync, ::UPTKMemset3DAsync,
 * ::UPTKMalloc3D, ::make_UPTKPitchedPtr,
 * ::make_UPTKExtent
 */
extern __host__ UPTKError_t UPTKMemset3D(struct UPTKPitchedPtr pitchedDevPtr, int value, struct UPTKExtent extent);

/**
 * \brief Initializes or sets device memory to a value
 *
 * Fills the first \p count bytes of the memory area pointed to by \p devPtr
 * with the constant byte value \p value.
 *
 * ::UPTKMemsetAsync() is asynchronous with respect to the host, so
 * the call may return before the memset is complete. The operation can optionally
 * be associated to a stream by passing a non-zero \p stream argument.
 * If \p stream is non-zero, the operation may overlap with operations in other streams.
 *
 * The device version of this function only handles device to device copies and
 * cannot be given local or shared pointers.
 *
 * \param devPtr - Pointer to device memory
 * \param value  - Value to set for each byte of specified memory
 * \param count  - Size in bytes to set
 * \param stream - Stream identifier
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * \notefnerr
 * \note_memset
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemset, ::UPTKMemset2D, ::UPTKMemset3D,
 * ::UPTKMemset2DAsync, ::UPTKMemset3DAsync,
 * ::cuMemsetD8Async,
 * ::cuMemsetD16Async,
 * ::cuMemsetD32Async
 */
extern __host__ UPTKError_t UPTKMemsetAsync(void *devPtr, int value, size_t count, UPTKStream_t stream __dv(0));

/**
 * \brief Initializes or sets device memory to a value
 *
 * Sets to the specified value \p value a matrix (\p height rows of \p width
 * bytes each) pointed to by \p dstPtr. \p pitch is the width in bytes of the
 * 2D array pointed to by \p dstPtr, including any padding added to the end
 * of each row. This function performs fastest when the pitch is one that has
 * been passed back by ::UPTKMallocPitch().
 *
 * ::UPTKMemset2DAsync() is asynchronous with respect to the host, so
 * the call may return before the memset is complete. The operation can optionally
 * be associated to a stream by passing a non-zero \p stream argument.
 * If \p stream is non-zero, the operation may overlap with operations in other streams.
 *
 * The device version of this function only handles device to device copies and
 * cannot be given local or shared pointers.
 *
 * \param devPtr - Pointer to 2D device memory
 * \param pitch  - Pitch in bytes of 2D device memory(Unused if \p height is 1)
 * \param value  - Value to set for each byte of specified memory
 * \param width  - Width of matrix set (columns in bytes)
 * \param height - Height of matrix set (rows)
 * \param stream - Stream identifier
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * \notefnerr
 * \note_memset
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemset, ::UPTKMemset2D, ::UPTKMemset3D,
 * ::UPTKMemsetAsync, ::UPTKMemset3DAsync,
 * ::cuMemsetD2D8Async,
 * ::cuMemsetD2D16Async,
 * ::cuMemsetD2D32Async
 */
extern __host__ UPTKError_t UPTKMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, UPTKStream_t stream __dv(0));

/**
 * \brief Initializes or sets device memory to a value
 *
 * Initializes each element of a 3D array to the specified value \p value.
 * The object to initialize is defined by \p pitchedDevPtr. The \p pitch field
 * of \p pitchedDevPtr is the width in memory in bytes of the 3D array pointed
 * to by \p pitchedDevPtr, including any padding added to the end of each row.
 * The \p xsize field specifies the logical width of each row in bytes, while
 * the \p ysize field specifies the height of each 2D slice in rows.
 * The \p pitch field of \p pitchedDevPtr is ignored when \p height and \p depth 
 * are both equal to 1. 
 *
 * The extents of the initialized region are specified as a \p width in bytes,
 * a \p height in rows, and a \p depth in slices.
 *
 * Extents with \p width greater than or equal to the \p xsize of
 * \p pitchedDevPtr may perform significantly faster than extents narrower
 * than the \p xsize. Secondarily, extents with \p height equal to the
 * \p ysize of \p pitchedDevPtr will perform faster than when the \p height is
 * shorter than the \p ysize.
 *
 * This function performs fastest when the \p pitchedDevPtr has been allocated
 * by ::UPTKMalloc3D().
 *
 * ::UPTKMemset3DAsync() is asynchronous with respect to the host, so
 * the call may return before the memset is complete. The operation can optionally
 * be associated to a stream by passing a non-zero \p stream argument.
 * If \p stream is non-zero, the operation may overlap with operations in other streams.
 *
 * The device version of this function only handles device to device copies and
 * cannot be given local or shared pointers.
 *
 * \param pitchedDevPtr - Pointer to pitched device memory
 * \param value         - Value to set for each byte of specified memory
 * \param extent        - Size parameters for where to set device memory (\p width field in bytes)
 * \param stream - Stream identifier
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * \notefnerr
 * \note_memset
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemset, ::UPTKMemset2D, ::UPTKMemset3D,
 * ::UPTKMemsetAsync, ::UPTKMemset2DAsync,
 * ::UPTKMalloc3D, ::make_UPTKPitchedPtr,
 * ::make_UPTKExtent
 */
extern __host__ UPTKError_t UPTKMemset3DAsync(struct UPTKPitchedPtr pitchedDevPtr, int value, struct UPTKExtent extent, UPTKStream_t stream __dv(0));

/**
 * \brief Finds the address associated with a UPTK symbol
 *
 * Returns in \p *devPtr the address of symbol \p symbol on the device.
 * \p symbol is a variable that resides in global or constant memory space.
 * If \p symbol cannot be found, or if \p symbol is not declared in the
 * global or constant memory space, \p *devPtr is unchanged and the error
 * ::UPTKErrorInvalidSymbol is returned.
 *
 * \param devPtr - Return device pointer associated with symbol
 * \param symbol - Device symbol address
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidSymbol,
 * ::UPTKErrorNoKernelImageForDevice
 * \notefnerr
 * \note_string_api_deprecation
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * \ref ::UPTKGetSymbolAddress(void**, const T&) "UPTKGetSymbolAddress (C++ API)",
 * \ref ::UPTKGetSymbolSize(size_t*, const void*) "UPTKGetSymbolSize (C API)",
 * ::cuModuleGetGlobal
 */
extern __host__ UPTKError_t UPTKGetSymbolAddress(void **devPtr, const void *symbol);

/**
 * \brief Finds the size of the object associated with a UPTK symbol
 *
 * Returns in \p *size the size of symbol \p symbol. \p symbol is a variable that
 * resides in global or constant memory space. If \p symbol cannot be found, or
 * if \p symbol is not declared in global or constant memory space, \p *size is
 * unchanged and the error ::UPTKErrorInvalidSymbol is returned.
 *
 * \param size   - Size of object associated with symbol
 * \param symbol - Device symbol address
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidSymbol,
 * ::UPTKErrorNoKernelImageForDevice
 * \notefnerr
 * \note_string_api_deprecation
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * \ref ::UPTKGetSymbolAddress(void**, const void*) "UPTKGetSymbolAddress (C API)",
 * \ref ::UPTKGetSymbolSize(size_t*, const T&) "UPTKGetSymbolSize (C++ API)",
 * ::cuModuleGetGlobal
 */
extern __host__ UPTKError_t UPTKGetSymbolSize(size_t *size, const void *symbol);

/**
 * \brief Prefetches memory to the specified destination device
 *
 * Prefetches memory to the specified destination device.  \p devPtr is the 
 * base device pointer of the memory to be prefetched and \p dstDevice is the 
 * destination device. \p count specifies the number of bytes to copy. \p stream
 * is the stream in which the operation is enqueued. The memory range must refer
 * to managed memory allocated via ::UPTKMallocManaged or declared via __managed__ variables,
 * or it may also refer to system-allocated memory on systems with non-zero
 * UPTKDevAttrPageableMemoryAccess.
 *
 * Passing in UPTKCpuDeviceId for \p dstDevice will prefetch the data to host memory. If
 * \p dstDevice is a GPU, then the device attribute ::UPTKDevAttrConcurrentManagedAccess
 * must be non-zero. Additionally, \p stream must be associated with a device that has a
 * non-zero value for the device attribute ::UPTKDevAttrConcurrentManagedAccess.
 *
 * The start address and end address of the memory range will be rounded down and rounded up
 * respectively to be aligned to CPU page size before the prefetch operation is enqueued
 * in the stream.
 *
 * If no physical memory has been allocated for this region, then this memory region
 * will be populated and mapped on the destination device. If there's insufficient
 * memory to prefetch the desired region, the Unified Memory driver may evict pages from other
 * ::UPTKMallocManaged allocations to host memory in order to make room. Device memory
 * allocated using ::UPTKMalloc or ::UPTKMallocArray will not be evicted.
 *
 * By default, any mappings to the previous location of the migrated pages are removed and
 * mappings for the new location are only setup on \p dstDevice. The exact behavior however
 * also depends on the settings applied to this memory range via ::UPTKMemAdvise as described
 * below:
 *
 * If ::UPTKMemAdviseSetReadMostly was set on any subset of this memory range,
 * then that subset will create a read-only copy of the pages on \p dstDevice.
 *
 * If ::UPTKMemAdviseSetPreferredLocation was called on any subset of this memory
 * range, then the pages will be migrated to \p dstDevice even if \p dstDevice is not the
 * preferred location of any pages in the memory range.
 *
 * If ::UPTKMemAdviseSetAccessedBy was called on any subset of this memory range,
 * then mappings to those pages from all the appropriate processors are updated to
 * refer to the new location if establishing such a mapping is possible. Otherwise,
 * those mappings are cleared.
 *
 * Note that this API is not required for functionality and only serves to improve performance
 * by allowing the application to migrate data to a suitable location before it is accessed.
 * Memory accesses to this range are always coherent and are allowed even when the data is
 * actively being migrated.
 *
 * Note that this function is asynchronous with respect to the host and all work
 * on other devices.
 *
 * \param devPtr    - Pointer to be prefetched
 * \param count     - Size in bytes
 * \param dstDevice - Destination device to prefetch to
 * \param stream    - Stream to enqueue prefetch operation
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidDevice
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpyPeer, ::UPTKMemcpyAsync,
 * ::UPTKMemcpy3DPeerAsync, ::UPTKMemAdvise, ::UPTKMemAdvise_v2
 * ::cuMemPrefetchAsync
 */
extern __host__ UPTKError_t UPTKMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, UPTKStream_t stream __dv(0));

/**
 * \brief Prefetches memory to the specified destination location
 *
 * Prefetches memory to the specified destination location.  \p devPtr is the
 * base device pointer of the memory to be prefetched and \p location specifies the
 * destination location. \p count specifies the number of bytes to copy. \p stream
 * is the stream in which the operation is enqueued. The memory range must refer
 * to managed memory allocated via ::UPTKMallocManaged or declared via __managed__ variables, 
 * or it may also refer to system-allocated memory on systems with non-zero
 * UPTKDevAttrPageableMemoryAccess.
 * 
 * Specifying ::UPTKMemLocationTypeDevice for ::UPTKMemLocation::type will prefetch memory to GPU
 * specified by device ordinal ::UPTKMemLocation::id which must have non-zero value for the device attribute
 * ::concurrentManagedAccess. Additionally, \p stream must be associated with a device
 * that has a non-zero value for the device attribute ::concurrentManagedAccess.
 * Specifying ::UPTKMemLocationTypeHost as ::UPTKMemLocation::type will prefetch data to host memory.
 * Applications can request prefetching memory to a specific host NUMA node by specifying
 * ::UPTKMemLocationTypeHostNuma for ::UPTKMemLocation::type and a valid host NUMA node id in ::UPTKMemLocation::id
 * Users can also request prefetching memory to the host NUMA node closest to the current thread's CPU by specifying
 * ::UPTKMemLocationTypeHostNumaCurrent for ::UPTKMemLocation::type. Note when ::UPTKMemLocation::type is etiher
 * ::UPTKMemLocationTypeHost OR ::UPTKMemLocationTypeHostNumaCurrent, ::UPTKMemLocation::id will be ignored.
 *
 * The start address and end address of the memory range will be rounded down and rounded up
 * respectively to be aligned to CPU page size before the prefetch operation is enqueued
 * in the stream.
 *
 * If no physical memory has been allocated for this region, then this memory region
 * will be populated and mapped on the destination device. If there's insufficient
 * memory to prefetch the desired region, the Unified Memory driver may evict pages from other
 * ::UPTKMallocManaged allocations to host memory in order to make room. Device memory
 * allocated using ::UPTKMalloc or ::UPTKMallocArray will not be evicted.
 *
 * By default, any mappings to the previous location of the migrated pages are removed and
 * mappings for the new location are only setup on the destination location. The exact behavior however
 * also depends on the settings applied to this memory range via ::cuMemAdvise as described
 * below:
 *
 * If ::UPTKMemAdviseSetReadMostly was set on any subset of this memory range,
 * then that subset will create a read-only copy of the pages on destination location.
 * If however the destination location is a host NUMA node, then any pages of that subset
 * that are already in another host NUMA node will be transferred to the destination.
 *
 * If ::UPTKMemAdviseSetPreferredLocation was called on any subset of this memory
 * range, then the pages will be migrated to \p location even if \p location is not the
 * preferred location of any pages in the memory range.
 *
 * If ::UPTKMemAdviseSetAccessedBy was called on any subset of this memory range,
 * then mappings to those pages from all the appropriate processors are updated to
 * refer to the new location if establishing such a mapping is possible. Otherwise,
 * those mappings are cleared.
 *
 * Note that this API is not required for functionality and only serves to improve performance
 * by allowing the application to migrate data to a suitable location before it is accessed.
 * Memory accesses to this range are always coherent and are allowed even when the data is
 * actively being migrated.
 *
 * Note that this function is asynchronous with respect to the host and all work
 * on other devices.
 *
 * \param devPtr    - Pointer to be prefetched
 * \param count     - Size in bytes
 * \param location  - location to prefetch to
 * \param flags     - flags for future use, must be zero now. 
 * \param stream    - Stream to enqueue prefetch operation
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidDevice
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpyPeer, ::UPTKMemcpyAsync,
 * ::UPTKMemcpy3DPeerAsync, ::UPTKMemAdvise, ::UPTKMemAdvise_v2
 * ::cuMemPrefetchAsync
 */
extern __host__ UPTKError_t UPTKMemPrefetchAsync_v2(const void *devPtr, size_t count, struct UPTKMemLocation location, unsigned int flags, UPTKStream_t stream __dv(0));

/**
 * \brief Advise about the usage of a given memory range
 *
 * Advise the Unified Memory subsystem about the usage pattern for the memory range
 * starting at \p devPtr with a size of \p count bytes. The start address and end address of the memory
 * range will be rounded down and rounded up respectively to be aligned to CPU page size before the
 * advice is applied. The memory range must refer to managed memory allocated via ::UPTKMallocManaged
 * or declared via __managed__ variables. The memory range could also refer to system-allocated pageable
 * memory provided it represents a valid, host-accessible region of memory and all additional constraints
 * imposed by \p advice as outlined below are also satisfied. Specifying an invalid system-allocated pageable
 * memory range results in an error being returned.
 *
 * The \p advice parameter can take the following values:
 * - ::UPTKMemAdviseSetReadMostly: This implies that the data is mostly going to be read
 * from and only occasionally written to. Any read accesses from any processor to this region will create a
 * read-only copy of at least the accessed pages in that processor's memory. Additionally, if ::UPTKMemPrefetchAsync
 * is called on this region, it will create a read-only copy of the data on the destination processor.
 * If any processor writes to this region, all copies of the corresponding page will be invalidated
 * except for the one where the write occurred. The \p device argument is ignored for this advice.
 * Note that for a page to be read-duplicated, the accessing processor must either be the CPU or a GPU
 * that has a non-zero value for the device attribute ::UPTKDevAttrConcurrentManagedAccess.
 * Also, if a context is created on a device that does not have the device attribute
 * ::UPTKDevAttrConcurrentManagedAccess set, then read-duplication will not occur until
 * all such contexts are destroyed.
 * If the memory region refers to valid system-allocated pageable memory, then the accessing device must
 * have a non-zero value for the device attribute ::UPTKDevAttrPageableMemoryAccess for a read-only
 * copy to be created on that device. Note however that if the accessing device also has a non-zero value for the
 * device attribute ::UPTKDevAttrPageableMemoryAccessUsesHostPageTables, then setting this advice
 * will not create a read-only copy when that device accesses this memory region.
 *
 * - ::UPTKMemAdviceUnsetReadMostly: Undoes the effect of ::UPTKMemAdviceReadMostly and also prevents the
 * Unified Memory driver from attempting heuristic read-duplication on the memory range. Any read-duplicated
 * copies of the data will be collapsed into a single copy. The location for the collapsed
 * copy will be the preferred location if the page has a preferred location and one of the read-duplicated
 * copies was resident at that location. Otherwise, the location chosen is arbitrary.
 *
 * - ::UPTKMemAdviseSetPreferredLocation: This advice sets the preferred location for the
 * data to be the memory belonging to \p device. Passing in UPTKCpuDeviceId for \p device sets the
 * preferred location as host memory. If \p device is a GPU, then it must have a non-zero value for the
 * device attribute ::UPTKDevAttrConcurrentManagedAccess. Setting the preferred location
 * does not cause data to migrate to that location immediately. Instead, it guides the migration policy
 * when a fault occurs on that memory region. If the data is already in its preferred location and the
 * faulting processor can establish a mapping without requiring the data to be migrated, then
 * data migration will be avoided. On the other hand, if the data is not in its preferred location
 * or if a direct mapping cannot be established, then it will be migrated to the processor accessing
 * it. It is important to note that setting the preferred location does not prevent data prefetching
 * done using ::UPTKMemPrefetchAsync.
 * Having a preferred location can override the page thrash detection and resolution logic in the Unified
 * Memory driver. Normally, if a page is detected to be constantly thrashing between for example host and device
 * memory, the page may eventually be pinned to host memory by the Unified Memory driver. But
 * if the preferred location is set as device memory, then the page will continue to thrash indefinitely.
 * If ::UPTKMemAdviseSetReadMostly is also set on this memory region or any subset of it, then the
 * policies associated with that advice will override the policies of this advice, unless read accesses from
 * \p device will not result in a read-only copy being created on that device as outlined in description for
 * the advice ::UPTKMemAdviseSetReadMostly.
 * If the memory region refers to valid system-allocated pageable memory, then \p device must have a non-zero
 * value for the device attribute ::UPTKDevAttrPageableMemoryAccess.
 *
 * - ::UPTKMemAdviseUnsetPreferredLocation: Undoes the effect of ::UPTKMemAdviseSetPreferredLocation
 * and changes the preferred location to none.
 *
 * - ::UPTKMemAdviseSetAccessedBy: This advice implies that the data will be accessed by \p device.
 * Passing in ::UPTKCpuDeviceId for \p device will set the advice for the CPU. If \p device is a GPU, then
 * the device attribute ::UPTKDevAttrConcurrentManagedAccess must be non-zero.
 * This advice does not cause data migration and has no impact on the location of the data per se. Instead,
 * it causes the data to always be mapped in the specified processor's page tables, as long as the
 * location of the data permits a mapping to be established. If the data gets migrated for any reason,
 * the mappings are updated accordingly.
 * This advice is recommended in scenarios where data locality is not important, but avoiding faults is.
 * Consider for example a system containing multiple GPUs with peer-to-peer access enabled, where the
 * data located on one GPU is occasionally accessed by peer GPUs. In such scenarios, migrating data
 * over to the other GPUs is not as important because the accesses are infrequent and the overhead of
 * migration may be too high. But preventing faults can still help improve performance, and so having
 * a mapping set up in advance is useful. Note that on CPU access of this data, the data may be migrated
 * to host memory because the CPU typically cannot access device memory directly. Any GPU that had the
 * ::UPTKMemAdviceSetAccessedBy flag set for this data will now have its mapping updated to point to the
 * page in host memory.
 * If ::UPTKMemAdviseSetReadMostly is also set on this memory region or any subset of it, then the
 * policies associated with that advice will override the policies of this advice. Additionally, if the
 * preferred location of this memory region or any subset of it is also \p device, then the policies
 * associated with ::UPTKMemAdviseSetPreferredLocation will override the policies of this advice.
 * If the memory region refers to valid system-allocated pageable memory, then \p device must have a non-zero
 * value for the device attribute ::UPTKDevAttrPageableMemoryAccess. Additionally, if \p device has
 * a non-zero value for the device attribute ::UPTKDevAttrPageableMemoryAccessUsesHostPageTables,
 * then this call has no effect.
 *
 * - ::UPTKMemAdviseUnsetAccessedBy: Undoes the effect of ::UPTKMemAdviseSetAccessedBy. Any mappings to
 * the data from \p device may be removed at any time causing accesses to result in non-fatal page faults.
 * If the memory region refers to valid system-allocated pageable memory, then \p device must have a non-zero
 * value for the device attribute ::UPTKDevAttrPageableMemoryAccess. Additionally, if \p device has
 * a non-zero value for the device attribute ::UPTKDevAttrPageableMemoryAccessUsesHostPageTables,
 * then this call has no effect.
 *
 * \param devPtr - Pointer to memory to set the advice for
 * \param count  - Size in bytes of the memory range
 * \param advice - Advice to be applied for the specified memory range
 * \param device - Device to apply the advice for
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidDevice
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpyPeer, ::UPTKMemcpyAsync,
 * ::UPTKMemcpy3DPeerAsync, ::UPTKMemPrefetchAsync,
 * ::cuMemAdvise
 */
extern __host__ UPTKError_t UPTKMemAdvise(const void *devPtr, size_t count, enum UPTKMemoryAdvise advice, int device);

/**
 * \brief Advise about the usage of a given memory range
 *
 * Advise the Unified Memory subsystem about the usage pattern for the memory range
 * starting at \p devPtr with a size of \p count bytes. The start address and end address of the memory
 * range will be rounded down and rounded up respectively to be aligned to CPU page size before the
 * advice is applied. The memory range must refer to managed memory allocated via ::UPTKMallocManaged
 * or declared via __managed__ variables. The memory range could also refer to system-allocated pageable
 * memory provided it represents a valid, host-accessible region of memory and all additional constraints
 * imposed by \p advice as outlined below are also satisfied. Specifying an invalid system-allocated pageable
 * memory range results in an error being returned.
 *
 * The \p advice parameter can take the following values:
 * - ::UPTKMemAdviseSetReadMostly: This implies that the data is mostly going to be read
 * from and only occasionally written to. Any read accesses from any processor to this region will create a
 * read-only copy of at least the accessed pages in that processor's memory. Additionally, if ::UPTKMemPrefetchAsync
 * or ::UPTKMemPrefetchAsync_v2 is called on this region, it will create a read-only copy of the data on the destination processor.
 * If the target location for ::UPTKMemPrefetchAsync_v2 is a host NUMA node and a read-only copy already exists on
 * another host NUMA node, that copy will be migrated to the targeted host NUMA node.
 * If any processor writes to this region, all copies of the corresponding page will be invalidated
 * except for the one where the write occurred. If the writing processor is the CPU and the preferred location of
 * the page is a host NUMA node, then the page will also be migrated to that host NUMA node. The \p location argument is ignored for this advice.
 * Note that for a page to be read-duplicated, the accessing processor must either be the CPU or a GPU
 * that has a non-zero value for the device attribute ::UPTKDevAttrConcurrentManagedAccess.
 * Also, if a context is created on a device that does not have the device attribute
 * ::UPTKDevAttrConcurrentManagedAccess set, then read-duplication will not occur until
 * all such contexts are destroyed.
 * If the memory region refers to valid system-allocated pageable memory, then the accessing device must
 * have a non-zero value for the device attribute ::UPTKDevAttrPageableMemoryAccess for a read-only
 * copy to be created on that device. Note however that if the accessing device also has a non-zero value for the
 * device attribute ::UPTKDevAttrPageableMemoryAccessUsesHostPageTables, then setting this advice
 * will not create a read-only copy when that device accesses this memory region.
 *
 * - ::UPTKMemAdviceUnsetReadMostly:  Undoes the effect of ::UPTKMemAdviseSetReadMostly and also prevents the
 * Unified Memory driver from attempting heuristic read-duplication on the memory range. Any read-duplicated
 * copies of the data will be collapsed into a single copy. The location for the collapsed
 * copy will be the preferred location if the page has a preferred location and one of the read-duplicated
 * copies was resident at that location. Otherwise, the location chosen is arbitrary.
 * Note: The \p location argument is ignored for this advice.
 *
 * - ::UPTKMemAdviseSetPreferredLocation: This advice sets the preferred location for the
 * data to be the memory belonging to \p location. When ::UPTKMemLocation::type is ::UPTKMemLocationTypeHost,
 * ::UPTKMemLocation::id is ignored and the preferred location is set to be host memory. To set the preferred location
 * to a specific host NUMA node, applications must set ::UPTKMemLocation::type to ::UPTKMemLocationTypeHostNuma and
 * ::UPTKMemLocation::id must specify the NUMA ID of the host NUMA node. If ::UPTKMemLocation::type is set to ::UPTKMemLocationTypeHostNumaCurrent,
 * ::UPTKMemLocation::id will be ignored and the host NUMA node closest to the calling thread's CPU will be used as the preferred location.
 * If ::UPTKMemLocation::type is a ::UPTKMemLocationTypeDevice, then ::UPTKMemLocation::id must be a valid device ordinal
 * and the device must have a non-zero value for the device attribute ::UPTKDevAttrConcurrentManagedAccess.
 * Setting the preferred location does not cause data to migrate to that location immediately. Instead, it guides the migration policy
 * when a fault occurs on that memory region. If the data is already in its preferred location and the
 * faulting processor can establish a mapping without requiring the data to be migrated, then
 * data migration will be avoided. On the other hand, if the data is not in its preferred location
 * or if a direct mapping cannot be established, then it will be migrated to the processor accessing
 * it. It is important to note that setting the preferred location does not prevent data prefetching
 * done using ::UPTKMemPrefetchAsync.
 * Having a preferred location can override the page thrash detection and resolution logic in the Unified
 * Memory driver. Normally, if a page is detected to be constantly thrashing between for example host and device
 * memory, the page may eventually be pinned to host memory by the Unified Memory driver. But
 * if the preferred location is set as device memory, then the page will continue to thrash indefinitely.
 * If ::UPTKMemAdviseSetReadMostly is also set on this memory region or any subset of it, then the
 * policies associated with that advice will override the policies of this advice, unless read accesses from
 * \p location will not result in a read-only copy being created on that procesor as outlined in description for
 * the advice ::UPTKMemAdviseSetReadMostly.
 * If the memory region refers to valid system-allocated pageable memory, and ::UPTKMemLocation::type is ::UPTKMemLocationTypeDevice
 * then ::UPTKMemLocation::id must be a valid device that has a non-zero alue for the device attribute ::UPTKDevAttrPageableMemoryAccess.
 *
 * - ::UPTKMemAdviseUnsetPreferredLocation: Undoes the effect of ::UPTKMemAdviseSetPreferredLocation
 * and changes the preferred location to none. The \p location argument is ignored for this advice.
 *
 * - ::UPTKMemAdviseSetAccessedBy: This advice implies that the data will be accessed by processor \p location.
 * The ::UPTKMemLocation::type must be either ::UPTKMemLocationTypeDevice with ::UPTKMemLocation::id representing a valid device
 * ordinal or ::UPTKMemLocationTypeHost and ::UPTKMemLocation::id will be ignored. All other location types are invalid.
 * If ::UPTKMemLocation::id is a GPU, then the device attribute ::UPTKDevAttrConcurrentManagedAccess must be non-zero.
 * This advice does not cause data migration and has no impact on the location of the data per se. Instead,
 * it causes the data to always be mapped in the specified processor's page tables, as long as the
 * location of the data permits a mapping to be established. If the data gets migrated for any reason,
 * the mappings are updated accordingly.
 * This advice is recommended in scenarios where data locality is not important, but avoiding faults is.
 * Consider for example a system containing multiple GPUs with peer-to-peer access enabled, where the
 * data located on one GPU is occasionally accessed by peer GPUs. In such scenarios, migrating data
 * over to the other GPUs is not as important because the accesses are infrequent and the overhead of
 * migration may be too high. But preventing faults can still help improve performance, and so having
 * a mapping set up in advance is useful. Note that on CPU access of this data, the data may be migrated
 * to host memory because the CPU typically cannot access device memory directly. Any GPU that had the
 * ::UPTKMemAdviseSetAccessedBy flag set for this data will now have its mapping updated to point to the
 * page in host memory.
 * If ::UPTKMemAdviseSetReadMostly is also set on this memory region or any subset of it, then the
 * policies associated with that advice will override the policies of this advice. Additionally, if the
 * preferred location of this memory region or any subset of it is also \p location, then the policies
 * associated with ::CU_MEM_ADVISE_SET_PREFERRED_LOCATION will override the policies of this advice.
 * If the memory region refers to valid system-allocated pageable memory, and ::UPTKMemLocation::type is ::UPTKMemLocationTypeDevice
 * then device in ::UPTKMemLocation::id must have a non-zero value for the device attribute ::UPTKDevAttrPageableMemoryAccess.
 * Additionally, if ::UPTKMemLocation::id has a non-zero value for the device attribute ::UPTKDevAttrPageableMemoryAccessUsesHostPageTables,
 * then this call has no effect.
 *
 * - ::CU_MEM_ADVISE_UNSET_ACCESSED_BY: Undoes the effect of ::UPTKMemAdviseSetAccessedBy. Any mappings to
 * the data from \p location may be removed at any time causing accesses to result in non-fatal page faults.
 * If the memory region refers to valid system-allocated pageable memory, and ::UPTKMemLocation::type is ::UPTKMemLocationTypeDevice
 * then device in ::UPTKMemLocation::id must have a non-zero value for the device attribute ::UPTKDevAttrPageableMemoryAccess.
 * Additionally, if ::UPTKMemLocation::id has a non-zero value for the device attribute ::UPTKDevAttrPageableMemoryAccessUsesHostPageTables,
 * then this call has no effect.
 *
 * \param devPtr   - Pointer to memory to set the advice for
 * \param count    - Size in bytes of the memory range
 * \param advice   - Advice to be applied for the specified memory range
 * \param location - location to apply the advice for
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidDevice
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpyPeer, ::UPTKMemcpyAsync,
 * ::UPTKMemcpy3DPeerAsync, ::UPTKMemPrefetchAsync,
 * ::cuMemAdvise, ::cuMemAdvise_v2
 */
extern __host__ UPTKError_t UPTKMemAdvise_v2(const void *devPtr, size_t count, enum UPTKMemoryAdvise advice, struct UPTKMemLocation location);

/**
* \brief Query an attribute of a given memory range
*
* Query an attribute about the memory range starting at \p devPtr with a size of \p count bytes. The
* memory range must refer to managed memory allocated via ::UPTKMallocManaged or declared via
* __managed__ variables.
*
* The \p attribute parameter can take the following values:
* - ::UPTKMemRangeAttributeReadMostly: If this attribute is specified, \p data will be interpreted
* as a 32-bit integer, and \p dataSize must be 4. The result returned will be 1 if all pages in the given
* memory range have read-duplication enabled, or 0 otherwise.
* - ::UPTKMemRangeAttributePreferredLocation: If this attribute is specified, \p data will be
* interpreted as a 32-bit integer, and \p dataSize must be 4. The result returned will be a GPU device
* id if all pages in the memory range have that GPU as their preferred location, or it will be UPTKCpuDeviceId
* if all pages in the memory range have the CPU as their preferred location, or it will be UPTKInvalidDeviceId
* if either all the pages don't have the same preferred location or some of the pages don't have a
* preferred location at all. Note that the actual location of the pages in the memory range at the time of
* the query may be different from the preferred location.
* - ::UPTKMemRangeAttributeAccessedBy: If this attribute is specified, \p data will be interpreted
* as an array of 32-bit integers, and \p dataSize must be a non-zero multiple of 4. The result returned
* will be a list of device ids that had ::UPTKMemAdviceSetAccessedBy set for that entire memory range.
* If any device does not have that advice set for the entire memory range, that device will not be included.
* If \p data is larger than the number of devices that have that advice set for that memory range,
* UPTKInvalidDeviceId will be returned in all the extra space provided. For ex., if \p dataSize is 12
* (i.e. \p data has 3 elements) and only device 0 has the advice set, then the result returned will be
* { 0, UPTKInvalidDeviceId, UPTKInvalidDeviceId }. If \p data is smaller than the number of devices that have
* that advice set, then only as many devices will be returned as can fit in the array. There is no
* guarantee on which specific devices will be returned, however.
* - ::UPTKMemRangeAttributeLastPrefetchLocation: If this attribute is specified, \p data will be
* interpreted as a 32-bit integer, and \p dataSize must be 4. The result returned will be the last location
* to which all pages in the memory range were prefetched explicitly via ::UPTKMemPrefetchAsync. This will either be
* a GPU id or UPTKCpuDeviceId depending on whether the last location for prefetch was a GPU or the CPU
* respectively. If any page in the memory range was never explicitly prefetched or if all pages were not
* prefetched to the same location, UPTKInvalidDeviceId will be returned. Note that this simply returns the
* last location that the applicaton requested to prefetch the memory range to. It gives no indication as to
* whether the prefetch operation to that location has completed or even begun.
 * - ::UPTKMemRangeAttributePreferredLocationType: If this attribute is specified, \p data will be
 * interpreted as a ::UPTKMemLocationType, and \p dataSize must be sizeof(UPTKMemLocationType). The ::UPTKMemLocationType returned will be
 * ::UPTKMemLocationTypeDevice if all pages in the memory range have the same GPU as their preferred location, or ::UPTKMemLocationType
 * will be ::UPTKMemLocationTypeHost if all pages in the memory range have the CPU as their preferred location, or or it will be ::UPTKMemLocationTypeHostNuma
 * if all the pages in the memory range have the same host NUMA node ID as their preferred location or it will be ::UPTKMemLocationTypeInvalid
 * if either all the pages don't have the same preferred location or some of the pages don't have a preferred location at all.
 * Note that the actual location type of the pages in the memory range at the time of the query may be different from the preferred location type.
 *  - ::UPTKMemRangeAttributePreferredLocationId: If this attribute is specified, \p data will be
 * interpreted as a 32-bit integer, and \p dataSize must be 4. If the ::UPTKMemRangeAttributePreferredLocationType query for the same address range
 * returns ::UPTKMemLocationTypeDevice, it will be a valid device ordinal or if it returns ::UPTKMemLocationTypeHostNuma, it will be a valid host NUMA node ID
 * or if it returns any other location type, the id should be ignored.
 * - ::UPTKMemRangeAttributeLastPrefetchLocationType: If this attribute is specified, \p data will be
 * interpreted as a ::UPTKMemLocationType, and \p dataSize must be sizeof(UPTKMemLocationType). The result returned will be the last location type
 * to which all pages in the memory range were prefetched explicitly via ::cuMemPrefetchAsync. The ::UPTKMemLocationType returned
 * will be ::UPTKMemLocationTypeDevice if the last prefetch location was the GPU or ::UPTKMemLocationTypeHost if it was the CPU or ::UPTKMemLocationTypeHostNuma if
 * the last prefetch location was a specific host NUMA node. If any page in the memory range was never explicitly prefetched or if all pages were not
 * prefetched to the same location, ::CUmemLocationType will be ::UPTKMemLocationTypeInvalid.
 * Note that this simply returns the last location type that the application requested to prefetch the memory range to. It gives no indication as to
 * whether the prefetch operation to that location has completed or even begun.
 *  - ::UPTKMemRangeAttributeLastPrefetchLocationId: If this attribute is specified, \p data will be
 * interpreted as a 32-bit integer, and \p dataSize must be 4. If the ::UPTKMemRangeAttributeLastPrefetchLocationType query for the same address range
 * returns ::UPTKMemLocationTypeDevice, it will be a valid device ordinal or if it returns ::UPTKMemLocationTypeHostNuma, it will be a valid host NUMA node ID
 * or if it returns any other location type, the id should be ignored.
*
* \param data      - A pointers to a memory location where the result
*                    of each attribute query will be written to.
* \param dataSize  - Array containing the size of data
* \param attribute - The attribute to query
* \param devPtr    - Start of the range to query
* \param count     - Size of the range to query
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemRangeGetAttributes, ::UPTKMemPrefetchAsync,
 * ::UPTKMemAdvise,
 * ::cuMemRangeGetAttribute
 */
extern __host__ UPTKError_t UPTKMemRangeGetAttribute(void *data, size_t dataSize, enum UPTKMemRangeAttribute attribute, const void *devPtr, size_t count);

/**
 * \brief Query attributes of a given memory range.
 *
 * Query attributes of the memory range starting at \p devPtr with a size of \p count bytes. The
 * memory range must refer to managed memory allocated via ::UPTKMallocManaged or declared via
 * __managed__ variables. The \p attributes array will be interpreted to have \p numAttributes
 * entries. The \p dataSizes array will also be interpreted to have \p numAttributes entries.
 * The results of the query will be stored in \p data.
 *
 * The list of supported attributes are given below. Please refer to ::UPTKMemRangeGetAttribute for
 * attribute descriptions and restrictions.
 *
 * - ::UPTKMemRangeAttributeReadMostly
 * - ::UPTKMemRangeAttributePreferredLocation
 * - ::UPTKMemRangeAttributeAccessedBy
 * - ::UPTKMemRangeAttributeLastPrefetchLocation
 * - :: UPTKMemRangeAttributePreferredLocationType
 * - :: UPTKMemRangeAttributePreferredLocationId
 * - :: UPTKMemRangeAttributeLastPrefetchLocationType
 * - :: UPTKMemRangeAttributeLastPrefetchLocationId
 *
 * \param data          - A two-dimensional array containing pointers to memory
 *                        locations where the result of each attribute query will be written to.
 * \param dataSizes     - Array containing the sizes of each result
 * \param attributes    - An array of attributes to query
 *                        (numAttributes and the number of attributes in this array should match)
 * \param numAttributes - Number of attributes to query
 * \param devPtr        - Start of the range to query
 * \param count         - Size of the range to query
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemRangeGetAttribute, ::UPTKMemAdvise,
 * ::UPTKMemPrefetchAsync,
 * ::cuMemRangeGetAttributes
 */
extern __host__ UPTKError_t UPTKMemRangeGetAttributes(void **data, size_t *dataSizes, enum UPTKMemRangeAttribute *attributes, size_t numAttributes, const void *devPtr, size_t count);

/** @} */ /* END UPTKRT_MEMORY */

/**
 * \defgroup UPTKRT_MEMORY_DEPRECATED Memory Management [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated memory management functions of the UPTK runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes deprecated memory management functions of the UPTK runtime
 * application programming interface.
 *
 * Some functions have overloaded C++ API template versions documented separately in the
 * \ref UPTKRT_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */

/**
 * \brief Copies data between host and device
 *
 * \deprecated
 *
 * Copies \p count bytes from the memory area pointed to by \p src to the
 * UPTK array \p dst starting at \p hOffset rows and \p wOffset bytes from
 * the upper left corner, where \p kind specifies the direction
 * of the copy, and must be one of ::UPTKMemcpyHostToHost,
 * ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToHost,
 * ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault. Passing
 * ::UPTKMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * \param dst     - Destination memory address
 * \param wOffset - Destination starting X offset (columns in bytes)
 * \param hOffset - Destination starting Y offset (rows)
 * \param src     - Source memory address
 * \param count   - Size in bytes to copy
 * \param kind    - Type of transfer
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_sync
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpy2D,
 * ::UPTKMemcpy2DToArray, ::UPTKMemcpyFromArray, ::UPTKMemcpy2DFromArray,
 * ::UPTKMemcpyArrayToArray, ::UPTKMemcpy2DArrayToArray, ::UPTKMemcpyToSymbol,
 * ::UPTKMemcpyFromSymbol, ::UPTKMemcpyAsync, ::UPTKMemcpy2DAsync,
 * ::UPTKMemcpyToArrayAsync, ::UPTKMemcpy2DToArrayAsync,
 * ::UPTKMemcpyFromArrayAsync, ::UPTKMemcpy2DFromArrayAsync,
 * ::UPTKMemcpyToSymbolAsync, ::UPTKMemcpyFromSymbolAsync,
 * ::cuMemcpyHtoA,
 * ::cuMemcpyDtoA
 */
extern __host__ UPTKError_t UPTKMemcpyToArray(UPTKArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum UPTKMemcpyKind kind);

/**
 * \brief Copies data between host and device
 *
 * \deprecated
 *
 * Copies \p count bytes from the UPTK array \p src starting at \p hOffset rows
 * and \p wOffset bytes from the upper left corner to the memory area pointed to
 * by \p dst, where \p kind specifies the direction of the copy, and must be one of
 * ::UPTKMemcpyHostToHost, ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToHost,
 * ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault. Passing
 * ::UPTKMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * \param dst     - Destination memory address
 * \param src     - Source memory address
 * \param wOffset - Source starting X offset (columns in bytes)
 * \param hOffset - Source starting Y offset (rows)
 * \param count   - Size in bytes to copy
 * \param kind    - Type of transfer
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_sync
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpy2D, ::UPTKMemcpyToArray,
 * ::UPTKMemcpy2DToArray, ::UPTKMemcpy2DFromArray,
 * ::UPTKMemcpyArrayToArray, ::UPTKMemcpy2DArrayToArray, ::UPTKMemcpyToSymbol,
 * ::UPTKMemcpyFromSymbol, ::UPTKMemcpyAsync, ::UPTKMemcpy2DAsync,
 * ::UPTKMemcpyToArrayAsync, ::UPTKMemcpy2DToArrayAsync,
 * ::UPTKMemcpyFromArrayAsync, ::UPTKMemcpy2DFromArrayAsync,
 * ::UPTKMemcpyToSymbolAsync, ::UPTKMemcpyFromSymbolAsync,
 * ::cuMemcpyAtoH,
 * ::cuMemcpyAtoD
 */
extern __host__ UPTKError_t UPTKMemcpyFromArray(void *dst, UPTKArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum UPTKMemcpyKind kind);

/**
 * \brief Copies data between host and device
 *
 * \deprecated
 *
 * Copies \p count bytes from the UPTK array \p src starting at \p hOffsetSrc
 * rows and \p wOffsetSrc bytes from the upper left corner to the UPTK array
 * \p dst starting at \p hOffsetDst rows and \p wOffsetDst bytes from the upper
 * left corner, where \p kind specifies the direction of the copy, and must be one of
 * ::UPTKMemcpyHostToHost, ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToHost,
 * ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault. Passing
 * ::UPTKMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * \param dst        - Destination memory address
 * \param wOffsetDst - Destination starting X offset (columns in bytes)
 * \param hOffsetDst - Destination starting Y offset (rows)
 * \param src        - Source memory address
 * \param wOffsetSrc - Source starting X offset (columns in bytes)
 * \param hOffsetSrc - Source starting Y offset (rows)
 * \param count      - Size in bytes to copy
 * \param kind       - Type of transfer
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpy2D, ::UPTKMemcpyToArray,
 * ::UPTKMemcpy2DToArray, ::UPTKMemcpyFromArray, ::UPTKMemcpy2DFromArray,
 * ::UPTKMemcpy2DArrayToArray, ::UPTKMemcpyToSymbol,
 * ::UPTKMemcpyFromSymbol, ::UPTKMemcpyAsync, ::UPTKMemcpy2DAsync,
 * ::UPTKMemcpyToArrayAsync, ::UPTKMemcpy2DToArrayAsync,
 * ::UPTKMemcpyFromArrayAsync, ::UPTKMemcpy2DFromArrayAsync,
 * ::UPTKMemcpyToSymbolAsync, ::UPTKMemcpyFromSymbolAsync,
 * ::cuMemcpyAtoA
 */
extern __host__ UPTKError_t UPTKMemcpyArrayToArray(UPTKArray_t dst, size_t wOffsetDst, size_t hOffsetDst, UPTKArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum UPTKMemcpyKind kind __dv(UPTKMemcpyDeviceToDevice));

/**
 * \brief Copies data between host and device
 *
 * \deprecated
 *
 * Copies \p count bytes from the memory area pointed to by \p src to the
 * UPTK array \p dst starting at \p hOffset rows and \p wOffset bytes from
 * the upper left corner, where \p kind specifies the
 * direction of the copy, and must be one of ::UPTKMemcpyHostToHost,
 * ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToHost,
 * ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault. Passing
 * ::UPTKMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * ::UPTKMemcpyToArrayAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If \p
 * kind is ::UPTKMemcpyHostToDevice or ::UPTKMemcpyDeviceToHost and \p stream
 * is non-zero, the copy may overlap with operations in other streams.
 *
 * \param dst     - Destination memory address
 * \param wOffset - Destination starting X offset (columns in bytes)
 * \param hOffset - Destination starting Y offset (rows)
 * \param src     - Source memory address
 * \param count   - Size in bytes to copy
 * \param kind    - Type of transfer
 * \param stream  - Stream identifier
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpy2D, ::UPTKMemcpyToArray,
 * ::UPTKMemcpy2DToArray, ::UPTKMemcpyFromArray, ::UPTKMemcpy2DFromArray,
 * ::UPTKMemcpyArrayToArray, ::UPTKMemcpy2DArrayToArray, ::UPTKMemcpyToSymbol,
 * ::UPTKMemcpyFromSymbol, ::UPTKMemcpyAsync, ::UPTKMemcpy2DAsync,
 * ::UPTKMemcpy2DToArrayAsync,
 * ::UPTKMemcpyFromArrayAsync, ::UPTKMemcpy2DFromArrayAsync,
 * ::UPTKMemcpyToSymbolAsync, ::UPTKMemcpyFromSymbolAsync,
 * ::cuMemcpyHtoAAsync,
 * ::cuMemcpy2DAsync
 */
extern __host__ UPTKError_t UPTKMemcpyToArrayAsync(UPTKArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum UPTKMemcpyKind kind, UPTKStream_t stream __dv(0));

/**
 * \brief Copies data between host and device
 *
 * \deprecated
 *
 * Copies \p count bytes from the UPTK array \p src starting at \p hOffset rows
 * and \p wOffset bytes from the upper left corner to the memory area pointed to
 * by \p dst, where \p kind specifies the direction of the copy, and must be one of
 * ::UPTKMemcpyHostToHost, ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToHost,
 * ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault. Passing
 * ::UPTKMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * ::UPTKMemcpyFromArrayAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If \p
 * kind is ::UPTKMemcpyHostToDevice or ::UPTKMemcpyDeviceToHost and \p stream
 * is non-zero, the copy may overlap with operations in other streams.
 *
 * \param dst     - Destination memory address
 * \param src     - Source memory address
 * \param wOffset - Source starting X offset (columns in bytes)
 * \param hOffset - Source starting Y offset (rows)
 * \param count   - Size in bytes to copy
 * \param kind    - Type of transfer
 * \param stream  - Stream identifier
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKMemcpy, ::UPTKMemcpy2D, ::UPTKMemcpyToArray,
 * ::UPTKMemcpy2DToArray, ::UPTKMemcpyFromArray, ::UPTKMemcpy2DFromArray,
 * ::UPTKMemcpyArrayToArray, ::UPTKMemcpy2DArrayToArray, ::UPTKMemcpyToSymbol,
 * ::UPTKMemcpyFromSymbol, ::UPTKMemcpyAsync, ::UPTKMemcpy2DAsync,
 * ::UPTKMemcpyToArrayAsync, ::UPTKMemcpy2DToArrayAsync,
 * ::UPTKMemcpy2DFromArrayAsync,
 * ::UPTKMemcpyToSymbolAsync, ::UPTKMemcpyFromSymbolAsync,
 * ::cuMemcpyAtoHAsync,
 * ::cuMemcpy2DAsync
 */
extern __host__ UPTKError_t UPTKMemcpyFromArrayAsync(void *dst, UPTKArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum UPTKMemcpyKind kind, UPTKStream_t stream __dv(0));

/** @} */ /* END UPTKRT_MEMORY_DEPRECATED */

/**
 * \defgroup UPTKRT_MEMORY_POOLS Stream Ordered Memory Allocator 
 *
 * ___MANBRIEF___ Functions for performing allocation and free operations in stream order.
 *                Functions for controlling the behavior of the underlying allocator.
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 * 
 *
 * @{
 *
 * \section UPTKRT_MEMORY_POOLS_overview overview
 *
 * The asynchronous allocator allows the user to allocate and free in stream order.
 * All asynchronous accesses of the allocation must happen between
 * the stream executions of the allocation and the free. If the memory is accessed
 * outside of the promised stream order, a use before allocation / use after free error
 * will cause undefined behavior.
 *
 * The allocator is free to reallocate the memory as long as it can guarantee
 * that compliant memory accesses will not overlap temporally.
 * The allocator may refer to internal stream ordering as well as inter-stream dependencies
 * (such as UPTK events and null stream dependencies) when establishing the temporal guarantee.
 * The allocator may also insert inter-stream dependencies to establish the temporal guarantee.
 *
 * \section UPTKRT_MEMORY_POOLS_support Supported Platforms
 *
 * Whether or not a device supports the integrated stream ordered memory allocator
 * may be queried by calling ::UPTKDeviceGetAttribute() with the device attribute
 * ::UPTKDevAttrMemoryPoolsSupported.
 */

/**
 * \brief Allocates memory with stream ordered semantics
 *
 * Inserts an allocation operation into \p hStream.
 * A pointer to the allocated memory is returned immediately in *dptr.
 * The allocation must not be accessed until the the allocation operation completes.
 * The allocation comes from the memory pool associated with the stream's device.
 *
 * \note The default memory pool of a device contains device memory from that device.
 * \note Basic stream ordering allows future work submitted into the same stream to use the allocation.
 *       Stream query, stream synchronize, and UPTK events can be used to guarantee that the allocation
 *       operation completes before work submitted in a separate stream runs.
 * \note During stream capture, this function results in the creation of an allocation node.  In this case,
 *       the allocation is owned by the graph instead of the memory pool. The memory pool's properties
 *       are used to set the node's creation parameters.
 *
 * \param[out] devPtr  - Returned device pointer
 * \param[in] size     - Number of bytes to allocate
 * \param[in] hStream  - The stream establishing the stream ordering contract and the memory pool to allocate from
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorNotSupported,
 * ::UPTKErrorOutOfMemory,
 * \notefnerr
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cuMemAllocAsync,
 * \ref ::UPTKMallocAsync(void** ptr, size_t size, UPTKMemPool_t memPool, UPTKStream_t stream)  "UPTKMallocAsync (C++ API)", 
 * ::UPTKMallocFromPoolAsync, ::UPTKFreeAsync, ::UPTKDeviceSetMemPool, ::UPTKDeviceGetDefaultMemPool, ::UPTKDeviceGetMemPool, ::UPTKMemPoolSetAccess, ::UPTKMemPoolSetAttribute, ::UPTKMemPoolGetAttribute
 */
extern __host__ UPTKError_t UPTKMallocAsync(void **devPtr, size_t size, UPTKStream_t hStream);

/**
 * \brief Frees memory with stream ordered semantics
 *
 * Inserts a free operation into \p hStream.
 * The allocation must not be accessed after stream execution reaches the free.
 * After this API returns, accessing the memory from any subsequent work launched on the GPU
 * or querying its pointer attributes results in undefined behavior.
 *
 * \note During stream capture, this function results in the creation of a free node and
 *       must therefore be passed the address of a graph allocation.
 *
 * \param dptr - memory to free
 * \param hStream - The stream establishing the stream ordering promise
 * \returns
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorNotSupported
 * \notefnerr
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cuMemFreeAsync, ::UPTKMallocAsync
 */
extern __host__ UPTKError_t UPTKFreeAsync(void *devPtr, UPTKStream_t hStream);

/**
 * \brief Tries to release memory back to the OS
 *
 * Releases memory back to the OS until the pool contains fewer than minBytesToKeep
 * reserved bytes, or there is no more memory that the allocator can safely release.
 * The allocator cannot release OS allocations that back outstanding asynchronous allocations.
 * The OS allocations may happen at different granularity from the user allocations.
 *
 * \note: Allocations that have not been freed count as outstanding.
 * \note: Allocations that have been asynchronously freed but whose completion has
 *        not been observed on the host (eg. by a synchronize) can count as outstanding.
 *
 * \param[in] pool           - The memory pool to trim
 * \param[in] minBytesToKeep - If the pool has less than minBytesToKeep reserved,
 * the TrimTo operation is a no-op.  Otherwise the pool will be guaranteed to have
 * at least minBytesToKeep bytes reserved after the operation.
 * \returns
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_callback
 *
 * \sa ::cuMemPoolTrimTo, ::UPTKMallocAsync, ::UPTKFreeAsync, ::UPTKDeviceGetDefaultMemPool, ::UPTKDeviceGetMemPool, ::UPTKMemPoolCreate
 */
extern __host__ UPTKError_t UPTKMemPoolTrimTo(UPTKMemPool_t memPool, size_t minBytesToKeep);

/**
 * \brief Sets attributes of a memory pool
 *
 * Supported attributes are:
 * - ::UPTKMemPoolAttrReleaseThreshold: (value type = cuuint64_t)
 *                    Amount of reserved memory in bytes to hold onto before trying
 *                    to release memory back to the OS. When more than the release
 *                    threshold bytes of memory are held by the memory pool, the
 *                    allocator will try to release memory back to the OS on the
 *                    next call to stream, event or context synchronize. (default 0)
 * - ::UPTKMemPoolReuseFollowEventDependencies: (value type = int)
 *                    Allow ::UPTKMallocAsync to use memory asynchronously freed
 *                    in another stream as long as a stream ordering dependency
 *                    of the allocating stream on the free action exists.
 *                    UPTK events and null stream interactions can create the required
 *                    stream ordered dependencies. (default enabled)
 * - ::UPTKMemPoolReuseAllowOpportunistic: (value type = int)
 *                    Allow reuse of already completed frees when there is no dependency
 *                    between the free and allocation. (default enabled)
 * - ::UPTKMemPoolReuseAllowInternalDependencies: (value type = int)
 *                    Allow ::UPTKMallocAsync to insert new stream dependencies
 *                    in order to establish the stream ordering required to reuse
 *                    a piece of memory released by ::UPTKFreeAsync (default enabled).
 * - ::UPTKMemPoolAttrReservedMemHigh: (value type = cuuint64_t)
 *                    Reset the high watermark that tracks the amount of backing memory that was
 *                    allocated for the memory pool. It is illegal to set this attribute to a non-zero value.
 * - ::UPTKMemPoolAttrUsedMemHigh: (value type = cuuint64_t)
 *                    Reset the high watermark that tracks the amount of used memory that was
 *                    allocated for the memory pool. It is illegal to set this attribute to a non-zero value.
 *
 * \param[in] pool  - The memory pool to modify
 * \param[in] attr  - The attribute to modify
 * \param[in] value - Pointer to the value to assign
 *
 * \returns
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_callback
 *
 * \sa ::cuMemPoolSetAttribute, ::UPTKMallocAsync, ::UPTKFreeAsync, ::UPTKDeviceGetDefaultMemPool, ::UPTKDeviceGetMemPool, ::UPTKMemPoolCreate

 */
extern __host__ UPTKError_t UPTKMemPoolSetAttribute(UPTKMemPool_t memPool, enum UPTKMemPoolAttr attr, void *value );

/**
 * \brief Gets attributes of a memory pool
 *
 * Supported attributes are:
 * - ::UPTKMemPoolAttrReleaseThreshold: (value type = cuuint64_t)
 *                    Amount of reserved memory in bytes to hold onto before trying
 *                    to release memory back to the OS. When more than the release
 *                    threshold bytes of memory are held by the memory pool, the
 *                    allocator will try to release memory back to the OS on the
 *                    next call to stream, event or context synchronize. (default 0)
 * - ::UPTKMemPoolReuseFollowEventDependencies: (value type = int)
 *                    Allow ::UPTKMallocAsync to use memory asynchronously freed
 *                    in another stream as long as a stream ordering dependency
 *                    of the allocating stream on the free action exists.
 *                    UPTK events and null stream interactions can create the required
 *                    stream ordered dependencies. (default enabled)
 * - ::UPTKMemPoolReuseAllowOpportunistic: (value type = int)
 *                    Allow reuse of already completed frees when there is no dependency
 *                    between the free and allocation. (default enabled)
 * - ::UPTKMemPoolReuseAllowInternalDependencies: (value type = int)
 *                    Allow ::UPTKMallocAsync to insert new stream dependencies
 *                    in order to establish the stream ordering required to reuse
 *                    a piece of memory released by ::UPTKFreeAsync (default enabled).
 * - ::UPTKMemPoolAttrReservedMemCurrent: (value type = cuuint64_t)
 *                    Amount of backing memory currently allocated for the mempool.
 * - ::UPTKMemPoolAttrReservedMemHigh: (value type = cuuint64_t)
 *                    High watermark of backing memory allocated for the mempool since
 *                    the last time it was reset.
 * - ::UPTKMemPoolAttrUsedMemCurrent: (value type = cuuint64_t)
 *                    Amount of memory from the pool that is currently in use by the application.
 * - ::UPTKMemPoolAttrUsedMemHigh: (value type = cuuint64_t)
 *                    High watermark of the amount of memory from the pool that was in use by the
 *                    application since the last time it was reset.
 *
 * \param[in] pool  - The memory pool to get attributes of 
 * \param[in] attr  - The attribute to get
 * \param[in] value - Retrieved value 
 *
 * \returns
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_callback
 *
 * \sa ::cuMemPoolGetAttribute, ::UPTKMallocAsync, ::UPTKFreeAsync, ::UPTKDeviceGetDefaultMemPool, ::UPTKDeviceGetMemPool, ::UPTKMemPoolCreate

 */
extern __host__ UPTKError_t UPTKMemPoolGetAttribute(UPTKMemPool_t memPool, enum UPTKMemPoolAttr attr, void *value );

/**
 * \brief Controls visibility of pools between devices
 *
 * \param[in] pool  - The pool being modified
 * \param[in] map   - Array of access descriptors. Each descriptor instructs the access to enable for a single gpu
 * \param[in] count - Number of descriptors in the map array.
 *
 * \returns
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 *
 * \sa ::cuMemPoolSetAccess, ::UPTKMemPoolGetAccess, ::UPTKMallocAsync, UPTKFreeAsync
 */
extern __host__ UPTKError_t UPTKMemPoolSetAccess(UPTKMemPool_t memPool, const struct UPTKMemAccessDesc *descList, size_t count);

/**
 * \brief Returns the accessibility of a pool from a device
 *
 * Returns the accessibility of the pool's memory from the specified location.
 *
 * \param[out] flags   - the accessibility of the pool from the specified location
 * \param[in] memPool  - the pool being queried
 * \param[in] location - the location accessing the pool
 *
 * \sa ::cuMemPoolGetAccess, ::UPTKMemPoolSetAccess
 */
extern __host__ UPTKError_t UPTKMemPoolGetAccess(enum UPTKMemAccessFlags *flags, UPTKMemPool_t memPool, struct UPTKMemLocation *location);

/**
 * \brief Creates a memory pool
 *
 * Creates a UPTK memory pool and returns the handle in \p pool.  The \p poolProps determines
 * the properties of the pool such as the backing device and IPC capabilities.
 *
 * To create a memory pool targeting a specific host NUMA node, applications must
 * set ::UPTKMemPoolProps::UPTKMemLocation::type to ::UPTKMemLocationTypeHostNuma and
 * ::UPTKMemPoolProps::UPTKMemLocation::id must specify the NUMA ID of the host memory node.
 * Specifying ::UPTKMemLocationTypeHostNumaCurrent or ::UPTKMemLocationTypeHost as the
 * ::UPTKMemPoolProps::UPTKMemLocation::type will result in ::UPTKErrorInvalidValue.
* By default, the pool's memory will be accessible from the device it is allocated on.
 * In the case of pools created with ::UPTKMemLocationTypeHostNuma, their default accessibility
 * will be from the host CPU.
 * Applications can control the maximum size of the pool by specifying a non-zero value for ::UPTKMemPoolProps::maxSize.
 * If set to 0, the maximum size of the pool will default to a system dependent value.
 *
 * Applications can set ::UPTKMemPoolProps::handleTypes to ::UPTKMemHandleTypeFabric
 * in order to create ::UPTKMemPool_t suitable for sharing within an IMEX domain.
 * An IMEX domain is either an OS instance or a group of securely connected OS instances
 * using the NVIDIA IMEX daemon. An IMEX channel is a global resource within the IMEX domain
 * that represents a logical entity that aims to provide fine grained accessibility control
 * for the participating processes. When exporter and importer UPTK processes have been
 * granted access to the same IMEX channel, they can securely share memory.
 * If the allocating process does not have access setup for an IMEX channel, attempting to export
 * a ::CUmemoryPool with ::UPTKMemHandleTypeFabric will result in ::UPTKErrorNotPermitted.
 * The nvidia-modprobe CLI provides more information regarding setting up of IMEX channels.
 *
 * \note Specifying UPTKMemHandleTypeNone creates a memory pool that will not support IPC.
 *
 * \returns
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorNotSupported
 *
 * \sa ::cuMemPoolCreate, ::UPTKDeviceSetMemPool, ::UPTKMallocFromPoolAsync, ::UPTKMemPoolExportToShareableHandle, ::UPTKDeviceGetDefaultMemPool, ::UPTKDeviceGetMemPool

 */
extern __host__ UPTKError_t UPTKMemPoolCreate(UPTKMemPool_t *memPool, const struct UPTKMemPoolProps *poolProps);

/**
 * \brief Destroys the specified memory pool 
 *
 * If any pointers obtained from this pool haven't been freed or
 * the pool has free operations that haven't completed
 * when ::UPTKMemPoolDestroy is invoked, the function will return immediately and the
 * resources associated with the pool will be released automatically
 * once there are no more outstanding allocations.
 *
 * Destroying the current mempool of a device sets the default mempool of
 * that device as the current mempool for that device.
 *
 * \note A device's default memory pool cannot be destroyed.
 *
 * \returns
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 *
 * \sa cuMemPoolDestroy, ::UPTKFreeAsync, ::UPTKDeviceSetMemPool, ::UPTKDeviceGetDefaultMemPool, ::UPTKDeviceGetMemPool, ::UPTKMemPoolCreate
 */
extern __host__ UPTKError_t UPTKMemPoolDestroy(UPTKMemPool_t memPool);

/**
 * \brief Allocates memory from a specified pool with stream ordered semantics.
 *
 * Inserts an allocation operation into \p hStream.
 * A pointer to the allocated memory is returned immediately in *dptr.
 * The allocation must not be accessed until the the allocation operation completes.
 * The allocation comes from the specified memory pool.
 *
 * \note
 *    -  The specified memory pool may be from a device different than that of the specified \p hStream.
 *
 *    -  Basic stream ordering allows future work submitted into the same stream to use the allocation.
 *       Stream query, stream synchronize, and UPTK events can be used to guarantee that the allocation
 *       operation completes before work submitted in a separate stream runs.
 *
 * \note During stream capture, this function results in the creation of an allocation node.  In this case,
 *       the allocation is owned by the graph instead of the memory pool. The memory pool's properties
 *       are used to set the node's creation parameters.
 *
 * \param[out] ptr     - Returned device pointer
 * \param[in] bytesize - Number of bytes to allocate
 * \param[in] memPool  - The pool to allocate from
 * \param[in] stream   - The stream establishing the stream ordering semantic
 *
 * \returns
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorNotSupported,
 * ::UPTKErrorOutOfMemory
 *
 * \sa ::cuMemAllocFromPoolAsync,
 * \ref ::UPTKMallocAsync(void** ptr, size_t size, UPTKMemPool_t memPool, UPTKStream_t stream)  "UPTKMallocAsync (C++ API)", 
 * ::UPTKMallocAsync, ::UPTKFreeAsync, ::UPTKDeviceGetDefaultMemPool, ::UPTKMemPoolCreate, ::UPTKMemPoolSetAccess, ::UPTKMemPoolSetAttribute
 */
extern __host__ UPTKError_t UPTKMallocFromPoolAsync(void **ptr, size_t size, UPTKMemPool_t memPool, UPTKStream_t stream);

/**
 * \brief Exports a memory pool to the requested handle type.
 *
 * Given an IPC capable mempool, create an OS handle to share the pool with another process.
 * A recipient process can convert the shareable handle into a mempool with ::UPTKMemPoolImportFromShareableHandle.
 * Individual pointers can then be shared with the ::UPTKMemPoolExportPointer and ::UPTKMemPoolImportPointer APIs.
 * The implementation of what the shareable handle is and how it can be transferred is defined by the requested
 * handle type.
 *
 * \note: To create an IPC capable mempool, create a mempool with a CUmemAllocationHandleType other than UPTKMemHandleTypeNone.
 *
 * \param[out] handle_out  - pointer to the location in which to store the requested handle 
 * \param[in] pool         - pool to export
 * \param[in] handleType   - the type of handle to create
 * \param[in] flags        - must be 0
 *
 * \returns
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorOutOfMemory
 *
 * \sa ::cuMemPoolExportToShareableHandle, ::UPTKMemPoolImportFromShareableHandle, ::UPTKMemPoolExportPointer, ::UPTKMemPoolImportPointer
 */
extern __host__ UPTKError_t UPTKMemPoolExportToShareableHandle(
    void                            *shareableHandle,
    UPTKMemPool_t                    memPool,
    enum UPTKMemAllocationHandleType handleType,
    unsigned int                     flags);

/**
 * \brief imports a memory pool from a shared handle.
 *
 * Specific allocations can be imported from the imported pool with ::UPTKMemPoolImportPointer.
 *
 * \note Imported memory pools do not support creating new allocations.
 *       As such imported memory pools may not be used in ::UPTKDeviceSetMemPool
 *       or ::UPTKMallocFromPoolAsync calls.
 *
 * \param[out] pool_out    - Returned memory pool
 * \param[in] handle       - OS handle of the pool to open
 * \param[in] handleType   - The type of handle being imported
 * \param[in] flags        - must be 0
 *
 * \returns
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorOutOfMemory
 *
 * \sa ::cuMemPoolImportFromShareableHandle, ::UPTKMemPoolExportToShareableHandle, ::UPTKMemPoolExportPointer, ::UPTKMemPoolImportPointer
 */
extern __host__ UPTKError_t UPTKMemPoolImportFromShareableHandle(
    UPTKMemPool_t                   *memPool,
    void                            *shareableHandle,
    enum UPTKMemAllocationHandleType handleType,
    unsigned int                     flags);

/**
 * \brief Export data to share a memory pool allocation between processes.
 *
 * Constructs \p shareData_out for sharing a specific allocation from an already shared memory pool.
 * The recipient process can import the allocation with the ::UPTKMemPoolImportPointer api.
 * The data is not a handle and may be shared through any IPC mechanism.
 *
 * \param[out] shareData_out - Returned export data
 * \param[in] ptr            - pointer to memory being exported
 *
 * \returns
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorOutOfMemory
 *
 * \sa ::cuMemPoolExportPointer, ::UPTKMemPoolExportToShareableHandle, ::UPTKMemPoolImportFromShareableHandle, ::UPTKMemPoolImportPointer
 */
extern __host__ UPTKError_t UPTKMemPoolExportPointer(struct UPTKMemPoolPtrExportData *exportData, void *ptr);

/**
 * \brief Import a memory pool allocation from another process.
 *
 * Returns in \p ptr_out a pointer to the imported memory.
 * The imported memory must not be accessed before the allocation operation completes
 * in the exporting process. The imported memory must be freed from all importing processes before
 * being freed in the exporting process. The pointer may be freed with UPTKFree
 * or UPTKFreeAsync.  If ::UPTKFreeAsync is used, the free must be completed
 * on the importing process before the free operation on the exporting process.
 *
 * \note The ::UPTKFreeAsync api may be used in the exporting process before
 *       the ::UPTKFreeAsync operation completes in its stream as long as the
 *       ::UPTKFreeAsync in the exporting process specifies a stream with
 *       a stream dependency on the importing process's ::UPTKFreeAsync.
 *
 * \param[out] ptr_out  - pointer to imported memory
 * \param[in] pool      - pool from which to import
 * \param[in] shareData - data specifying the memory to import
 *
 * \returns
 * ::UPTK_SUCCESS,
 * ::UPTK_ERROR_INVALID_VALUE,
 * ::UPTK_ERROR_NOT_INITIALIZED,
 * ::UPTK_ERROR_OUT_OF_MEMORY
 *
 * \sa ::cuMemPoolImportPointer, ::UPTKMemPoolExportToShareableHandle, ::UPTKMemPoolImportFromShareableHandle, ::UPTKMemPoolExportPointer
 */
extern __host__ UPTKError_t UPTKMemPoolImportPointer(void **ptr, UPTKMemPool_t memPool, struct UPTKMemPoolPtrExportData *exportData);

/** @} */ /* END UPTKRT_MEMORY_POOLS */

/**
 * \defgroup UPTKRT_UNIFIED Unified Addressing
 *
 * ___MANBRIEF___ unified addressing functions of the UPTK runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the unified addressing functions of the UPTK 
 * runtime application programming interface.
 *
 * @{
 *
 * \section UPTKRT_UNIFIED_overview Overview
 *
 * UPTK devices can share a unified address space with the host.  
 * For these devices there is no distinction between a device
 * pointer and a host pointer -- the same pointer value may be 
 * used to access memory from the host program and from a kernel 
 * running on the device (with exceptions enumerated below).
 *
 * \section UPTKRT_UNIFIED_support Supported Platforms
 * 
 * Whether or not a device supports unified addressing may be 
 * queried by calling ::UPTKGetDeviceProperties() with the device 
 * property ::UPTKDeviceProp::unifiedAddressing.
 *
 * Unified addressing is automatically enabled in 64-bit processes .
 *
 * \section UPTKRT_UNIFIED_lookup Looking Up Information from Pointer Values
 *
 * It is possible to look up information about the memory which backs a 
 * pointer value.  For instance, one may want to know if a pointer points
 * to host or device memory.  As another example, in the case of device 
 * memory, one may want to know on which UPTK device the memory 
 * resides.  These properties may be queried using the function 
 * ::UPTKPointerGetAttributes()
 *
 * Since pointers are unique, it is not necessary to specify information
 * about the pointers specified to ::UPTKMemcpy() and other copy functions.  
 * The copy direction ::UPTKMemcpyDefault may be used to specify that the 
 * UPTK runtime should infer the location of the pointer from its value.
 *
 * \section UPTKRT_UNIFIED_automaphost Automatic Mapping of Host Allocated Host Memory
 *
 * All host memory allocated through all devices using ::UPTKMallocHost() and
 * ::UPTKHostAlloc() is always directly accessible from all devices that 
 * support unified addressing.  This is the case regardless of whether or 
 * not the flags ::UPTKHostAllocPortable and ::UPTKHostAllocMapped are 
 * specified.
 *
 * The pointer value through which allocated host memory may be accessed 
 * in kernels on all devices that support unified addressing is the same 
 * as the pointer value through which that memory is accessed on the host.
 * It is not necessary to call ::UPTKHostGetDevicePointer() to get the device 
 * pointer for these allocations.  
 *
 * Note that this is not the case for memory allocated using the flag
 * ::UPTKHostAllocWriteCombined, as discussed below.
 *
 * \section UPTKRT_UNIFIED_autopeerregister Direct Access of Peer Memory
 
 * Upon enabling direct access from a device that supports unified addressing 
 * to another peer device that supports unified addressing using 
 * ::UPTKDeviceEnablePeerAccess() all memory allocated in the peer device using 
 * ::UPTKMalloc() and ::UPTKMallocPitch() will immediately be accessible 
 * by the current device.  The device pointer value through 
 * which any peer's memory may be accessed in the current device 
 * is the same pointer value through which that memory may be 
 * accessed from the peer device. 
 *
 * \section UPTKRT_UNIFIED_exceptions Exceptions, Disjoint Addressing
 * 
 * Not all memory may be accessed on devices through the same pointer
 * value through which they are accessed on the host.  These exceptions
 * are host memory registered using ::UPTKHostRegister() and host memory
 * allocated using the flag ::UPTKHostAllocWriteCombined.  For these 
 * exceptions, there exists a distinct host and device address for the
 * memory.  The device address is guaranteed to not overlap any valid host
 * pointer range and is guaranteed to have the same value across all devices
 * that support unified addressing.  
 * 
 * This device address may be queried using ::UPTKHostGetDevicePointer() 
 * when a device using unified addressing is current.  Either the host 
 * or the unified device pointer value may be used to refer to this memory 
 * in ::UPTKMemcpy() and similar functions using the ::UPTKMemcpyDefault 
 * memory direction.
 *
 */

/**
 * \brief Returns attributes about a specified pointer
 *
 * Returns in \p *attributes the attributes of the pointer \p ptr.
 * If pointer was not allocated in, mapped by or registered with context
 * supporting unified addressing ::UPTKErrorInvalidValue is returned.
 *
 * \note In UPTK 11.0 forward passing host pointer will return ::UPTKMemoryTypeUnregistered
 * in ::UPTKPointerAttributes::type and call will return ::UPTKSuccess.
 *
 * The ::UPTKPointerAttributes structure is defined as:
 * \code
    struct UPTKPointerAttributes {
        enum UPTKMemoryType type;
        int device;
        void *devicePointer;
        void *hostPointer;
    }
    \endcode
 * In this structure, the individual fields mean
 *
 * - \ref ::UPTKPointerAttributes::type identifies type of memory. It can be
 *    ::UPTKMemoryTypeUnregistered for unregistered host memory,
 *    ::UPTKMemoryTypeHost for registered host memory, ::UPTKMemoryTypeDevice for device
 *    memory or  ::UPTKMemoryTypeManaged for managed memory.
 *
 * - \ref ::UPTKPointerAttributes::device "device" is the device against which
 *   \p ptr was allocated.  If \p ptr has memory type ::UPTKMemoryTypeDevice
 *   then this identifies the device on which the memory referred to by \p ptr
 *   physically resides.  If \p ptr has memory type ::UPTKMemoryTypeHost then this
 *   identifies the device which was current when the allocation was made
 *   (and if that device is deinitialized then this allocation will vanish
 *   with that device's state).
 *
 * - \ref ::UPTKPointerAttributes::devicePointer "devicePointer" is
 *   the device pointer alias through which the memory referred to by \p ptr
 *   may be accessed on the current device.
 *   If the memory referred to by \p ptr cannot be accessed directly by the 
 *   current device then this is NULL.  
 *
 * - \ref ::UPTKPointerAttributes::hostPointer "hostPointer" is
 *   the host pointer alias through which the memory referred to by \p ptr
 *   may be accessed on the host.
 *   If the memory referred to by \p ptr cannot be accessed directly by the
 *   host then this is NULL.
 *
 * \param attributes - Attributes for the specified pointer
 * \param ptr        - Pointer to get attributes for
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDevice,
 * ::UPTKErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKGetDeviceCount, ::UPTKGetDevice, ::UPTKSetDevice,
 * ::UPTKChooseDevice,
 * ::UPTKInitDevice,
 * ::cuPointerGetAttributes
 */
extern __host__ UPTKError_t UPTKPointerGetAttributes(struct UPTKPointerAttributes *attributes, const void *ptr);

/** @} */ /* END UPTKRT_UNIFIED */

/**
 * \defgroup UPTKRT_PEER Peer Device Memory Access
 *
 * ___MANBRIEF___ peer device memory access functions of the UPTK runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the peer device memory access functions of the UPTK runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Queries if a device may directly access a peer device's memory.
 *
 * Returns in \p *canAccessPeer a value of 1 if device \p device is capable of
 * directly accessing memory from \p peerDevice and 0 otherwise.  If direct
 * access of \p peerDevice from \p device is possible, then access may be
 * enabled by calling ::UPTKDeviceEnablePeerAccess().
 *
 * \param canAccessPeer - Returned access capability
 * \param device        - Device from which allocations on \p peerDevice are to
 *                        be directly accessed.
 * \param peerDevice    - Device on which the allocations to be directly accessed 
 *                        by \p device reside.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDevice
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKDeviceEnablePeerAccess,
 * ::UPTKDeviceDisablePeerAccess,
 * ::cuDeviceCanAccessPeer
 */
extern __host__ UPTKError_t UPTKDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice);

/**
 * \brief Enables direct access to memory allocations on a peer device.
 *
 * On success, all allocations from \p peerDevice will immediately be accessible by
 * the current device.  They will remain accessible until access is explicitly
 * disabled using ::UPTKDeviceDisablePeerAccess() or either device is reset using
 * ::UPTKDeviceReset().
 *
 * Note that access granted by this call is unidirectional and that in order to access
 * memory on the current device from \p peerDevice, a separate symmetric call 
 * to ::UPTKDeviceEnablePeerAccess() is required.
 *
 * Note that there are both device-wide and system-wide limitations per system
 * configuration, as noted in the UPTK Programming Guide under the section
 * "Peer-to-Peer Memory Access".
 *
 * Returns ::UPTKErrorInvalidDevice if ::UPTKDeviceCanAccessPeer() indicates
 * that the current device cannot directly access memory from \p peerDevice.
 *
 * Returns ::UPTKErrorPeerAccessAlreadyEnabled if direct access of
 * \p peerDevice from the current device has already been enabled.
 *
 * Returns ::UPTKErrorInvalidValue if \p flags is not 0.
 *
 * \param peerDevice  - Peer device to enable direct access to from the current device
 * \param flags       - Reserved for future use and must be set to 0
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDevice,
 * ::UPTKErrorPeerAccessAlreadyEnabled,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKDeviceCanAccessPeer,
 * ::UPTKDeviceDisablePeerAccess,
 * ::cuCtxEnablePeerAccess
 */
extern __host__ UPTKError_t UPTKDeviceEnablePeerAccess(int peerDevice, unsigned int flags);

/**
 * \brief Disables direct access to memory allocations on a peer device.
 *
 * Returns ::UPTKErrorPeerAccessNotEnabled if direct access to memory on
 * \p peerDevice has not yet been enabled from the current device.
 *
 * \param peerDevice - Peer device to disable direct access to
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorPeerAccessNotEnabled,
 * ::UPTKErrorInvalidDevice
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::UPTKDeviceCanAccessPeer,
 * ::UPTKDeviceEnablePeerAccess,
 * ::cuCtxDisablePeerAccess
 */
extern __host__ UPTKError_t UPTKDeviceDisablePeerAccess(int peerDevice);

/** @} */ /* END UPTKRT_PEER */

/** \defgroup UPTKRT_OPENGL OpenGL Interoperability */

/** \defgroup UPTKRT_OPENGL_DEPRECATED OpenGL Interoperability [DEPRECATED] */

/** \defgroup UPTKRT_D3D9 Direct3D 9 Interoperability */

/** \defgroup UPTKRT_D3D9_DEPRECATED Direct3D 9 Interoperability [DEPRECATED] */

/** \defgroup UPTKRT_D3D10 Direct3D 10 Interoperability */

/** \defgroup UPTKRT_D3D10_DEPRECATED Direct3D 10 Interoperability [DEPRECATED] */

/** \defgroup UPTKRT_D3D11 Direct3D 11 Interoperability */

/** \defgroup UPTKRT_D3D11_DEPRECATED Direct3D 11 Interoperability [DEPRECATED] */

/** \defgroup UPTKRT_VDPAU VDPAU Interoperability */

/** \defgroup UPTKRT_EGL EGL Interoperability */

/**
 * \defgroup UPTKRT_INTEROP Graphics Interoperability
 *
 * ___MANBRIEF___ graphics interoperability functions of the UPTK runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the graphics interoperability functions of the UPTK
 * runtime application programming interface.
 *
 * @{
 */

/**
 * \brief Unregisters a graphics resource for access by UPTK
 *
 * Unregisters the graphics resource \p resource so it is not accessible by
 * UPTK unless registered again.
 *
 * If \p resource is invalid then ::UPTKErrorInvalidResourceHandle is
 * returned.
 *
 * \param resource - Resource to unregister
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorUnknown
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_destroy_ub
 *
 * \sa
 * ::UPTKGraphicsD3D9RegisterResource,
 * ::UPTKGraphicsD3D10RegisterResource,
 * ::UPTKGraphicsD3D11RegisterResource,
 * ::UPTKGraphicsGLRegisterBuffer,
 * ::UPTKGraphicsGLRegisterImage,
 * ::cuGraphicsUnregisterResource
 */
extern __host__ UPTKError_t UPTKGraphicsUnregisterResource(UPTKGraphicsResource_t resource);

/**
 * \brief Set usage flags for mapping a graphics resource
 *
 * Set \p flags for mapping the graphics resource \p resource.
 *
 * Changes to \p flags will take effect the next time \p resource is mapped.
 * The \p flags argument may be any of the following:
 * - ::UPTKGraphicsMapFlagsNone: Specifies no hints about how \p resource will
 *     be used. It is therefore assumed that UPTK may read from or write to \p resource.
 * - ::UPTKGraphicsMapFlagsReadOnly: Specifies that UPTK will not write to \p resource.
 * - ::UPTKGraphicsMapFlagsWriteDiscard: Specifies UPTK will not read from \p resource and will
 *   write over the entire contents of \p resource, so none of the data
 *   previously stored in \p resource will be preserved.
 *
 * If \p resource is presently mapped for access by UPTK then ::UPTKErrorUnknown is returned.
 * If \p flags is not one of the above values then ::UPTKErrorInvalidValue is returned.
 *
 * \param resource - Registered resource to set flags for
 * \param flags    - Parameters for resource mapping
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorUnknown,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphicsMapResources,
 * ::cuGraphicsResourceSetMapFlags
 */
extern __host__ UPTKError_t UPTKGraphicsResourceSetMapFlags(UPTKGraphicsResource_t resource, unsigned int flags);

/**
 * \brief Map graphics resources for access by UPTK
 *
 * Maps the \p count graphics resources in \p resources for access by UPTK.
 *
 * The resources in \p resources may be accessed by UPTK until they
 * are unmapped. The graphics API from which \p resources were registered
 * should not access any resources while they are mapped by UPTK. If an
 * application does so, the results are undefined.
 *
 * This function provides the synchronization guarantee that any graphics calls
 * issued before ::UPTKGraphicsMapResources() will complete before any subsequent UPTK
 * work issued in \p stream begins.
 *
 * If \p resources contains any duplicate entries then ::UPTKErrorInvalidResourceHandle
 * is returned. If any of \p resources are presently mapped for access by
 * UPTK then ::UPTKErrorUnknown is returned.
 *
 * \param count     - Number of resources to map
 * \param resources - Resources to map for UPTK
 * \param stream    - Stream for synchronization
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorUnknown
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphicsResourceGetMappedPointer,
 * ::UPTKGraphicsSubResourceGetMappedArray,
 * ::UPTKGraphicsUnmapResources,
 * ::cuGraphicsMapResources
 */
extern __host__ UPTKError_t UPTKGraphicsMapResources(int count, UPTKGraphicsResource_t *resources, UPTKStream_t stream __dv(0));

/**
 * \brief Unmap graphics resources.
 *
 * Unmaps the \p count graphics resources in \p resources.
 *
 * Once unmapped, the resources in \p resources may not be accessed by UPTK
 * until they are mapped again.
 *
 * This function provides the synchronization guarantee that any UPTK work issued
 * in \p stream before ::UPTKGraphicsUnmapResources() will complete before any
 * subsequently issued graphics work begins.
 *
 * If \p resources contains any duplicate entries then ::UPTKErrorInvalidResourceHandle
 * is returned. If any of \p resources are not presently mapped for access by
 * UPTK then ::UPTKErrorUnknown is returned.
 *
 * \param count     - Number of resources to unmap
 * \param resources - Resources to unmap
 * \param stream    - Stream for synchronization
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorUnknown
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphicsMapResources,
 * ::cuGraphicsUnmapResources
 */
extern __host__ UPTKError_t UPTKGraphicsUnmapResources(int count, UPTKGraphicsResource_t *resources, UPTKStream_t stream __dv(0));

/**
 * \brief Get an device pointer through which to access a mapped graphics resource.
 *
 * Returns in \p *devPtr a pointer through which the mapped graphics resource
 * \p resource may be accessed.
 * Returns in \p *size the size of the memory in bytes which may be accessed from that pointer.
 * The value set in \p devPtr may change every time that \p resource is mapped.
 *
 * If \p resource is not a buffer then it cannot be accessed via a pointer and
 * ::UPTKErrorUnknown is returned.
 * If \p resource is not mapped then ::UPTKErrorUnknown is returned.
 * *
 * \param devPtr     - Returned pointer through which \p resource may be accessed
 * \param size       - Returned size of the buffer accessible starting at \p *devPtr
 * \param resource   - Mapped resource to access
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorUnknown
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphicsMapResources,
 * ::UPTKGraphicsSubResourceGetMappedArray,
 * ::cuGraphicsResourceGetMappedPointer
 */
extern __host__ UPTKError_t UPTKGraphicsResourceGetMappedPointer(void **devPtr, size_t *size, UPTKGraphicsResource_t resource);

/**
 * \brief Get an array through which to access a subresource of a mapped graphics resource.
 *
 * Returns in \p *array an array through which the subresource of the mapped
 * graphics resource \p resource which corresponds to array index \p arrayIndex
 * and mipmap level \p mipLevel may be accessed.  The value set in \p array may
 * change every time that \p resource is mapped.
 *
 * If \p resource is not a texture then it cannot be accessed via an array and
 * ::UPTKErrorUnknown is returned.
 * If \p arrayIndex is not a valid array index for \p resource then
 * ::UPTKErrorInvalidValue is returned.
 * If \p mipLevel is not a valid mipmap level for \p resource then
 * ::UPTKErrorInvalidValue is returned.
 * If \p resource is not mapped then ::UPTKErrorUnknown is returned.
 *
 * \param array       - Returned array through which a subresource of \p resource may be accessed
 * \param resource    - Mapped resource to access
 * \param arrayIndex  - Array index for array textures or cubemap face
 *                      index as defined by ::UPTKGraphicsCubeFace for
 *                      cubemap textures for the subresource to access
 * \param mipLevel    - Mipmap level for the subresource to access
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorUnknown
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphicsResourceGetMappedPointer,
 * ::cuGraphicsSubResourceGetMappedArray
 */
extern __host__ UPTKError_t UPTKGraphicsSubResourceGetMappedArray(UPTKArray_t *array, UPTKGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel);

/**
 * \brief Get a mipmapped array through which to access a mapped graphics resource.
 *
 * Returns in \p *mipmappedArray a mipmapped array through which the mapped
 * graphics resource \p resource may be accessed. The value set in \p mipmappedArray may
 * change every time that \p resource is mapped.
 *
 * If \p resource is not a texture then it cannot be accessed via an array and
 * ::UPTKErrorUnknown is returned.
 * If \p resource is not mapped then ::UPTKErrorUnknown is returned.
 *
 * \param mipmappedArray - Returned mipmapped array through which \p resource may be accessed
 * \param resource       - Mapped resource to access
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorUnknown
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphicsResourceGetMappedPointer,
 * ::cuGraphicsResourceGetMappedMipmappedArray
 */
extern __host__ UPTKError_t UPTKGraphicsResourceGetMappedMipmappedArray(UPTKMipmappedArray_t *mipmappedArray, UPTKGraphicsResource_t resource);

/** @} */ /* END UPTKRT_INTEROP */

/**
 * \defgroup UPTKRT_TEXTURE_OBJECT Texture Object Management
 *
 * ___MANBRIEF___ texture object management functions of the UPTK runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the low level texture object management functions
 * of the UPTK runtime application programming interface. The texture
 * object API is only supported on devices of compute capability 3.0 or higher.
 *
 * @{
 */

/**
 * \brief Get the channel descriptor of an array
 *
 * Returns in \p *desc the channel descriptor of the UPTK array \p array.
 *
 * \param desc  - Channel format
 * \param array - Memory array on device
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::UPTKCreateChannelDesc(int, int, int, int, UPTKChannelFormatKind) "UPTKCreateChannelDesc (C API)",
 * ::UPTKCreateTextureObject, ::UPTKCreateSurfaceObject
 */
extern __host__ UPTKError_t UPTKGetChannelDesc(struct UPTKChannelFormatDesc *desc, UPTKArray_const_t array);

/**
 * \brief Returns a channel descriptor using the specified format
 *
 * Returns a channel descriptor with format \p f and number of bits of each
 * component \p x, \p y, \p z, and \p w.  The ::UPTKChannelFormatDesc is
 * defined as:
 * \code
  struct UPTKChannelFormatDesc {
    int x, y, z, w;
    enum UPTKChannelFormatKind f;
  };
 * \endcode
 *
 * where ::UPTKChannelFormatKind is one of ::UPTKChannelFormatKindSigned,
 * ::UPTKChannelFormatKindUnsigned, or ::UPTKChannelFormatKindFloat.
 *
 * \param x - X component
 * \param y - Y component
 * \param z - Z component
 * \param w - W component
 * \param f - Channel format
 *
 * \return
 * Channel descriptor with format \p f
 *
 * \sa \ref ::UPTKCreateChannelDesc(void) "UPTKCreateChannelDesc (C++ API)",
 * ::UPTKGetChannelDesc, ::UPTKCreateTextureObject, ::UPTKCreateSurfaceObject
 */
extern __host__ struct UPTKChannelFormatDesc UPTKCreateChannelDesc(int x, int y, int z, int w, enum UPTKChannelFormatKind f);

/**
 * \brief Creates a texture object
 *
 * Creates a texture object and returns it in \p pTexObject. \p pResDesc describes
 * the data to texture from. \p pTexDesc describes how the data should be sampled.
 * \p pResViewDesc is an optional argument that specifies an alternate format for
 * the data described by \p pResDesc, and also describes the subresource region
 * to restrict access to when texturing. \p pResViewDesc can only be specified if
 * the type of resource is a UPTK array or a UPTK mipmapped array not in a block
 * compressed format.
 *
 * Texture objects are only supported on devices of compute capability 3.0 or higher.
 * Additionally, a texture object is an opaque value, and, as such, should only be
 * accessed through UPTK API calls.
 *
 * The ::UPTKResourceDesc structure is defined as:
 * \code
        struct UPTKResourceDesc {
            enum UPTKResourceType resType;
            
            union {
                struct {
                    UPTKArray_t array;
                } array;
                struct {
                    UPTKMipmappedArray_t mipmap;
                } mipmap;
                struct {
                    void *devPtr;
                    struct UPTKChannelFormatDesc desc;
                    size_t sizeInBytes;
                } linear;
                struct {
                    void *devPtr;
                    struct UPTKChannelFormatDesc desc;
                    size_t width;
                    size_t height;
                    size_t pitchInBytes;
                } pitch2D;
            } res;
        };
 * \endcode
 * where:
 * - ::UPTKResourceDesc::resType specifies the type of resource to texture from.
 * CUresourceType is defined as:
 * \code
        enum UPTKResourceType {
            UPTKResourceTypeArray          = 0x00,
            UPTKResourceTypeMipmappedArray = 0x01,
            UPTKResourceTypeLinear         = 0x02,
            UPTKResourceTypePitch2D        = 0x03
        };
 * \endcode
 *
 * \par
 * If ::UPTKResourceDesc::resType is set to ::UPTKResourceTypeArray, ::UPTKResourceDesc::res::array::array
 * must be set to a valid UPTK array handle.
 *
 * \par
 * If ::UPTKResourceDesc::resType is set to ::UPTKResourceTypeMipmappedArray, ::UPTKResourceDesc::res::mipmap::mipmap
 * must be set to a valid UPTK mipmapped array handle and ::UPTKTextureDesc::normalizedCoords must be set to true.
 *
 * \par
 * If ::UPTKResourceDesc::resType is set to ::UPTKResourceTypeLinear, ::UPTKResourceDesc::res::linear::devPtr
 * must be set to a valid device pointer, that is aligned to ::UPTKDeviceProp::textureAlignment.
 * ::UPTKResourceDesc::res::linear::desc describes the format and the number of components per array element. ::UPTKResourceDesc::res::linear::sizeInBytes
 * specifies the size of the array in bytes. The total number of elements in the linear address range cannot exceed 
 * ::UPTKDeviceProp::maxTexture1DLinear. The number of elements is computed as (sizeInBytes / sizeof(desc)).
 *
 * \par
 * If ::UPTKResourceDesc::resType is set to ::UPTKResourceTypePitch2D, ::UPTKResourceDesc::res::pitch2D::devPtr
 * must be set to a valid device pointer, that is aligned to ::UPTKDeviceProp::textureAlignment.
 * ::UPTKResourceDesc::res::pitch2D::desc describes the format and the number of components per array element. ::UPTKResourceDesc::res::pitch2D::width
 * and ::UPTKResourceDesc::res::pitch2D::height specify the width and height of the array in elements, and cannot exceed
 * ::UPTKDeviceProp::maxTexture2DLinear[0] and ::UPTKDeviceProp::maxTexture2DLinear[1] respectively.
 * ::UPTKResourceDesc::res::pitch2D::pitchInBytes specifies the pitch between two rows in bytes and has to be aligned to 
 * ::UPTKDeviceProp::texturePitchAlignment. Pitch cannot exceed ::UPTKDeviceProp::maxTexture2DLinear[2].
 *
 *
 * The ::UPTKTextureDesc struct is defined as
 * \code
        struct UPTKTextureDesc {
            enum UPTKTextureAddressMode addressMode[3];
            enum UPTKTextureFilterMode  filterMode;
            enum UPTKTextureReadMode    readMode;
            int                         sRGB;
            float                       borderColor[4];
            int                         normalizedCoords;
            unsigned int                maxAnisotropy;
            enum UPTKTextureFilterMode  mipmapFilterMode;
            float                       mipmapLevelBias;
            float                       minMipmapLevelClamp;
            float                       maxMipmapLevelClamp;
            int                         disableTrilinearOptimization;
            int                         seamlessCubemap;
        };
 * \endcode
 * where
 * - ::UPTKTextureDesc::addressMode specifies the addressing mode for each dimension of the texture data. ::UPTKTextureAddressMode is defined as:
 *   \code
        enum UPTKTextureAddressMode {
            UPTKAddressModeWrap   = 0,
            UPTKAddressModeClamp  = 1,
            UPTKAddressModeMirror = 2,
            UPTKAddressModeBorder = 3
        };
 *   \endcode
 *   This is ignored if ::UPTKResourceDesc::resType is ::UPTKResourceTypeLinear. Also, if ::UPTKTextureDesc::normalizedCoords
 *   is set to zero, ::UPTKAddressModeWrap and ::UPTKAddressModeMirror won't be supported and will be switched to ::UPTKAddressModeClamp.
 *
 * - ::UPTKTextureDesc::filterMode specifies the filtering mode to be used when fetching from the texture. ::UPTKTextureFilterMode is defined as:
 *   \code
        enum UPTKTextureFilterMode {
            UPTKFilterModePoint  = 0,
            UPTKFilterModeLinear = 1
        };
 *   \endcode
 *   This is ignored if ::UPTKResourceDesc::resType is ::UPTKResourceTypeLinear.
 *
 * - ::UPTKTextureDesc::readMode specifies whether integer data should be converted to floating point or not. ::UPTKTextureReadMode is defined as:
 *   \code
        enum UPTKTextureReadMode {
            UPTKReadModeElementType     = 0,
            UPTKReadModeNormalizedFloat = 1
        };
 *   \endcode
 *   Note that this applies only to 8-bit and 16-bit integer formats. 32-bit integer format would not be promoted, regardless of 
 *   whether or not this ::UPTKTextureDesc::readMode is set ::UPTKReadModeNormalizedFloat is specified.
 *
 * - ::UPTKTextureDesc::sRGB specifies whether sRGB to linear conversion should be performed during texture fetch.
 *
 * - ::UPTKTextureDesc::borderColor specifies the float values of color. where:
 *   ::UPTKTextureDesc::borderColor[0] contains value of 'R', 
 *   ::UPTKTextureDesc::borderColor[1] contains value of 'G',
 *   ::UPTKTextureDesc::borderColor[2] contains value of 'B', 
 *   ::UPTKTextureDesc::borderColor[3] contains value of 'A'
 *   Note that application using integer border color values will need to <reinterpret_cast> these values to float.
 *   The values are set only when the addressing mode specified by ::UPTKTextureDesc::addressMode is UPTKAddressModeBorder.
 *
 * - ::UPTKTextureDesc::normalizedCoords specifies whether the texture coordinates will be normalized or not.
 *
 * - ::UPTKTextureDesc::maxAnisotropy specifies the maximum anistropy ratio to be used when doing anisotropic filtering. This value will be
 *   clamped to the range [1,16].
 *
 * - ::UPTKTextureDesc::mipmapFilterMode specifies the filter mode when the calculated mipmap level lies between two defined mipmap levels.
 *
 * - ::UPTKTextureDesc::mipmapLevelBias specifies the offset to be applied to the calculated mipmap level.
 *
 * - ::UPTKTextureDesc::minMipmapLevelClamp specifies the lower end of the mipmap level range to clamp access to.
 *
 * - ::UPTKTextureDesc::maxMipmapLevelClamp specifies the upper end of the mipmap level range to clamp access to.
 *
 * - ::UPTKTextureDesc::disableTrilinearOptimization specifies whether the trilinear filtering optimizations will be disabled.
 *
 * - ::UPTKTextureDesc::seamlessCubemap specifies whether seamless cube map filtering is enabled. This flag can only be specified if the 
 *   underlying resource is a UPTK array or a UPTK mipmapped array that was created with the flag ::UPTKArrayCubemap.
 *   When seamless cube map filtering is enabled, texture address modes specified by ::UPTKTextureDesc::addressMode are ignored.
 *   Instead, if the ::UPTKTextureDesc::filterMode is set to ::UPTKFilterModePoint the address mode ::UPTKAddressModeClamp will be applied for all dimensions.
 *   If the ::UPTKTextureDesc::filterMode is set to ::UPTKFilterModeLinear seamless cube map filtering will be performed when sampling along the cube face borders.
 *
 * The ::UPTKResourceViewDesc struct is defined as
 * \code
        struct UPTKResourceViewDesc {
            enum UPTKResourceViewFormat format;
            size_t                      width;
            size_t                      height;
            size_t                      depth;
            unsigned int                firstMipmapLevel;
            unsigned int                lastMipmapLevel;
            unsigned int                firstLayer;
            unsigned int                lastLayer;
        };
 * \endcode
 * where:
 * - ::UPTKResourceViewDesc::format specifies how the data contained in the UPTK array or UPTK mipmapped array should
 *   be interpreted. Note that this can incur a change in size of the texture data. If the resource view format is a block
 *   compressed format, then the underlying UPTK array or UPTK mipmapped array has to have a 32-bit unsigned integer format
 *   with 2 or 4 channels, depending on the block compressed format. For ex., BC1 and BC4 require the underlying UPTK array to have
 *   a 32-bit unsigned int with 2 channels. The other BC formats require the underlying resource to have the same 32-bit unsigned int
 *   format but with 4 channels.
 *
 * - ::UPTKResourceViewDesc::width specifies the new width of the texture data. If the resource view format is a block
 *   compressed format, this value has to be 4 times the original width of the resource. For non block compressed formats,
 *   this value has to be equal to that of the original resource.
 *
 * - ::UPTKResourceViewDesc::height specifies the new height of the texture data. If the resource view format is a block
 *   compressed format, this value has to be 4 times the original height of the resource. For non block compressed formats,
 *   this value has to be equal to that of the original resource.
 *
 * - ::UPTKResourceViewDesc::depth specifies the new depth of the texture data. This value has to be equal to that of the
 *   original resource.
 *
 * - ::UPTKResourceViewDesc::firstMipmapLevel specifies the most detailed mipmap level. This will be the new mipmap level zero.
 *   For non-mipmapped resources, this value has to be zero.::UPTKTextureDesc::minMipmapLevelClamp and ::UPTKTextureDesc::maxMipmapLevelClamp
 *   will be relative to this value. For ex., if the firstMipmapLevel is set to 2, and a minMipmapLevelClamp of 1.2 is specified,
 *   then the actual minimum mipmap level clamp will be 3.2.
 *
 * - ::UPTKResourceViewDesc::lastMipmapLevel specifies the least detailed mipmap level. For non-mipmapped resources, this value
 *   has to be zero.
 *
 * - ::UPTKResourceViewDesc::firstLayer specifies the first layer index for layered textures. This will be the new layer zero.
 *   For non-layered resources, this value has to be zero.
 *
 * - ::UPTKResourceViewDesc::lastLayer specifies the last layer index for layered textures. For non-layered resources, 
 *   this value has to be zero.
 *
 *
 * \param pTexObject   - Texture object to create
 * \param pResDesc     - Resource descriptor
 * \param pTexDesc     - Texture descriptor
 * \param pResViewDesc - Resource view descriptor
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKDestroyTextureObject,
 * ::cuTexObjectCreate
 */

extern __host__ UPTKError_t UPTKCreateTextureObject(UPTKTextureObject_t *pTexObject, const struct UPTKResourceDesc *pResDesc, const struct UPTKTextureDesc *pTexDesc, const struct UPTKResourceViewDesc *pResViewDesc);

/**
 * \brief Destroys a texture object
 *
 * Destroys the texture object specified by \p texObject.
 *
 * \param texObject - Texture object to destroy
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_init_rt
 * \note_callback
 * \note_destroy_ub
 *
 * \sa
 * ::UPTKCreateTextureObject,
 * ::cuTexObjectDestroy
 */
extern __host__ UPTKError_t UPTKDestroyTextureObject(UPTKTextureObject_t texObject);

/**
 * \brief Returns a texture object's resource descriptor
 *
 * Returns the resource descriptor for the texture object specified by \p texObject.
 *
 * \param pResDesc  - Resource descriptor
 * \param texObject - Texture object
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKCreateTextureObject,
 * ::cuTexObjectGetResourceDesc
 */
extern __host__ UPTKError_t UPTKGetTextureObjectResourceDesc(struct UPTKResourceDesc *pResDesc, UPTKTextureObject_t texObject);

/**
 * \brief Returns a texture object's texture descriptor
 *
 * Returns the texture descriptor for the texture object specified by \p texObject.
 *
 * \param pTexDesc  - Texture descriptor
 * \param texObject - Texture object
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKCreateTextureObject,
 * ::cuTexObjectGetTextureDesc
 */
extern __host__ UPTKError_t UPTKGetTextureObjectTextureDesc(struct UPTKTextureDesc *pTexDesc, UPTKTextureObject_t texObject);

/**
 * \brief Returns a texture object's resource view descriptor
 *
 * Returns the resource view descriptor for the texture object specified by \p texObject.
 * If no resource view was specified, ::UPTKErrorInvalidValue is returned.
 *
 * \param pResViewDesc - Resource view descriptor
 * \param texObject    - Texture object
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKCreateTextureObject,
 * ::cuTexObjectGetResourceViewDesc
 */
extern __host__ UPTKError_t UPTKGetTextureObjectResourceViewDesc(struct UPTKResourceViewDesc *pResViewDesc, UPTKTextureObject_t texObject);

/** @} */ /* END UPTKRT_TEXTURE_OBJECT */

/**
 * \defgroup UPTKRT_SURFACE_OBJECT Surface Object Management
 *
 * ___MANBRIEF___ surface object management functions of the UPTK runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the low level texture object management functions
 * of the UPTK runtime application programming interface. The surface object 
 * API is only supported on devices of compute capability 3.0 or higher.
 *
 * @{
 */

/**
 * \brief Creates a surface object
 *
 * Creates a surface object and returns it in \p pSurfObject. \p pResDesc describes
 * the data to perform surface load/stores on. ::UPTKResourceDesc::resType must be 
 * ::UPTKResourceTypeArray and  ::UPTKResourceDesc::res::array::array
 * must be set to a valid UPTK array handle.
 *
 * Surface objects are only supported on devices of compute capability 3.0 or higher.
 * Additionally, a surface object is an opaque value, and, as such, should only be
 * accessed through UPTK API calls.
 *
 * \param pSurfObject - Surface object to create
 * \param pResDesc    - Resource descriptor
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidChannelDescriptor,
 * ::UPTKErrorInvalidResourceHandle
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKDestroySurfaceObject,
 * ::cuSurfObjectCreate
 */

extern __host__ UPTKError_t UPTKCreateSurfaceObject(UPTKSurfaceObject_t *pSurfObject, const struct UPTKResourceDesc *pResDesc);

/**
 * \brief Destroys a surface object
 *
 * Destroys the surface object specified by \p surfObject.
 *
 * \param surfObject - Surface object to destroy
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_init_rt
 * \note_callback
 * \note_destroy_ub
 *
 * \sa
 * ::UPTKCreateSurfaceObject,
 * ::cuSurfObjectDestroy
 */
extern __host__ UPTKError_t UPTKDestroySurfaceObject(UPTKSurfaceObject_t surfObject);

/**
 * \brief Returns a surface object's resource descriptor
 * Returns the resource descriptor for the surface object specified by \p surfObject.
 *
 * \param pResDesc   - Resource descriptor
 * \param surfObject - Surface object
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKCreateSurfaceObject,
 * ::cuSurfObjectGetResourceDesc
 */
extern __host__ UPTKError_t UPTKGetSurfaceObjectResourceDesc(struct UPTKResourceDesc *pResDesc, UPTKSurfaceObject_t surfObject);

/** @} */ /* END UPTKRT_SURFACE_OBJECT */

/**
 * \defgroup UPTKRT__VERSION Version Management
 *
 * @{
 */

/**
 * \brief Returns the latest version of UPTK supported by the driver
 *
 * Returns in \p *driverVersion the latest version of UPTK supported by
 * the driver. The version is returned as (1000 &times; major + 10 &times; minor).
 * For example, UPTK 9.2 would be represented by 9020. If no driver is installed,
 * then 0 is returned as the driver version.
 *
 * This function automatically returns ::UPTKErrorInvalidValue
 * if \p driverVersion is NULL.
 *
 * \param driverVersion - Returns the UPTK driver version.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKRuntimeGetVersion,
 * ::cuDriverGetVersion
 */
extern __host__ UPTKError_t UPTKDriverGetVersion(int *driverVersion);

/**
 * \brief Returns the UPTK Runtime version
 *
 * Returns in \p *runtimeVersion the version number of the current UPTK
 * Runtime instance. The version is returned as
 * (1000 &times; major + 10 &times; minor). For example,
 * UPTK 9.2 would be represented by 9020.
 *
 * As of UPTK 12.0, this function no longer initializes UPTK. The purpose
 * of this API is solely to return a compile-time constant stating the
 * UPTK Toolkit version in the above format.
 *
 * This function automatically returns ::UPTKErrorInvalidValue if
 * the \p runtimeVersion argument is NULL.
 *
 * \param runtimeVersion - Returns the UPTK Runtime version.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKDriverGetVersion,
 * ::cuDriverGetVersion
 */
extern __host__ UPTKError_t UPTKRuntimeGetVersion(int *runtimeVersion);

/** @} */ /* END UPTKRT__VERSION */

/**
 * \defgroup UPTKRT_GRAPH Graph Management
 *
 * ___MANBRIEF___ graph management functions of the UPTK runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the graph management functions of UPTK
 * runtime application programming interface.
 *
 * @{
 */

/**
 * \brief Creates a graph
 *
 * Creates an empty graph, which is returned via \p pGraph.
 *
 * \param pGraph - Returns newly created graph
 * \param flags   - Graph creation flags, must be 0
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorMemoryAllocation
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddChildGraphNode,
 * ::UPTKGraphAddEmptyNode,
 * ::UPTKGraphAddKernelNode,
 * ::UPTKGraphAddHostNode,
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphAddMemsetNode,
 * ::UPTKGraphInstantiate,
 * ::UPTKGraphDestroy,
 * ::UPTKGraphGetNodes,
 * ::UPTKGraphGetRootNodes,
 * ::UPTKGraphGetEdges,
 * ::UPTKGraphClone
 */
extern __host__ UPTKError_t UPTKGraphCreate(UPTKGraph_t *pGraph, unsigned int flags);

/**
 * \brief Creates a kernel execution node and adds it to a graph
 *
 * Creates a new kernel execution node and adds it to \p graph with \p numDependencies
 * dependencies specified via \p pDependencies and arguments specified in \p pNodeParams.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p pGraphNode.
 *
 * The UPTKKernelNodeParams structure is defined as:
 *
 * \code
 *  struct UPTKKernelNodeParams
 *  {
 *      void* func;
 *      dim3 gridDim;
 *      dim3 blockDim;
 *      unsigned int sharedMemBytes;
 *      void **kernelParams;
 *      void **extra;
 *  };
 * \endcode
 *
 * When the graph is launched, the node will invoke kernel \p func on a (\p gridDim.x x
 * \p gridDim.y x \p gridDim.z) grid of blocks. Each block contains
 * (\p blockDim.x x \p blockDim.y x \p blockDim.z) threads.
 *
 * \p sharedMem sets the amount of dynamic shared memory that will be
 * available to each thread block.
 *
 * Kernel parameters to \p func can be specified in one of two ways:
 *
 * 1) Kernel parameters can be specified via \p kernelParams. If the kernel has N
 * parameters, then \p kernelParams needs to be an array of N pointers. Each pointer,
 * from \p kernelParams[0] to \p kernelParams[N-1], points to the region of memory from which the actual
 * parameter will be copied. The number of kernel parameters and their offsets and sizes do not need
 * to be specified as that information is retrieved directly from the kernel's image.
 *
 * 2) Kernel parameters can also be packaged by the application into a single buffer that is passed in
 * via \p extra. This places the burden on the application of knowing each kernel
 * parameter's size and alignment/padding within the buffer. The \p extra parameter exists
 * to allow this function to take additional less commonly used arguments. \p extra specifies
 * a list of names of extra settings and their corresponding values. Each extra setting name is
 * immediately followed by the corresponding value. The list must be terminated with either NULL or
 * CU_LAUNCH_PARAM_END.
 *
 * - ::CU_LAUNCH_PARAM_END, which indicates the end of the \p extra
 *   array;
 * - ::CU_LAUNCH_PARAM_BUFFER_POINTER, which specifies that the next
 *   value in \p extra will be a pointer to a buffer
 *   containing all the kernel parameters for launching kernel
 *   \p func;
 * - ::CU_LAUNCH_PARAM_BUFFER_SIZE, which specifies that the next
 *   value in \p extra will be a pointer to a size_t
 *   containing the size of the buffer specified with
 *   ::CU_LAUNCH_PARAM_BUFFER_POINTER;
 *
 * The error ::UPTKErrorInvalidValue will be returned if kernel parameters are specified with both
 * \p kernelParams and \p extra (i.e. both \p kernelParams and
 * \p extra are non-NULL).
 *
 * The \p kernelParams or \p extra array, as well as the argument values it points to,
 * are copied during this call.
 *
 * \note Kernels launched using graphs must not use texture and surface references. Reading or
 *       writing through any texture or surface reference is undefined behavior.
 *       This restriction does not apply to texture and surface objects.
 *
 * \param pGraphNode     - Returns newly created node
 * \param graph          - Graph to which to add the node
 * \param pDependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param pNodeParams      - Parameters for the GPU execution node
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidDeviceFunction
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddNode,
 * ::UPTKLaunchKernel,
 * ::UPTKGraphKernelNodeGetParams,
 * ::UPTKGraphKernelNodeSetParams,
 * ::UPTKGraphCreate,
 * ::UPTKGraphDestroyNode,
 * ::UPTKGraphAddChildGraphNode,
 * ::UPTKGraphAddEmptyNode,
 * ::UPTKGraphAddHostNode,
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphAddMemsetNode
 */
extern __host__ UPTKError_t UPTKGraphAddKernelNode(UPTKGraphNode_t *pGraphNode, UPTKGraph_t graph, const UPTKGraphNode_t *pDependencies, size_t numDependencies, const struct UPTKKernelNodeParams *pNodeParams);

/**
 * \brief Returns a kernel node's parameters
 *
 * Returns the parameters of kernel node \p node in \p pNodeParams.
 * The \p kernelParams or \p extra array returned in \p pNodeParams,
 * as well as the argument values it points to, are owned by the node.
 * This memory remains valid until the node is destroyed or its
 * parameters are modified, and should not be modified
 * directly. Use ::UPTKGraphKernelNodeSetParams to update the
 * parameters of this node.
 *
 * The params will contain either \p kernelParams or \p extra,
 * according to which of these was most recently set on the node.
 *
 * \param node        - Node to get the parameters for
 * \param pNodeParams - Pointer to return the parameters
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidDeviceFunction
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKLaunchKernel,
 * ::UPTKGraphAddKernelNode,
 * ::UPTKGraphKernelNodeSetParams
 */
extern __host__ UPTKError_t UPTKGraphKernelNodeGetParams(UPTKGraphNode_t node, struct UPTKKernelNodeParams *pNodeParams);

/**
 * \brief Sets a kernel node's parameters
 *
 * Sets the parameters of kernel node \p node to \p pNodeParams.
 *
 * \param node        - Node to set the parameters for
 * \param pNodeParams - Parameters to copy
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle,
 * ::UPTKErrorMemoryAllocation
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphNodeSetParams,
 * ::UPTKLaunchKernel,
 * ::UPTKGraphAddKernelNode,
 * ::UPTKGraphKernelNodeGetParams
 */
extern __host__ UPTKError_t UPTKGraphKernelNodeSetParams(UPTKGraphNode_t node, const struct UPTKKernelNodeParams *pNodeParams);

/**
 * \brief Copies attributes from source node to destination node.
 *
 * Copies attributes from source node \p src to destination node \p dst.
 * Both node must have the same context.
 *
 * \param[out] dst Destination node
 * \param[in] src Source node
 * For list of attributes see ::UPTKKernelNodeAttrID
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidContext
 * \notefnerr
 *
 * \sa
 * ::UPTKAccessPolicyWindow
 */
extern __host__ UPTKError_t UPTKGraphKernelNodeCopyAttributes(
        UPTKGraphNode_t hSrc,
        UPTKGraphNode_t hDst);

/**
 * \brief Queries node attribute.
 *
 * Queries attribute \p attr from node \p hNode and stores it in corresponding
 * member of \p value_out.
 *
 * \param[in] hNode
 * \param[in] attr
 * \param[out] value_out
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle
 * \notefnerr
 *
 * \sa
 * ::UPTKAccessPolicyWindow
 */
extern __host__ UPTKError_t UPTKGraphKernelNodeGetAttribute(
    UPTKGraphNode_t hNode,
    UPTKKernelNodeAttrID attr,
    UPTKKernelNodeAttrValue *value_out);

/**
 * \brief Sets node attribute.
 *
 * Sets attribute \p attr on node \p hNode from corresponding attribute of
 * \p value.
 *
 * \param[out] hNode
 * \param[in] attr
 * \param[out] value
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidResourceHandle
 * \notefnerr
 *
 * \sa
 * ::UPTKAccessPolicyWindow
 */
extern __host__ UPTKError_t UPTKGraphKernelNodeSetAttribute(
    UPTKGraphNode_t hNode,
    UPTKKernelNodeAttrID attr,
    const UPTKKernelNodeAttrValue *value);

/**
 * \brief Creates a memcpy node and adds it to a graph
 *
 * Creates a new memcpy node and adds it to \p graph with \p numDependencies
 * dependencies specified via \p pDependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p pGraphNode.
 *
 * When the graph is launched, the node will perform the memcpy described by \p pCopyParams.
 * See ::UPTKMemcpy3D() for a description of the structure and its restrictions.
 *
 * Memcpy nodes have some additional restrictions with regards to managed memory, if the
 * system contains at least one device which has a zero value for the device attribute
 * ::UPTKDevAttrConcurrentManagedAccess.
 *
 * \param pGraphNode     - Returns newly created node
 * \param graph          - Graph to which to add the node
 * \param pDependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param pCopyParams      - Parameters for the memory copy
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddNode,
 * ::UPTKMemcpy3D,
 * ::UPTKGraphAddMemcpyNodeToSymbol,
 * ::UPTKGraphAddMemcpyNodeFromSymbol,
 * ::UPTKGraphAddMemcpyNode1D,
 * ::UPTKGraphMemcpyNodeGetParams,
 * ::UPTKGraphMemcpyNodeSetParams,
 * ::UPTKGraphCreate,
 * ::UPTKGraphDestroyNode,
 * ::UPTKGraphAddChildGraphNode,
 * ::UPTKGraphAddEmptyNode,
 * ::UPTKGraphAddKernelNode,
 * ::UPTKGraphAddHostNode,
 * ::UPTKGraphAddMemsetNode
 */
extern __host__ UPTKError_t UPTKGraphAddMemcpyNode(UPTKGraphNode_t *pGraphNode, UPTKGraph_t graph, const UPTKGraphNode_t *pDependencies, size_t numDependencies, const struct UPTKMemcpy3DParms *pCopyParams);

/**
 * \brief Creates a memcpy node to copy to a symbol on the device and adds it to a graph
 *
 * Creates a new memcpy node to copy to \p symbol and adds it to \p graph with
 * \p numDependencies dependencies specified via \p pDependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p pGraphNode.
 *
 * When the graph is launched, the node will copy \p count bytes from the memory area
 * pointed to by \p src to the memory area pointed to by \p offset bytes from the start
 * of symbol \p symbol. The memory areas may not overlap. \p symbol is a variable that
 * resides in global or constant memory space. \p kind can be either
 * ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault.
 * Passing ::UPTKMemcpyDefault is recommended, in which case the type of
 * transfer is inferred from the pointer values. However, ::UPTKMemcpyDefault
 * is only allowed on systems that support unified virtual addressing.
 *
 * Memcpy nodes have some additional restrictions with regards to managed memory, if the
 * system contains at least one device which has a zero value for the device attribute
 * ::UPTKDevAttrConcurrentManagedAccess.
 *
 * \param pGraphNode      - Returns newly created node
 * \param graph           - Graph to which to add the node
 * \param pDependencies   - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param symbol          - Device symbol address
 * \param src             - Source memory address
 * \param count           - Size in bytes to copy
 * \param offset          - Offset from start of symbol in bytes
 * \param kind            - Type of transfer
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKMemcpyToSymbol,
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphAddMemcpyNodeFromSymbol,
 * ::UPTKGraphMemcpyNodeGetParams,
 * ::UPTKGraphMemcpyNodeSetParams,
 * ::UPTKGraphMemcpyNodeSetParamsToSymbol,
 * ::UPTKGraphMemcpyNodeSetParamsFromSymbol,
 * ::UPTKGraphCreate,
 * ::UPTKGraphDestroyNode,
 * ::UPTKGraphAddChildGraphNode,
 * ::UPTKGraphAddEmptyNode,
 * ::UPTKGraphAddKernelNode,
 * ::UPTKGraphAddHostNode,
 * ::UPTKGraphAddMemsetNode
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKGraphAddMemcpyNodeToSymbol(
    UPTKGraphNode_t *pGraphNode,
    UPTKGraph_t graph,
    const UPTKGraphNode_t *pDependencies,
    size_t numDependencies,
    const void* symbol,
    const void* src,
    size_t count,
    size_t offset,
    enum UPTKMemcpyKind kind);
#endif

/**
 * \brief Creates a memcpy node to copy from a symbol on the device and adds it to a graph
 *
 * Creates a new memcpy node to copy from \p symbol and adds it to \p graph with
 * \p numDependencies dependencies specified via \p pDependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p pGraphNode.
 *
 * When the graph is launched, the node will copy \p count bytes from the memory area
 * pointed to by \p offset bytes from the start of symbol \p symbol to the memory area
 *  pointed to by \p dst. The memory areas may not overlap. \p symbol is a variable
 *  that resides in global or constant memory space. \p kind can be either
 * ::UPTKMemcpyDeviceToHost, ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault.
 * Passing ::UPTKMemcpyDefault is recommended, in which case the type of transfer
 * is inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * Memcpy nodes have some additional restrictions with regards to managed memory, if the
 * system contains at least one device which has a zero value for the device attribute
 * ::UPTKDevAttrConcurrentManagedAccess.
 *
 * \param pGraphNode      - Returns newly created node
 * \param graph           - Graph to which to add the node
 * \param pDependencies   - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param dst             - Destination memory address
 * \param symbol          - Device symbol address
 * \param count           - Size in bytes to copy
 * \param offset          - Offset from start of symbol in bytes
 * \param kind            - Type of transfer
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKMemcpyFromSymbol,
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphAddMemcpyNodeToSymbol,
 * ::UPTKGraphMemcpyNodeGetParams,
 * ::UPTKGraphMemcpyNodeSetParams,
 * ::UPTKGraphMemcpyNodeSetParamsFromSymbol,
 * ::UPTKGraphMemcpyNodeSetParamsToSymbol,
 * ::UPTKGraphCreate,
 * ::UPTKGraphDestroyNode,
 * ::UPTKGraphAddChildGraphNode,
 * ::UPTKGraphAddEmptyNode,
 * ::UPTKGraphAddKernelNode,
 * ::UPTKGraphAddHostNode,
 * ::UPTKGraphAddMemsetNode
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKGraphAddMemcpyNodeFromSymbol(
    UPTKGraphNode_t* pGraphNode,
    UPTKGraph_t graph,
    const UPTKGraphNode_t* pDependencies,
    size_t numDependencies,
    void* dst,
    const void* symbol,
    size_t count,
    size_t offset,
    enum UPTKMemcpyKind kind);
#endif

/**
 * \brief Creates a 1D memcpy node and adds it to a graph
 *
 * Creates a new 1D memcpy node and adds it to \p graph with \p numDependencies
 * dependencies specified via \p pDependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p pGraphNode.
 *
 * When the graph is launched, the node will copy \p count bytes from the memory
 * area pointed to by \p src to the memory area pointed to by \p dst, where
 * \p kind specifies the direction of the copy, and must be one of
 * ::UPTKMemcpyHostToHost, ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToHost,
 * ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault. Passing
 * ::UPTKMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing. Launching a
 * memcpy node with dst and src pointers that do not match the direction of
 * the copy results in an undefined behavior.
 *
 * Memcpy nodes have some additional restrictions with regards to managed memory, if the
 * system contains at least one device which has a zero value for the device attribute
 * ::UPTKDevAttrConcurrentManagedAccess.
 *
 * \param pGraphNode      - Returns newly created node
 * \param graph           - Graph to which to add the node
 * \param pDependencies   - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param dst             - Destination memory address
 * \param src             - Source memory address
 * \param count           - Size in bytes to copy
 * \param kind            - Type of transfer
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKMemcpy,
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphMemcpyNodeGetParams,
 * ::UPTKGraphMemcpyNodeSetParams,
 * ::UPTKGraphMemcpyNodeSetParams1D,
 * ::UPTKGraphCreate,
 * ::UPTKGraphDestroyNode,
 * ::UPTKGraphAddChildGraphNode,
 * ::UPTKGraphAddEmptyNode,
 * ::UPTKGraphAddKernelNode,
 * ::UPTKGraphAddHostNode,
 * ::UPTKGraphAddMemsetNode
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKGraphAddMemcpyNode1D(
    UPTKGraphNode_t *pGraphNode,
    UPTKGraph_t graph,
    const UPTKGraphNode_t *pDependencies,
    size_t numDependencies,
    void* dst,
    const void* src,
    size_t count,
    enum UPTKMemcpyKind kind);
#endif

/**
 * \brief Returns a memcpy node's parameters
 *
 * Returns the parameters of memcpy node \p node in \p pNodeParams.
 *
 * \param node        - Node to get the parameters for
 * \param pNodeParams - Pointer to return the parameters
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKMemcpy3D,
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphMemcpyNodeSetParams
 */
extern __host__ UPTKError_t UPTKGraphMemcpyNodeGetParams(UPTKGraphNode_t node, struct UPTKMemcpy3DParms *pNodeParams);

/**
 * \brief Sets a memcpy node's parameters
 *
 * Sets the parameters of memcpy node \p node to \p pNodeParams.
 *
 * \param node        - Node to set the parameters for
 * \param pNodeParams - Parameters to copy
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphNodeSetParams,
 * ::UPTKMemcpy3D,
 * ::UPTKGraphMemcpyNodeSetParamsToSymbol,
 * ::UPTKGraphMemcpyNodeSetParamsFromSymbol,
 * ::UPTKGraphMemcpyNodeSetParams1D,
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphMemcpyNodeGetParams
 */
extern __host__ UPTKError_t UPTKGraphMemcpyNodeSetParams(UPTKGraphNode_t node, const struct UPTKMemcpy3DParms *pNodeParams);

/**
 * \brief Sets a memcpy node's parameters to copy to a symbol on the device
 *
 * Sets the parameters of memcpy node \p node to the copy described by the provided parameters.
 *
 * When the graph is launched, the node will copy \p count bytes from the memory area
 * pointed to by \p src to the memory area pointed to by \p offset bytes from the start
 * of symbol \p symbol. The memory areas may not overlap. \p symbol is a variable that
 * resides in global or constant memory space. \p kind can be either
 * ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault.
 * Passing ::UPTKMemcpyDefault is recommended, in which case the type of
 * transfer is inferred from the pointer values. However, ::UPTKMemcpyDefault
 * is only allowed on systems that support unified virtual addressing.
 *
 * \param node            - Node to set the parameters for
 * \param symbol          - Device symbol address
 * \param src             - Source memory address
 * \param count           - Size in bytes to copy
 * \param offset          - Offset from start of symbol in bytes
 * \param kind            - Type of transfer
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKMemcpyToSymbol,
 * ::UPTKGraphMemcpyNodeSetParams,
 * ::UPTKGraphMemcpyNodeSetParamsFromSymbol,
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphMemcpyNodeGetParams
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKGraphMemcpyNodeSetParamsToSymbol(
    UPTKGraphNode_t node,
    const void* symbol,
    const void* src,
    size_t count,
    size_t offset,
    enum UPTKMemcpyKind kind);
#endif

/**
 * \brief Sets a memcpy node's parameters to copy from a symbol on the device
 *
 * Sets the parameters of memcpy node \p node to the copy described by the provided parameters.
 *
 * When the graph is launched, the node will copy \p count bytes from the memory area
 * pointed to by \p offset bytes from the start of symbol \p symbol to the memory area
 *  pointed to by \p dst. The memory areas may not overlap. \p symbol is a variable
 *  that resides in global or constant memory space. \p kind can be either
 * ::UPTKMemcpyDeviceToHost, ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault.
 * Passing ::UPTKMemcpyDefault is recommended, in which case the type of transfer
 * is inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * \param node            - Node to set the parameters for
 * \param dst             - Destination memory address
 * \param symbol          - Device symbol address
 * \param count           - Size in bytes to copy
 * \param offset          - Offset from start of symbol in bytes
 * \param kind            - Type of transfer
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKMemcpyFromSymbol,
 * ::UPTKGraphMemcpyNodeSetParams,
 * ::UPTKGraphMemcpyNodeSetParamsToSymbol,
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphMemcpyNodeGetParams
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKGraphMemcpyNodeSetParamsFromSymbol(
    UPTKGraphNode_t node,
    void* dst,
    const void* symbol,
    size_t count,
    size_t offset,
    enum UPTKMemcpyKind kind);
#endif

/**
 * \brief Sets a memcpy node's parameters to perform a 1-dimensional copy
 *
 * Sets the parameters of memcpy node \p node to the copy described by the provided parameters.
 *
 * When the graph is launched, the node will copy \p count bytes from the memory
 * area pointed to by \p src to the memory area pointed to by \p dst, where
 * \p kind specifies the direction of the copy, and must be one of
 * ::UPTKMemcpyHostToHost, ::UPTKMemcpyHostToDevice, ::UPTKMemcpyDeviceToHost,
 * ::UPTKMemcpyDeviceToDevice, or ::UPTKMemcpyDefault. Passing
 * ::UPTKMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::UPTKMemcpyDefault is only
 * allowed on systems that support unified virtual addressing. Launching a
 * memcpy node with dst and src pointers that do not match the direction of
 * the copy results in an undefined behavior.
 *
 * \param node            - Node to set the parameters for
 * \param dst             - Destination memory address
 * \param src             - Source memory address
 * \param count           - Size in bytes to copy
 * \param kind            - Type of transfer
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKMemcpy,
 * ::UPTKGraphMemcpyNodeSetParams,
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphMemcpyNodeGetParams
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKGraphMemcpyNodeSetParams1D(
    UPTKGraphNode_t node,
    void* dst,
    const void* src,
    size_t count,
    enum UPTKMemcpyKind kind);
#endif

/**
 * \brief Creates a memset node and adds it to a graph
 *
 * Creates a new memset node and adds it to \p graph with \p numDependencies
 * dependencies specified via \p pDependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p pGraphNode.
 *
 * The element size must be 1, 2, or 4 bytes.
 * When the graph is launched, the node will perform the memset described by \p pMemsetParams.
 *
 * \param pGraphNode     - Returns newly created node
 * \param graph          - Graph to which to add the node
 * \param pDependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param pMemsetParams    - Parameters for the memory set
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidDevice
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddNode,
 * ::UPTKMemset2D,
 * ::UPTKGraphMemsetNodeGetParams,
 * ::UPTKGraphMemsetNodeSetParams,
 * ::UPTKGraphCreate,
 * ::UPTKGraphDestroyNode,
 * ::UPTKGraphAddChildGraphNode,
 * ::UPTKGraphAddEmptyNode,
 * ::UPTKGraphAddKernelNode,
 * ::UPTKGraphAddHostNode,
 * ::UPTKGraphAddMemcpyNode
 */
extern __host__ UPTKError_t UPTKGraphAddMemsetNode(UPTKGraphNode_t *pGraphNode, UPTKGraph_t graph, const UPTKGraphNode_t *pDependencies, size_t numDependencies, const struct UPTKMemsetParams *pMemsetParams);

/**
 * \brief Returns a memset node's parameters
 *
 * Returns the parameters of memset node \p node in \p pNodeParams.
 *
 * \param node        - Node to get the parameters for
 * \param pNodeParams - Pointer to return the parameters
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKMemset2D,
 * ::UPTKGraphAddMemsetNode,
 * ::UPTKGraphMemsetNodeSetParams
 */
extern __host__ UPTKError_t UPTKGraphMemsetNodeGetParams(UPTKGraphNode_t node, struct UPTKMemsetParams *pNodeParams);

/**
 * \brief Sets a memset node's parameters
 *
 * Sets the parameters of memset node \p node to \p pNodeParams.
 *
 * \param node        - Node to set the parameters for
 * \param pNodeParams - Parameters to copy
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphNodeSetParams,
 * ::UPTKMemset2D,
 * ::UPTKGraphAddMemsetNode,
 * ::UPTKGraphMemsetNodeGetParams
 */
extern __host__ UPTKError_t UPTKGraphMemsetNodeSetParams(UPTKGraphNode_t node, const struct UPTKMemsetParams *pNodeParams);

/**
 * \brief Creates a host execution node and adds it to a graph
 *
 * Creates a new CPU execution node and adds it to \p graph with \p numDependencies
 * dependencies specified via \p pDependencies and arguments specified in \p pNodeParams.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p pGraphNode.
 *
 * When the graph is launched, the node will invoke the specified CPU function.
 * Host nodes are not supported under MPS with pre-Volta GPUs.
 *
 * \param pGraphNode     - Returns newly created node
 * \param graph          - Graph to which to add the node
 * \param pDependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param pNodeParams      - Parameters for the host node
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorNotSupported,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddNode,
 * ::UPTKLaunchHostFunc,
 * ::UPTKGraphHostNodeGetParams,
 * ::UPTKGraphHostNodeSetParams,
 * ::UPTKGraphCreate,
 * ::UPTKGraphDestroyNode,
 * ::UPTKGraphAddChildGraphNode,
 * ::UPTKGraphAddEmptyNode,
 * ::UPTKGraphAddKernelNode,
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphAddMemsetNode
 */
extern __host__ UPTKError_t UPTKGraphAddHostNode(UPTKGraphNode_t *pGraphNode, UPTKGraph_t graph, const UPTKGraphNode_t *pDependencies, size_t numDependencies, const struct UPTKHostNodeParams *pNodeParams);

/**
 * \brief Returns a host node's parameters
 *
 * Returns the parameters of host node \p node in \p pNodeParams.
 *
 * \param node        - Node to get the parameters for
 * \param pNodeParams - Pointer to return the parameters
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKLaunchHostFunc,
 * ::UPTKGraphAddHostNode,
 * ::UPTKGraphHostNodeSetParams
 */
extern __host__ UPTKError_t UPTKGraphHostNodeGetParams(UPTKGraphNode_t node, struct UPTKHostNodeParams *pNodeParams);

/**
 * \brief Sets a host node's parameters
 *
 * Sets the parameters of host node \p node to \p nodeParams.
 *
 * \param node        - Node to set the parameters for
 * \param pNodeParams - Parameters to copy
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphNodeSetParams,
 * ::UPTKLaunchHostFunc,
 * ::UPTKGraphAddHostNode,
 * ::UPTKGraphHostNodeGetParams
 */
extern __host__ UPTKError_t UPTKGraphHostNodeSetParams(UPTKGraphNode_t node, const struct UPTKHostNodeParams *pNodeParams);

/**
 * \brief Creates a child graph node and adds it to a graph
 *
 * Creates a new node which executes an embedded graph, and adds it to \p graph with
 * \p numDependencies dependencies specified via \p pDependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p pGraphNode.
 *
 * If \p hGraph contains allocation or free nodes, this call will return an error.
 *
 * The node executes an embedded child graph. The child graph is cloned in this call.
 *
 * \param pGraphNode     - Returns newly created node
 * \param graph          - Graph to which to add the node
 * \param pDependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param childGraph      - The graph to clone into this node
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddNode,
 * ::UPTKGraphChildGraphNodeGetGraph,
 * ::UPTKGraphCreate,
 * ::UPTKGraphDestroyNode,
 * ::UPTKGraphAddEmptyNode,
 * ::UPTKGraphAddKernelNode,
 * ::UPTKGraphAddHostNode,
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphAddMemsetNode,
 * ::UPTKGraphClone
 */
extern __host__ UPTKError_t UPTKGraphAddChildGraphNode(UPTKGraphNode_t *pGraphNode, UPTKGraph_t graph, const UPTKGraphNode_t *pDependencies, size_t numDependencies, UPTKGraph_t childGraph);

/**
 * \brief Gets a handle to the embedded graph of a child graph node
 *
 * Gets a handle to the embedded graph in a child graph node. This call
 * does not clone the graph. Changes to the graph will be reflected in
 * the node, and the node retains ownersUPTK of the graph.
 *
 * Allocation and free nodes cannot be added to the returned graph.
 * Attempting to do so will return an error.
 *
 * \param node   - Node to get the embedded graph for
 * \param pGraph - Location to store a handle to the graph
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddChildGraphNode,
 * ::UPTKGraphNodeFindInClone
 */
extern __host__ UPTKError_t UPTKGraphChildGraphNodeGetGraph(UPTKGraphNode_t node, UPTKGraph_t *pGraph);

/**
 * \brief Creates an empty node and adds it to a graph
 *
 * Creates a new node which performs no operation, and adds it to \p graph with
 * \p numDependencies dependencies specified via \p pDependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p pGraphNode.
 *
 * An empty node performs no operation during execution, but can be used for
 * transitive ordering. For example, a phased execution graph with 2 groups of n
 * nodes with a barrier between them can be represented using an empty node and
 * 2*n dependency edges, rather than no empty node and n^2 dependency edges.
 *
 * \param pGraphNode     - Returns newly created node
 * \param graph          - Graph to which to add the node
 * \param pDependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddNode,
 * ::UPTKGraphCreate,
 * ::UPTKGraphDestroyNode,
 * ::UPTKGraphAddChildGraphNode,
 * ::UPTKGraphAddKernelNode,
 * ::UPTKGraphAddHostNode,
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphAddMemsetNode
 */
extern __host__ UPTKError_t UPTKGraphAddEmptyNode(UPTKGraphNode_t *pGraphNode, UPTKGraph_t graph, const UPTKGraphNode_t *pDependencies, size_t numDependencies);

/**
 * \brief Creates an event record node and adds it to a graph
 *
 * Creates a new event record node and adds it to \p hGraph with \p numDependencies
 * dependencies specified via \p dependencies and event specified in \p event.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p phGraphNode.
 *
 * Each launch of the graph will record \p event to capture execution of the
 * node's dependencies.
 *
 * These nodes may not be used in loops or conditionals.
 *
 * \param phGraphNode     - Returns newly created node
 * \param hGraph          - Graph to which to add the node
 * \param dependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param event           - Event for the node
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddNode,
 * ::UPTKGraphAddEventWaitNode,
 * ::UPTKEventRecordWithFlags,
 * ::UPTKStreamWaitEvent,
 * ::UPTKGraphCreate,
 * ::UPTKGraphDestroyNode,
 * ::UPTKGraphAddChildGraphNode,
 * ::UPTKGraphAddEmptyNode,
 * ::UPTKGraphAddKernelNode,
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphAddMemsetNode
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKGraphAddEventRecordNode(UPTKGraphNode_t *pGraphNode, UPTKGraph_t graph, const UPTKGraphNode_t *pDependencies, size_t numDependencies, UPTKEvent_t event);
#endif

/**
 * \brief Returns the event associated with an event record node
 *
 * Returns the event of event record node \p hNode in \p event_out.
 *
 * \param hNode     - Node to get the event for
 * \param event_out - Pointer to return the event
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddEventRecordNode,
 * ::UPTKGraphEventRecordNodeSetEvent,
 * ::UPTKGraphEventWaitNodeGetEvent,
 * ::UPTKEventRecordWithFlags,
 * ::UPTKStreamWaitEvent
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKGraphEventRecordNodeGetEvent(UPTKGraphNode_t node, UPTKEvent_t *event_out);
#endif

/**
 * \brief Sets an event record node's event
 *
 * Sets the event of event record node \p hNode to \p event.
 *
 * \param hNode - Node to set the event for
 * \param event - Event to use
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphNodeSetParams,
 * ::UPTKGraphAddEventRecordNode,
 * ::UPTKGraphEventRecordNodeGetEvent,
 * ::UPTKGraphEventWaitNodeSetEvent,
 * ::UPTKEventRecordWithFlags,
 * ::UPTKStreamWaitEvent
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKGraphEventRecordNodeSetEvent(UPTKGraphNode_t node, UPTKEvent_t event);
#endif

/**
 * \brief Creates an event wait node and adds it to a graph
 *
 * Creates a new event wait node and adds it to \p hGraph with \p numDependencies
 * dependencies specified via \p dependencies and event specified in \p event.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p phGraphNode.
 *
 * The graph node will wait for all work captured in \p event.  See ::cuEventRecord()
 * for details on what is captured by an event.  The synchronization will be performed
 * efficiently on the device when applicable.  \p event may be from a different context
 * or device than the launch stream.
 *
 * These nodes may not be used in loops or conditionals.
 *
 * \param phGraphNode     - Returns newly created node
 * \param hGraph          - Graph to which to add the node
 * \param dependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param event           - Event for the node
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddNode,
 * ::UPTKGraphAddEventRecordNode,
 * ::UPTKEventRecordWithFlags,
 * ::UPTKStreamWaitEvent,
 * ::UPTKGraphCreate,
 * ::UPTKGraphDestroyNode,
 * ::UPTKGraphAddChildGraphNode,
 * ::UPTKGraphAddEmptyNode,
 * ::UPTKGraphAddKernelNode,
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphAddMemsetNode
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKGraphAddEventWaitNode(UPTKGraphNode_t *pGraphNode, UPTKGraph_t graph, const UPTKGraphNode_t *pDependencies, size_t numDependencies, UPTKEvent_t event);
#endif

/**
 * \brief Returns the event associated with an event wait node
 *
 * Returns the event of event wait node \p hNode in \p event_out.
 *
 * \param hNode     - Node to get the event for
 * \param event_out - Pointer to return the event
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddEventWaitNode,
 * ::UPTKGraphEventWaitNodeSetEvent,
 * ::UPTKGraphEventRecordNodeGetEvent,
 * ::UPTKEventRecordWithFlags,
 * ::UPTKStreamWaitEvent
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKGraphEventWaitNodeGetEvent(UPTKGraphNode_t node, UPTKEvent_t *event_out);
#endif

/**
 * \brief Sets an event wait node's event
 *
 * Sets the event of event wait node \p hNode to \p event.
 *
 * \param hNode - Node to set the event for
 * \param event - Event to use
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphNodeSetParams,
 * ::UPTKGraphAddEventWaitNode,
 * ::UPTKGraphEventWaitNodeGetEvent,
 * ::UPTKGraphEventRecordNodeSetEvent,
 * ::UPTKEventRecordWithFlags,
 * ::UPTKStreamWaitEvent
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKGraphEventWaitNodeSetEvent(UPTKGraphNode_t node, UPTKEvent_t event);
#endif

/**
 * \brief Creates an external semaphore signal node and adds it to a graph
 *
 * Creates a new external semaphore signal node and adds it to \p graph with \p
 * numDependencies dependencies specified via \p dependencies and arguments specified
 * in \p nodeParams. It is possible for \p numDependencies to be 0, in which case the
 * node will be placed at the root of the graph. \p dependencies may not have any
 * duplicate entries. A handle to the new node will be returned in \p pGraphNode.
 *
 * Performs a signal operation on a set of externally allocated semaphore objects
 * when the node is launched.  The operation(s) will occur after all of the node's
 * dependencies have completed.
 *
 * \param pGraphNode      - Returns newly created node
 * \param graph           - Graph to which to add the node
 * \param pDependencies   - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param nodeParams      - Parameters for the node
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddNode,
 * ::UPTKGraphExternalSemaphoresSignalNodeGetParams,
 * ::UPTKGraphExternalSemaphoresSignalNodeSetParams,
 * ::UPTKGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::UPTKGraphAddExternalSemaphoresWaitNode,
 * ::UPTKImportExternalSemaphore,
 * ::UPTKSignalExternalSemaphoresAsync,
 * ::UPTKWaitExternalSemaphoresAsync,
 * ::UPTKGraphCreate,
 * ::UPTKGraphDestroyNode,
 * ::UPTKGraphAddEventRecordNode,
 * ::UPTKGraphAddEventWaitNode,
 * ::UPTKGraphAddChildGraphNode,
 * ::UPTKGraphAddEmptyNode,
 * ::UPTKGraphAddKernelNode,
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphAddMemsetNode
 */
#if __UPTKRT_API_VERSION >= 11020
extern __host__ UPTKError_t UPTKGraphAddExternalSemaphoresSignalNode(UPTKGraphNode_t *pGraphNode, UPTKGraph_t graph, const UPTKGraphNode_t *pDependencies, size_t numDependencies, const struct UPTKExternalSemaphoreSignalNodeParams *nodeParams);
#endif

/**
 * \brief Returns an external semaphore signal node's parameters
 *
 * Returns the parameters of an external semaphore signal node \p hNode in \p params_out.
 * The \p extSemArray and \p paramsArray returned in \p params_out,
 * are owned by the node.  This memory remains valid until the node is destroyed or its
 * parameters are modified, and should not be modified
 * directly. Use ::UPTKGraphExternalSemaphoresSignalNodeSetParams to update the
 * parameters of this node.
 *
 * \param hNode      - Node to get the parameters for
 * \param params_out - Pointer to return the parameters
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKLaunchKernel,
 * ::UPTKGraphAddExternalSemaphoresSignalNode,
 * ::UPTKGraphExternalSemaphoresSignalNodeSetParams,
 * ::UPTKGraphAddExternalSemaphoresWaitNode,
 * ::UPTKSignalExternalSemaphoresAsync,
 * ::UPTKWaitExternalSemaphoresAsync
 */
#if __UPTKRT_API_VERSION >= 11020
extern __host__ UPTKError_t UPTKGraphExternalSemaphoresSignalNodeGetParams(UPTKGraphNode_t hNode, struct UPTKExternalSemaphoreSignalNodeParams *params_out);
#endif

/**
 * \brief Sets an external semaphore signal node's parameters
 *
 * Sets the parameters of an external semaphore signal node \p hNode to \p nodeParams.
 *
 * \param hNode      - Node to set the parameters for
 * \param nodeParams - Parameters to copy
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphNodeSetParams,
 * ::UPTKGraphAddExternalSemaphoresSignalNode,
 * ::UPTKGraphExternalSemaphoresSignalNodeSetParams,
 * ::UPTKGraphAddExternalSemaphoresWaitNode,
 * ::UPTKSignalExternalSemaphoresAsync,
 * ::UPTKWaitExternalSemaphoresAsync
 */
#if __UPTKRT_API_VERSION >= 11020
extern __host__ UPTKError_t UPTKGraphExternalSemaphoresSignalNodeSetParams(UPTKGraphNode_t hNode, const struct UPTKExternalSemaphoreSignalNodeParams *nodeParams);
#endif

/**
 * \brief Creates an external semaphore wait node and adds it to a graph
 *
 * Creates a new external semaphore wait node and adds it to \p graph with \p numDependencies
 * dependencies specified via \p dependencies and arguments specified in \p nodeParams.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries. A handle
 * to the new node will be returned in \p pGraphNode.
 *
 * Performs a wait operation on a set of externally allocated semaphore objects
 * when the node is launched.  The node's dependencies will not be launched until
 * the wait operation has completed.
 *
 * \param pGraphNode      - Returns newly created node
 * \param graph           - Graph to which to add the node
 * \param pDependencies   - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param nodeParams      - Parameters for the node
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddNode,
 * ::UPTKGraphExternalSemaphoresWaitNodeGetParams,
 * ::UPTKGraphExternalSemaphoresWaitNodeSetParams,
 * ::UPTKGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::UPTKGraphAddExternalSemaphoresSignalNode,
 * ::UPTKImportExternalSemaphore,
 * ::UPTKSignalExternalSemaphoresAsync,
 * ::UPTKWaitExternalSemaphoresAsync,
 * ::UPTKGraphCreate,
 * ::UPTKGraphDestroyNode,
 * ::UPTKGraphAddEventRecordNode,
 * ::UPTKGraphAddEventWaitNode,
 * ::UPTKGraphAddChildGraphNode,
 * ::UPTKGraphAddEmptyNode,
 * ::UPTKGraphAddKernelNode,
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphAddMemsetNode
 */
#if __UPTKRT_API_VERSION >= 11020
extern __host__ UPTKError_t UPTKGraphAddExternalSemaphoresWaitNode(UPTKGraphNode_t *pGraphNode, UPTKGraph_t graph, const UPTKGraphNode_t *pDependencies, size_t numDependencies, const struct UPTKExternalSemaphoreWaitNodeParams *nodeParams);
#endif

/**
 * \brief Returns an external semaphore wait node's parameters
 *
 * Returns the parameters of an external semaphore wait node \p hNode in \p params_out.
 * The \p extSemArray and \p paramsArray returned in \p params_out,
 * are owned by the node.  This memory remains valid until the node is destroyed or its
 * parameters are modified, and should not be modified
 * directly. Use ::UPTKGraphExternalSemaphoresSignalNodeSetParams to update the
 * parameters of this node.
 *
 * \param hNode      - Node to get the parameters for
 * \param params_out - Pointer to return the parameters
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKLaunchKernel,
 * ::UPTKGraphAddExternalSemaphoresWaitNode,
 * ::UPTKGraphExternalSemaphoresWaitNodeSetParams,
 * ::UPTKGraphAddExternalSemaphoresWaitNode,
 * ::UPTKSignalExternalSemaphoresAsync,
 * ::UPTKWaitExternalSemaphoresAsync
 */
#if __UPTKRT_API_VERSION >= 11020
extern __host__ UPTKError_t UPTKGraphExternalSemaphoresWaitNodeGetParams(UPTKGraphNode_t hNode, struct UPTKExternalSemaphoreWaitNodeParams *params_out);
#endif

/**
 * \brief Sets an external semaphore wait node's parameters
 *
 * Sets the parameters of an external semaphore wait node \p hNode to \p nodeParams.
 *
 * \param hNode      - Node to set the parameters for
 * \param nodeParams - Parameters to copy
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphNodeSetParams,
 * ::UPTKGraphAddExternalSemaphoresWaitNode,
 * ::UPTKGraphExternalSemaphoresWaitNodeSetParams,
 * ::UPTKGraphAddExternalSemaphoresWaitNode,
 * ::UPTKSignalExternalSemaphoresAsync,
 * ::UPTKWaitExternalSemaphoresAsync
 */
#if __UPTKRT_API_VERSION >= 11020
extern __host__ UPTKError_t UPTKGraphExternalSemaphoresWaitNodeSetParams(UPTKGraphNode_t hNode, const struct UPTKExternalSemaphoreWaitNodeParams *nodeParams);
#endif

/**
 * \brief Creates an allocation node and adds it to a graph
 *
 * Creates a new allocation node and adds it to \p graph with \p numDependencies
 * dependencies specified via \p pDependencies and arguments specified in \p nodeParams.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries. A handle
 * to the new node will be returned in \p pGraphNode.
 *
 * \param pGraphNode      - Returns newly created node
 * \param graph           - Graph to which to add the node
 * \param pDependencies   - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param nodeParams      - Parameters for the node
 *
 * When ::UPTKGraphAddMemAllocNode creates an allocation node, it returns the address of the allocation in
 * \p nodeParams.dptr.  The allocation's address remains fixed across instantiations and launches.
 *
 * If the allocation is freed in the same graph, by creating a free node using ::UPTKGraphAddMemFreeNode,
 * the allocation can be accessed by nodes ordered after the allocation node but before the free node.
 * These allocations cannot be freed outside the owning graph, and they can only be freed once in the
 * owning graph.
 *
 * If the allocation is not freed in the same graph, then it can be accessed not only by nodes in the
 * graph which are ordered after the allocation node, but also by stream operations ordered after the
 * graph's execution but before the allocation is freed.
 *
 * Allocations which are not freed in the same graph can be freed by:
 * - passing the allocation to ::UPTKMemFreeAsync or ::UPTKMemFree;
 * - launching a graph with a free node for that allocation; or
 * - specifying ::UPTKGraphInstantiateFlagAutoFreeOnLaunch during instantiation, which makes
 *   each launch behave as though it called ::UPTKMemFreeAsync for every unfreed allocation.
 *
 * It is not possible to free an allocation in both the owning graph and another graph.  If the allocation
 * is freed in the same graph, a free node cannot be added to another graph.  If the allocation is freed
 * in another graph, a free node can no longer be added to the owning graph.
 *
 * The following restrictions apply to graphs which contain allocation and/or memory free nodes:
 * - Nodes and edges of the graph cannot be deleted.
 * - The graph cannot be used in a child node.
 * - Only one instantiation of the graph may exist at any point in time.
 * - The graph cannot be cloned.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorUPTKrtUnloading,
 * ::UPTKErrorInitializationError,
 * ::UPTKErrorNotSupported,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorOutOfMemory
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::UPTKGraphAddNode,
 * ::UPTKGraphAddMemFreeNode,
 * ::UPTKGraphMemAllocNodeGetParams,
 * ::UPTKDeviceGraphMemTrim,
 * ::UPTKDeviceGetGraphMemAttribute,
 * ::UPTKDeviceSetGraphMemAttribute,
 * ::UPTKMallocAsync,
 * ::UPTKFreeAsync,
 * ::UPTKGraphCreate,
 * ::UPTKGraphDestroyNode,
 * ::UPTKGraphAddChildGraphNode,
 * ::UPTKGraphAddEmptyNode,
 * ::UPTKGraphAddEventRecordNode,
 * ::UPTKGraphAddEventWaitNode,
 * ::UPTKGraphAddExternalSemaphoresSignalNode,
 * ::UPTKGraphAddExternalSemaphoresWaitNode,
 * ::UPTKGraphAddKernelNode,
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphAddMemsetNode
 */
#if __UPTKRT_API_VERSION >= 11040
extern __host__ UPTKError_t UPTKGraphAddMemAllocNode(UPTKGraphNode_t *pGraphNode, UPTKGraph_t graph, const UPTKGraphNode_t *pDependencies, size_t numDependencies, struct UPTKMemAllocNodeParams *nodeParams);
#endif

/**
 * \brief Returns a memory alloc node's parameters
 *
 * Returns the parameters of a memory alloc node \p hNode in \p params_out.
 * The \p poolProps and \p accessDescs returned in \p params_out, are owned by the
 * node.  This memory remains valid until the node is destroyed.  The returned
 * parameters must not be modified.
 *
 * \param node       - Node to get the parameters for
 * \param params_out - Pointer to return the parameters
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddMemAllocNode,
 * ::UPTKGraphMemFreeNodeGetParams
 */
#if __UPTKRT_API_VERSION >= 11040
extern __host__ UPTKError_t UPTKGraphMemAllocNodeGetParams(UPTKGraphNode_t node, struct UPTKMemAllocNodeParams *params_out);
#endif

/**
 * \brief Creates a memory free node and adds it to a graph
 *
 * Creates a new memory free node and adds it to \p graph with \p numDependencies
 * dependencies specified via \p pDependencies and address specified in \p dptr.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries. A handle
 * to the new node will be returned in \p pGraphNode.
 *
 * \param pGraphNode      - Returns newly created node
 * \param graph           - Graph to which to add the node
 * \param pDependencies   - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param dptr            - Address of memory to free
 *
 * ::UPTKGraphAddMemFreeNode will return ::UPTKErrorInvalidValue if the user attempts to free:
 * - an allocation twice in the same graph.
 * - an address that was not returned by an allocation node.
 * - an invalid address.
 *
 * The following restrictions apply to graphs which contain allocation and/or memory free nodes:
 * - Nodes and edges of the graph cannot be deleted.
 * - The graph cannot be used in a child node.
 * - Only one instantiation of the graph may exist at any point in time.
 * - The graph cannot be cloned.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorUPTKrtUnloading,
 * ::UPTKErrorInitializationError,
 * ::UPTKErrorNotSupported,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorOutOfMemory
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::UPTKGraphAddNode,
 * ::UPTKGraphAddMemAllocNode,
 * ::UPTKGraphMemFreeNodeGetParams,
 * ::UPTKDeviceGraphMemTrim,
 * ::UPTKDeviceGetGraphMemAttribute,
 * ::UPTKDeviceSetGraphMemAttribute,
 * ::UPTKMallocAsync,
 * ::UPTKFreeAsync,
 * ::UPTKGraphCreate,
 * ::UPTKGraphDestroyNode,
 * ::UPTKGraphAddChildGraphNode,
 * ::UPTKGraphAddEmptyNode,
 * ::UPTKGraphAddEventRecordNode,
 * ::UPTKGraphAddEventWaitNode,
 * ::UPTKGraphAddExternalSemaphoresSignalNode,
 * ::UPTKGraphAddExternalSemaphoresWaitNode,
 * ::UPTKGraphAddKernelNode,
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphAddMemsetNode
 */
#if __UPTKRT_API_VERSION >= 11040
extern __host__ UPTKError_t UPTKGraphAddMemFreeNode(UPTKGraphNode_t *pGraphNode, UPTKGraph_t graph, const UPTKGraphNode_t *pDependencies, size_t numDependencies, void *dptr);
#endif

/**
 * \brief Returns a memory free node's parameters
 *
 * Returns the address of a memory free node \p hNode in \p dptr_out.
 *
 * \param node     - Node to get the parameters for
 * \param dptr_out - Pointer to return the device address
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddMemFreeNode,
 * ::UPTKGraphMemFreeNodeGetParams
 */
#if __UPTKRT_API_VERSION >= 11040
extern __host__ UPTKError_t UPTKGraphMemFreeNodeGetParams(UPTKGraphNode_t node, void *dptr_out);
#endif

/**
 * \brief Free unused memory that was cached on the specified device for use with graphs back to the OS.
 *
 * Blocks which are not in use by a graph that is either currently executing or scheduled to execute are
 * freed back to the operating system.
 *
 * \param device - The device for which cached memory should be freed.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddMemAllocNode,
 * ::UPTKGraphAddMemFreeNode,
 * ::UPTKDeviceGetGraphMemAttribute,
 * ::UPTKDeviceSetGraphMemAttribute,
 * ::UPTKMallocAsync,
 * ::UPTKFreeAsync
 */
#if __UPTKRT_API_VERSION >= 11040
extern __host__ UPTKError_t UPTKDeviceGraphMemTrim(int device);
#endif

/**
 * \brief Query asynchronous allocation attributes related to graphs
 *
 * Valid attributes are:
 *
 * - ::UPTKGraphMemAttrUsedMemCurrent: Amount of memory, in bytes, currently associated with graphs
 * - ::UPTKGraphMemAttrUsedMemHigh: High watermark of memory, in bytes, associated with graphs since the
 *   last time it was reset.  High watermark can only be reset to zero.
 * - ::UPTKGraphMemAttrReservedMemCurrent: Amount of memory, in bytes, currently allocated for use by
 *   the UPTK graphs asynchronous allocator.
 * - ::UPTKGraphMemAttrReservedMemHigh: High watermark of memory, in bytes, currently allocated for use by
 *   the UPTK graphs asynchronous allocator.
 *
 * \param device - Specifies the scope of the query
 * \param attr - attribute to get
 * \param value - retrieved value
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDevice
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKDeviceSetGraphMemAttribute,
 * ::UPTKGraphAddMemAllocNode,
 * ::UPTKGraphAddMemFreeNode,
 * ::UPTKDeviceGraphMemTrim,
 * ::UPTKMallocAsync,
 * ::UPTKFreeAsync
 */
#if __UPTKRT_API_VERSION >= 11040
extern __host__ UPTKError_t UPTKDeviceGetGraphMemAttribute(int device, enum UPTKGraphMemAttributeType attr, void* value);
#endif

/**
 * \brief Set asynchronous allocation attributes related to graphs
 *
 * Valid attributes are:
 *
 * - ::UPTKGraphMemAttrUsedMemHigh: High watermark of memory, in bytes, associated with graphs since the
 *   last time it was reset.  High watermark can only be reset to zero.
 * - ::UPTKGraphMemAttrReservedMemHigh: High watermark of memory, in bytes, currently allocated for use by
 *   the UPTK graphs asynchronous allocator.
 *
 * \param device - Specifies the scope of the query
 * \param attr - attribute to get
 * \param value - pointer to value to set
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidDevice
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKDeviceGetGraphMemAttribute,
 * ::UPTKGraphAddMemAllocNode,
 * ::UPTKGraphAddMemFreeNode,
 * ::UPTKDeviceGraphMemTrim,
 * ::UPTKMallocAsync,
 * ::UPTKFreeAsync
 */
#if __UPTKRT_API_VERSION >= 11040
extern __host__ UPTKError_t UPTKDeviceSetGraphMemAttribute(int device, enum UPTKGraphMemAttributeType attr, void* value);
#endif

/**
 * \brief Clones a graph
 *
 * This function creates a copy of \p originalGraph and returns it in \p pGraphClone.
 * All parameters are copied into the cloned graph. The original graph may be modified 
 * after this call without affecting the clone.
 *
 * Child graph nodes in the original graph are recursively copied into the clone.
 *
 * \param pGraphClone  - Returns newly created cloned graph
 * \param originalGraph - Graph to clone
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorMemoryAllocation
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphCreate,
 * ::UPTKGraphNodeFindInClone
 */
extern __host__ UPTKError_t UPTKGraphClone(UPTKGraph_t *pGraphClone, UPTKGraph_t originalGraph);

/**
 * \brief Finds a cloned version of a node
 *
 * This function returns the node in \p clonedGraph corresponding to \p originalNode 
 * in the original graph.
 *
 * \p clonedGraph must have been cloned from \p originalGraph via ::UPTKGraphClone. 
 * \p originalNode must have been in \p originalGraph at the time of the call to 
 * ::UPTKGraphClone, and the corresponding cloned node in \p clonedGraph must not have 
 * been removed. The cloned node is then returned via \p pClonedNode.
 *
 * \param pNode  - Returns handle to the cloned node
 * \param originalNode - Handle to the original node
 * \param clonedGraph - Cloned graph to query
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphClone
 */
extern __host__ UPTKError_t UPTKGraphNodeFindInClone(UPTKGraphNode_t *pNode, UPTKGraphNode_t originalNode, UPTKGraph_t clonedGraph);

/**
 * \brief Returns a node's type
 *
 * Returns the node type of \p node in \p pType.
 *
 * \param node - Node to query
 * \param pType  - Pointer to return the node type
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphGetNodes,
 * ::UPTKGraphGetRootNodes,
 * ::UPTKGraphChildGraphNodeGetGraph,
 * ::UPTKGraphKernelNodeGetParams,
 * ::UPTKGraphKernelNodeSetParams,
 * ::UPTKGraphHostNodeGetParams,
 * ::UPTKGraphHostNodeSetParams,
 * ::UPTKGraphMemcpyNodeGetParams,
 * ::UPTKGraphMemcpyNodeSetParams,
 * ::UPTKGraphMemsetNodeGetParams,
 * ::UPTKGraphMemsetNodeSetParams
 */
extern __host__ UPTKError_t UPTKGraphNodeGetType(UPTKGraphNode_t node, enum UPTKGraphNodeType *pType);

/**
 * \brief Returns a graph's nodes
 *
 * Returns a list of \p graph's nodes. \p nodes may be NULL, in which case this
 * function will return the number of nodes in \p numNodes. Otherwise,
 * \p numNodes entries will be filled in. If \p numNodes is higher than the actual
 * number of nodes, the remaining entries in \p nodes will be set to NULL, and the
 * number of nodes actually obtained will be returned in \p numNodes.
 *
 * \param graph    - Graph to query
 * \param nodes    - Pointer to return the nodes
 * \param numNodes - See description
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphCreate,
 * ::UPTKGraphGetRootNodes,
 * ::UPTKGraphGetEdges,
 * ::UPTKGraphNodeGetType,
 * ::UPTKGraphNodeGetDependencies,
 * ::UPTKGraphNodeGetDependentNodes
 */
extern __host__ UPTKError_t UPTKGraphGetNodes(UPTKGraph_t graph, UPTKGraphNode_t *nodes, size_t *numNodes);

/**
 * \brief Returns a graph's root nodes
 *
 * Returns a list of \p graph's root nodes. \p pRootNodes may be NULL, in which case this
 * function will return the number of root nodes in \p pNumRootNodes. Otherwise,
 * \p pNumRootNodes entries will be filled in. If \p pNumRootNodes is higher than the actual
 * number of root nodes, the remaining entries in \p pRootNodes will be set to NULL, and the
 * number of nodes actually obtained will be returned in \p pNumRootNodes.
 *
 * \param graph       - Graph to query
 * \param pRootNodes    - Pointer to return the root nodes
 * \param pNumRootNodes - See description
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphCreate,
 * ::UPTKGraphGetNodes,
 * ::UPTKGraphGetEdges,
 * ::UPTKGraphNodeGetType,
 * ::UPTKGraphNodeGetDependencies,
 * ::UPTKGraphNodeGetDependentNodes
 */
extern __host__ UPTKError_t UPTKGraphGetRootNodes(UPTKGraph_t graph, UPTKGraphNode_t *pRootNodes, size_t *pNumRootNodes);

/**
 * \brief Returns a graph's dependency edges
 *
 * Returns a list of \p graph's dependency edges. Edges are returned via corresponding
 * indices in \p from and \p to; that is, the node in \p to[i] has a dependency on the
 * node in \p from[i]. \p from and \p to may both be NULL, in which
 * case this function only returns the number of edges in \p numEdges. Otherwise,
 * \p numEdges entries will be filled in. If \p numEdges is higher than the actual
 * number of edges, the remaining entries in \p from and \p to will be set to NULL, and
 * the number of edges actually returned will be written to \p numEdges.
 *
 * \param graph    - Graph to get the edges from
 * \param from     - Location to return edge endpoints
 * \param to       - Location to return edge endpoints
 * \param numEdges - See description
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphGetNodes,
 * ::UPTKGraphGetRootNodes,
 * ::UPTKGraphAddDependencies,
 * ::UPTKGraphRemoveDependencies,
 * ::UPTKGraphNodeGetDependencies,
 * ::UPTKGraphNodeGetDependentNodes
 */
extern __host__ UPTKError_t UPTKGraphGetEdges(UPTKGraph_t graph, UPTKGraphNode_t *from, UPTKGraphNode_t *to, size_t *numEdges);

/**
 * \brief Returns a graph's dependency edges (12.3+)
 *
 * Returns a list of \p graph's dependency edges. Edges are returned via corresponding
 * indices in \p from, \p to and \p edgeData; that is, the node in \p to[i] has a
 * dependency on the node in \p from[i] with data \p edgeData[i]. \p from and \p to may
 * both be NULL, in which case this function only returns the number of edges in
 * \p numEdges. Otherwise, \p numEdges entries will be filled in. If \p numEdges is higher
 * than the actual number of edges, the remaining entries in \p from and \p to will be
 * set to NULL, and the number of edges actually returned will be written to \p numEdges.
 * \p edgeData may alone be NULL, in which case the edges must all have default (zeroed)
 * edge data. Attempting a losst query via NULL \p edgeData will result in
 * ::UPTKErrorLossyQuery. If \p edgeData is non-NULL then \p from and \p to must be as
 * well.
 *
 * \param graph    - Graph to get the edges from
 * \param from     - Location to return edge endpoints
 * \param to       - Location to return edge endpoints
 * \param edgeData - Optional location to return edge data
 * \param numEdges - See description
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorLossyQuery,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphGetNodes,
 * ::UPTKGraphGetRootNodes,
 * ::UPTKGraphAddDependencies,
 * ::UPTKGraphRemoveDependencies,
 * ::UPTKGraphNodeGetDependencies,
 * ::UPTKGraphNodeGetDependentNodes
 */
extern __host__ UPTKError_t UPTKGraphGetEdges_v2(UPTKGraph_t graph, UPTKGraphNode_t *from, UPTKGraphNode_t *to, UPTKGraphEdgeData *edgeData, size_t *numEdges);

/**
 * \brief Returns a node's dependencies
 *
 * Returns a list of \p node's dependencies. \p pDependencies may be NULL, in which case this
 * function will return the number of dependencies in \p pNumDependencies. Otherwise,
 * \p pNumDependencies entries will be filled in. If \p pNumDependencies is higher than the actual
 * number of dependencies, the remaining entries in \p pDependencies will be set to NULL, and the
 * number of nodes actually obtained will be returned in \p pNumDependencies.
 *
 * \param node           - Node to query
 * \param pDependencies    - Pointer to return the dependencies
 * \param pNumDependencies - See description
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphNodeGetDependentNodes,
 * ::UPTKGraphGetNodes,
 * ::UPTKGraphGetRootNodes,
 * ::UPTKGraphGetEdges,
 * ::UPTKGraphAddDependencies,
 * ::UPTKGraphRemoveDependencies
 */
extern __host__ UPTKError_t UPTKGraphNodeGetDependencies(UPTKGraphNode_t node, UPTKGraphNode_t *pDependencies, size_t *pNumDependencies);

/**
 * \brief Returns a node's dependencies (12.3+)
 *
 * Returns a list of \p node's dependencies. \p pDependencies may be NULL, in which case this
 * function will return the number of dependencies in \p pNumDependencies. Otherwise,
 * \p pNumDependencies entries will be filled in. If \p pNumDependencies is higher than the actual
 * number of dependencies, the remaining entries in \p pDependencies will be set to NULL, and the
 * number of nodes actually obtained will be returned in \p pNumDependencies.
 *
 * Note that if an edge has non-zero (non-default) edge data and \p edgeData is NULL,
 * this API will return ::UPTKErrorLossyQuery. If \p edgeData is non-NULL, then
 * \p pDependencies must be as well.
 *
 * \param node             - Node to query
 * \param pDependencies    - Pointer to return the dependencies
 * \param edgeData         - Optional array to return edge data for each dependency
 * \param pNumDependencies - See description
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorLossyQuery,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphNodeGetDependentNodes,
 * ::UPTKGraphGetNodes,
 * ::UPTKGraphGetRootNodes,
 * ::UPTKGraphGetEdges,
 * ::UPTKGraphAddDependencies,
 * ::UPTKGraphRemoveDependencies
 */
extern __host__ UPTKError_t UPTKGraphNodeGetDependencies_v2(UPTKGraphNode_t node, UPTKGraphNode_t *pDependencies, UPTKGraphEdgeData *edgeData, size_t *pNumDependencies);

/**
 * \brief Returns a node's dependent nodes
 *
 * Returns a list of \p node's dependent nodes. \p pDependentNodes may be NULL, in which
 * case this function will return the number of dependent nodes in \p pNumDependentNodes.
 * Otherwise, \p pNumDependentNodes entries will be filled in. If \p pNumDependentNodes is
 * higher than the actual number of dependent nodes, the remaining entries in
 * \p pDependentNodes will be set to NULL, and the number of nodes actually obtained will
 * be returned in \p pNumDependentNodes.
 *
 * \param node             - Node to query
 * \param pDependentNodes    - Pointer to return the dependent nodes
 * \param pNumDependentNodes - See description
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphNodeGetDependencies,
 * ::UPTKGraphGetNodes,
 * ::UPTKGraphGetRootNodes,
 * ::UPTKGraphGetEdges,
 * ::UPTKGraphAddDependencies,
 * ::UPTKGraphRemoveDependencies
 */
extern __host__ UPTKError_t UPTKGraphNodeGetDependentNodes(UPTKGraphNode_t node, UPTKGraphNode_t *pDependentNodes, size_t *pNumDependentNodes);

/**
 * \brief Returns a node's dependent nodes (12.3+)
 *
 * Returns a list of \p node's dependent nodes. \p pDependentNodes may be NULL, in which
 * case this function will return the number of dependent nodes in \p pNumDependentNodes.
 * Otherwise, \p pNumDependentNodes entries will be filled in. If \p pNumDependentNodes is
 * higher than the actual number of dependent nodes, the remaining entries in
 * \p pDependentNodes will be set to NULL, and the number of nodes actually obtained will
 * be returned in \p pNumDependentNodes.
 *
 * Note that if an edge has non-zero (non-default) edge data and \p edgeData is NULL,
 * this API will return ::UPTKErrorLossyQuery. If \p edgeData is non-NULL, then
 * \p pDependentNodes must be as well.
 *
 * \param node               - Node to query
 * \param pDependentNodes    - Pointer to return the dependent nodes
 * \param edgeData           - Optional pointer to return edge data for dependent nodes
 * \param pNumDependentNodes - See description
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorLossyQuery,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphNodeGetDependencies,
 * ::UPTKGraphGetNodes,
 * ::UPTKGraphGetRootNodes,
 * ::UPTKGraphGetEdges,
 * ::UPTKGraphAddDependencies,
 * ::UPTKGraphRemoveDependencies
 */
extern __host__ UPTKError_t UPTKGraphNodeGetDependentNodes_v2(UPTKGraphNode_t node, UPTKGraphNode_t *pDependentNodes, UPTKGraphEdgeData *edgeData, size_t *pNumDependentNodes);

/**
 * \brief Adds dependency edges to a graph.
 *
 * The number of dependencies to be added is defined by \p numDependencies
 * Elements in \p pFrom and \p pTo at corresponding indices define a dependency.
 * Each node in \p pFrom and \p pTo must belong to \p graph.
 *
 * If \p numDependencies is 0, elements in \p pFrom and \p pTo will be ignored.
 * Specifying an existing dependency will return an error.
 *
 * \param graph - Graph to which dependencies are added
 * \param from - Array of nodes that provide the dependencies
 * \param to - Array of dependent nodes
 * \param numDependencies - Number of dependencies to be added
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphRemoveDependencies,
 * ::UPTKGraphGetEdges,
 * ::UPTKGraphNodeGetDependencies,
 * ::UPTKGraphNodeGetDependentNodes
 */
extern __host__ UPTKError_t UPTKGraphAddDependencies(UPTKGraph_t graph, const UPTKGraphNode_t *from, const UPTKGraphNode_t *to, size_t numDependencies);

/**
 * \brief Adds dependency edges to a graph. (12.3+)
 *
 * The number of dependencies to be added is defined by \p numDependencies
 * Elements in \p pFrom and \p pTo at corresponding indices define a dependency.
 * Each node in \p pFrom and \p pTo must belong to \p graph.
 *
 * If \p numDependencies is 0, elements in \p pFrom and \p pTo will be ignored.
 * Specifying an existing dependency will return an error.
 *
 * \param graph - Graph to which dependencies are added
 * \param from - Array of nodes that provide the dependencies
 * \param to - Array of dependent nodes
 * \param edgeData - Optional array of edge data. If NULL, default (zeroed) edge data is assumed.
 * \param numDependencies - Number of dependencies to be added
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphRemoveDependencies,
 * ::UPTKGraphGetEdges,
 * ::UPTKGraphNodeGetDependencies,
 * ::UPTKGraphNodeGetDependentNodes
 */
extern __host__ UPTKError_t UPTKGraphAddDependencies_v2(UPTKGraph_t graph, const UPTKGraphNode_t *from, const UPTKGraphNode_t *to, const UPTKGraphEdgeData *edgeData, size_t numDependencies);

/**
 * \brief Removes dependency edges from a graph.
 *
 * The number of \p pDependencies to be removed is defined by \p numDependencies.
 * Elements in \p pFrom and \p pTo at corresponding indices define a dependency.
 * Each node in \p pFrom and \p pTo must belong to \p graph.
 *
 * If \p numDependencies is 0, elements in \p pFrom and \p pTo will be ignored.
 * Specifying a non-existing dependency will return an error.
 *
 * \param graph - Graph from which to remove dependencies
 * \param from - Array of nodes that provide the dependencies
 * \param to - Array of dependent nodes
 * \param numDependencies - Number of dependencies to be removed
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddDependencies,
 * ::UPTKGraphGetEdges,
 * ::UPTKGraphNodeGetDependencies,
 * ::UPTKGraphNodeGetDependentNodes
 */
extern __host__ UPTKError_t UPTKGraphRemoveDependencies(UPTKGraph_t graph, const UPTKGraphNode_t *from, const UPTKGraphNode_t *to, size_t numDependencies);

/**
 * \brief Removes dependency edges from a graph. (12.3+)
 *
 * The number of \p pDependencies to be removed is defined by \p numDependencies.
 * Elements in \p pFrom and \p pTo at corresponding indices define a dependency.
 * Each node in \p pFrom and \p pTo must belong to \p graph.
 *
 * If \p numDependencies is 0, elements in \p pFrom and \p pTo will be ignored.
 * Specifying an edge that does not exist in the graph, with data matching
 * \p edgeData, results in an error. \p edgeData is nullable, which is equivalent
 * to passing default (zeroed) data for each edge.
 *
 * \param graph - Graph from which to remove dependencies
 * \param from - Array of nodes that provide the dependencies
 * \param to - Array of dependent nodes
 * \param edgeData - Optional array of edge data. If NULL, edge data is assumed to
 *                   be default (zeroed).
 * \param numDependencies - Number of dependencies to be removed
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddDependencies,
 * ::UPTKGraphGetEdges,
 * ::UPTKGraphNodeGetDependencies,
 * ::UPTKGraphNodeGetDependentNodes
 */
extern __host__ UPTKError_t UPTKGraphRemoveDependencies_v2(UPTKGraph_t graph, const UPTKGraphNode_t *from, const UPTKGraphNode_t *to, const UPTKGraphEdgeData *edgeData, size_t numDependencies);

/**
 * \brief Remove a node from the graph
 *
 * Removes \p node from its graph. This operation also severs any dependencies of other nodes 
 * on \p node and vice versa.
 *
 * Dependencies cannot be removed from graphs which contain allocation or free nodes.
 * Any attempt to do so will return an error.
 *
 * \param node  - Node to remove
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_destroy_ub
 *
 * \sa
 * ::UPTKGraphAddChildGraphNode,
 * ::UPTKGraphAddEmptyNode,
 * ::UPTKGraphAddKernelNode,
 * ::UPTKGraphAddHostNode,
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphAddMemsetNode
 */
extern __host__ UPTKError_t UPTKGraphDestroyNode(UPTKGraphNode_t node);

/**
 * \brief Creates an executable graph from a graph
 *
 * Instantiates \p graph as an executable graph. The graph is validated for any
 * structural constraints or intra-node constraints which were not previously
 * validated. If instantiation is successful, a handle to the instantiated graph
 * is returned in \p pGraphExec.
 *
 * The \p flags parameter controls the behavior of instantiation and subsequent
 * graph launches.  Valid flags are:
 *
 * - ::UPTKGraphInstantiateFlagAutoFreeOnLaunch, which configures a
 * graph containing memory allocation nodes to automatically free any
 * unfreed memory allocations before the graph is relaunched.
 *
 * - ::UPTKGraphInstantiateFlagDeviceLaunch, which configures the graph for launch
 * from the device. If this flag is passed, the executable graph handle returned can be
 * used to launch the graph from both the host and device. This flag cannot be used in
 * conjunction with ::UPTKGraphInstantiateFlagAutoFreeOnLaunch.
 *
 * - ::UPTKGraphInstantiateFlagUseNodePriority, which causes the graph
 * to use the priorities from the per-node attributes rather than the priority
 * of the launch stream during execution. Note that priorities are only available
 * on kernel nodes, and are copied from stream priority during stream capture.
 *
 * If \p graph contains any allocation or free nodes, there can be at most one
 * executable graph in existence for that graph at a time. An attempt to
 * instantiate a second executable graph before destroying the first with
 * ::UPTKGraphExecDestroy will result in an error.
 * The same also applies if \p graph contains any device-updatable kernel nodes.
 * 
 * Graphs instantiated for launch on the device have additional restrictions which do not
 * apply to host graphs:
 *
 * - The graph's nodes must reside on a single device.
 * - The graph can only contain kernel nodes, memcpy nodes, memset nodes, and child graph nodes.
 * - The graph cannot be empty and must contain at least one kernel, memcpy, or memset node.
 *   Operation-specific restrictions are outlined below.
 * - Kernel nodes:
 *   - Use of UPTK Dynamic Parallelism is not permitted.
 *   - Cooperative launches are permitted as long as MPS is not in use.
 * - Memcpy nodes:
 *   - Only copies involving device memory and/or pinned device-mapped host memory are permitted.
 *   - Copies involving UPTK arrays are not permitted.
 *   - Both operands must be accessible from the current device, and the current device must
 *     match the device of other nodes in the graph.
 *
 * If \p graph is not instantiated for launch on the device but contains kernels which
 * call device-side UPTKGraphLaunch() from multiple devices, this will result in an error.
 *
 * \param pGraphExec - Returns instantiated graph
 * \param graph      - Graph to instantiate
 * \param flags      - Flags to control instantiation.  See ::CUgraphInstantiate_flags.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphInstantiateWithFlags,
 * ::UPTKGraphCreate,
 * ::UPTKGraphUpload,
 * ::UPTKGraphLaunch,
 * ::UPTKGraphExecDestroy
 */
extern __host__ UPTKError_t UPTKGraphInstantiate(UPTKGraphExec_t *pGraphExec, UPTKGraph_t graph, unsigned long long flags __dv(0));

/**
 * \brief Creates an executable graph from a graph
 *
 * Instantiates \p graph as an executable graph. The graph is validated for any
 * structural constraints or intra-node constraints which were not previously
 * validated. If instantiation is successful, a handle to the instantiated graph
 * is returned in \p pGraphExec.
 *
 * The \p flags parameter controls the behavior of instantiation and subsequent
 * graph launches.  Valid flags are:
 *
 * - ::UPTKGraphInstantiateFlagAutoFreeOnLaunch, which configures a
 * graph containing memory allocation nodes to automatically free any
 * unfreed memory allocations before the graph is relaunched.
 *
 * - ::UPTKGraphInstantiateFlagDeviceLaunch, which configures the graph for launch
 * from the device. If this flag is passed, the executable graph handle returned can be
 * used to launch the graph from both the host and device. This flag can only be used
 * on platforms which support unified addressing. This flag cannot be used in
 * conjunction with ::UPTKGraphInstantiateFlagAutoFreeOnLaunch.
 *
 * - ::UPTKGraphInstantiateFlagUseNodePriority, which causes the graph
 * to use the priorities from the per-node attributes rather than the priority
 * of the launch stream during execution. Note that priorities are only available
 * on kernel nodes, and are copied from stream priority during stream capture.
 *
 * If \p graph contains any allocation or free nodes, there can be at most one
 * executable graph in existence for that graph at a time. An attempt to
 * instantiate a second executable graph before destroying the first with
 * ::UPTKGraphExecDestroy will result in an error.
 * The same also applies if \p graph contains any device-updatable kernel nodes.
 *
 * If \p graph contains kernels which call device-side UPTKGraphLaunch() from multiple
 * devices, this will result in an error.
 *
 * Graphs instantiated for launch on the device have additional restrictions which do not
 * apply to host graphs:
 *
 * - The graph's nodes must reside on a single device.
 * - The graph can only contain kernel nodes, memcpy nodes, memset nodes, and child graph nodes.
 * - The graph cannot be empty and must contain at least one kernel, memcpy, or memset node.
 *   Operation-specific restrictions are outlined below.
 * - Kernel nodes:
 *   - Use of UPTK Dynamic Parallelism is not permitted.
 *   - Cooperative launches are permitted as long as MPS is not in use.
 * - Memcpy nodes:
 *   - Only copies involving device memory and/or pinned device-mapped host memory are permitted.
 *   - Copies involving UPTK arrays are not permitted.
 *   - Both operands must be accessible from the current device, and the current device must
 *     match the device of other nodes in the graph.
 *
 * \param pGraphExec - Returns instantiated graph
 * \param graph      - Graph to instantiate
 * \param flags      - Flags to control instantiation.  See ::CUgraphInstantiate_flags.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphInstantiate,
 * ::UPTKGraphCreate,
 * ::UPTKGraphUpload,
 * ::UPTKGraphLaunch,
 * ::UPTKGraphExecDestroy
 */
#if __UPTKRT_API_VERSION >= 11040
extern __host__ UPTKError_t UPTKGraphInstantiateWithFlags(UPTKGraphExec_t *pGraphExec, UPTKGraph_t graph, unsigned long long flags __dv(0));
#endif

/**
 * \brief Creates an executable graph from a graph
 *
 * Instantiates \p graph as an executable graph according to the \p instantiateParams structure.
 * The graph is validated for any structural constraints or intra-node constraints
 * which were not previously validated. If instantiation is successful, a handle to
 * the instantiated graph is returned in \p pGraphExec.
 *
 * \p instantiateParams controls the behavior of instantiation and subsequent
 * graph launches, as well as returning more detailed information in the event of an error.
 * ::UPTKGraphInstantiateParams is defined as:
 *
 * \code
    typedef struct {
        unsigned long long flags;
        UPTKStream_t uploadStream;
        UPTKGraphNode_t errNode_out;
        UPTKGraphInstantiateResult result_out;
    } UPTKGraphInstantiateParams;
 * \endcode
 *
 * The \p flags field controls the behavior of instantiation and subsequent
 * graph launches. Valid flags are:
 *
 * - ::UPTKGraphInstantiateFlagAutoFreeOnLaunch, which configures a
 * graph containing memory allocation nodes to automatically free any
 * unfreed memory allocations before the graph is relaunched.
 *
 * - ::UPTKGraphInstantiateFlagUpload, which will perform an upload of the graph
 * into \p uploadStream once the graph has been instantiated.
 *
 * - ::UPTKGraphInstantiateFlagDeviceLaunch, which configures the graph for launch
 * from the device. If this flag is passed, the executable graph handle returned can be
 * used to launch the graph from both the host and device. This flag can only be used
 * on platforms which support unified addressing. This flag cannot be used in
 * conjunction with ::UPTKGraphInstantiateFlagAutoFreeOnLaunch.
 *
 * - ::UPTKGraphInstantiateFlagUseNodePriority, which causes the graph
 * to use the priorities from the per-node attributes rather than the priority
 * of the launch stream during execution. Note that priorities are only available
 * on kernel nodes, and are copied from stream priority during stream capture.
 *
 * If \p graph contains any allocation or free nodes, there can be at most one
 * executable graph in existence for that graph at a time. An attempt to instantiate a
 * second executable graph before destroying the first with ::UPTKGraphExecDestroy will
 * result in an error.
 * The same also applies if \p graph contains any device-updatable kernel nodes.
 *
 * If \p graph contains kernels which call device-side UPTKGraphLaunch() from multiple
 * devices, this will result in an error.
 *
 * Graphs instantiated for launch on the device have additional restrictions which do not
 * apply to host graphs:
 *
 * - The graph's nodes must reside on a single device.
 * - The graph can only contain kernel nodes, memcpy nodes, memset nodes, and child graph nodes.
 * - The graph cannot be empty and must contain at least one kernel, memcpy, or memset node.
 *   Operation-specific restrictions are outlined below.
 * - Kernel nodes:
 *   - Use of UPTK Dynamic Parallelism is not permitted.
 *   - Cooperative launches are permitted as long as MPS is not in use.
 * - Memcpy nodes:
 *   - Only copies involving device memory and/or pinned device-mapped host memory are permitted.
 *   - Copies involving UPTK arrays are not permitted.
 *   - Both operands must be accessible from the current device, and the current device must
 *     match the device of other nodes in the graph.
 *
 * In the event of an error, the \p result_out and \p errNode_out fields will contain more
 * information about the nature of the error. Possible error reporting includes:
 *
 * - ::UPTKGraphInstantiateError, if passed an invalid value or if an unexpected error occurred
 *   which is described by the return value of the function. \p errNode_out will be set to NULL.
 * - ::UPTKGraphInstantiateInvalidStructure, if the graph structure is invalid. \p errNode_out
 *   will be set to one of the offending nodes.
 * - ::UPTKGraphInstantiateNodeOperationNotSupported, if the graph is instantiated for device
 *   launch but contains a node of an unsupported node type, or a node which performs unsupported
 *   operations, such as use of UPTK dynamic parallelism within a kernel node. \p errNode_out will
 *   be set to this node.
 * - ::UPTKGraphInstantiateMultipleDevicesNotSupported, if the graph is instantiated for device
 *   launch but a node’s device differs from that of another node. This error can also be returned
 *   if a graph is not instantiated for device launch and it contains kernels which call device-side
 *   UPTKGraphLaunch() from multiple devices. \p errNode_out will be set to this node.
 *
 * If instantiation is successful, \p result_out will be set to ::UPTKGraphInstantiateSuccess,
 * and \p hErrNode_out will be set to NULL.
 *
 * \param pGraphExec       - Returns instantiated graph
 * \param graph            - Graph to instantiate
 * \param instantiateParams - Instantiation parameters
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphCreate,
 * ::UPTKGraphInstantiate,
 * ::UPTKGraphInstantiateWithFlags,
 * ::UPTKGraphExecDestroy
 */
extern __host__ UPTKError_t UPTKGraphInstantiateWithParams(UPTKGraphExec_t *pGraphExec, UPTKGraph_t graph, UPTKGraphInstantiateParams *instantiateParams);

/**
 * \brief Query the instantiation flags of an executable graph
 *
 * Returns the flags that were passed to instantiation for the given executable graph.
 * ::UPTKGraphInstantiateFlagUpload will not be returned by this API as it does
 * not affect the resulting executable graph.
 *
 * \param graphExec - The executable graph to query
 * \param flags     - Returns the instantiation flags
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphInstantiate,
 * ::UPTKGraphInstantiateWithFlags,
 * ::UPTKGraphInstantiateWithParams
 */
extern __host__ UPTKError_t UPTKGraphExecGetFlags(UPTKGraphExec_t graphExec, unsigned long long *flags);

/**
 * \brief Sets the parameters for a kernel node in the given graphExec
 *
 * Sets the parameters of a kernel node in an executable graph \p hGraphExec. 
 * The node is identified by the corresponding node \p node in the 
 * non-executable graph, from which the executable graph was instantiated. 
 *
 * \p node must not have been removed from the original graph. All \p nodeParams 
 * fields may change, but the following restrictions apply to \p func updates: 
 *
 *   - The owning device of the function cannot change.
 *   - A node whose function originally did not use UPTK dynamic parallelism cannot be updated
 *     to a function which uses CDP
 *   - A node whose function originally did not make device-side update calls cannot be updated
 *     to a function which makes device-side update calls.
 *   - If \p hGraphExec was not instantiated for device launch, a node whose function originally
 *     did not use device-side UPTKGraphLaunch() cannot be updated to a function which uses
 *     device-side UPTKGraphLaunch() unless the node resides on the same device as nodes which
 *     contained such calls at instantiate-time. If no such calls were present at instantiation,
 *     these updates cannot be performed at all.
 *
 * The modifications only affect future launches of \p hGraphExec. Already 
 * enqueued or running launches of \p hGraphExec are not affected by this call. 
 * \p node is also not modified by this call.
 *
 * If \p node is a device-updatable kernel node, the next upload/launch of \p hGraphExec
 * will overwrite any previous device-side updates. Additionally, applying host updates to a
 * device-updatable kernel node while it is being updated from the device will result in
 * undefined behavior.
 *
 * \param hGraphExec  - The executable graph in which to set the specified node
 * \param node        - kernel node from the graph from which graphExec was instantiated
 * \param pNodeParams - Updated Parameters to set
 * 
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphExecNodeSetParams,
 * ::UPTKGraphAddKernelNode,
 * ::UPTKGraphKernelNodeSetParams,
 * ::UPTKGraphExecMemcpyNodeSetParams,
 * ::UPTKGraphExecMemsetNodeSetParams,
 * ::UPTKGraphExecHostNodeSetParams,
 * ::UPTKGraphExecChildGraphNodeSetParams,
 * ::UPTKGraphExecEventRecordNodeSetEvent,
 * ::UPTKGraphExecEventWaitNodeSetEvent,
 * ::UPTKGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::UPTKGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::UPTKGraphExecUpdate,
 * ::UPTKGraphInstantiate
 */
extern __host__ UPTKError_t UPTKGraphExecKernelNodeSetParams(UPTKGraphExec_t hGraphExec, UPTKGraphNode_t node, const struct UPTKKernelNodeParams *pNodeParams);

/**
 * \brief Sets the parameters for a memcpy node in the given graphExec.
 *
 * Updates the work represented by \p node in \p hGraphExec as though \p node had 
 * contained \p pNodeParams at instantiation.  \p node must remain in the graph which was 
 * used to instantiate \p hGraphExec.  Changed edges to and from \p node are ignored.
 *
 * The source and destination memory in \p pNodeParams must be allocated from the same 
 * contexts as the original source and destination memory.  Both the instantiation-time 
 * memory operands and the memory operands in \p pNodeParams must be 1-dimensional.
 * Zero-length operations are not supported.
 *
 * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
 * or running launches of \p hGraphExec are not affected by this call.  \p node is also 
 * not modified by this call.
 *
 * Returns ::UPTKErrorInvalidValue if the memory operands' mappings changed or
 * either the original or new memory operands are multidimensional.
 *
 * \param hGraphExec  - The executable graph in which to set the specified node
 * \param node        - Memcpy node from the graph which was used to instantiate graphExec
 * \param pNodeParams - Updated Parameters to set
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphExecNodeSetParams,
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphMemcpyNodeSetParams,
 * ::UPTKGraphExecMemcpyNodeSetParamsToSymbol,
 * ::UPTKGraphExecMemcpyNodeSetParamsFromSymbol,
 * ::UPTKGraphExecMemcpyNodeSetParams1D,
 * ::UPTKGraphExecKernelNodeSetParams,
 * ::UPTKGraphExecMemsetNodeSetParams,
 * ::UPTKGraphExecHostNodeSetParams,
 * ::UPTKGraphExecChildGraphNodeSetParams,
 * ::UPTKGraphExecEventRecordNodeSetEvent,
 * ::UPTKGraphExecEventWaitNodeSetEvent,
 * ::UPTKGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::UPTKGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::UPTKGraphExecUpdate,
 * ::UPTKGraphInstantiate
 */
extern __host__ UPTKError_t UPTKGraphExecMemcpyNodeSetParams(UPTKGraphExec_t hGraphExec, UPTKGraphNode_t node, const struct UPTKMemcpy3DParms *pNodeParams);

/**
 * \brief Sets the parameters for a memcpy node in the given graphExec to copy to a symbol on the device
 *
 * Updates the work represented by \p node in \p hGraphExec as though \p node had 
 * contained the given params at instantiation.  \p node must remain in the graph which was 
 * used to instantiate \p hGraphExec.  Changed edges to and from \p node are ignored.
 *
 * \p src and \p symbol must be allocated from the same contexts as the original source and
 * destination memory.  The instantiation-time memory operands must be 1-dimensional.
 * Zero-length operations are not supported.
 *
 * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
 * or running launches of \p hGraphExec are not affected by this call.  \p node is also 
 * not modified by this call.
 *
 * Returns ::UPTKErrorInvalidValue if the memory operands' mappings changed or
 * the original memory operands are multidimensional.
 *
 * \param hGraphExec      - The executable graph in which to set the specified node
 * \param node            - Memcpy node from the graph which was used to instantiate graphExec
 * \param symbol          - Device symbol address
 * \param src             - Source memory address
 * \param count           - Size in bytes to copy
 * \param offset          - Offset from start of symbol in bytes
 * \param kind            - Type of transfer
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphAddMemcpyNodeToSymbol,
 * ::UPTKGraphMemcpyNodeSetParams,
 * ::UPTKGraphMemcpyNodeSetParamsToSymbol,
 * ::UPTKGraphExecMemcpyNodeSetParams,
 * ::UPTKGraphExecMemcpyNodeSetParamsFromSymbol,
 * ::UPTKGraphExecKernelNodeSetParams,
 * ::UPTKGraphExecMemsetNodeSetParams,
 * ::UPTKGraphExecHostNodeSetParams,
 * ::UPTKGraphExecChildGraphNodeSetParams,
 * ::UPTKGraphExecEventRecordNodeSetEvent,
 * ::UPTKGraphExecEventWaitNodeSetEvent,
 * ::UPTKGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::UPTKGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::UPTKGraphExecUpdate,
 * ::UPTKGraphInstantiate
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKGraphExecMemcpyNodeSetParamsToSymbol(
    UPTKGraphExec_t hGraphExec,
    UPTKGraphNode_t node,
    const void* symbol,
    const void* src,
    size_t count,
    size_t offset,
    enum UPTKMemcpyKind kind);
#endif

/**
 * \brief Sets the parameters for a memcpy node in the given graphExec to copy from a symbol on the device
 *
 * Updates the work represented by \p node in \p hGraphExec as though \p node had 
 * contained the given params at instantiation.  \p node must remain in the graph which was 
 * used to instantiate \p hGraphExec.  Changed edges to and from \p node are ignored.
 *
 * \p symbol and \p dst must be allocated from the same contexts as the original source and
 * destination memory.  The instantiation-time memory operands must be 1-dimensional.
 * Zero-length operations are not supported.
 *
 * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
 * or running launches of \p hGraphExec are not affected by this call.  \p node is also 
 * not modified by this call.
 *
 * Returns ::UPTKErrorInvalidValue if the memory operands' mappings changed or
 * the original memory operands are multidimensional.
 *
 * \param hGraphExec      - The executable graph in which to set the specified node
 * \param node            - Memcpy node from the graph which was used to instantiate graphExec
 * \param dst             - Destination memory address
 * \param symbol          - Device symbol address
 * \param count           - Size in bytes to copy
 * \param offset          - Offset from start of symbol in bytes
 * \param kind            - Type of transfer
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphAddMemcpyNodeFromSymbol,
 * ::UPTKGraphMemcpyNodeSetParams,
 * ::UPTKGraphMemcpyNodeSetParamsFromSymbol,
 * ::UPTKGraphExecMemcpyNodeSetParams,
 * ::UPTKGraphExecMemcpyNodeSetParamsToSymbol,
 * ::UPTKGraphExecKernelNodeSetParams,
 * ::UPTKGraphExecMemsetNodeSetParams,
 * ::UPTKGraphExecHostNodeSetParams,
 * ::UPTKGraphExecChildGraphNodeSetParams,
 * ::UPTKGraphExecEventRecordNodeSetEvent,
 * ::UPTKGraphExecEventWaitNodeSetEvent,
 * ::UPTKGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::UPTKGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::UPTKGraphExecUpdate,
 * ::UPTKGraphInstantiate
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKGraphExecMemcpyNodeSetParamsFromSymbol(
    UPTKGraphExec_t hGraphExec,
    UPTKGraphNode_t node,
    void* dst,
    const void* symbol,
    size_t count,
    size_t offset,
    enum UPTKMemcpyKind kind);
#endif

/**
 * \brief Sets the parameters for a memcpy node in the given graphExec to perform a 1-dimensional copy
 *
 * Updates the work represented by \p node in \p hGraphExec as though \p node had 
 * contained the given params at instantiation.  \p node must remain in the graph which was 
 * used to instantiate \p hGraphExec.  Changed edges to and from \p node are ignored.
 *
 * \p src and \p dst must be allocated from the same contexts as the original source
 * and destination memory.  The instantiation-time memory operands must be 1-dimensional.
 * Zero-length operations are not supported.
 *
 * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
 * or running launches of \p hGraphExec are not affected by this call.  \p node is also 
 * not modified by this call.
 *
 * Returns ::UPTKErrorInvalidValue if the memory operands' mappings changed or
 * the original memory operands are multidimensional.
 *
 * \param hGraphExec      - The executable graph in which to set the specified node
 * \param node            - Memcpy node from the graph which was used to instantiate graphExec
 * \param dst             - Destination memory address
 * \param src             - Source memory address
 * \param count           - Size in bytes to copy
 * \param kind            - Type of transfer
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddMemcpyNode,
 * ::UPTKGraphAddMemcpyNode1D,
 * ::UPTKGraphMemcpyNodeSetParams,
 * ::UPTKGraphMemcpyNodeSetParams1D,
 * ::UPTKGraphExecMemcpyNodeSetParams,
 * ::UPTKGraphExecKernelNodeSetParams,
 * ::UPTKGraphExecMemsetNodeSetParams,
 * ::UPTKGraphExecHostNodeSetParams,
 * ::UPTKGraphExecChildGraphNodeSetParams,
 * ::UPTKGraphExecEventRecordNodeSetEvent,
 * ::UPTKGraphExecEventWaitNodeSetEvent,
 * ::UPTKGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::UPTKGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::UPTKGraphExecUpdate,
 * ::UPTKGraphInstantiate
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKGraphExecMemcpyNodeSetParams1D(
    UPTKGraphExec_t hGraphExec,
    UPTKGraphNode_t node,
    void* dst,
    const void* src,
    size_t count,
    enum UPTKMemcpyKind kind);
#endif

/**
 * \brief Sets the parameters for a memset node in the given graphExec.
 *
 * Updates the work represented by \p node in \p hGraphExec as though \p node had 
 * contained \p pNodeParams at instantiation.  \p node must remain in the graph which was 
 * used to instantiate \p hGraphExec.  Changed edges to and from \p node are ignored.
 *
 * Zero sized operations are not supported.
 *
 * The new destination pointer in \p pNodeParams must be to the same kind of allocation
 * as the original destination pointer and have the same context association and device mapping
 * as the original destination pointer.
 *
 * Both the value and pointer address may be updated.  
 * Changing other aspects of the memset (width, height, element size or pitch) may cause the update to be rejected.
 * Specifically, for 2d memsets, all dimension changes are rejected.
 * For 1d memsets, changes in height are explicitly rejected and other changes are oportunistically allowed
 * if the resulting work maps onto the work resources already allocated for the node.

 * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
 * or running launches of \p hGraphExec are not affected by this call.  \p node is also 
 * not modified by this call.
 *
 * \param hGraphExec  - The executable graph in which to set the specified node
 * \param node        - Memset node from the graph which was used to instantiate graphExec
 * \param pNodeParams - Updated Parameters to set
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphExecNodeSetParams,
 * ::UPTKGraphAddMemsetNode,
 * ::UPTKGraphMemsetNodeSetParams,
 * ::UPTKGraphExecKernelNodeSetParams,
 * ::UPTKGraphExecMemcpyNodeSetParams,
 * ::UPTKGraphExecHostNodeSetParams,
 * ::UPTKGraphExecChildGraphNodeSetParams,
 * ::UPTKGraphExecEventRecordNodeSetEvent,
 * ::UPTKGraphExecEventWaitNodeSetEvent,
 * ::UPTKGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::UPTKGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::UPTKGraphExecUpdate,
 * ::UPTKGraphInstantiate
 */
extern __host__ UPTKError_t UPTKGraphExecMemsetNodeSetParams(UPTKGraphExec_t hGraphExec, UPTKGraphNode_t node, const struct UPTKMemsetParams *pNodeParams);

/**
 * \brief Sets the parameters for a host node in the given graphExec.
 *
 * Updates the work represented by \p node in \p hGraphExec as though \p node had 
 * contained \p pNodeParams at instantiation.  \p node must remain in the graph which was 
 * used to instantiate \p hGraphExec.  Changed edges to and from \p node are ignored.
 *
 * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
 * or running launches of \p hGraphExec are not affected by this call.  \p node is also 
 * not modified by this call.
 *
 * \param hGraphExec  - The executable graph in which to set the specified node
 * \param node        - Host node from the graph which was used to instantiate graphExec
 * \param pNodeParams - Updated Parameters to set
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphExecNodeSetParams,
 * ::UPTKGraphAddHostNode,
 * ::UPTKGraphHostNodeSetParams,
 * ::UPTKGraphExecKernelNodeSetParams,
 * ::UPTKGraphExecMemcpyNodeSetParams,
 * ::UPTKGraphExecMemsetNodeSetParams,
 * ::UPTKGraphExecChildGraphNodeSetParams,
 * ::UPTKGraphExecEventRecordNodeSetEvent,
 * ::UPTKGraphExecEventWaitNodeSetEvent,
 * ::UPTKGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::UPTKGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::UPTKGraphExecUpdate,
 * ::UPTKGraphInstantiate
 */
extern __host__ UPTKError_t UPTKGraphExecHostNodeSetParams(UPTKGraphExec_t hGraphExec, UPTKGraphNode_t node, const struct UPTKHostNodeParams *pNodeParams);

/**
 * \brief Updates node parameters in the child graph node in the given graphExec.
 *
 * Updates the work represented by \p node in \p hGraphExec as though the nodes contained
 * in \p node's graph had the parameters contained in \p childGraph's nodes at instantiation.
 * \p node must remain in the graph which was used to instantiate \p hGraphExec.
 * Changed edges to and from \p node are ignored.
 *
 * The modifications only affect future launches of \p hGraphExec.  Already enqueued
 * or running launches of \p hGraphExec are not affected by this call.  \p node is also
 * not modified by this call.
 *
 * The topology of \p childGraph, as well as the node insertion order,  must match that
 * of the graph contained in \p node.  See ::UPTKGraphExecUpdate() for a list of restrictions
 * on what can be updated in an instantiated graph.  The update is recursive, so child graph
 * nodes contained within the top level child graph will also be updated.

 * \param hGraphExec - The executable graph in which to set the specified node
 * \param node       - Host node from the graph which was used to instantiate graphExec
 * \param childGraph - The graph supplying the updated parameters
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphExecNodeSetParams,
 * ::UPTKGraphAddChildGraphNode,
 * ::UPTKGraphChildGraphNodeGetGraph,
 * ::UPTKGraphExecKernelNodeSetParams,
 * ::UPTKGraphExecMemcpyNodeSetParams,
 * ::UPTKGraphExecMemsetNodeSetParams,
 * ::UPTKGraphExecHostNodeSetParams,
 * ::UPTKGraphExecEventRecordNodeSetEvent,
 * ::UPTKGraphExecEventWaitNodeSetEvent,
 * ::UPTKGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::UPTKGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::UPTKGraphExecUpdate,
 * ::UPTKGraphInstantiate
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKGraphExecChildGraphNodeSetParams(UPTKGraphExec_t hGraphExec, UPTKGraphNode_t node, UPTKGraph_t childGraph);
#endif

/**
 * \brief Sets the event for an event record node in the given graphExec
 *
 * Sets the event of an event record node in an executable graph \p hGraphExec.
 * The node is identified by the corresponding node \p hNode in the
 * non-executable graph, from which the executable graph was instantiated.
 *
 * The modifications only affect future launches of \p hGraphExec. Already
 * enqueued or running launches of \p hGraphExec are not affected by this call.
 * \p hNode is also not modified by this call.
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - Event record node from the graph from which graphExec was instantiated
 * \param event      - Updated event to use
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphExecNodeSetParams,
 * ::UPTKGraphAddEventRecordNode,
 * ::UPTKGraphEventRecordNodeGetEvent,
 * ::UPTKGraphEventWaitNodeSetEvent,
 * ::UPTKEventRecordWithFlags,
 * ::UPTKStreamWaitEvent,
 * ::UPTKGraphExecKernelNodeSetParams,
 * ::UPTKGraphExecMemcpyNodeSetParams,
 * ::UPTKGraphExecMemsetNodeSetParams,
 * ::UPTKGraphExecHostNodeSetParams,
 * ::UPTKGraphExecChildGraphNodeSetParams,
 * ::UPTKGraphExecEventWaitNodeSetEvent,
 * ::UPTKGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::UPTKGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::UPTKGraphExecUpdate,
 * ::UPTKGraphInstantiate
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKGraphExecEventRecordNodeSetEvent(UPTKGraphExec_t hGraphExec, UPTKGraphNode_t hNode, UPTKEvent_t event);
#endif

/**
 * \brief Sets the event for an event wait node in the given graphExec
 *
 * Sets the event of an event wait node in an executable graph \p hGraphExec.
 * The node is identified by the corresponding node \p hNode in the
 * non-executable graph, from which the executable graph was instantiated.
 *
 * The modifications only affect future launches of \p hGraphExec. Already
 * enqueued or running launches of \p hGraphExec are not affected by this call.
 * \p hNode is also not modified by this call.
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - Event wait node from the graph from which graphExec was instantiated
 * \param event      - Updated event to use
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphExecNodeSetParams,
 * ::UPTKGraphAddEventWaitNode,
 * ::UPTKGraphEventWaitNodeGetEvent,
 * ::UPTKGraphEventRecordNodeSetEvent,
 * ::UPTKEventRecordWithFlags,
 * ::UPTKStreamWaitEvent,
 * ::UPTKGraphExecKernelNodeSetParams,
 * ::UPTKGraphExecMemcpyNodeSetParams,
 * ::UPTKGraphExecMemsetNodeSetParams,
 * ::UPTKGraphExecHostNodeSetParams,
 * ::UPTKGraphExecChildGraphNodeSetParams,
 * ::UPTKGraphExecEventRecordNodeSetEvent,
 * ::UPTKGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::UPTKGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::UPTKGraphExecUpdate,
 * ::UPTKGraphInstantiate
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKGraphExecEventWaitNodeSetEvent(UPTKGraphExec_t hGraphExec, UPTKGraphNode_t hNode, UPTKEvent_t event);
#endif

/**
 * \brief Sets the parameters for an external semaphore signal node in the given graphExec
 *
 * Sets the parameters of an external semaphore signal node in an executable graph \p hGraphExec.
 * The node is identified by the corresponding node \p hNode in the
 * non-executable graph, from which the executable graph was instantiated.
 *
 * \p hNode must not have been removed from the original graph.
 *
 * The modifications only affect future launches of \p hGraphExec. Already
 * enqueued or running launches of \p hGraphExec are not affected by this call.
 * \p hNode is also not modified by this call.
 *
 * Changing \p nodeParams->numExtSems is not supported.
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - semaphore signal node from the graph from which graphExec was instantiated
 * \param nodeParams - Updated Parameters to set
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphExecNodeSetParams,
 * ::UPTKGraphAddExternalSemaphoresSignalNode,
 * ::UPTKImportExternalSemaphore,
 * ::UPTKSignalExternalSemaphoresAsync,
 * ::UPTKWaitExternalSemaphoresAsync,
 * ::UPTKGraphExecKernelNodeSetParams,
 * ::UPTKGraphExecMemcpyNodeSetParams,
 * ::UPTKGraphExecMemsetNodeSetParams,
 * ::UPTKGraphExecHostNodeSetParams,
 * ::UPTKGraphExecChildGraphNodeSetParams,
 * ::UPTKGraphExecEventRecordNodeSetEvent,
 * ::UPTKGraphExecEventWaitNodeSetEvent,
 * ::UPTKGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::UPTKGraphExecUpdate,
 * ::UPTKGraphInstantiate
 */
#if __UPTKRT_API_VERSION >= 11020
extern __host__ UPTKError_t UPTKGraphExecExternalSemaphoresSignalNodeSetParams(UPTKGraphExec_t hGraphExec, UPTKGraphNode_t hNode, const struct UPTKExternalSemaphoreSignalNodeParams *nodeParams);
#endif

/**
 * \brief Sets the parameters for an external semaphore wait node in the given graphExec
 *
 * Sets the parameters of an external semaphore wait node in an executable graph \p hGraphExec.
 * The node is identified by the corresponding node \p hNode in the
 * non-executable graph, from which the executable graph was instantiated.
 *
 * \p hNode must not have been removed from the original graph.
 *
 * The modifications only affect future launches of \p hGraphExec. Already
 * enqueued or running launches of \p hGraphExec are not affected by this call.
 * \p hNode is also not modified by this call.
 *
 * Changing \p nodeParams->numExtSems is not supported.
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - semaphore wait node from the graph from which graphExec was instantiated
 * \param nodeParams - Updated Parameters to set
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphExecNodeSetParams,
 * ::UPTKGraphAddExternalSemaphoresWaitNode,
 * ::UPTKImportExternalSemaphore,
 * ::UPTKSignalExternalSemaphoresAsync,
 * ::UPTKWaitExternalSemaphoresAsync,
 * ::UPTKGraphExecKernelNodeSetParams,
 * ::UPTKGraphExecMemcpyNodeSetParams,
 * ::UPTKGraphExecMemsetNodeSetParams,
 * ::UPTKGraphExecHostNodeSetParams,
 * ::UPTKGraphExecChildGraphNodeSetParams,
 * ::UPTKGraphExecEventRecordNodeSetEvent,
 * ::UPTKGraphExecEventWaitNodeSetEvent,
 * ::UPTKGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::UPTKGraphExecUpdate,
 * ::UPTKGraphInstantiate
 */
#if __UPTKRT_API_VERSION >= 11020
extern __host__ UPTKError_t UPTKGraphExecExternalSemaphoresWaitNodeSetParams(UPTKGraphExec_t hGraphExec, UPTKGraphNode_t hNode, const struct UPTKExternalSemaphoreWaitNodeParams *nodeParams);
#endif

/**
 * \brief Enables or disables the specified node in the given graphExec
 *
 * Sets \p hNode to be either enabled or disabled. Disabled nodes are functionally equivalent 
 * to empty nodes until they are reenabled. Existing node parameters are not affected by 
 * disabling/enabling the node.
 *  
 * The node is identified by the corresponding node \p hNode in the non-executable 
 * graph, from which the executable graph was instantiated.   
 *
 * \p hNode must not have been removed from the original graph.
 *
 * The modifications only affect future launches of \p hGraphExec. Already
 * enqueued or running launches of \p hGraphExec are not affected by this call.
 * \p hNode is also not modified by this call.
 *
 * \note Currently only kernel, memset and memcpy nodes are supported. 
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - Node from the graph from which graphExec was instantiated
 * \param isEnabled  - Node is enabled if != 0, otherwise the node is disabled
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphNodeGetEnabled,
 * ::UPTKGraphExecUpdate,
 * ::UPTKGraphInstantiate
 * ::UPTKGraphLaunch
 */
#if __UPTKRT_API_VERSION >= 11060
extern __host__ UPTKError_t UPTKGraphNodeSetEnabled(UPTKGraphExec_t hGraphExec, UPTKGraphNode_t hNode, unsigned int isEnabled);
#endif

/**
 * \brief Query whether a node in the given graphExec is enabled
 *
 * Sets isEnabled to 1 if \p hNode is enabled, or 0 if \p hNode is disabled.
 *
 * The node is identified by the corresponding node \p hNode in the non-executable 
 * graph, from which the executable graph was instantiated.   
 *
 * \p hNode must not have been removed from the original graph.
 *
 * \note Currently only kernel, memset and memcpy nodes are supported. 
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - Node from the graph from which graphExec was instantiated
 * \param isEnabled  - Location to return the enabled status of the node
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphNodeSetEnabled,
 * ::UPTKGraphExecUpdate,
 * ::UPTKGraphInstantiate
 * ::UPTKGraphLaunch
 */
#if __UPTKRT_API_VERSION >= 11060
extern __host__ UPTKError_t UPTKGraphNodeGetEnabled(UPTKGraphExec_t hGraphExec, UPTKGraphNode_t hNode, unsigned int *isEnabled);
#endif

/**
 * \brief Check whether an executable graph can be updated with a graph and perform the update if possible
 *
 * Updates the node parameters in the instantiated graph specified by \p hGraphExec with the
 * node parameters in a topologically identical graph specified by \p hGraph.
 *
 * Limitations:
 *
 * - Kernel nodes:
 *   - The owning context of the function cannot change.
 *   - A node whose function originally did not use UPTK dynamic parallelism cannot be updated
 *     to a function which uses CDP.
 *   - A node whose function originally did not make device-side update calls cannot be updated
 *     to a function which makes device-side update calls.
 *   - A cooperative node cannot be updated to a non-cooperative node, and vice-versa.
 *   - If the graph was instantiated with UPTKGraphInstantiateFlagUseNodePriority, the
 *     priority attribute cannot change. Equality is checked on the originally requested
 *     priority values, before they are clamped to the device's supported range.
 *   - If \p hGraphExec was not instantiated for device launch, a node whose function originally
 *     did not use device-side UPTKGraphLaunch() cannot be updated to a function which uses
 *     device-side UPTKGraphLaunch() unless the node resides on the same device as nodes which
 *     contained such calls at instantiate-time. If no such calls were present at instantiation,
 *     these updates cannot be performed at all.
 *   - Neither \p hGraph nor \p hGraphExec may contain device-updatable kernel nodes.
 * - Memset and memcpy nodes:
 *   - The UPTK device(s) to which the operand(s) was allocated/mapped cannot change.
 *   - The source/destination memory must be allocated from the same contexts as the original
 *     source/destination memory.
 *   - For 2d memsets, only address and assinged value may be updated.
 *   - For 1d memsets, updating dimensions is also allowed, but may fail if the resulting operation doesn't
 *     map onto the work resources already allocated for the node. 
 * - Additional memcpy node restrictions:
 *   - Changing either the source or destination memory type(i.e. CU_MEMORYTYPE_DEVICE,
 *     CU_MEMORYTYPE_ARRAY, etc.) is not supported.
 * - Conditional nodes:
 *   - Changing node parameters is not supported.
 *   - Changeing parameters of nodes within the conditional body graph is subject to the rules above.
 *   - Conditional handle flags and default values are updated as part of the graph update.
 *
 * Note:  The API may add further restrictions in future releases.  The return code should always be checked.
 *
 * UPTKGraphExecUpdate sets the result member of \p resultInfo to UPTKGraphExecUpdateErrorTopologyChanged
 * under the following conditions:
 * - The count of nodes directly in \p hGraphExec and \p hGraph differ, in which case resultInfo->errorNode
 *   is set to NULL.
 * - \p hGraph has more exit nodes than \p hGraph, in which case resultInfo->errorNode is set to one of
 *   the exit nodes in hGraph. 
 * - A node in \p hGraph has a different number of dependencies than the node from \p hGraphExec it is paired with,
 *   in which case resultInfo->errorNode is set to the node from \p hGraph.
 * - A node in \p hGraph has a dependency that does not match with the corresponding dependency of the paired node
 *   from \p hGraphExec. resultInfo->errorNode will be set to the node from \p hGraph. resultInfo->errorFromNode
 *   will be set to the mismatched dependency. The dependencies are paired based on edge order and a dependency
 *   does not match when the nodes are already paired based on other edges examined in the graph.
 *
 * UPTKGraphExecUpdate sets \p the result member of \p resultInfo to:
 * - UPTKGraphExecUpdateError if passed an invalid value.
 * - UPTKGraphExecUpdateErrorTopologyChanged if the graph topology changed
 * - UPTKGraphExecUpdateErrorNodeTypeChanged if the type of a node changed, in which case
 *   \p hErrorNode_out is set to the node from \p hGraph.
 * - UPTKGraphExecUpdateErrorFunctionChanged if the function of a kernel node changed (UPTK driver < 11.2)
 * - UPTKGraphExecUpdateErrorUnsupportedFunctionChange if the func field of a kernel changed in an
 *   unsupported way(see note above), in which case \p hErrorNode_out is set to the node from \p hGraph
 * - UPTKGraphExecUpdateErrorParametersChanged if any parameters to a node changed in a way 
 *   that is not supported, in which case \p hErrorNode_out is set to the node from \p hGraph
 * - UPTKGraphExecUpdateErrorAttributesChanged if any attributes of a node changed in a way 
 *   that is not supported, in which case \p hErrorNode_out is set to the node from \p hGraph
 * - UPTKGraphExecUpdateErrorNotSupported if something about a node is unsupported, like 
 *   the node's type or configuration, in which case \p hErrorNode_out is set to the node from \p hGraph
 *
 * If the update fails for a reason not listed above, the result member of \p resultInfo will be set
 * to UPTKGraphExecUpdateError. If the update succeeds, the result member will be set to UPTKGraphExecUpdateSuccess.
 *
 * UPTKGraphExecUpdate returns UPTKSuccess when the updated was performed successfully.  It returns
 * UPTKErrorGraphExecUpdateFailure if the graph update was not performed because it included 
 * changes which violated constraints specific to instantiated graph update.
 *
 * \param hGraphExec The instantiated graph to be updated
 * \param hGraph The graph containing the updated parameters
   \param resultInfo the error info structure
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorGraphExecUpdateFailure,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphInstantiate
 */
extern __host__ UPTKError_t UPTKGraphExecUpdate(UPTKGraphExec_t hGraphExec, UPTKGraph_t hGraph, UPTKGraphExecUpdateResultInfo *resultInfo);

/**
 * \brief Uploads an executable graph in a stream
 *
 * Uploads \p hGraphExec to the device in \p hStream without executing it. Uploads of
 * the same \p hGraphExec will be serialized. Each upload is ordered behind both any
 * previous work in \p hStream and any previous launches of \p hGraphExec.
 * Uses memory cached by \p stream to back the allocations owned by \p graphExec.
 *
 * \param hGraphExec - Executable graph to upload
 * \param hStream    - Stream in which to upload the graph
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * \notefnerr
 * \note_init_rt
 *
 * \sa
 * ::UPTKGraphInstantiate,
 * ::UPTKGraphLaunch,
 * ::UPTKGraphExecDestroy
 */
#if __UPTKRT_API_VERSION >= 11010
 extern __host__ UPTKError_t UPTKGraphUpload(UPTKGraphExec_t graphExec, UPTKStream_t stream);
#endif

/**
 * \brief Launches an executable graph in a stream
 *
 * Executes \p graphExec in \p stream. Only one instance of \p graphExec may be executing
 * at a time. Each launch is ordered behind both any previous work in \p stream
 * and any previous launches of \p graphExec. To execute a graph concurrently, it must be
 * instantiated multiple times into multiple executable graphs.
 *
 * If any allocations created by \p graphExec remain unfreed (from a previous launch) and
 * \p graphExec was not instantiated with ::UPTKGraphInstantiateFlagAutoFreeOnLaunch,
 * the launch will fail with ::UPTKErrorInvalidValue.
 *
 * \param graphExec - Executable graph to launch
 * \param stream    - Stream in which to launch the graph
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphInstantiate,
 * ::UPTKGraphUpload,
 * ::UPTKGraphExecDestroy
 */
extern __host__ UPTKError_t UPTKGraphLaunch(UPTKGraphExec_t graphExec, UPTKStream_t stream);

/**
 * \brief Destroys an executable graph
 *
 * Destroys the executable graph specified by \p graphExec.
 *
 * \param graphExec - Executable graph to destroy
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_destroy_ub
 *
 * \sa
 * ::UPTKGraphInstantiate,
 * ::UPTKGraphUpload,
 * ::UPTKGraphLaunch
 */
extern __host__ UPTKError_t UPTKGraphExecDestroy(UPTKGraphExec_t graphExec);

/**
 * \brief Destroys a graph
 *
 * Destroys the graph specified by \p graph, as well as all of its nodes.
 *
 * \param graph - Graph to destroy
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_destroy_ub
 *
 * \sa
 * ::UPTKGraphCreate
 */
extern __host__ UPTKError_t UPTKGraphDestroy(UPTKGraph_t graph);

/**
 * \brief Write a DOT file describing graph structure
 *
 * Using the provided \p graph, write to \p path a DOT formatted description of the graph.
 * By default this includes the graph topology, node types, node id, kernel names and memcpy direction.
 * \p flags can be specified to write more detailed information about each node type such as
 * parameter values, kernel attributes, node and function handles.
 *
 * \param graph - The graph to create a DOT file from
 * \param path  - The path to write the DOT file to
 * \param flags - Flags from UPTKGraphDebugDotFlags for specifying which additional node information to write
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorOperatingSystem
 */
extern __host__ UPTKError_t UPTKGraphDebugDotPrint(UPTKGraph_t graph, const char *path, unsigned int flags);

/**
 * \brief Create a user object
 *
 * Create a user object with the specified destructor callback and initial reference count. The
 * initial references are owned by the caller.
 *
 * Destructor callbacks cannot make UPTK API calls and should avoid blocking behavior, as they
 * are executed by a shared internal thread. Another thread may be signaled to perform such
 * actions, if it does not block forward progress of tasks scheduled through UPTK.
 *
 * See UPTK User Objects in the UPTK C++ Programming Guide for more information on user objects.
 *
 * \param object_out      - Location to return the user object handle
 * \param ptr             - The pointer to pass to the destroy function
 * \param destroy         - Callback to free the user object when it is no longer in use
 * \param initialRefcount - The initial refcount to create the object with, typically 1. The
 *                          initial references are owned by the calling thread.
 * \param flags           - Currently it is required to pass ::UPTKUserObjectNoDestructorSync,
 *                          which is the only defined flag. This indicates that the destroy
 *                          callback cannot be waited on by any UPTK API. Users requiring
 *                          synchronization of the callback should signal its completion
 *                          manually.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 *
 * \sa
 * ::UPTKUserObjectRetain,
 * ::UPTKUserObjectRelease,
 * ::UPTKGraphRetainUserObject,
 * ::UPTKGraphReleaseUserObject,
 * ::UPTKGraphCreate
 */
extern __host__ UPTKError_t UPTKUserObjectCreate(UPTKUserObject_t *object_out, void *ptr, UPTKHostFn_t destroy, unsigned int initialRefcount, unsigned int flags);

/**
 * \brief Retain a reference to a user object
 *
 * Retains new references to a user object. The new references are owned by the caller.
 *
 * See UPTK User Objects in the UPTK C++ Programming Guide for more information on user objects.
 *
 * \param object - The object to retain
 * \param count  - The number of references to retain, typically 1. Must be nonzero
 *                 and not larger than INT_MAX.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 *
 * \sa
 * ::UPTKUserObjectCreate,
 * ::UPTKUserObjectRelease,
 * ::UPTKGraphRetainUserObject,
 * ::UPTKGraphReleaseUserObject,
 * ::UPTKGraphCreate
 */
extern __host__ UPTKError_t UPTKUserObjectRetain(UPTKUserObject_t object, unsigned int count __dv(1));

/**
 * \brief Release a reference to a user object
 *
 * Releases user object references owned by the caller. The object's destructor is invoked if
 * the reference count reaches zero.
 *
 * It is undefined behavior to release references not owned by the caller, or to use a user
 * object handle after all references are released.
 *
 * See UPTK User Objects in the UPTK C++ Programming Guide for more information on user objects.
 *
 * \param object - The object to release
 * \param count  - The number of references to release, typically 1. Must be nonzero
 *                 and not larger than INT_MAX.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 *
 * \sa
 * ::UPTKUserObjectCreate,
 * ::UPTKUserObjectRetain,
 * ::UPTKGraphRetainUserObject,
 * ::UPTKGraphReleaseUserObject,
 * ::UPTKGraphCreate
 */
extern __host__ UPTKError_t UPTKUserObjectRelease(UPTKUserObject_t object, unsigned int count __dv(1));

/**
 * \brief Retain a reference to a user object from a graph
 *
 * Creates or moves user object references that will be owned by a UPTK graph.
 *
 * See UPTK User Objects in the UPTK C++ Programming Guide for more information on user objects.
 *
 * \param graph  - The graph to associate the reference with
 * \param object - The user object to retain a reference for
 * \param count  - The number of references to add to the graph, typically 1. Must be
 *                 nonzero and not larger than INT_MAX.
 * \param flags  - The optional flag ::UPTKGraphUserObjectMove transfers references
 *                 from the calling thread, rather than create new references. Pass 0
 *                 to create new references.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 *
 * \sa
 * ::UPTKUserObjectCreate
 * ::UPTKUserObjectRetain,
 * ::UPTKUserObjectRelease,
 * ::UPTKGraphReleaseUserObject,
 * ::UPTKGraphCreate
 */
extern __host__ UPTKError_t UPTKGraphRetainUserObject(UPTKGraph_t graph, UPTKUserObject_t object, unsigned int count __dv(1), unsigned int flags __dv(0));

/**
 * \brief Release a user object reference from a graph
 *
 * Releases user object references owned by a graph.
 *
 * See UPTK User Objects in the UPTK C++ Programming Guide for more information on user objects.
 *
 * \param graph  - The graph that will release the reference
 * \param object - The user object to release a reference for
 * \param count  - The number of references to release, typically 1. Must be nonzero
 *                 and not larger than INT_MAX.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue
 *
 * \sa
 * ::UPTKUserObjectCreate
 * ::UPTKUserObjectRetain,
 * ::UPTKUserObjectRelease,
 * ::UPTKGraphRetainUserObject,
 * ::UPTKGraphCreate
 */
extern __host__ UPTKError_t UPTKGraphReleaseUserObject(UPTKGraph_t graph, UPTKUserObject_t object, unsigned int count __dv(1));

/**
 * \brief Adds a node of arbitrary type to a graph
 *
 * Creates a new node in \p graph described by \p nodeParams with \p numDependencies
 * dependencies specified via \p pDependencies. \p numDependencies may be 0.
 * \p pDependencies may be null if \p numDependencies is 0. \p pDependencies may not have
 * any duplicate entries.
 *
 * \p nodeParams is a tagged union. The node type should be specified in the \p type field,
 * and type-specific parameters in the corresponding union member. All unused bytes - that
 * is, \p reserved0 and all bytes past the utilized union member - must be set to zero.
 * It is recommended to use brace initialization or memset to ensure all bytes are
 * initialized.
 *
 * Note that for some node types, \p nodeParams may contain "out parameters" which are
 * modified during the call, such as \p nodeParams->alloc.dptr.
 *
 * A handle to the new node will be returned in \p phGraphNode.
 *
 * \param pGraphNode      - Returns newly created node
 * \param graph           - Graph to which to add the node
 * \param pDependencies   - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param nodeParams      - Specification of the node
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidDeviceFunction,
 * ::UPTKErrorNotSupported
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphCreate,
 * ::UPTKGraphNodeSetParams,
 * ::UPTKGraphExecNodeSetParams
 */
extern __host__ UPTKError_t UPTKGraphAddNode(UPTKGraphNode_t *pGraphNode, UPTKGraph_t graph, const UPTKGraphNode_t *pDependencies, size_t numDependencies, struct UPTKGraphNodeParams *nodeParams);

/**
 * \brief Adds a node of arbitrary type to a graph (12.3+)
 *
 * Creates a new node in \p graph described by \p nodeParams with \p numDependencies
 * dependencies specified via \p pDependencies. \p numDependencies may be 0.
 * \p pDependencies may be null if \p numDependencies is 0. \p pDependencies may not have
 * any duplicate entries.
 *
 * \p nodeParams is a tagged union. The node type should be specified in the \p type field,
 * and type-specific parameters in the corresponding union member. All unused bytes - that
 * is, \p reserved0 and all bytes past the utilized union member - must be set to zero.
 * It is recommended to use brace initialization or memset to ensure all bytes are
 * initialized.
 *
 * Note that for some node types, \p nodeParams may contain "out parameters" which are
 * modified during the call, such as \p nodeParams->alloc.dptr.
 *
 * A handle to the new node will be returned in \p phGraphNode.
 *
 * \param pGraphNode      - Returns newly created node
 * \param graph           - Graph to which to add the node
 * \param pDependencies   - Dependencies of the node
 * \param dependencyData  - Optional edge data for the dependencies. If NULL, the data is
 *                          assumed to be default (zeroed) for all dependencies.
 * \param numDependencies - Number of dependencies
 * \param nodeParams      - Specification of the node
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidDeviceFunction,
 * ::UPTKErrorNotSupported
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphCreate,
 * ::UPTKGraphNodeSetParams,
 * ::UPTKGraphExecNodeSetParams
 */
extern __host__ UPTKError_t UPTKGraphAddNode_v2(UPTKGraphNode_t *pGraphNode, UPTKGraph_t graph, const UPTKGraphNode_t *pDependencies, const UPTKGraphEdgeData *dependencyData, size_t numDependencies, struct UPTKGraphNodeParams *nodeParams);

/**
 * \brief Update's a graph node's parameters
 *
 * Sets the parameters of graph node \p node to \p nodeParams. The node type specified by
 * \p nodeParams->type must match the type of \p node. \p nodeParams must be fully
 * initialized and all unused bytes (reserved, padding) zeroed.
 *
 * Modifying parameters is not supported for node types UPTKGraphNodeTypeMemAlloc and
 * UPTKGraphNodeTypeMemFree.
 *
 * \param node       - Node to set the parameters for
 * \param nodeParams - Parameters to copy
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidDeviceFunction,
 * ::UPTKErrorNotSupported
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddNode,
 * ::UPTKGraphExecNodeSetParams
 */
extern __host__ UPTKError_t UPTKGraphNodeSetParams(UPTKGraphNode_t node, struct UPTKGraphNodeParams *nodeParams);

/**
 * \brief Update's a graph node's parameters in an instantiated graph
 *
 * Sets the parameters of a node in an executable graph \p graphExec. The node is identified
 * by the corresponding node \p node in the non-executable graph from which the executable
 * graph was instantiated. \p node must not have been removed from the original graph.
 *
 * The modifications only affect future launches of \p graphExec. Already
 * enqueued or running launches of \p graphExec are not affected by this call.
 * \p node is also not modified by this call.
 *
 * Allowed changes to parameters on executable graphs are as follows:
 * <table>
 *   <tr><th>Node type<th>Allowed changes
 *   <tr><td>kernel<td>See ::UPTKGraphExecKernelNodeSetParams
 *   <tr><td>memcpy<td>Addresses for 1-dimensional copies if allocated in same context; see ::UPTKGraphExecMemcpyNodeSetParams
 *   <tr><td>memset<td>Addresses for 1-dimensional memsets if allocated in same context; see ::UPTKGraphExecMemsetNodeSetParams
 *   <tr><td>host<td>Unrestricted
 *   <tr><td>child graph<td>Topology must match and restrictions apply recursively; see ::UPTKGraphExecUpdate
 *   <tr><td>event wait<td>Unrestricted
 *   <tr><td>event record<td>Unrestricted
 *   <tr><td>external semaphore signal<td>Number of semaphore operations cannot change
 *   <tr><td>external semaphore wait<td>Number of semaphore operations cannot change
 *   <tr><td>memory allocation<td>API unsupported
 *   <tr><td>memory free<td>API unsupported
 * </table>
 *
 * \param graphExec  - The executable graph in which to update the specified node
 * \param node       - Corresponding node from the graph from which graphExec was instantiated
 * \param nodeParams - Updated Parameters to set
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorInvalidDeviceFunction,
 * ::UPTKErrorNotSupported
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::UPTKGraphAddNode,
 * ::UPTKGraphNodeSetParams
 * ::UPTKGraphExecUpdate,
 * ::UPTKGraphInstantiate
 */
extern __host__ UPTKError_t UPTKGraphExecNodeSetParams(UPTKGraphExec_t graphExec, UPTKGraphNode_t node, struct UPTKGraphNodeParams *nodeParams);

/**
 * \brief Create a conditional handle
 *
 * Creates a conditional handle associated with \p hGraph.
 *
 * The conditional handle must be associated with a conditional node in this graph or one of its children.
 *  
 * Handles not associated with a conditional node may cause graph instantiation to fail. 
 *
 * \param pHandle_out        - Pointer used to return the handle to the caller.
 * \param hGraph             - Graph which will contain the conditional node using this handle.
 * \param defaultLaunchValue - Optional initial value for the conditional variable.
 * \param flags              - Currently must be UPTKGraphCondAssignDefault or 0.
 *
 * \return
 * ::UPTK_SUCCESS,
 * ::UPTK_ERROR_INVALID_VALUE,
 * ::UPTK_ERROR_NOT_SUPPORTED
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cuGraphAddNode,
 */
extern __host__ UPTKError_t UPTKGraphConditionalHandleCreate(UPTKGraphConditionalHandle *pHandle_out, UPTKGraph_t graph, unsigned int defaultLaunchValue __dv(0), unsigned int flags __dv(0));

/** @} */ /* END UPTKRT_GRAPH */

/**
 * \defgroup UPTKRT_DRIVER_ENTRY_POINT Driver Entry Point Access
 *
 * ___MANBRIEF___ driver entry point access functions of the UPTK runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the driver entry point access functions of UPTK
 * runtime application programming interface.
 *
 * @{
 */

/**
 * \brief Returns the requested driver API function pointer
 *
 * Returns in \p **funcPtr the address of the UPTK driver function for the requested flags.
 *
 * For a requested driver symbol, if the UPTK version in which the driver symbol was
 * introduced is less than or equal to the UPTK runtime version, the API will return
 * the function pointer to the corresponding versioned driver function.
 *
 * The pointer returned by the API should be cast to a function pointer matching the
 * requested driver function's definition in the API header file. The function pointer
 * typedef can be picked up from the corresponding typedefs header file. For example,
 * UPTKTypedefs.h consists of function pointer typedefs for driver APIs defined in UPTK.h.
 *
 * The API will return ::UPTKSuccess and set the returned \p funcPtr if the
 * requested driver function is valid and supported on the platform.
 *
 * The API will return ::UPTKSuccess and set the returned \p funcPtr to NULL if the
 * requested driver function is not supported on the platform, no ABI
 * compatible driver function exists for the UPTK runtime version or if the
 * driver symbol is invalid.
 *
 * It will also set the optional \p driverStatus to one of the values in 
 * ::UPTKDriverEntryPointQueryResult with the following meanings:
 * - ::UPTKDriverEntryPointSuccess - The requested symbol was succesfully found based
 *   on input arguments and \p pfn is valid
 * - ::UPTKDriverEntryPointSymbolNotFound - The requested symbol was not found
 * - ::UPTKDriverEntryPointVersionNotSufficent - The requested symbol was found but is
 *   not supported by the current runtime version (UPTKRT_VERSION)
 *
 * The requested flags can be:
 * - ::UPTKEnableDefault: This is the default mode. This is equivalent to
 *   ::UPTKEnablePerThreadDefaultStream if the code is compiled with
 *   --default-stream per-thread compilation flag or the macro UPTK_API_PER_THREAD_DEFAULT_STREAM
 *   is defined; ::UPTKEnableLegacyStream otherwise.
 * - ::UPTKEnableLegacyStream: This will enable the search for all driver symbols
 *   that match the requested driver symbol name except the corresponding per-thread versions.
 * - ::UPTKEnablePerThreadDefaultStream: This will enable the search for all
 *   driver symbols that match the requested driver symbol name including the per-thread
 *   versions. If a per-thread version is not found, the API will return the legacy version
 *   of the driver function.
 *
 * \param symbol - The base name of the driver API function to look for. As an example,
 *                 for the driver API ::cuMemAlloc_v2, \p symbol would be cuMemAlloc.
 *                 Note that the API will use the UPTK runtime version to return the
 *                 address to the most recent ABI compatible driver symbol, ::cuMemAlloc
 *                 or ::cuMemAlloc_v2.
 * \param funcPtr - Location to return the function pointer to the requested driver function
 * \param flags -  Flags to specify search options.
 * \param driverStatus - Optional location to store the status of finding the symbol from
 *                       the driver. See ::UPTKDriverEntryPointQueryResult for 
 *                       possible values.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorNotSupported
 * \note_version_mixing
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cuGetProcAddress
 */
#if defined(__cplusplus)
extern __host__ UPTKError_t UPTKGetDriverEntryPoint(const char *symbol, void **funcPtr, unsigned long long flags, enum UPTKDriverEntryPointQueryResult *driverStatus = NULL);
#else
extern __host__ UPTKError_t UPTKGetDriverEntryPoint(const char *symbol, void **funcPtr, unsigned long long flags, enum UPTKDriverEntryPointQueryResult *driverStatus);
#endif

/**
 * \brief Returns the requested driver API function pointer by UPTK version
 *
 * Returns in \p **funcPtr the address of the UPTK driver function for the requested flags and UPTK driver version.
 *
 * The UPTK version is specified as (1000 * major + 10 * minor), so UPTK 11.2
 * should be specified as 11020. For a requested driver symbol, if the specified
 * UPTK version is greater than or equal to the UPTK version in which the driver symbol
 * was introduced, this API will return the function pointer to the corresponding
 * versioned function.
 *
 * The pointer returned by the API should be cast to a function pointer matching the
 * requested driver function's definition in the API header file. The function pointer
 * typedef can be picked up from the corresponding typedefs header file. For example,
 * UPTKTypedefs.h consists of function pointer typedefs for driver APIs defined in UPTK.h.
 *
 * For the case where the UPTK version requested is greater than the UPTK Toolkit 
 * installed, there may not be an appropriate function pointer typedef in the
 * corresponding header file and may need a custom typedef to match the driver
 * function signature returned. This can be done by getting the typedefs from a later
 * toolkit or creating appropriately matching custom function typedefs.
 *
 * The API will return ::UPTKSuccess and set the returned \p funcPtr if the
 * requested driver function is valid and supported on the platform.
 *
 * The API will return ::UPTKSuccess and set the returned \p funcPtr to NULL if the
 * requested driver function is not supported on the platform, no ABI
 * compatible driver function exists for the requested version or if the
 * driver symbol is invalid.
 *
 * It will also set the optional \p driverStatus to one of the values in 
 * ::UPTKDriverEntryPointQueryResult with the following meanings:
 * - ::UPTKDriverEntryPointSuccess - The requested symbol was succesfully found based
 *   on input arguments and \p pfn is valid
 * - ::UPTKDriverEntryPointSymbolNotFound - The requested symbol was not found
 * - ::UPTKDriverEntryPointVersionNotSufficent - The requested symbol was found but is
 *   not supported by the specified version \p UPTKVersion
 *
 * The requested flags can be:
 * - ::UPTKEnableDefault: This is the default mode. This is equivalent to
 *   ::UPTKEnablePerThreadDefaultStream if the code is compiled with
 *   --default-stream per-thread compilation flag or the macro UPTK_API_PER_THREAD_DEFAULT_STREAM
 *   is defined; ::UPTKEnableLegacyStream otherwise.
 * - ::UPTKEnableLegacyStream: This will enable the search for all driver symbols
 *   that match the requested driver symbol name except the corresponding per-thread versions.
 * - ::UPTKEnablePerThreadDefaultStream: This will enable the search for all
 *   driver symbols that match the requested driver symbol name including the per-thread
 *   versions. If a per-thread version is not found, the API will return the legacy version
 *   of the driver function.
 *
 * \param symbol - The base name of the driver API function to look for. As an example,
 *                 for the driver API ::cuMemAlloc_v2, \p symbol would be cuMemAlloc.
 * \param funcPtr - Location to return the function pointer to the requested driver function
 * \param UPTKVersion - The UPTK version to look for the requested driver symbol 
 * \param flags -  Flags to specify search options.
 * \param driverStatus - Optional location to store the status of finding the symbol from
 *                       the driver. See ::UPTKDriverEntryPointQueryResult for 
 *                       possible values.
 *
 * \return
 * ::UPTKSuccess,
 * ::UPTKErrorInvalidValue,
 * ::UPTKErrorNotSupported
 * \note_version_mixing
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cuGetProcAddress
 */
#if defined(__cplusplus)
extern __host__ UPTKError_t UPTKGetDriverEntryPointByVersion(const char *symbol, void **funcPtr, unsigned int UPTKVersion, unsigned long long flags, enum UPTKDriverEntryPointQueryResult *driverStatus = NULL);
#else
extern __host__ UPTKError_t UPTKGetDriverEntryPointByVersion(const char *symbol, void **funcPtr, unsigned int UPTKVersion, unsigned long long flags, enum UPTKDriverEntryPointQueryResult *driverStatus);
#endif

/** @} */ /* END UPTKRT_DRIVER_ENTRY_POINT */

/** \cond impl_private */
extern __host__ UPTKError_t UPTKGetExportTable(const void **ppExportTable, const UPTKUUID_t *pExportTableId);
/** \endcond impl_private */

/**
 * \defgroup UPTKRT_HIGHLEVEL C++ API Routines
 *
 * ___MANBRIEF___ C++ high level API functions of the UPTK runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the C++ high level API functions of the UPTK runtime
 * application programming interface. To use these functions, your
 * application needs to be compiled with the \p nvcc compiler.
 *
 * \brief C++-style interface built on top of UPTK runtime API
 */

/**
 * \defgroup UPTKRT_DRIVER Interactions with the UPTK Driver API
 *
 * ___MANBRIEF___ interactions between UPTK Driver API and UPTK Runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the interactions between the UPTK Driver API and the UPTK Runtime API
 *
 * @{
 *
 * \section UPTKRT_UPTK_primary Primary Contexts
 *
 * There exists a one to one relationsUPTK between UPTK devices in the UPTK Runtime
 * API and ::CUcontext s in the UPTK Driver API within a process.  The specific
 * context which the UPTK Runtime API uses for a device is called the device's
 * primary context.  From the perspective of the UPTK Runtime API, a device and 
 * its primary context are synonymous.
 *
 * \section UPTKRT_UPTK_init Initialization and Tear-Down
 *
 * UPTK Runtime API calls operate on the UPTK Driver API ::CUcontext which is current to
 * to the calling host thread.
 * 
 * The function ::UPTKInitDevice() ensures that the primary context is initialized
 * for the requested device but does not make it current to the calling thread. 
 *
 * The function ::UPTKSetDevice() initializes the primary context for the
 * specified device and makes it current to the calling thread by calling ::cuCtxSetCurrent().
 *
 * The UPTK Runtime API will automatically initialize the primary context for
 * a device at the first UPTK Runtime API call which requires an active context.
 * If no ::CUcontext is current to the calling thread when a UPTK Runtime API call 
 * which requires an active context is made, then the primary context for a device 
 * will be selected, made current to the calling thread, and initialized.
 *
 * The context which the UPTK Runtime API initializes will be initialized using 
 * the parameters specified by the UPTK Runtime API functions
 * ::UPTKSetDeviceFlags(), 
 * ::UPTKD3D9SetDirect3DDevice(), 
 * ::UPTKD3D10SetDirect3DDevice(), 
 * ::UPTKD3D11SetDirect3DDevice(), 
 * ::UPTKGLSetGLDevice(), and
 * ::UPTKVDPAUSetVDPAUDevice().
 * Note that these functions will fail with ::UPTKErrorSetOnActiveProcess if they are 
 * called when the primary context for the specified device has already been initialized.
 * (or if the current device has already been initialized, in the case of 
 * ::UPTKSetDeviceFlags()). 
 *
 * Primary contexts will remain active until they are explicitly deinitialized 
 * using ::UPTKDeviceReset().  The function ::UPTKDeviceReset() will deinitialize the 
 * primary context for the calling thread's current device immediately.  The context 
 * will remain current to all of the threads that it was current to.  The next UPTK 
 * Runtime API call on any thread which requires an active context will trigger the 
 * reinitialization of that device's primary context.
 *
 * Note that primary contexts are shared resources. It is recommended that
 * the primary context not be reset except just before exit or to recover from an
 * unspecified launch failure.
 * 
 * \section UPTKRT_UPTK_context Context Interoperability
 *
 * Note that the use of multiple ::CUcontext s per device within a single process 
 * will substantially degrade performance and is strongly discouraged.  Instead,
 * it is highly recommended that the implicit one-to-one device-to-context mapping
 * for the process provided by the UPTK Runtime API be used.
 *
 * If a non-primary ::CUcontext created by the UPTK Driver API is current to a
 * thread then the UPTK Runtime API calls to that thread will operate on that 
 * ::CUcontext, with some exceptions listed below.  Interoperability between data
 * types is discussed in the following sections.
 *
 * The function ::UPTKPointerGetAttributes() will return the error 
 * ::UPTKErrorIncompatibleDriverContext if the pointer being queried was allocated by a 
 * non-primary context.  The function ::UPTKDeviceEnablePeerAccess() and the rest of 
 * the peer access API may not be called when a non-primary ::CUcontext is current.  
 * To use the pointer query and peer access APIs with a context created using the 
 * UPTK Driver API, it is necessary that the UPTK Driver API be used to access
 * these features.
 *
 * All UPTK Runtime API state (e.g, global variables' addresses and values) travels
 * with its underlying ::CUcontext.  In particular, if a ::CUcontext is moved from one 
 * thread to another then all UPTK Runtime API state will move to that thread as well.
 *
 * Please note that attaching to legacy contexts (those with a version of 3010 as returned
 * by ::cuCtxGetApiVersion()) is not possible. The UPTK Runtime will return
 * ::UPTKErrorIncompatibleDriverContext in such cases.
 *
 * \section UPTKRT_UPTK_stream Interactions between CUstream and UPTKStream_t
 *
 * The types ::CUstream and ::UPTKStream_t are identical and may be used interchangeably.
 *
 * \section UPTKRT_UPTK_event Interactions between CUevent and UPTKEvent_t
 *
 * The types ::CUevent and ::UPTKEvent_t are identical and may be used interchangeably.
 *
 * \section UPTKRT_UPTK_array Interactions between CUarray and UPTKArray_t 
 *
 * The types ::CUarray and struct ::UPTKArray * represent the same data type and may be used
 * interchangeably by casting the two types between each other.
 *
 * In order to use a ::CUarray in a UPTK Runtime API function which takes a struct ::UPTKArray *,
 * it is necessary to explicitly cast the ::CUarray to a struct ::UPTKArray *.
 *
 * In order to use a struct ::UPTKArray * in a UPTK Driver API function which takes a ::CUarray,
 * it is necessary to explicitly cast the struct ::UPTKArray * to a ::CUarray .
 *
 * \section UPTKRT_UPTK_graphicsResource Interactions between CUgraphicsResource and UPTKGraphicsResource_t
 *
 * The types ::CUgraphicsResource and ::UPTKGraphicsResource_t represent the same data type and may be used
 * interchangeably by casting the two types between each other.
 *
 * In order to use a ::CUgraphicsResource in a UPTK Runtime API function which takes a 
 * ::UPTKGraphicsResource_t, it is necessary to explicitly cast the ::CUgraphicsResource 
 * to a ::UPTKGraphicsResource_t.
 *
 * In order to use a ::UPTKGraphicsResource_t in a UPTK Driver API function which takes a
 * ::CUgraphicsResource, it is necessary to explicitly cast the ::UPTKGraphicsResource_t 
 * to a ::CUgraphicsResource.
 *
 * \section UPTKRT_UPTK_texture_objects Interactions between CUtexObject and UPTKTextureObject_t
 *
 * The types ::CUtexObject and ::UPTKTextureObject_t represent the same data type and may be used
 * interchangeably by casting the two types between each other.
 *
 * In order to use a ::CUtexObject in a UPTK Runtime API function which takes a ::UPTKTextureObject_t,
 * it is necessary to explicitly cast the ::CUtexObject to a ::UPTKTextureObject_t.
 *
 * In order to use a ::UPTKTextureObject_t in a UPTK Driver API function which takes a ::CUtexObject,
 * it is necessary to explicitly cast the ::UPTKTextureObject_t to a ::CUtexObject.
 *
 * \section UPTKRT_UPTK_surface_objects Interactions between CUsurfObject and UPTKSurfaceObject_t
 *
 * The types ::CUsurfObject and ::UPTKSurfaceObject_t represent the same data type and may be used
 * interchangeably by casting the two types between each other.
 *
 * In order to use a ::CUsurfObject in a UPTK Runtime API function which takes a ::UPTKSurfaceObject_t,
 * it is necessary to explicitly cast the ::CUsurfObject to a ::UPTKSurfaceObject_t.
 *
 * In order to use a ::UPTKSurfaceObject_t in a UPTK Driver API function which takes a ::CUsurfObject,
 * it is necessary to explicitly cast the ::UPTKSurfaceObject_t to a ::CUsurfObject.
 *
 * \section UPTKRT_UPTK_module Interactions between CUfunction and UPTKFunction_t
 *
 * The types ::CUfunction and ::UPTKFunction_t represent the same data type and may be used
 * interchangeably by casting the two types between each other.
 *
 * In order to use a ::UPTKFunction_t in a UPTK Driver API function which takes a ::CUfunction,
 * it is necessary to explicitly cast the ::UPTKFunction_t to a ::CUfunction.
 *
 */

 /**
  * \brief Get pointer to device entry function that matches entry function \p symbolPtr
  *
  * Returns in \p functionPtr the device entry function corresponding to the symbol \p symbolPtr.
  *
  * \param functionPtr     - Returns the device entry function
  * \param symbolPtr       - Pointer to device entry function to search for
  *
  * \return
  * ::UPTKSuccess
  *
  */
extern __host__ UPTKError_t UPTKGetFuncBySymbol(UPTKFunction_t* functionPtr, const void* symbolPtr);

/**
 * \brief Get pointer to device kernel that matches entry function \p entryFuncAddr
  *
  * Returns in \p kernelPtr the device kernel corresponding to the entry function \p entryFuncAddr.
  *
  * \param kernelPtr          - Returns the device kernel
  * \param entryFuncAddr      - Address of device entry function to search kernel for
  *
  * \return
  * ::UPTKSuccess
  *
  * \sa
  * \ref ::UPTKGetKernel(UPTKKernel_t *kernelPtr, const T *entryFuncAddr) "UPTKGetKernel (C++ API)"
  */
extern __host__ UPTKError_t UPTKGetKernel(UPTKKernel_t *kernelPtr, const void *entryFuncAddr);

extern __host__ UPTKError_t UPTKOccupancyMaxPotentialBlockSize_internal(int* gridSize, int* blockSize, const void* f, size_t dynSharedMemPerBlk, int blockSizeLimit);


/** @} */ /* END UPTKRT_DRIVER */

#if defined(__UPTK_API_VERSION_INTERNAL)
    #undef UPTKMemcpy
    #undef UPTKMemcpyToSymbol
    #undef UPTKMemcpyFromSymbol
    #undef UPTKMemcpy2D
    #undef UPTKMemcpyToArray
    #undef UPTKMemcpy2DToArray
    #undef UPTKMemcpyFromArray
    #undef UPTKMemcpy2DFromArray
    #undef UPTKMemcpyArrayToArray
    #undef UPTKMemcpy2DArrayToArray
    #undef UPTKMemcpy3D
    #undef UPTKMemcpy3DPeer
    #undef UPTKMemset
    #undef UPTKMemset2D
    #undef UPTKMemset3D
    #undef UPTKMemcpyAsync
    #undef UPTKMemcpyToSymbolAsync
    #undef UPTKMemcpyFromSymbolAsync
    #undef UPTKMemcpy2DAsync
    #undef UPTKMemcpyToArrayAsync
    #undef UPTKMemcpy2DToArrayAsync
    #undef UPTKMemcpyFromArrayAsync
    #undef UPTKMemcpy2DFromArrayAsync
    #undef UPTKMemcpy3DAsync
    #undef UPTKMemcpy3DPeerAsync
    #undef UPTKMemsetAsync
    #undef UPTKMemset2DAsync
    #undef UPTKMemset3DAsync
    #undef UPTKStreamQuery
    #undef UPTKStreamGetFlags
    #undef UPTKStreamGetId
    #undef UPTKStreamGetPriority
    #undef UPTKEventRecord
    #undef UPTKEventRecordWithFlags
    #undef UPTKStreamWaitEvent
    #undef UPTKStreamAddCallback
    #undef UPTKStreamAttachMemAsync
    #undef UPTKStreamSynchronize
    #undef UPTKLaunchKernel
    #undef UPTKLaunchKernelExC
    #undef UPTKLaunchHostFunc
    #undef UPTKMemPrefetchAsync
    #undef UPTKMemPrefetchAsync_v2
    #undef UPTKLaunchCooperativeKernel
    #undef UPTKSignalExternalSemaphoresAsync
    #undef UPTKWaitExternalSemaphoresAsync
    #undef UPTKGraphInstantiateWithParams
    #undef UPTKGraphUpload
    #undef UPTKGraphLaunch
    #undef UPTKStreamBeginCapture
    #undef UPTKStreamBeginCaptureToGraph
    #undef UPTKStreamEndCapture
    #undef UPTKStreamIsCapturing
    #undef UPTKStreamGetCaptureInfo
    #undef UPTKStreamGetCaptureInfo_v2
    #undef UPTKStreamGetCaptureInfo_v3
    #undef UPTKStreamUpdateCaptureDependencies
    #undef UPTKStreamUpdateCaptureDependencies_v2
    #undef UPTKStreamCopyAttributes
    #undef UPTKStreamGetAttribute
    #undef UPTKStreamSetAttribute
    #undef UPTKMallocAsync
    #undef UPTKFreeAsync
    #undef UPTKMallocFromPoolAsync
    #undef UPTKGetDriverEntryPoint
    #undef UPTKGetDriverEntryPointByVersion

    #undef UPTKGetDeviceProperties

    extern __host__ UPTKError_t UPTKMemcpy(void *dst, const void *src, size_t count, enum UPTKMemcpyKind kind);
    extern __host__ UPTKError_t UPTKMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset __dv(0), enum UPTKMemcpyKind kind __dv(UPTKMemcpyHostToDevice));
    extern __host__ UPTKError_t UPTKMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset __dv(0), enum UPTKMemcpyKind kind __dv(UPTKMemcpyDeviceToHost));
    extern __host__ UPTKError_t UPTKMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum UPTKMemcpyKind kind);
    extern __host__ UPTKError_t UPTKMemcpyToArray(UPTKArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum UPTKMemcpyKind kind);
    extern __host__ UPTKError_t UPTKMemcpy2DToArray(UPTKArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum UPTKMemcpyKind kind);
    extern __host__ UPTKError_t UPTKMemcpyFromArray(void *dst, UPTKArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum UPTKMemcpyKind kind);
    extern __host__ UPTKError_t UPTKMemcpy2DFromArray(void *dst, size_t dpitch, UPTKArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum UPTKMemcpyKind kind);
    extern __host__ UPTKError_t UPTKMemcpyArrayToArray(UPTKArray_t dst, size_t wOffsetDst, size_t hOffsetDst, UPTKArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum UPTKMemcpyKind kind __dv(UPTKMemcpyDeviceToDevice));
    extern __host__ UPTKError_t UPTKMemcpy2DArrayToArray(UPTKArray_t dst, size_t wOffsetDst, size_t hOffsetDst, UPTKArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum UPTKMemcpyKind kind __dv(UPTKMemcpyDeviceToDevice));
    extern __host__ UPTKError_t UPTKMemcpy3D(const struct UPTKMemcpy3DParms *p);
    extern __host__ UPTKError_t UPTKMemcpy3DPeer(const struct UPTKMemcpy3DPeerParms *p);
    extern __host__ UPTKError_t UPTKMemset(void *devPtr, int value, size_t count);
    extern __host__ UPTKError_t UPTKMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height);
    extern __host__ UPTKError_t UPTKMemset3D(struct UPTKPitchedPtr pitchedDevPtr, int value, struct UPTKExtent extent);
    extern __host__ UPTKError_t UPTKMemcpyAsync(void *dst, const void *src, size_t count, enum UPTKMemcpyKind kind, UPTKStream_t stream __dv(0));
    extern __host__ UPTKError_t UPTKMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset, enum UPTKMemcpyKind kind, UPTKStream_t stream __dv(0));
    extern __host__ UPTKError_t UPTKMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset, enum UPTKMemcpyKind kind, UPTKStream_t stream __dv(0));
    extern __host__ UPTKError_t UPTKMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum UPTKMemcpyKind kind, UPTKStream_t stream __dv(0));
    extern __host__ UPTKError_t UPTKMemcpyToArrayAsync(UPTKArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum UPTKMemcpyKind kind, UPTKStream_t stream __dv(0));
    extern __host__ UPTKError_t UPTKMemcpy2DToArrayAsync(UPTKArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum UPTKMemcpyKind kind, UPTKStream_t stream __dv(0));
    extern __host__ UPTKError_t UPTKMemcpyFromArrayAsync(void *dst, UPTKArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum UPTKMemcpyKind kind, UPTKStream_t stream __dv(0));
    extern __host__ UPTKError_t UPTKMemcpy2DFromArrayAsync(void *dst, size_t dpitch, UPTKArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum UPTKMemcpyKind kind, UPTKStream_t stream __dv(0));
    extern __host__ UPTKError_t UPTKMemcpy3DAsync(const struct UPTKMemcpy3DParms *p, UPTKStream_t stream __dv(0));
    extern __host__ UPTKError_t UPTKMemcpy3DPeerAsync(const struct UPTKMemcpy3DPeerParms *p, UPTKStream_t stream __dv(0));
    extern __host__ UPTKError_t UPTKMemsetAsync(void *devPtr, int value, size_t count, UPTKStream_t stream __dv(0));
    extern __host__ UPTKError_t UPTKMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, UPTKStream_t stream __dv(0));
    extern __host__ UPTKError_t UPTKMemset3DAsync(struct UPTKPitchedPtr pitchedDevPtr, int value, struct UPTKExtent extent, UPTKStream_t stream __dv(0));
    extern __host__ UPTKError_t UPTKStreamQuery(UPTKStream_t stream);
    extern __host__ UPTKError_t UPTKStreamGetFlags(UPTKStream_t hStream, unsigned int *flags);
    extern __host__ UPTKError_t UPTKStreamGetId(UPTKStream_t hStream, unsigned long long *streamId);
    extern __host__ UPTKError_t UPTKStreamGetPriority(UPTKStream_t hStream, int *priority);
    extern __host__ UPTKError_t UPTKEventRecord(UPTKEvent_t event, UPTKStream_t stream __dv(0));
    extern __host__ UPTKError_t UPTKEventRecordWithFlags(UPTKEvent_t event, UPTKStream_t stream __dv(0), unsigned int flags __dv(0));
    extern __host__ UPTKError_t UPTKStreamWaitEvent(UPTKStream_t stream, UPTKEvent_t event, unsigned int flags);
    extern __host__ UPTKError_t UPTKStreamAddCallback(UPTKStream_t stream, UPTKStreamCallback_t callback, void *userData, unsigned int flags);
    extern __host__ UPTKError_t UPTKStreamAttachMemAsync(UPTKStream_t stream, void *devPtr, size_t length, unsigned int flags);
    extern __host__ UPTKError_t UPTKStreamSynchronize(UPTKStream_t stream);
    extern __host__ UPTKError_t UPTKLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, UPTKStream_t stream);
    extern __host__ UPTKError_t UPTKLaunchKernelExC(const UPTKLaunchConfig_t *config, const void *func, void **args);
    extern __host__ UPTKError_t UPTKLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, UPTKStream_t stream);
    extern __host__ UPTKError_t UPTKLaunchHostFunc(UPTKStream_t stream, UPTKHostFn_t fn, void *userData);
    extern __host__ UPTKError_t UPTKMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, UPTKStream_t stream);
    extern __host__ UPTKError_t UPTKMemPrefetchAsync_v2(const void *devPtr, size_t count, struct UPTKMemLocation location, unsigned int flags, UPTKStream_t stream);
    extern __host__ UPTKError_t UPTKSignalExternalSemaphoresAsync(const UPTKExternalSemaphore_t *extSemArray, const struct UPTKExternalSemaphoreSignalParams_v1 *paramsArray, unsigned int numExtSems, UPTKStream_t stream __dv(0));
    extern __host__ UPTKError_t UPTKSignalExternalSemaphoresAsync_ptsz(const UPTKExternalSemaphore_t *extSemArray, const struct UPTKExternalSemaphoreSignalParams_v1 *paramsArray, unsigned int numExtSems, UPTKStream_t stream __dv(0));
    extern __host__ UPTKError_t UPTKSignalExternalSemaphoresAsync_v2(const UPTKExternalSemaphore_t *extSemArray, const struct UPTKExternalSemaphoreSignalParams *paramsArray, unsigned int numExtSems, UPTKStream_t stream __dv(0));
    extern __host__ UPTKError_t UPTKWaitExternalSemaphoresAsync(const UPTKExternalSemaphore_t *extSemArray, const struct UPTKExternalSemaphoreWaitParams_v1 *paramsArray, unsigned int numExtSems, UPTKStream_t stream __dv(0));
    extern __host__ UPTKError_t UPTKWaitExternalSemaphoresAsync_ptsz(const UPTKExternalSemaphore_t *extSemArray, const struct UPTKExternalSemaphoreWaitParams_v1 *paramsArray, unsigned int numExtSems, UPTKStream_t stream __dv(0));
    extern __host__ UPTKError_t UPTKWaitExternalSemaphoresAsync_v2(const UPTKExternalSemaphore_t *extSemArray, const struct UPTKExternalSemaphoreWaitParams *paramsArray, unsigned int numExtSems, UPTKStream_t stream __dv(0));
    extern __host__ UPTKError_t UPTKGraphInstantiateWithParams(UPTKGraphExec_t *pGraphExec, UPTKGraph_t graph, UPTKGraphInstantiateParams *instantiateParams);
    extern __host__ UPTKError_t UPTKGraphUpload(UPTKGraphExec_t graphExec, UPTKStream_t stream);
    extern __host__ UPTKError_t UPTKGraphLaunch(UPTKGraphExec_t graphExec, UPTKStream_t stream);
    extern __host__ UPTKError_t UPTKStreamBeginCapture(UPTKStream_t stream, enum UPTKStreamCaptureMode mode);
    extern __host__ UPTKError_t UPTKStreamBeginCaptureToGraph(UPTKStream_t stream, UPTKGraph_t graph, const UPTKGraphNode_t *dependencies, const UPTKGraphEdgeData *dependencyData, size_t numDependencies, enum UPTKStreamCaptureMode mode);
    extern __host__ UPTKError_t UPTKStreamEndCapture(UPTKStream_t stream, UPTKGraph_t *pGraph);
    extern __host__ UPTKError_t UPTKStreamIsCapturing(UPTKStream_t stream, enum UPTKStreamCaptureStatus *pCaptureStatus);
    extern __host__ UPTKError_t UPTKStreamGetCaptureInfo(UPTKStream_t stream, enum UPTKStreamCaptureStatus *captureStatus_out, unsigned long long *id_out);
    extern __host__ UPTKError_t UPTKStreamGetCaptureInfo_ptsz(UPTKStream_t stream, enum UPTKStreamCaptureStatus *captureStatus_out, unsigned long long *id_out);
    extern __host__ UPTKError_t UPTKStreamGetCaptureInfo_v2(UPTKStream_t stream, enum UPTKStreamCaptureStatus *captureStatus_out, unsigned long long *id_out __dv(0), UPTKGraph_t *graph_out __dv(0), const UPTKGraphNode_t **dependencies_out __dv(0), size_t *numDependencies_out __dv(0));
    extern __host__ UPTKError_t UPTKStreamGetCaptureInfo_v3(UPTKStream_t stream, enum UPTKStreamCaptureStatus *captureStatus_out, unsigned long long *id_out __dv(0), UPTKGraph_t *graph_out __dv(0), const UPTKGraphNode_t **dependencies_out __dv(0), const UPTKGraphEdgeData **edgeData_out __dv(0), size_t *numDependencies_out __dv(0));
    extern __host__ UPTKError_t UPTKStreamUpdateCaptureDependencies(UPTKStream_t stream, UPTKGraphNode_t *dependencies, size_t numDependencies, unsigned int flags __dv(0));
    extern __host__ UPTKError_t UPTKStreamUpdateCaptureDependencies_v2(UPTKStream_t stream, UPTKGraphNode_t *dependencies, const UPTKGraphEdgeData *dependencyData, size_t numDependencies, unsigned int flags __dv(0));
    extern __host__ UPTKError_t UPTKStreamCopyAttributes(UPTKStream_t dstStream, UPTKStream_t srcStream);
    extern __host__ UPTKError_t UPTKStreamGetAttribute(UPTKStream_t stream, UPTKStreamAttrID attr, UPTKStreamAttrValue *value);
    extern __host__ UPTKError_t UPTKStreamSetAttribute(UPTKStream_t stream, UPTKStreamAttrID attr, const UPTKStreamAttrValue *param);

    extern __host__ UPTKError_t UPTKMallocAsync(void **devPtr, size_t size, UPTKStream_t hStream);
    extern __host__ UPTKError_t UPTKFreeAsync(void *devPtr, UPTKStream_t hStream);
    extern __host__ UPTKError_t UPTKMallocFromPoolAsync(void **ptr, size_t size, UPTKMemPool_t memPool, UPTKStream_t stream);
    extern __host__ UPTKError_t UPTKGetDriverEntryPoint(const char *symbol, void **funcPtr, unsigned long long flags, enum UPTKDriverEntryPointQueryResult *driverStatus);
    extern __host__ UPTKError_t UPTKGetDriverEntryPointByVersion(const char *symbol, void **funcPtr, unsigned int UPTKVersion, unsigned long long flags, enum UPTKDriverEntryPointQueryResult *driverStatus);

    extern __host__ UPTKError_t UPTKGetDeviceProperties(struct UPTKDeviceProp *prop, int device);

#elif defined(__UPTKRT_API_PER_THREAD_DEFAULT_STREAM)
    // nvcc stubs reference the 'UPTKLaunch'/'UPTKLaunchKernel' identifier even if it was defined
    // to 'UPTKLaunch_ptsz'/'UPTKLaunchKernel_ptsz'. Redirect through a static inline function.
    #undef UPTKLaunchKernel
    static __inline__ __host__ UPTKError_t UPTKLaunchKernel(const void *func, 
                                                            dim3 gridDim, dim3 blockDim, 
                                                            void **args, size_t sharedMem, 
                                                            UPTKStream_t stream)
    {
        return UPTKLaunchKernel_ptsz(func, gridDim, blockDim, args, sharedMem, stream);
    }
    #define UPTKLaunchKernel __UPTKRT_API_PTSZ(UPTKLaunchKernel)
    #undef UPTKLaunchKernelExC
    static __inline__ __host__ UPTKError_t UPTKLaunchKernelExC(const UPTKLaunchConfig_t *config,
                                                               const void *func,
                                                                  void **args)
    {
        return UPTKLaunchKernelExC_ptsz(config, func, args);
    }
    #define UPTKLaunchKernelExC __UPTKRT_API_PTSZ(UPTKLaunchKernelExC)
#endif

#if defined(__cplusplus)
}

#endif /* __cplusplus */

#undef UPTK_EXCLUDE_FROM_RTC
#endif /* !__UPTKCC_RTC__ */

#undef __dv

#if defined(__UNDEF_UPTK_INCLUDE_COMPILER_INTERNAL_HEADERS_UPTK_RUNTIME_API_H__)
#undef __UPTK_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_UPTK_INCLUDE_COMPILER_INTERNAL_HEADERS_UPTK_RUNTIME_API_H__
#endif

#endif /* !__UPTK_RUNTIME_API_H__ */

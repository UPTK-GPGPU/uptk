/*
 * Copyright 1993-2023 NVIDIA Corporation.  All rights reserved.
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

#if !defined(__UPTK_DRIVER_TYPES_H__)
#define __UPTK_DRIVER_TYPES_H__

#if !defined(__UPTK_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __UPTK_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_UPTK_INCLUDE_COMPILER_INTERNAL_HEADERS_DRIVER_TYPES_H__
#endif

#ifndef __CUDACC_RTC_MINIMAL__
/**
 * \defgroup UPTKRT_TYPES Data types used by UPTK Runtime
 * \ingroup UPTKRT
 *
 * @{
 */

/*******************************************************************************
*                                                                              *
*  TYPE DEFINITIONS USED BY RUNTIME API                                        *
*                                                                              *
*******************************************************************************/

#if !defined(__UPTK_INTERNAL_COMPILATION__)


#if !defined(__CUDACC_RTC__)
#include <limits.h>
#include <stddef.h>
#endif /* !defined(__CUDACC_RTC__) */

#define UPTKHostAllocDefault                0x00  /**< Default page-locked allocation flag */
#define UPTKHostAllocPortable               0x01  /**< Pinned memory accessible by all UPTK contexts */
#define UPTKHostAllocMapped                 0x02  /**< Map allocation into device space */
#define UPTKHostAllocWriteCombined          0x04  /**< Write-combined memory */

#define UPTKHostRegisterDefault             0x00  /**< Default host memory registration flag */
#define UPTKHostRegisterPortable            0x01  /**< Pinned memory accessible by all UPTK contexts */
#define UPTKHostRegisterMapped              0x02  /**< Map registered memory into device space */
#define UPTKHostRegisterIoMemory            0x04  /**< Memory-mapped I/O space */
#define UPTKHostRegisterReadOnly            0x08  /**< Memory-mapped read-only */

#define UPTKPeerAccessDefault               0x00  /**< Default peer addressing enable flag */

#define UPTKStreamDefault                   0x00  /**< Default stream flag */
#define UPTKStreamNonBlocking               0x01  /**< Stream does not synchronize with stream 0 (the NULL stream) */

 /**
 * Legacy stream handle
 *
 * Stream handle that can be passed as a UPTKStream_t to use an implicit stream
 * with legacy synchronization behavior.
 *
 * See details of the \link_sync_behavior
 */
#define UPTKStreamLegacy                    ((UPTKStream_t)0x1)


/**
 * Per-thread stream handle
 *
 * Stream handle that can be passed as a UPTKStream_t to use an implicit stream
 * with per-thread synchronization behavior.
 *
 * See details of the \link_sync_behavior
 */
#define UPTKStreamPerThread                 ((UPTKStream_t)0x2)

#define UPTKEventDefault                    0x00  /**< Default event flag */
#define UPTKEventBlockingSync               0x01  /**< Event uses blocking synchronization */
#define UPTKEventDisableTiming              0x02  /**< Event will not record timing data */
#define UPTKEventInterprocess               0x04  /**< Event is suitable for interprocess use. UPTKEventDisableTiming must be set */

#define UPTKEventRecordDefault              0x00  /**< Default event record flag */
#define UPTKEventRecordExternal             0x01  /**< Event is captured in the graph as an external event node when performing stream capture */

#define UPTKEventWaitDefault                0x00  /**< Default event wait flag */
#define UPTKEventWaitExternal               0x01  /**< Event is captured in the graph as an external event node when performing stream capture */

#define UPTKDeviceScheduleAuto              0x00  /**< Device flag - Automatic scheduling */
#define UPTKDeviceScheduleSpin              0x01  /**< Device flag - Spin default scheduling */
#define UPTKDeviceScheduleYield             0x02  /**< Device flag - Yield default scheduling */
#define UPTKDeviceScheduleBlockingSync      0x04  /**< Device flag - Use blocking synchronization */
#define UPTKDeviceBlockingSync              0x04  /**< Device flag - Use blocking synchronization 
                                                    *  \deprecated This flag was deprecated as of UPTK 4.0 and
                                                    *  replaced with ::UPTKDeviceScheduleBlockingSync. */
#define UPTKDeviceScheduleMask              0x07  /**< Device schedule flags mask */
#define UPTKDeviceMapHost                   0x08  /**< Device flag - Support mapped pinned allocations */
#define UPTKDeviceLmemResizeToMax           0x10  /**< Device flag - Keep local memory allocation after launch */
#define UPTKDeviceSyncMemops                0x80  /**< Device flag - Ensure synchronous memory operations on this context will synchronize */
#define UPTKDeviceMask                      0xff  /**< Device flags mask */

#define UPTKArrayDefault                    0x00  /**< Default UPTK array allocation flag */
#define UPTKArrayLayered                    0x01  /**< Must be set in UPTKMalloc3DArray to create a layered UPTK array */
#define UPTKArraySurfaceLoadStore           0x02  /**< Must be set in UPTKMallocArray or UPTKMalloc3DArray in order to bind surfaces to the UPTK array */
#define UPTKArrayCubemap                    0x04  /**< Must be set in UPTKMalloc3DArray to create a cubemap UPTK array */
#define UPTKArrayTextureGather              0x08  /**< Must be set in UPTKMallocArray or UPTKMalloc3DArray in order to perform texture gather operations on the UPTK array */
#define UPTKArrayColorAttachment            0x20  /**< Must be set in UPTKExternalMemoryGetMappedMipmappedArray if the mipmapped array is used as a color target in a graphics API */
#define UPTKArraySparse                     0x40  /**< Must be set in UPTKMallocArray, UPTKMalloc3DArray or UPTKMallocMipmappedArray in order to create a sparse UPTK array or UPTK mipmapped array */
#define UPTKArrayDeferredMapping            0x80  /**< Must be set in UPTKMallocArray, UPTKMalloc3DArray or UPTKMallocMipmappedArray in order to create a deferred mapping UPTK array or UPTK mipmapped array */

#define UPTKIpcMemLazyEnablePeerAccess      0x01  /**< Automatically enable peer access between remote devices as needed */

#define UPTKMemAttachGlobal                 0x01  /**< Memory can be accessed by any stream on any device*/
#define UPTKMemAttachHost                   0x02  /**< Memory cannot be accessed by any stream on any device */
#define UPTKMemAttachSingle                 0x04  /**< Memory can only be accessed by a single stream on the associated device */

#define UPTKOccupancyDefault                0x00  /**< Default behavior */
#define UPTKOccupancyDisableCachingOverride 0x01  /**< Assume global caching is enabled and cannot be automatically turned off */

#define UPTKCpuDeviceId                     ((int)-1) /**< Device id that represents the CPU */
#define UPTKInvalidDeviceId                 ((int)-2) /**< Device id that represents an invalid device */
#define UPTKInitDeviceFlagsAreValid         0x01  /**< Tell the UPTK runtime that DeviceFlags is being set in UPTKInitDevice call */
/**
 * If set, each kernel launched as part of ::UPTKLaunchCooperativeKernelMultiDevice only
 * waits for prior work in the stream corresponding to that GPU to complete before the
 * kernel begins execution.
 */
#define UPTKCooperativeLaunchMultiDeviceNoPreSync  0x01

/**
 * If set, any subsequent work pushed in a stream that participated in a call to
 * ::UPTKLaunchCooperativeKernelMultiDevice will only wait for the kernel launched on
 * the GPU corresponding to that stream to complete before it begins execution.
 */
#define UPTKCooperativeLaunchMultiDeviceNoPostSync 0x02

#endif /* !__UPTK_INTERNAL_COMPILATION__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

/**
 * UPTK error types
 */
enum UPTKError
{
    /**
     * The API call returned with no errors. In the case of query calls, this
     * also means that the operation being queried is complete (see
     * ::UPTKEventQuery() and ::UPTKStreamQuery()).
     */
    UPTKSuccess                           =      0,
  
    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    UPTKErrorInvalidValue                 =     1,
  
    /**
     * The API call failed because it was unable to allocate enough memory or
     * other resources to perform the requested operation.
     */
    UPTKErrorMemoryAllocation             =      2,
  
    /**
     * The API call failed because the UPTK driver and runtime could not be
     * initialized.
     */
    UPTKErrorInitializationError          =      3,
  
    /**
     * This indicates that a UPTK Runtime API call cannot be executed because
     * it is being called during process shut down, at a point in time after
     * UPTK driver has been unloaded.
     */
    UPTKErrorCudartUnloading              =     4,

    /**
     * This indicates profiler is not initialized for this run. This can
     * happen when the application is running with external profiling tools
     * like visual profiler.
     */
    UPTKErrorProfilerDisabled             =     5,

    /**
     * \deprecated
     * This error return is deprecated as of UPTK 5.0. It is no longer an error
     * to attempt to enable/disable the profiling via ::UPTKProfilerStart or
     * ::UPTKProfilerStop without initialization.
     */
    UPTKErrorProfilerNotInitialized       =     6,

    /**
     * \deprecated
     * This error return is deprecated as of UPTK 5.0. It is no longer an error
     * to call UPTKProfilerStart() when profiling is already enabled.
     */
    UPTKErrorProfilerAlreadyStarted       =     7,

    /**
     * \deprecated
     * This error return is deprecated as of UPTK 5.0. It is no longer an error
     * to call UPTKProfilerStop() when profiling is already disabled.
     */
     UPTKErrorProfilerAlreadyStopped       =    8,
    /**
     * This indicates that a kernel launch is requesting resources that can
     * never be satisfied by the current device. Requesting more shared memory
     * per block than the device supports will trigger this error, as will
     * requesting too many threads or blocks. See ::UPTKDeviceProp for more
     * device limitations.
     */
    UPTKErrorInvalidConfiguration         =      9,
  
    /**
     * This indicates that one or more of the pitch-related parameters passed
     * to the API call is not within the acceptable range for pitch.
     */
    UPTKErrorInvalidPitchValue            =     12,
  
    /**
     * This indicates that the symbol name/identifier passed to the API call
     * is not a valid name or identifier.
     */
    UPTKErrorInvalidSymbol                =     13,

    /**
     * This indicates that at least one host pointer passed to the API call is
     * not a valid host pointer.
     * \deprecated
     * This error return is deprecated as of UPTK 10.1.
     */
    UPTKErrorInvalidHostPointer           =     16,
  
    /**
     * This indicates that at least one device pointer passed to the API call is
     * not a valid device pointer.
     * \deprecated
     * This error return is deprecated as of UPTK 10.1.
     */
    UPTKErrorInvalidDevicePointer         =     17,
    /**
     * This indicates that the texture passed to the API call is not a valid
     * texture.
     */
    UPTKErrorInvalidTexture               =     18,
  
    /**
     * This indicates that the texture binding is not valid. This occurs if you
     * call ::UPTKGetTextureAlignmentOffset() with an unbound texture.
     */
    UPTKErrorInvalidTextureBinding        =     19,
  
    /**
     * This indicates that the channel descriptor passed to the API call is not
     * valid. This occurs if the format is not one of the formats specified by
     * ::UPTKChannelFormatKind, or if one of the dimensions is invalid.
     */
    UPTKErrorInvalidChannelDescriptor     =     20,
  
    /**
     * This indicates that the direction of the memcpy passed to the API call is
     * not one of the types specified by ::UPTKMemcpyKind.
     */
    UPTKErrorInvalidMemcpyDirection       =     21,

    /**
     * This indicated that the user has taken the address of a constant variable,
     * which was forbidden up until the UPTK 3.1 release.
     * \deprecated
     * This error return is deprecated as of UPTK 3.1. Variables in constant
     * memory may now have their address taken by the runtime via
     * ::UPTKGetSymbolAddress().
     */
    UPTKErrorAddressOfConstant            =     22,
  
    /**
     * This indicated that a texture fetch was not able to be performed.
     * This was previously used for device emulation of texture operations.
     * \deprecated
     * This error return is deprecated as of UPTK 3.1. Device emulation mode was
     * removed with the UPTK 3.1 release.
     */
    UPTKErrorTextureFetchFailed           =     23,
  
    /**
     * This indicated that a texture was not bound for access.
     * This was previously used for device emulation of texture operations.
     * \deprecated
     * This error return is deprecated as of UPTK 3.1. Device emulation mode was
     * removed with the UPTK 3.1 release.
     */
    UPTKErrorTextureNotBound              =     24,
  
    /**
     * This indicated that a synchronization operation had failed.
     * This was previously used for some device emulation functions.
     * \deprecated
     * This error return is deprecated as of UPTK 3.1. Device emulation mode was
     * removed with the UPTK 3.1 release.
     */
    UPTKErrorSynchronizationError         =     25,
    /**
     * This indicates that a non-float texture was being accessed with linear
     * filtering. This is not supported by UPTK.
     */
    UPTKErrorInvalidFilterSetting         =     26,
  
    /**
     * This indicates that an attempt was made to read a non-float texture as a
     * normalized float. This is not supported by UPTK.
     */
    UPTKErrorInvalidNormSetting           =     27,

    /**
     * Mixing of device and device emulation code was not allowed.
     * \deprecated
     * This error return is deprecated as of UPTK 3.1. Device emulation mode was
     * removed with the UPTK 3.1 release.
     */
    UPTKErrorMixedDeviceExecution         =     28,

    /**
     * This indicates that the API call is not yet implemented. Production
     * releases of UPTK will never return this error.
     * \deprecated
     * This error return is deprecated as of UPTK 4.1.
     */
    UPTKErrorNotYetImplemented            =     31,
  
    /**
     * This indicated that an emulated device pointer exceeded the 32-bit address
     * range.
     * \deprecated
     * This error return is deprecated as of UPTK 3.1. Device emulation mode was
     * removed with the UPTK 3.1 release.
     */
    UPTKErrorMemoryValueTooLarge          =     32,
    /**
     * This indicates that the UPTK driver that the application has loaded is a
     * stub library. Applications that run with the stub rather than a real
     * driver loaded will result in UPTK API returning this error.
     */
    UPTKErrorStubLibrary                  =     34,

    /**
     * This indicates that the installed NVIDIA UPTK driver is older than the
     * UPTK runtime library. This is not a supported configuration. Users should
     * install an updated NVIDIA display driver to allow the application to run.
     */
    UPTKErrorInsufficientDriver           =     35,

    /**
     * This indicates that the API call requires a newer UPTK driver than the one
     * currently installed. Users should install an updated NVIDIA UPTK driver
     * to allow the API call to succeed.
     */
    UPTKErrorCallRequiresNewerDriver      =     36,
  
    /**
     * This indicates that the surface passed to the API call is not a valid
     * surface.
     */
    UPTKErrorInvalidSurface               =     37,
  
    /**
     * This indicates that multiple global or constant variables (across separate
     * UPTK source files in the application) share the same string name.
     */
    UPTKErrorDuplicateVariableName        =     43,
  
    /**
     * This indicates that multiple textures (across separate UPTK source
     * files in the application) share the same string name.
     */
    UPTKErrorDuplicateTextureName         =     44,
  
    /**
     * This indicates that multiple surfaces (across separate UPTK source
     * files in the application) share the same string name.
     */
    UPTKErrorDuplicateSurfaceName         =     45,
  
    /**
     * This indicates that all UPTK devices are busy or unavailable at the current
     * time. Devices are often busy/unavailable due to use of
     * ::UPTKComputeModeProhibited, ::UPTKComputeModeExclusiveProcess, or when long
     * running UPTK kernels have filled up the GPU and are blocking new work
     * from starting. They can also be unavailable due to memory constraints
     * on a device that already has active UPTK work being performed.
     */
    UPTKErrorDevicesUnavailable           =     46,
  
    /**
     * This indicates that the current context is not compatible with this
     * the UPTK Runtime. This can only occur if you are using UPTK
     * Runtime/Driver interoperability and have created an existing Driver
     * context using the driver API. The Driver context may be incompatible
     * either because the Driver context was created using an older version 
     * of the API, because the Runtime API call expects a primary driver 
     * context and the Driver context is not primary, or because the Driver 
     * context has been destroyed. Please see \ref UPTKRT_DRIVER "Interactions 
     * with the UPTK Driver API" for more information.
     */
    UPTKErrorIncompatibleDriverContext    =     49,
    
    /**
     * The device function being invoked (usually via ::UPTKLaunchKernel()) was not
     * previously configured via the ::UPTKConfigureCall() function.
     */
    UPTKErrorMissingConfiguration         =      52,

    /**
     * This indicated that a previous kernel launch failed. This was previously
     * used for device emulation of kernel launches.
     * \deprecated
     * This error return is deprecated as of UPTK 3.1. Device emulation mode was
     * removed with the UPTK 3.1 release.
     */
    UPTKErrorPriorLaunchFailure           =      53,
    /**
     * This error indicates that a device runtime grid launch did not occur 
     * because the depth of the child grid would exceed the maximum supported
     * number of nested grid launches. 
     */
    UPTKErrorLaunchMaxDepthExceeded       =     65,

    /**
     * This error indicates that a grid launch did not occur because the kernel 
     * uses file-scoped textures which are unsupported by the device runtime. 
     * Kernels launched via the device runtime only support textures created with 
     * the Texture Object API's.
     */
    UPTKErrorLaunchFileScopedTex          =     66,

    /**
     * This error indicates that a grid launch did not occur because the kernel 
     * uses file-scoped surfaces which are unsupported by the device runtime.
     * Kernels launched via the device runtime only support surfaces created with
     * the Surface Object API's.
     */
    UPTKErrorLaunchFileScopedSurf         =     67,

    /**
     * This error indicates that a call to ::UPTKDeviceSynchronize made from
     * the device runtime failed because the call was made at grid depth greater
     * than than either the default (2 levels of grids) or user specified device
     * limit ::UPTKLimitDevRuntimeSyncDepth. To be able to synchronize on
     * launched grids at a greater depth successfully, the maximum nested
     * depth at which ::UPTKDeviceSynchronize will be called must be specified
     * with the ::UPTKLimitDevRuntimeSyncDepth limit to the ::UPTKDeviceSetLimit
     * api before the host-side launch of a kernel using the device runtime.
     * Keep in mind that additional levels of sync depth require the runtime
     * to reserve large amounts of device memory that cannot be used for
     * user allocations. Note that ::UPTKDeviceSynchronize made from device
     * runtime is only supported on devices of compute capability < 9.0.
     */
    UPTKErrorSyncDepthExceeded            =     68,

    /**
     * This error indicates that a device runtime grid launch failed because
     * the launch would exceed the limit ::UPTKLimitDevRuntimePendingLaunchCount.
     * For this launch to proceed successfully, ::UPTKDeviceSetLimit must be
     * called to set the ::UPTKLimitDevRuntimePendingLaunchCount to be higher 
     * than the upper bound of outstanding launches that can be issued to the
     * device runtime. Keep in mind that raising the limit of pending device
     * runtime launches will require the runtime to reserve device memory that
     * cannot be used for user allocations.
     */
    UPTKErrorLaunchPendingCountExceeded   =     69,
  
    /**
     * The requested device function does not exist or is not compiled for the
     * proper device architecture.
     */
    UPTKErrorInvalidDeviceFunction        =      98,
  
    /**
     * This indicates that no UPTK-capable devices were detected by the installed
     * UPTK driver.
     */
    UPTKErrorNoDevice                     =     100,
  
    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid UPTK device or that the action requested is
     * invalid for the specified device.
     */
    UPTKErrorInvalidDevice                =     101,

    /**
     * This indicates that the device doesn't have a valid Grid License.
     */
    UPTKErrorDeviceNotLicensed            =     102,

   /**
    * By default, the UPTK runtime may perform a minimal set of self-tests,
    * as well as UPTK driver tests, to establish the validity of both.
    * Introduced in UPTK 11.2, this error return indicates that at least one
    * of these tests has failed and the validity of either the runtime
    * or the driver could not be established.
    */
   UPTKErrorSoftwareValidityNotEstablished  =     103,

    /**
     * This indicates an internal startup failure in the UPTK runtime.
     */
    UPTKErrorStartupFailure               =    127,
  
    /**
     * This indicates that the device kernel image is invalid.
     */
    UPTKErrorInvalidKernelImage           =     200,

    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::cuCtxDestroy() invoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::cuCtxGetApiVersion() for more details.
     */
    UPTKErrorDeviceUninitialized          =     201,

    /**
     * This indicates that the buffer object could not be mapped.
     */
    UPTKErrorMapBufferObjectFailed        =     205,
  
    /**
     * This indicates that the buffer object could not be unmapped.
     */
    UPTKErrorUnmapBufferObjectFailed      =     206,

    /**
     * This indicates that the specified array is currently mapped and thus
     * cannot be destroyed.
     */
    UPTKErrorArrayIsMapped                =     207,

    /**
     * This indicates that the resource is already mapped.
     */
    UPTKErrorAlreadyMapped                =     208,
  
    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular UPTK source file that do not include the
     * corresponding device configuration.
     */
    UPTKErrorNoKernelImageForDevice       =     209,

    /**
     * This indicates that a resource has already been acquired.
     */
    UPTKErrorAlreadyAcquired              =     210,

    /**
     * This indicates that a resource is not mapped.
     */
    UPTKErrorNotMapped                    =     211,

    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    UPTKErrorNotMappedAsArray             =     212,

    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    UPTKErrorNotMappedAsPointer           =     213,
  
    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    UPTKErrorECCUncorrectable             =     214,
  
    /**
     * This indicates that the ::UPTKLimit passed to the API call is not
     * supported by the active device.
     */
    UPTKErrorUnsupportedLimit             =     215,
    
    /**
     * This indicates that a call tried to access an exclusive-thread device that 
     * is already in use by a different thread.
     */
    UPTKErrorDeviceAlreadyInUse           =     216,

    /**
     * This error indicates that P2P access is not supported across the given
     * devices.
     */
    UPTKErrorPeerAccessUnsupported        =     217,

    /**
     * A PTX compilation failed. The runtime may fall back to compiling PTX if
     * an application does not contain a suitable binary for the current device.
     */
    UPTKErrorInvalidPtx                   =     218,

    /**
     * This indicates an error with the OpenGL or DirectX context.
     */
    UPTKErrorInvalidGraphicsContext       =     219,

    /**
     * This indicates that an uncorrectable NVLink error was detected during the
     * execution.
     */
    UPTKErrorNvlinkUncorrectable          =     220,

    /**
     * This indicates that the PTX JIT compiler library was not found. The JIT Compiler
     * library is used for PTX compilation. The runtime may fall back to compiling PTX
     * if an application does not contain a suitable binary for the current device.
     */
    UPTKErrorJitCompilerNotFound          =     221,

    /**
     * This indicates that the provided PTX was compiled with an unsupported toolchain.
     * The most common reason for this, is the PTX was generated by a compiler newer
     * than what is supported by the UPTK driver and PTX JIT compiler.
     */
    UPTKErrorUnsupportedPtxVersion        =     222,

    /**
     * This indicates that the JIT compilation was disabled. The JIT compilation compiles
     * PTX. The runtime may fall back to compiling PTX if an application does not contain
     * a suitable binary for the current device.
     */
    UPTKErrorJitCompilationDisabled       =     223,

    /**
     * This indicates that the provided execution affinity is not supported by the device.
     */
    UPTKErrorUnsupportedExecAffinity      =     224,

    /**
     * This indicates that the code to be compiled by the PTX JIT contains
     * unsupported call to UPTKDeviceSynchronize.
     */
    UPTKErrorUnsupportedDevSideSync       =     225,

    /**
     * This indicates that the device kernel source is invalid.
     */
    UPTKErrorInvalidSource                =     300,

    /**
     * This indicates that the file specified was not found.
     */
    UPTKErrorFileNotFound                 =     301,
  
    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    UPTKErrorSharedObjectSymbolNotFound   =     302,
  
    /**
     * This indicates that initialization of a shared object failed.
     */
    UPTKErrorSharedObjectInitFailed       =     303,

    /**
     * This error indicates that an OS call failed.
     */
    UPTKErrorOperatingSystem              =     304,
  
    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::UPTKStream_t and
     * ::UPTKEvent_t.
     */
    UPTKErrorInvalidResourceHandle        =     400,

    /**
     * This indicates that a resource required by the API call is not in a
     * valid state to perform the requested operation.
     */
    UPTKErrorIllegalState                 =     401,

    /**
     * This indicates an attempt was made to introspect an object in a way that
     * would discard semantically important information. This is either due to
     * the object using funtionality newer than the API version used to
     * introspect it or omission of optional return arguments.
     */
    UPTKErrorLossyQuery                   =     402,

    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, driver function names, texture names,
     * and surface names.
     */
    UPTKErrorSymbolNotFound               =     500,
  
    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::UPTKSuccess (which indicates completion). Calls that
     * may return this value include ::UPTKEventQuery() and ::UPTKStreamQuery().
     */
    UPTKErrorNotReady                     =     600,

    /**
     * The device encountered a load or store instruction on an invalid memory address.
     * This leaves the process in an inconsistent state and any further UPTK work
     * will return the same error. To continue using UPTK, the process must be terminated
     * and relaunched.
     */
    UPTKErrorIllegalAddress               =     700,
  
    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. Although this error is similar to
     * ::UPTKErrorInvalidConfiguration, this error usually indicates that the
     * user has attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register count.
     */
    UPTKErrorLaunchOutOfResources         =      701,
  
    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device property
     * \ref ::UPTKDeviceProp::kernelExecTimeoutEnabled "kernelExecTimeoutEnabled"
     * for more information.
     * This leaves the process in an inconsistent state and any further UPTK work
     * will return the same error. To continue using UPTK, the process must be terminated
     * and relaunched.
     */
    UPTKErrorLaunchTimeout                =      702,

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    UPTKErrorLaunchIncompatibleTexturing  =     703,
      
    /**
     * This error indicates that a call to ::UPTKDeviceEnablePeerAccess() is
     * trying to re-enable peer addressing on from a context which has already
     * had peer addressing enabled.
     */
    UPTKErrorPeerAccessAlreadyEnabled     =     704,
    
    /**
     * This error indicates that ::UPTKDeviceDisablePeerAccess() is trying to 
     * disable peer addressing which has not been enabled yet via 
     * ::UPTKDeviceEnablePeerAccess().
     */
    UPTKErrorPeerAccessNotEnabled         =     705,
  
    /**
     * This indicates that the user has called ::UPTKSetValidDevices(),
     * ::UPTKSetDeviceFlags(), ::UPTKD3D9SetDirect3DDevice(),
     * ::UPTKD3D10SetDirect3DDevice, ::UPTKD3D11SetDirect3DDevice(), or
     * ::UPTKVDPAUSetVDPAUDevice() after initializing the UPTK runtime by
     * calling non-device management operations (allocating memory and
     * launching kernels are examples of non-device management operations).
     * This error can also be returned if using runtime/driver
     * interoperability and there is an existing ::CUcontext active on the
     * host thread.
     */
    UPTKErrorSetOnActiveProcess           =     708,

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::cuCtxDestroy, or is a primary context which
     * has not yet been initialized.
     */
    UPTKErrorContextIsDestroyed           =     709,

    /**
     * An assert triggered in device code during kernel execution. The device
     * cannot be used again. All existing allocations are invalid. To continue
     * using UPTK, the process must be terminated and relaunched.
     */
    UPTKErrorAssert                        =    710,
  
    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices 
     * passed to ::UPTKEnablePeerAccess().
     */
    UPTKErrorTooManyPeers                 =     711,
  
    /**
     * This error indicates that the memory range passed to ::UPTKHostRegister()
     * has already been registered.
     */
    UPTKErrorHostMemoryAlreadyRegistered  =     712,
        
    /**
     * This error indicates that the pointer passed to ::UPTKHostUnregister()
     * does not correspond to any currently registered memory region.
     */
    UPTKErrorHostMemoryNotRegistered      =     713,

    /**
     * Device encountered an error in the call stack during kernel execution,
     * possibly due to stack corruption or exceeding the stack size limit.
     * This leaves the process in an inconsistent state and any further UPTK work
     * will return the same error. To continue using UPTK, the process must be terminated
     * and relaunched.
     */
    UPTKErrorHardwareStackError           =     714,

    /**
     * The device encountered an illegal instruction during kernel execution
     * This leaves the process in an inconsistent state and any further UPTK work
     * will return the same error. To continue using UPTK, the process must be terminated
     * and relaunched.
     */
    UPTKErrorIllegalInstruction           =     715,

    /**
     * The device encountered a load or store instruction
     * on a memory address which is not aligned.
     * This leaves the process in an inconsistent state and any further UPTK work
     * will return the same error. To continue using UPTK, the process must be terminated
     * and relaunched.
     */
    UPTKErrorMisalignedAddress            =     716,

    /**
     * While executing a kernel, the device encountered an instruction
     * which can only operate on memory locations in certain address spaces
     * (global, shared, or local), but was supplied a memory address not
     * belonging to an allowed address space.
     * This leaves the process in an inconsistent state and any further UPTK work
     * will return the same error. To continue using UPTK, the process must be terminated
     * and relaunched.
     */
    UPTKErrorInvalidAddressSpace          =     717,

    /**
     * The device encountered an invalid program counter.
     * This leaves the process in an inconsistent state and any further UPTK work
     * will return the same error. To continue using UPTK, the process must be terminated
     * and relaunched.
     */
    UPTKErrorInvalidPc                    =     718,
  
    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. Less common cases can be system specific - more
     * information about these cases can be found in the system specific user guide.
     * This leaves the process in an inconsistent state and any further UPTK work
     * will return the same error. To continue using UPTK, the process must be terminated
     * and relaunched.
     */
    UPTKErrorLaunchFailure                =      719,

    /**
     * This error indicates that the number of blocks launched per grid for a kernel that was
     * launched via either ::UPTKLaunchCooperativeKernel or ::UPTKLaunchCooperativeKernelMultiDevice
     * exceeds the maximum number of blocks as allowed by ::UPTKOccupancyMaxActiveBlocksPerMultiprocessor
     * or ::UPTKOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
     * as specified by the device attribute ::UPTKDevAttrMultiProcessorCount.
     */
    UPTKErrorCooperativeLaunchTooLarge    =     720,
    
    /**
     * This error indicates the attempted operation is not permitted.
     */
    UPTKErrorNotPermitted                 =     800,

    /**
     * This error indicates the attempted operation is not supported
     * on the current system or device.
     */
    UPTKErrorNotSupported                 =     801,

    /**
     * This error indicates that the system is not yet ready to start any UPTK
     * work.  To continue using UPTK, verify the system configuration is in a
     * valid state and all required driver daemons are actively running.
     * More information about this error can be found in the system specific
     * user guide.
     */
    UPTKErrorSystemNotReady               =     802,

    /**
     * This error indicates that there is a mismatch between the versions of
     * the display driver and the UPTK driver. Refer to the compatibility documentation
     * for supported versions.
     */
    UPTKErrorSystemDriverMismatch         =     803,

    /**
     * This error indicates that the system was upgraded to run with forward compatibility
     * but the visible hardware detected by UPTK does not support this configuration.
     * Refer to the compatibility documentation for the supported hardware matrix or ensure
     * that only supported hardware is visible during initialization via the UPTK_VISIBLE_DEVICES
     * environment variable.
     */
    UPTKErrorCompatNotSupportedOnDevice   =     804,

    /**
     * This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.
     */
    UPTKErrorMpsConnectionFailed          =     805,

    /**
     * This error indicates that the remote procedural call between the MPS server and the MPS client failed.
     */
    UPTKErrorMpsRpcFailure                =     806,

    /**
     * This error indicates that the MPS server is not ready to accept new MPS client requests.
     * This error can be returned when the MPS server is in the process of recovering from a fatal failure.
     */
    UPTKErrorMpsServerNotReady            =     807,

    /**
     * This error indicates that the hardware resources required to create MPS client have been exhausted.
     */
    UPTKErrorMpsMaxClientsReached         =     808,

    /**
     * This error indicates the the hardware resources required to device connections have been exhausted.
     */
    UPTKErrorMpsMaxConnectionsReached     =     809,

    /**
     * This error indicates that the MPS client has been terminated by the server. To continue using UPTK, the process must be terminated and relaunched.
     */
    UPTKErrorMpsClientTerminated          =     810,

    /**
     * This error indicates, that the program is using UPTK Dynamic Parallelism, but the current configuration, like MPS, does not support it.
     */
    UPTKErrorCdpNotSupported              =     811,

    /**
     * This error indicates, that the program contains an unsupported interaction between different versions of UPTK Dynamic Parallelism.
     */
    UPTKErrorCdpVersionMismatch           =     812,

    /**
     * The operation is not permitted when the stream is capturing.
     */
    UPTKErrorStreamCaptureUnsupported     =    900,

    /**
     * The current capture sequence on the stream has been invalidated due to
     * a previous error.
     */
    UPTKErrorStreamCaptureInvalidated     =    901,

    /**
     * The operation would have resulted in a merge of two independent capture
     * sequences.
     */
    UPTKErrorStreamCaptureMerge           =    902,

    /**
     * The capture was not initiated in this stream.
     */
    UPTKErrorStreamCaptureUnmatched       =    903,

    /**
     * The capture sequence contains a fork that was not joined to the primary
     * stream.
     */
    UPTKErrorStreamCaptureUnjoined        =    904,

    /**
     * A dependency would have been created which crosses the capture sequence
     * boundary. Only implicit in-stream ordering dependencies are allowed to
     * cross the boundary.
     */
    UPTKErrorStreamCaptureIsolation       =    905,

    /**
     * The operation would have resulted in a disallowed implicit dependency on
     * a current capture sequence from UPTKStreamLegacy.
     */
    UPTKErrorStreamCaptureImplicit        =    906,

    /**
     * The operation is not permitted on an event which was last recorded in a
     * capturing stream.
     */
    UPTKErrorCapturedEvent                =    907,
  
    /**
     * A stream capture sequence not initiated with the ::UPTKStreamCaptureModeRelaxed
     * argument to ::UPTKStreamBeginCapture was passed to ::UPTKStreamEndCapture in a
     * different thread.
     */
    UPTKErrorStreamCaptureWrongThread     =    908,

    /**
     * This indicates that the wait operation has timed out.
     */
    UPTKErrorTimeout                      =    909,

    /**
     * This error indicates that the graph update was not performed because it included 
     * changes which violated constraints specific to instantiated graph update.
     */
    UPTKErrorGraphExecUpdateFailure       =    910,

    /**
     * This indicates that an async error has occurred in a device outside of UPTK.
     * If UPTK was waiting for an external device's signal before consuming shared data,
     * the external device signaled an error indicating that the data is not valid for
     * consumption. This leaves the process in an inconsistent state and any further UPTK
     * work will return the same error. To continue using UPTK, the process must be
     * terminated and relaunched.
     */
    UPTKErrorExternalDevice               =    911,

    /**
     * This indicates that a kernel launch error has occurred due to cluster
     * misconfiguration.
     */
    UPTKErrorInvalidClusterSize           =    912,

    /**
     * Indiciates a function handle is not loaded when calling an API that requires
     * a loaded function.
     */
    UPTKErrorFunctionNotLoaded            =    913,

    /**
     * This error indicates one or more resources passed in are not valid resource
     * types for the operation.
     */
    UPTKErrorInvalidResourceType          =    914,

    /**
     * This error indicates one or more resources are insufficient or non-applicable for
     * the operation.
     */
    UPTKErrorInvalidResourceConfiguration =    915,

    /**
     * This indicates that an unknown internal error has occurred.
     */
    UPTKErrorUnknown                      =    999

    /**
     * Any unhandled UPTK driver error is added to this value and returned via
     * the runtime. Production releases of UPTK should not return such errors.
     * \deprecated
     * This error return is deprecated as of UPTK 4.1.
     */
    , UPTKErrorApiFailureBase               =  10000
};

/**
 * Channel format kind
 */
enum UPTKChannelFormatKind
{
    UPTKChannelFormatKindSigned                         =   0,      /**< Signed channel format */
    UPTKChannelFormatKindUnsigned                       =   1,      /**< Unsigned channel format */
    UPTKChannelFormatKindFloat                          =   2,      /**< Float channel format */
    UPTKChannelFormatKindNone                           =   3,      /**< No channel format */
    UPTKChannelFormatKindNV12                           =   4,      /**< Unsigned 8-bit integers, planar 4:2:0 YUV format */
    UPTKChannelFormatKindUnsignedNormalized8X1          =   5,      /**< 1 channel unsigned 8-bit normalized integer */
    UPTKChannelFormatKindUnsignedNormalized8X2          =   6,      /**< 2 channel unsigned 8-bit normalized integer */
    UPTKChannelFormatKindUnsignedNormalized8X4          =   7,      /**< 4 channel unsigned 8-bit normalized integer */
    UPTKChannelFormatKindUnsignedNormalized16X1         =   8,      /**< 1 channel unsigned 16-bit normalized integer */
    UPTKChannelFormatKindUnsignedNormalized16X2         =   9,      /**< 2 channel unsigned 16-bit normalized integer */
    UPTKChannelFormatKindUnsignedNormalized16X4         =   10,     /**< 4 channel unsigned 16-bit normalized integer */
    UPTKChannelFormatKindSignedNormalized8X1            =   11,     /**< 1 channel signed 8-bit normalized integer */
    UPTKChannelFormatKindSignedNormalized8X2            =   12,     /**< 2 channel signed 8-bit normalized integer */
    UPTKChannelFormatKindSignedNormalized8X4            =   13,     /**< 4 channel signed 8-bit normalized integer */
    UPTKChannelFormatKindSignedNormalized16X1           =   14,     /**< 1 channel signed 16-bit normalized integer */
    UPTKChannelFormatKindSignedNormalized16X2           =   15,     /**< 2 channel signed 16-bit normalized integer */
    UPTKChannelFormatKindSignedNormalized16X4           =   16,     /**< 4 channel signed 16-bit normalized integer */
    UPTKChannelFormatKindUnsignedBlockCompressed1       =   17,     /**< 4 channel unsigned normalized block-compressed (BC1 compression) format */
    UPTKChannelFormatKindUnsignedBlockCompressed1SRGB   =   18,     /**< 4 channel unsigned normalized block-compressed (BC1 compression) format with sRGB encoding*/
    UPTKChannelFormatKindUnsignedBlockCompressed2       =   19,     /**< 4 channel unsigned normalized block-compressed (BC2 compression) format */
    UPTKChannelFormatKindUnsignedBlockCompressed2SRGB   =   20,     /**< 4 channel unsigned normalized block-compressed (BC2 compression) format with sRGB encoding */
    UPTKChannelFormatKindUnsignedBlockCompressed3       =   21,     /**< 4 channel unsigned normalized block-compressed (BC3 compression) format */
    UPTKChannelFormatKindUnsignedBlockCompressed3SRGB   =   22,     /**< 4 channel unsigned normalized block-compressed (BC3 compression) format with sRGB encoding */
    UPTKChannelFormatKindUnsignedBlockCompressed4       =   23,     /**< 1 channel unsigned normalized block-compressed (BC4 compression) format */
    UPTKChannelFormatKindSignedBlockCompressed4         =   24,     /**< 1 channel signed normalized block-compressed (BC4 compression) format */
    UPTKChannelFormatKindUnsignedBlockCompressed5       =   25,     /**< 2 channel unsigned normalized block-compressed (BC5 compression) format */
    UPTKChannelFormatKindSignedBlockCompressed5         =   26,     /**< 2 channel signed normalized block-compressed (BC5 compression) format */
    UPTKChannelFormatKindUnsignedBlockCompressed6H      =   27,     /**< 3 channel unsigned half-float block-compressed (BC6H compression) format */
    UPTKChannelFormatKindSignedBlockCompressed6H        =   28,     /**< 3 channel signed half-float block-compressed (BC6H compression) format */
    UPTKChannelFormatKindUnsignedBlockCompressed7       =   29,     /**< 4 channel unsigned normalized block-compressed (BC7 compression) format */
    UPTKChannelFormatKindUnsignedBlockCompressed7SRGB   =   30      /**< 4 channel unsigned normalized block-compressed (BC7 compression) format with sRGB encoding */
};

/**
 * UPTK Channel format descriptor
 */
struct UPTKChannelFormatDesc
{
    int                        x; /**< x */
    int                        y; /**< y */
    int                        z; /**< z */
    int                        w; /**< w */
    enum UPTKChannelFormatKind f; /**< Channel format kind */
};

/**
 * UPTK array
 */
typedef struct UPTKArray *UPTKArray_t;

/**
 * UPTK array (as source copy argument)
 */
typedef const struct UPTKArray *UPTKArray_const_t;

struct UPTKArray;

/**
 * UPTK mipmapped array
 */
typedef struct UPTKMipmappedArray *UPTKMipmappedArray_t;

/**
 * UPTK mipmapped array (as source argument)
 */
typedef const struct UPTKMipmappedArray *UPTKMipmappedArray_const_t;

struct UPTKMipmappedArray;

/**
 * Indicates that the layered sparse UPTK array or UPTK mipmapped array has a single mip tail region for all layers
 */
#define UPTKArraySparsePropertiesSingleMipTail   0x1

/**
 * Sparse UPTK array and UPTK mipmapped array properties
 */
struct UPTKArraySparseProperties {
    struct {
        unsigned int width;             /**< Tile width in elements */
        unsigned int height;            /**< Tile height in elements */
        unsigned int depth;             /**< Tile depth in elements */
    } tileExtent;
    unsigned int miptailFirstLevel;     /**< First mip level at which the mip tail begins */   
    unsigned long long miptailSize;     /**< Total size of the mip tail. */
    unsigned int flags;                 /**< Flags will either be zero or ::UPTKArraySparsePropertiesSingleMipTail */
    unsigned int reserved[4];
};

/**
 * UPTK array and UPTK mipmapped array memory requirements
 */
struct UPTKArrayMemoryRequirements {
    size_t size;                    /**< Total size of the array. */
    size_t alignment;               /**< Alignment necessary for mapping the array. */
    unsigned int reserved[4];
};

/**
 * UPTK memory types
 */
enum UPTKMemoryType
{
    UPTKMemoryTypeUnregistered = 0, /**< Unregistered memory */
    UPTKMemoryTypeHost         = 1, /**< Host memory */
    UPTKMemoryTypeDevice       = 2, /**< Device memory */
    UPTKMemoryTypeManaged      = 3  /**< Managed memory */
};

/**
 * UPTK memory copy types
 */
enum UPTKMemcpyKind
{
    UPTKMemcpyHostToHost          =   0,      /**< Host   -> Host */
    UPTKMemcpyHostToDevice        =   1,      /**< Host   -> Device */
    UPTKMemcpyDeviceToHost        =   2,      /**< Device -> Host */
    UPTKMemcpyDeviceToDevice      =   3,      /**< Device -> Device */
    UPTKMemcpyDefault             =   4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};

/**
 * UPTK Pitched memory pointer
 *
 * \sa ::make_UPTKPitchedPtr
 */
struct UPTKPitchedPtr
{
    void   *ptr;      /**< Pointer to allocated memory */
    size_t  pitch;    /**< Pitch of allocated memory in bytes */
    size_t  xsize;    /**< Logical width of allocation in elements */
    size_t  ysize;    /**< Logical height of allocation in elements */
};

/**
 * UPTK extent
 *
 * \sa ::make_UPTKExtent
 */
struct UPTKExtent
{
    size_t width;     /**< Width in elements when referring to array memory, in bytes when referring to linear memory */
    size_t height;    /**< Height in elements */
    size_t depth;     /**< Depth in elements */
};

/**
 * UPTK 3D position
 *
 * \sa ::make_UPTKPos
 */
struct UPTKPos
{
    size_t x;     /**< x */
    size_t y;     /**< y */
    size_t z;     /**< z */
};

/**
 * UPTK 3D memory copying parameters
 */
struct UPTKMemcpy3DParms
{
    UPTKArray_t            srcArray;  /**< Source memory address */
    struct UPTKPos         srcPos;    /**< Source position offset */
    struct UPTKPitchedPtr  srcPtr;    /**< Pitched source memory address */
  
    UPTKArray_t            dstArray;  /**< Destination memory address */
    struct UPTKPos         dstPos;    /**< Destination position offset */
    struct UPTKPitchedPtr  dstPtr;    /**< Pitched destination memory address */
  
    struct UPTKExtent      extent;    /**< Requested memory copy size */
    enum UPTKMemcpyKind    kind;      /**< Type of transfer */
};

/**
 * Memcpy node parameters
 */
struct UPTKMemcpyNodeParams {
    int flags;                            /**< Must be zero */
    int reserved[3];                      /**< Must be zero */
    struct UPTKMemcpy3DParms copyParams;  /**< Parameters for the memory copy */
};

/**
 * UPTK 3D cross-device memory copying parameters
 */
struct UPTKMemcpy3DPeerParms
{
    UPTKArray_t            srcArray;  /**< Source memory address */
    struct UPTKPos         srcPos;    /**< Source position offset */
    struct UPTKPitchedPtr  srcPtr;    /**< Pitched source memory address */
    int                    srcDevice; /**< Source device */
  
    UPTKArray_t            dstArray;  /**< Destination memory address */
    struct UPTKPos         dstPos;    /**< Destination position offset */
    struct UPTKPitchedPtr  dstPtr;    /**< Pitched destination memory address */
    int                    dstDevice; /**< Destination device */
  
    struct UPTKExtent      extent;    /**< Requested memory copy size */
};

/**
 * UPTK Memset node parameters
 */
struct  UPTKMemsetParams {
    void *dst;                              /**< Destination device pointer */
    size_t pitch;                           /**< Pitch of destination device pointer. Unused if height is 1 */
    unsigned int value;                     /**< Value to be set */
    unsigned int elementSize;               /**< Size of each element in bytes. Must be 1, 2, or 4. */
    size_t width;                           /**< Width of the row in elements */
    size_t height;                          /**< Number of rows */
};

/**
 * UPTK Memset node parameters
 */
struct  UPTKMemsetParamsV2 {
    void *dst;                              /**< Destination device pointer */
    size_t pitch;                           /**< Pitch of destination device pointer. Unused if height is 1 */
    unsigned int value;                     /**< Value to be set */
    unsigned int elementSize;               /**< Size of each element in bytes. Must be 1, 2, or 4. */
    size_t width;                           /**< Width of the row in elements */
    size_t height;                          /**< Number of rows */
};

/**
 * Specifies performance hint with ::UPTKAccessPolicyWindow for hitProp and missProp members.
 */
enum  UPTKAccessProperty {
    UPTKAccessPropertyNormal = 0,       /**< Normal cache persistence. */
    UPTKAccessPropertyStreaming = 1,    /**< Streaming access is less likely to persit from cache. */
    UPTKAccessPropertyPersisting = 2    /**< Persisting access is more likely to persist in cache.*/
};

/**
 * Specifies an access policy for a window, a contiguous extent of memory
 * beginning at base_ptr and ending at base_ptr + num_bytes.
 * Partition into many segments and assign segments such that.
 * sum of "hit segments" / window == approx. ratio.
 * sum of "miss segments" / window == approx 1-ratio.
 * Segments and ratio specifications are fitted to the capabilities of
 * the architecture.
 * Accesses in a hit segment apply the hitProp access policy.
 * Accesses in a miss segment apply the missProp access policy.
 */
struct UPTKAccessPolicyWindow {
    void *base_ptr;                     /**< Starting address of the access policy window. UPTK driver may align it. */
    size_t num_bytes;                   /**< Size in bytes of the window policy. UPTK driver may restrict the maximum size and alignment. */
    float hitRatio;                     /**< hitRatio specifies percentage of lines assigned hitProp, rest are assigned missProp. */
    enum UPTKAccessProperty hitProp;    /**< ::CUaccessProperty set for hit. */
    enum UPTKAccessProperty missProp;   /**< ::CUaccessProperty set for miss. Must be either NORMAL or STREAMING. */
};

#ifdef _WIN32
#define UPTKRT_CB __stdcall
#else
#define UPTKRT_CB
#endif

/**
 * UPTK host function
 * \param userData Argument value passed to the function
 */
typedef void (UPTKRT_CB *UPTKHostFn_t)(void *userData);

/**
 * UPTK host node parameters
 */
struct UPTKHostNodeParams {
    UPTKHostFn_t fn;    /**< The function to call when the node executes */
    void* userData; /**< Argument to pass to the function */
};

/**
 * UPTK host node parameters
 */
struct UPTKHostNodeParamsV2 {
    UPTKHostFn_t fn;    /**< The function to call when the node executes */
    void* userData; /**< Argument to pass to the function */
};

/**
 * Possible stream capture statuses returned by ::UPTKStreamIsCapturing
 */
enum UPTKStreamCaptureStatus {
    UPTKStreamCaptureStatusNone        = 0, /**< Stream is not capturing */
    UPTKStreamCaptureStatusActive      = 1, /**< Stream is actively capturing */
    UPTKStreamCaptureStatusInvalidated = 2  /**< Stream is part of a capture sequence that
                                                   has been invalidated, but not terminated */
};

/**
 * Possible modes for stream capture thread interactions. For more details see
 * ::UPTKStreamBeginCapture and ::UPTKThreadExchangeStreamCaptureMode
 */
enum UPTKStreamCaptureMode {
    UPTKStreamCaptureModeGlobal      = 0,
    UPTKStreamCaptureModeThreadLocal = 1,
    UPTKStreamCaptureModeRelaxed     = 2
};

enum UPTKSynchronizationPolicy {
    UPTKSyncPolicyAuto = 1,
    UPTKSyncPolicySpin = 2,
    UPTKSyncPolicyYield = 3,
    UPTKSyncPolicyBlockingSync = 4
};

/**
 * Cluster scheduling policies. These may be passed to ::UPTKFuncSetAttribute
 */
enum UPTKClusterSchedulingPolicy {
    UPTKClusterSchedulingPolicyDefault       = 0, /**< the default policy */
    UPTKClusterSchedulingPolicySpread        = 1, /**< spread the blocks within a cluster to the SMs */
    UPTKClusterSchedulingPolicyLoadBalancing = 2  /**< allow the hardware to load-balance the blocks in a cluster to the SMs */
};

/**
 * Flags for ::UPTKStreamUpdateCaptureDependencies
 */
enum UPTKStreamUpdateCaptureDependenciesFlags {
    UPTKStreamAddCaptureDependencies = 0x0, /**< Add new nodes to the dependency set */
    UPTKStreamSetCaptureDependencies = 0x1  /**< Replace the dependency set with the new nodes */
};

/**
 * Flags for user objects for graphs
 */
enum UPTKUserObjectFlags {
    UPTKUserObjectNoDestructorSync = 0x1  /**< Indicates the destructor execution is not synchronized by any UPTK handle. */
};

/**
 * Flags for retaining user object references for graphs
 */
enum UPTKUserObjectRetainFlags {
    UPTKGraphUserObjectMove = 0x1  /**< Transfer references from the caller rather than creating new references. */
};

/**
 * UPTK graphics interop resource
 */
struct UPTKGraphicsResource;

/**
 * UPTK graphics interop register flags
 */
enum UPTKGraphicsRegisterFlags
{
    UPTKGraphicsRegisterFlagsNone             = 0,  /**< Default */
    UPTKGraphicsRegisterFlagsReadOnly         = 1,  /**< UPTK will not write to this resource */ 
    UPTKGraphicsRegisterFlagsWriteDiscard     = 2,  /**< UPTK will only write to and will not read from this resource */
    UPTKGraphicsRegisterFlagsSurfaceLoadStore = 4,  /**< UPTK will bind this resource to a surface reference */
    UPTKGraphicsRegisterFlagsTextureGather    = 8   /**< UPTK will perform texture gather operations on this resource */
};

/**
 * UPTK graphics interop map flags
 */
enum UPTKGraphicsMapFlags
{
    UPTKGraphicsMapFlagsNone         = 0,  /**< Default; Assume resource can be read/written */
    UPTKGraphicsMapFlagsReadOnly     = 1,  /**< UPTK will not write to this resource */
    UPTKGraphicsMapFlagsWriteDiscard = 2   /**< UPTK will only write to and will not read from this resource */
};

/**
 * UPTK graphics interop array indices for cube maps
 */
enum UPTKGraphicsCubeFace 
{
    UPTKGraphicsCubeFacePositiveX = 0x00, /**< Positive X face of cubemap */
    UPTKGraphicsCubeFaceNegativeX = 0x01, /**< Negative X face of cubemap */
    UPTKGraphicsCubeFacePositiveY = 0x02, /**< Positive Y face of cubemap */
    UPTKGraphicsCubeFaceNegativeY = 0x03, /**< Negative Y face of cubemap */
    UPTKGraphicsCubeFacePositiveZ = 0x04, /**< Positive Z face of cubemap */
    UPTKGraphicsCubeFaceNegativeZ = 0x05  /**< Negative Z face of cubemap */
};

/**
 * UPTK resource types
 */
enum UPTKResourceType
{
    UPTKResourceTypeArray          = 0x00, /**< Array resource */
    UPTKResourceTypeMipmappedArray = 0x01, /**< Mipmapped array resource */
    UPTKResourceTypeLinear         = 0x02, /**< Linear resource */
    UPTKResourceTypePitch2D        = 0x03  /**< Pitch 2D resource */
};

/**
 * UPTK texture resource view formats
 */
enum UPTKResourceViewFormat
{
    UPTKResViewFormatNone                      = 0x00, /**< No resource view format (use underlying resource format) */
    UPTKResViewFormatUnsignedChar1             = 0x01, /**< 1 channel unsigned 8-bit integers */
    UPTKResViewFormatUnsignedChar2             = 0x02, /**< 2 channel unsigned 8-bit integers */
    UPTKResViewFormatUnsignedChar4             = 0x03, /**< 4 channel unsigned 8-bit integers */
    UPTKResViewFormatSignedChar1               = 0x04, /**< 1 channel signed 8-bit integers */
    UPTKResViewFormatSignedChar2               = 0x05, /**< 2 channel signed 8-bit integers */
    UPTKResViewFormatSignedChar4               = 0x06, /**< 4 channel signed 8-bit integers */
    UPTKResViewFormatUnsignedShort1            = 0x07, /**< 1 channel unsigned 16-bit integers */
    UPTKResViewFormatUnsignedShort2            = 0x08, /**< 2 channel unsigned 16-bit integers */
    UPTKResViewFormatUnsignedShort4            = 0x09, /**< 4 channel unsigned 16-bit integers */
    UPTKResViewFormatSignedShort1              = 0x0a, /**< 1 channel signed 16-bit integers */
    UPTKResViewFormatSignedShort2              = 0x0b, /**< 2 channel signed 16-bit integers */
    UPTKResViewFormatSignedShort4              = 0x0c, /**< 4 channel signed 16-bit integers */
    UPTKResViewFormatUnsignedInt1              = 0x0d, /**< 1 channel unsigned 32-bit integers */
    UPTKResViewFormatUnsignedInt2              = 0x0e, /**< 2 channel unsigned 32-bit integers */
    UPTKResViewFormatUnsignedInt4              = 0x0f, /**< 4 channel unsigned 32-bit integers */
    UPTKResViewFormatSignedInt1                = 0x10, /**< 1 channel signed 32-bit integers */
    UPTKResViewFormatSignedInt2                = 0x11, /**< 2 channel signed 32-bit integers */
    UPTKResViewFormatSignedInt4                = 0x12, /**< 4 channel signed 32-bit integers */
    UPTKResViewFormatHalf1                     = 0x13, /**< 1 channel 16-bit floating point */
    UPTKResViewFormatHalf2                     = 0x14, /**< 2 channel 16-bit floating point */
    UPTKResViewFormatHalf4                     = 0x15, /**< 4 channel 16-bit floating point */
    UPTKResViewFormatFloat1                    = 0x16, /**< 1 channel 32-bit floating point */
    UPTKResViewFormatFloat2                    = 0x17, /**< 2 channel 32-bit floating point */
    UPTKResViewFormatFloat4                    = 0x18, /**< 4 channel 32-bit floating point */
    UPTKResViewFormatUnsignedBlockCompressed1  = 0x19, /**< Block compressed 1 */
    UPTKResViewFormatUnsignedBlockCompressed2  = 0x1a, /**< Block compressed 2 */
    UPTKResViewFormatUnsignedBlockCompressed3  = 0x1b, /**< Block compressed 3 */
    UPTKResViewFormatUnsignedBlockCompressed4  = 0x1c, /**< Block compressed 4 unsigned */
    UPTKResViewFormatSignedBlockCompressed4    = 0x1d, /**< Block compressed 4 signed */
    UPTKResViewFormatUnsignedBlockCompressed5  = 0x1e, /**< Block compressed 5 unsigned */
    UPTKResViewFormatSignedBlockCompressed5    = 0x1f, /**< Block compressed 5 signed */
    UPTKResViewFormatUnsignedBlockCompressed6H = 0x20, /**< Block compressed 6 unsigned half-float */
    UPTKResViewFormatSignedBlockCompressed6H   = 0x21, /**< Block compressed 6 signed half-float */
    UPTKResViewFormatUnsignedBlockCompressed7  = 0x22  /**< Block compressed 7 */
};

/**
 * UPTK resource descriptor
 */
struct UPTKResourceDesc {
    enum UPTKResourceType resType;             /**< Resource type */
    
    union {
        struct {
            UPTKArray_t array;                 /**< UPTK array */
        } array;
        struct {
            UPTKMipmappedArray_t mipmap;       /**< UPTK mipmapped array */
        } mipmap;
        struct {
            void *devPtr;                      /**< Device pointer */
            struct UPTKChannelFormatDesc desc; /**< Channel descriptor */
            size_t sizeInBytes;                /**< Size in bytes */
        } linear;
        struct {
            void *devPtr;                      /**< Device pointer */
            struct UPTKChannelFormatDesc desc; /**< Channel descriptor */
            size_t width;                      /**< Width of the array in elements */
            size_t height;                     /**< Height of the array in elements */
            size_t pitchInBytes;               /**< Pitch between two rows in bytes */
        } pitch2D;
    } res;
};

/**
 * UPTK resource view descriptor
 */
struct UPTKResourceViewDesc
{
    enum UPTKResourceViewFormat format;           /**< Resource view format */
    size_t                      width;            /**< Width of the resource view */
    size_t                      height;           /**< Height of the resource view */
    size_t                      depth;            /**< Depth of the resource view */
    unsigned int                firstMipmapLevel; /**< First defined mipmap level */
    unsigned int                lastMipmapLevel;  /**< Last defined mipmap level */
    unsigned int                firstLayer;       /**< First layer index */
    unsigned int                lastLayer;        /**< Last layer index */
};

/**
 * UPTK pointer attributes
 */
struct UPTKPointerAttributes
{
    /**
     * The type of memory - ::UPTKMemoryTypeUnregistered, ::UPTKMemoryTypeHost,
     * ::UPTKMemoryTypeDevice or ::UPTKMemoryTypeManaged.
     */
    enum UPTKMemoryType type;

    /** 
     * The device against which the memory was allocated or registered.
     * If the memory type is ::UPTKMemoryTypeDevice then this identifies 
     * the device on which the memory referred physically resides.  If
     * the memory type is ::UPTKMemoryTypeHost or::UPTKMemoryTypeManaged then
     * this identifies the device which was current when the memory was allocated
     * or registered (and if that device is deinitialized then this allocation
     * will vanish with that device's state).
     */
    int device;

    /**
     * The address which may be dereferenced on the current device to access 
     * the memory or NULL if no such address exists.
     */
    void *devicePointer;

    /**
     * The address which may be dereferenced on the host to access the
     * memory or NULL if no such address exists.
     *
     * \note UPTK doesn't check if unregistered memory is allocated so this field
     * may contain invalid pointer if an invalid pointer has been passed to UPTK.
     */
    void *hostPointer;
};

/**
 * UPTK function attributes
 */
struct UPTKFuncAttributes
{
   /**
    * The size in bytes of statically-allocated shared memory per block
    * required by this function. This does not include dynamically-allocated
    * shared memory requested by the user at runtime.
    */
   size_t sharedSizeBytes;

   /**
    * The size in bytes of user-allocated constant memory required by this
    * function.
    */
   size_t constSizeBytes;

   /**
    * The size in bytes of local memory used by each thread of this function.
    */
   size_t localSizeBytes;

   /**
    * The maximum number of threads per block, beyond which a launch of the
    * function would fail. This number depends on both the function and the
    * device on which the function is currently loaded.
    */
   int maxThreadsPerBlock;

   /**
    * The number of registers used by each thread of this function.
    */
   int numRegs;

   /**
    * The PTX virtual architecture version for which the function was
    * compiled. This value is the major PTX version * 10 + the minor PTX
    * version, so a PTX version 1.3 function would return the value 13.
    */
   int ptxVersion;

   /**
    * The binary architecture version for which the function was compiled.
    * This value is the major binary version * 10 + the minor binary version,
    * so a binary version 1.3 function would return the value 13.
    */
   int binaryVersion;

   /**
    * The attribute to indicate whether the function has been compiled with 
    * user specified option "-Xptxas --dlcm=ca" set.
    */
   int cacheModeCA;

   /**
    * The maximum size in bytes of dynamic shared memory per block for 
    * this function. Any launch must have a dynamic shared memory size
    * smaller than this value.
    */
   int maxDynamicSharedSizeBytes;

   /**
    * On devices where the L1 cache and shared memory use the same hardware resources, 
    * this sets the shared memory carveout preference, in percent of the maximum shared memory. 
    * Refer to ::UPTKDevAttrMaxSharedMemoryPerMultiprocessor.
    * This is only a hint, and the driver can choose a different ratio if required to execute the function.
    * See ::UPTKFuncSetAttribute
    */
   int preferredShmemCarveout;

   /**
    * If this attribute is set, the kernel must launch with a valid cluster dimension
    * specified.
    */
   int clusterDimMustBeSet;

   /**
    * The required cluster width/height/depth in blocks. The values must either
    * all be 0 or all be positive. The validity of the cluster dimensions is
    * otherwise checked at launch time.
    *
    * If the value is set during compile time, it cannot be set at runtime.
    * Setting it at runtime should return UPTKErrorNotPermitted.
    * See ::UPTKFuncSetAttribute
    */
   int requiredClusterWidth;
   int requiredClusterHeight;
   int requiredClusterDepth;

   /**
    * The block scheduling policy of a function.
    * See ::UPTKFuncSetAttribute
    */
   int clusterSchedulingPolicyPreference;

   /**
    * Whether the function can be launched with non-portable cluster size. 1 is
    * allowed, 0 is disallowed. A non-portable cluster size may only function
    * on the specific SKUs the program is tested on. The launch might fail if
    * the program is run on a different hardware platform.
    *
    * UPTK API provides ::UPTKOccupancyMaxActiveClusters to assist with checking
    * whether the desired size can be launched on the current device.
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
    * See ::UPTKFuncSetAttribute
    */
   int nonPortableClusterSizeAllowed;

   /**
    * Reserved for future use.
    */
   int reserved[16];
};

/**
 * UPTK function attributes that can be set using ::UPTKFuncSetAttribute
 */
enum UPTKFuncAttribute
{
    UPTKFuncAttributeMaxDynamicSharedMemorySize = 8, /**< Maximum dynamic shared memory size */
    UPTKFuncAttributePreferredSharedMemoryCarveout = 9, /**< Preferred shared memory-L1 cache split */
    UPTKFuncAttributeClusterDimMustBeSet = 10, /**< Indicator to enforce valid cluster dimension specification on kernel launch */
    UPTKFuncAttributeRequiredClusterWidth = 11, /**< Required cluster width */
    UPTKFuncAttributeRequiredClusterHeight = 12, /**< Required cluster height */
    UPTKFuncAttributeRequiredClusterDepth = 13, /**< Required cluster depth */
    UPTKFuncAttributeNonPortableClusterSizeAllowed = 14, /**< Whether non-portable cluster scheduling policy is supported */
    UPTKFuncAttributeClusterSchedulingPolicyPreference = 15, /**< Required cluster scheduling policy preference */
    UPTKFuncAttributeMax
};

/**
 * UPTK function cache configurations
 */
enum UPTKFuncCache
{
    UPTKFuncCachePreferNone   = 0,    /**< Default function cache configuration, no preference */
    UPTKFuncCachePreferShared = 1,    /**< Prefer larger shared memory and smaller L1 cache  */
    UPTKFuncCachePreferL1     = 2,    /**< Prefer larger L1 cache and smaller shared memory */
    UPTKFuncCachePreferEqual  = 3     /**< Prefer equal size L1 cache and shared memory */
};

/**
 * UPTK shared memory configuration
 * \deprecated
 */
enum UPTKSharedMemConfig
{
    UPTKSharedMemBankSizeDefault   = 0,
    UPTKSharedMemBankSizeFourByte  = 1,
    UPTKSharedMemBankSizeEightByte = 2
};

/** 
 * Shared memory carveout configurations. These may be passed to UPTKFuncSetAttribute
 */
enum UPTKSharedCarveout {
    UPTKSharedmemCarveoutDefault      = -1,  /**< No preference for shared memory or L1 (default) */
    UPTKSharedmemCarveoutMaxShared    = 100, /**< Prefer maximum available shared memory, minimum L1 cache */
    UPTKSharedmemCarveoutMaxL1        = 0    /**< Prefer maximum available L1 cache, minimum shared memory */
};

/**
 * UPTK device compute modes
 */
enum UPTKComputeMode
{
    UPTKComputeModeDefault          = 0,  /**< Default compute mode (Multiple threads can use ::UPTKSetDevice() with this device) */
    UPTKComputeModeExclusive        = 1,  /**< Compute-exclusive-thread mode (Only one thread in one process will be able to use ::UPTKSetDevice() with this device) */
    UPTKComputeModeProhibited       = 2,  /**< Compute-prohibited mode (No threads can use ::UPTKSetDevice() with this device) */
    UPTKComputeModeExclusiveProcess = 3   /**< Compute-exclusive-process mode (Many threads in one process will be able to use ::UPTKSetDevice() with this device) */
};

/**
 * UPTK Limits
 */
enum UPTKLimit
{
    UPTKLimitStackSize                    = 0x00, /**< GPU thread stack size */
    UPTKLimitPrintfFifoSize               = 0x01, /**< GPU printf FIFO size */
    UPTKLimitMallocHeapSize               = 0x02, /**< GPU malloc heap size */
    UPTKLimitDevRuntimeSyncDepth          = 0x03, /**< GPU device runtime synchronize depth */
    UPTKLimitDevRuntimePendingLaunchCount = 0x04, /**< GPU device runtime pending launch count */
    UPTKLimitMaxL2FetchGranularity        = 0x05, /**< A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint */
    UPTKLimitPersistingL2CacheSize        = 0x06  /**< A size in bytes for L2 persisting lines cache size */
};

/**
 * UPTK Memory Advise values
 */
enum UPTKMemoryAdvise
{
    UPTKMemAdviseSetReadMostly          = 1, /**< Data will mostly be read and only occassionally be written to */
    UPTKMemAdviseUnsetReadMostly        = 2, /**< Undo the effect of ::UPTKMemAdviseSetReadMostly */
    UPTKMemAdviseSetPreferredLocation   = 3, /**< Set the preferred location for the data as the specified device */
    UPTKMemAdviseUnsetPreferredLocation = 4, /**< Clear the preferred location for the data */
    UPTKMemAdviseSetAccessedBy          = 5, /**< Data will be accessed by the specified device, so prevent page faults as much as possible */
    UPTKMemAdviseUnsetAccessedBy        = 6  /**< Let the Unified Memory subsystem decide on the page faulting policy for the specified device */
};

/**
 * UPTK range attributes
 */
enum UPTKMemRangeAttribute
{
    UPTKMemRangeAttributeReadMostly                 = 1, /**< Whether the range will mostly be read and only occassionally be written to */
    UPTKMemRangeAttributePreferredLocation          = 2, /**< The preferred location of the range */
    UPTKMemRangeAttributeAccessedBy                 = 3, /**< Memory range has ::UPTKMemAdviseSetAccessedBy set for specified device */
    UPTKMemRangeAttributeLastPrefetchLocation       = 4  /**< The last location to which the range was prefetched */
    , UPTKMemRangeAttributePreferredLocationType    = 5  /**< The preferred location type of the range */
    , UPTKMemRangeAttributePreferredLocationId      = 6  /**< The preferred location id of the range */
    , UPTKMemRangeAttributeLastPrefetchLocationType = 7  /**< The last location type to which the range was prefetched */
    , UPTKMemRangeAttributeLastPrefetchLocationId   = 8  /**< The last location id to which the range was prefetched */
};

/**
 * UPTK GPUDirect RDMA flush writes APIs supported on the device
 */
enum UPTKFlushGPUDirectRDMAWritesOptions {
    UPTKFlushGPUDirectRDMAWritesOptionHost   = 1<<0, /**< ::UPTKDeviceFlushGPUDirectRDMAWrites() and its UPTK Driver API counterpart are supported on the device. */
    UPTKFlushGPUDirectRDMAWritesOptionMemOps = 1<<1  /**< The ::CU_STREAM_WAIT_VALUE_FLUSH flag and the ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the UPTK device. */
};

/**
 * UPTK GPUDirect RDMA flush writes ordering features of the device
 */
enum UPTKGPUDirectRDMAWritesOrdering {
    UPTKGPUDirectRDMAWritesOrderingNone       = 0,   /**< The device does not natively support ordering of GPUDirect RDMA writes. ::UPTKFlushGPUDirectRDMAWrites() can be leveraged if supported. */
    UPTKGPUDirectRDMAWritesOrderingOwner      = 100, /**< Natively, the device can consistently consume GPUDirect RDMA writes, although other UPTK devices may not. */
    UPTKGPUDirectRDMAWritesOrderingAllDevices = 200  /**< Any UPTK device in the system can consistently consume GPUDirect RDMA writes to this device. */
};

/**
 * UPTK GPUDirect RDMA flush writes scopes
 */
enum UPTKFlushGPUDirectRDMAWritesScope {
    UPTKFlushGPUDirectRDMAWritesToOwner      = 100, /**< Blocks until remote writes are visible to the UPTK device context owning the data. */
    UPTKFlushGPUDirectRDMAWritesToAllDevices = 200  /**< Blocks until remote writes are visible to all UPTK device contexts. */
};

/**
 * UPTK GPUDirect RDMA flush writes targets
 */
enum UPTKFlushGPUDirectRDMAWritesTarget {
    UPTKFlushGPUDirectRDMAWritesTargetCurrentDevice /**< Sets the target for ::UPTKDeviceFlushGPUDirectRDMAWrites() to the currently active UPTK device context. */
};


/**
 * UPTK device attributes
 */
enum UPTKDeviceAttr
{
    UPTKDevAttrMaxThreadsPerBlock             = 1,  /**< Maximum number of threads per block */
    UPTKDevAttrMaxBlockDimX                   = 2,  /**< Maximum block dimension X */
    UPTKDevAttrMaxBlockDimY                   = 3,  /**< Maximum block dimension Y */
    UPTKDevAttrMaxBlockDimZ                   = 4,  /**< Maximum block dimension Z */
    UPTKDevAttrMaxGridDimX                    = 5,  /**< Maximum grid dimension X */
    UPTKDevAttrMaxGridDimY                    = 6,  /**< Maximum grid dimension Y */
    UPTKDevAttrMaxGridDimZ                    = 7,  /**< Maximum grid dimension Z */
    UPTKDevAttrMaxSharedMemoryPerBlock        = 8,  /**< Maximum shared memory available per block in bytes */
    UPTKDevAttrTotalConstantMemory            = 9,  /**< Memory available on device for __constant__ variables in a UPTK C kernel in bytes */
    UPTKDevAttrWarpSize                       = 10, /**< Warp size in threads */
    UPTKDevAttrMaxPitch                       = 11, /**< Maximum pitch in bytes allowed by memory copies */
    UPTKDevAttrMaxRegistersPerBlock           = 12, /**< Maximum number of 32-bit registers available per block */
    UPTKDevAttrClockRate                      = 13, /**< Peak clock frequency in kilohertz */
    UPTKDevAttrTextureAlignment               = 14, /**< Alignment requirement for textures */
    UPTKDevAttrGpuOverlap                     = 15, /**< Device can possibly copy memory and execute a kernel concurrently */
    UPTKDevAttrMultiProcessorCount            = 16, /**< Number of multiprocessors on device */
    UPTKDevAttrKernelExecTimeout              = 17, /**< Specifies whether there is a run time limit on kernels */
    UPTKDevAttrIntegrated                     = 18, /**< Device is integrated with host memory */
    UPTKDevAttrCanMapHostMemory               = 19, /**< Device can map host memory into UPTK address space */
    UPTKDevAttrComputeMode                    = 20, /**< Compute mode (See ::UPTKComputeMode for details) */
    UPTKDevAttrMaxTexture1DWidth              = 21, /**< Maximum 1D texture width */
    UPTKDevAttrMaxTexture2DWidth              = 22, /**< Maximum 2D texture width */
    UPTKDevAttrMaxTexture2DHeight             = 23, /**< Maximum 2D texture height */
    UPTKDevAttrMaxTexture3DWidth              = 24, /**< Maximum 3D texture width */
    UPTKDevAttrMaxTexture3DHeight             = 25, /**< Maximum 3D texture height */
    UPTKDevAttrMaxTexture3DDepth              = 26, /**< Maximum 3D texture depth */
    UPTKDevAttrMaxTexture2DLayeredWidth       = 27, /**< Maximum 2D layered texture width */
    UPTKDevAttrMaxTexture2DLayeredHeight      = 28, /**< Maximum 2D layered texture height */
    UPTKDevAttrMaxTexture2DLayeredLayers      = 29, /**< Maximum layers in a 2D layered texture */
    UPTKDevAttrSurfaceAlignment               = 30, /**< Alignment requirement for surfaces */
    UPTKDevAttrConcurrentKernels              = 31, /**< Device can possibly execute multiple kernels concurrently */
    UPTKDevAttrEccEnabled                     = 32, /**< Device has ECC support enabled */
    UPTKDevAttrPciBusId                       = 33, /**< PCI bus ID of the device */
    UPTKDevAttrPciDeviceId                    = 34, /**< PCI device ID of the device */
    UPTKDevAttrTccDriver                      = 35, /**< Device is using TCC driver model */
    UPTKDevAttrMemoryClockRate                = 36, /**< Peak memory clock frequency in kilohertz */
    UPTKDevAttrGlobalMemoryBusWidth           = 37, /**< Global memory bus width in bits */
    UPTKDevAttrL2CacheSize                    = 38, /**< Size of L2 cache in bytes */
    UPTKDevAttrMaxThreadsPerMultiProcessor    = 39, /**< Maximum resident threads per multiprocessor */
    UPTKDevAttrAsyncEngineCount               = 40, /**< Number of asynchronous engines */
    UPTKDevAttrUnifiedAddressing              = 41, /**< Device shares a unified address space with the host */    
    UPTKDevAttrMaxTexture1DLayeredWidth       = 42, /**< Maximum 1D layered texture width */
    UPTKDevAttrMaxTexture1DLayeredLayers      = 43, /**< Maximum layers in a 1D layered texture */
    UPTKDevAttrMaxTexture2DGatherWidth        = 45, /**< Maximum 2D texture width if UPTKArrayTextureGather is set */
    UPTKDevAttrMaxTexture2DGatherHeight       = 46, /**< Maximum 2D texture height if UPTKArrayTextureGather is set */
    UPTKDevAttrMaxTexture3DWidthAlt           = 47, /**< Alternate maximum 3D texture width */
    UPTKDevAttrMaxTexture3DHeightAlt          = 48, /**< Alternate maximum 3D texture height */
    UPTKDevAttrMaxTexture3DDepthAlt           = 49, /**< Alternate maximum 3D texture depth */
    UPTKDevAttrPciDomainId                    = 50, /**< PCI domain ID of the device */
    UPTKDevAttrTexturePitchAlignment          = 51, /**< Pitch alignment requirement for textures */
    UPTKDevAttrMaxTextureCubemapWidth         = 52, /**< Maximum cubemap texture width/height */
    UPTKDevAttrMaxTextureCubemapLayeredWidth  = 53, /**< Maximum cubemap layered texture width/height */
    UPTKDevAttrMaxTextureCubemapLayeredLayers = 54, /**< Maximum layers in a cubemap layered texture */
    UPTKDevAttrMaxSurface1DWidth              = 55, /**< Maximum 1D surface width */
    UPTKDevAttrMaxSurface2DWidth              = 56, /**< Maximum 2D surface width */
    UPTKDevAttrMaxSurface2DHeight             = 57, /**< Maximum 2D surface height */
    UPTKDevAttrMaxSurface3DWidth              = 58, /**< Maximum 3D surface width */
    UPTKDevAttrMaxSurface3DHeight             = 59, /**< Maximum 3D surface height */
    UPTKDevAttrMaxSurface3DDepth              = 60, /**< Maximum 3D surface depth */
    UPTKDevAttrMaxSurface1DLayeredWidth       = 61, /**< Maximum 1D layered surface width */
    UPTKDevAttrMaxSurface1DLayeredLayers      = 62, /**< Maximum layers in a 1D layered surface */
    UPTKDevAttrMaxSurface2DLayeredWidth       = 63, /**< Maximum 2D layered surface width */
    UPTKDevAttrMaxSurface2DLayeredHeight      = 64, /**< Maximum 2D layered surface height */
    UPTKDevAttrMaxSurface2DLayeredLayers      = 65, /**< Maximum layers in a 2D layered surface */
    UPTKDevAttrMaxSurfaceCubemapWidth         = 66, /**< Maximum cubemap surface width */
    UPTKDevAttrMaxSurfaceCubemapLayeredWidth  = 67, /**< Maximum cubemap layered surface width */
    UPTKDevAttrMaxSurfaceCubemapLayeredLayers = 68, /**< Maximum layers in a cubemap layered surface */
    UPTKDevAttrMaxTexture1DLinearWidth        = 69, /**< Maximum 1D linear texture width */
    UPTKDevAttrMaxTexture2DLinearWidth        = 70, /**< Maximum 2D linear texture width */
    UPTKDevAttrMaxTexture2DLinearHeight       = 71, /**< Maximum 2D linear texture height */
    UPTKDevAttrMaxTexture2DLinearPitch        = 72, /**< Maximum 2D linear texture pitch in bytes */
    UPTKDevAttrMaxTexture2DMipmappedWidth     = 73, /**< Maximum mipmapped 2D texture width */
    UPTKDevAttrMaxTexture2DMipmappedHeight    = 74, /**< Maximum mipmapped 2D texture height */
    UPTKDevAttrComputeCapabilityMajor         = 75, /**< Major compute capability version number */ 
    UPTKDevAttrComputeCapabilityMinor         = 76, /**< Minor compute capability version number */
    UPTKDevAttrMaxTexture1DMipmappedWidth     = 77, /**< Maximum mipmapped 1D texture width */
    UPTKDevAttrStreamPrioritiesSupported      = 78, /**< Device supports stream priorities */
    UPTKDevAttrGlobalL1CacheSupported         = 79, /**< Device supports caching globals in L1 */
    UPTKDevAttrLocalL1CacheSupported          = 80, /**< Device supports caching locals in L1 */
    UPTKDevAttrMaxSharedMemoryPerMultiprocessor = 81, /**< Maximum shared memory available per multiprocessor in bytes */
    UPTKDevAttrMaxRegistersPerMultiprocessor  = 82, /**< Maximum number of 32-bit registers available per multiprocessor */
    UPTKDevAttrManagedMemory                  = 83, /**< Device can allocate managed memory on this system */
    UPTKDevAttrIsMultiGpuBoard                = 84, /**< Device is on a multi-GPU board */
    UPTKDevAttrMultiGpuBoardGroupID           = 85, /**< Unique identifier for a group of devices on the same multi-GPU board */
    UPTKDevAttrHostNativeAtomicSupported      = 86, /**< Link between the device and the host supports native atomic operations */
    UPTKDevAttrSingleToDoublePrecisionPerfRatio = 87, /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    UPTKDevAttrPageableMemoryAccess           = 88, /**< Device supports coherently accessing pageable memory without calling UPTKHostRegister on it */
    UPTKDevAttrConcurrentManagedAccess        = 89, /**< Device can coherently access managed memory concurrently with the CPU */
    UPTKDevAttrComputePreemptionSupported     = 90, /**< Device supports Compute Preemption */
    UPTKDevAttrCanUseHostPointerForRegisteredMem = 91, /**< Device can access host registered memory at the same virtual address as the CPU */
    UPTKDevAttrReserved92                     = 92,
    UPTKDevAttrReserved93                     = 93,
    UPTKDevAttrReserved94                     = 94,
    UPTKDevAttrCooperativeLaunch              = 95, /**< Device supports launching cooperative kernels via ::UPTKLaunchCooperativeKernel*/
    UPTKDevAttrCooperativeMultiDeviceLaunch   = 96, /**< Deprecated, UPTKLaunchCooperativeKernelMultiDevice is deprecated. */
    UPTKDevAttrMaxSharedMemoryPerBlockOptin   = 97, /**< The maximum optin shared memory per block. This value may vary by chip. See ::UPTKFuncSetAttribute */
    UPTKDevAttrCanFlushRemoteWrites           = 98, /**< Device supports flushing of outstanding remote writes. */
    UPTKDevAttrHostRegisterSupported          = 99, /**< Device supports host memory registration via ::UPTKHostRegister. */
    UPTKDevAttrPageableMemoryAccessUsesHostPageTables = 100, /**< Device accesses pageable memory via the host's page tables. */
    UPTKDevAttrDirectManagedMemAccessFromHost = 101, /**< Host can directly access managed memory on the device without migration. */
    UPTKDevAttrMaxBlocksPerMultiprocessor     = 106, /**< Maximum number of blocks per multiprocessor */
    UPTKDevAttrMaxPersistingL2CacheSize       = 108, /**< Maximum L2 persisting lines capacity setting in bytes. */
    UPTKDevAttrMaxAccessPolicyWindowSize      = 109, /**< Maximum value of UPTKAccessPolicyWindow::num_bytes. */
    UPTKDevAttrReservedSharedMemoryPerBlock   = 111, /**< Shared memory reserved by UPTK driver per block in bytes */
    UPTKDevAttrSparseUPTKArraySupported       = 112, /**< Device supports sparse UPTK arrays and sparse UPTK mipmapped arrays */
    UPTKDevAttrHostRegisterReadOnlySupported  = 113,  /**< Device supports using the ::UPTKHostRegister flag UPTKHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU */
    UPTKDevAttrTimelineSemaphoreInteropSupported = 114,  /**< External timeline semaphore interop is supported on the device */
    UPTKDevAttrMaxTimelineSemaphoreInteropSupported = 114,  /**< Deprecated, External timeline semaphore interop is supported on the device */
    UPTKDevAttrMemoryPoolsSupported           = 115, /**< Device supports using the ::UPTKMallocAsync and ::UPTKMemPool family of APIs */
    UPTKDevAttrGPUDirectRDMASupported         = 116, /**< Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/UPTK/gpudirect-rdma for more information) */
    UPTKDevAttrGPUDirectRDMAFlushWritesOptions = 117, /**< The returned attribute shall be interpreted as a bitmask, where the individual bits are listed in the ::UPTKFlushGPUDirectRDMAWritesOptions enum */
    UPTKDevAttrGPUDirectRDMAWritesOrdering    = 118, /**< GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See ::UPTKGPUDirectRDMAWritesOrdering for the numerical values returned here. */
    UPTKDevAttrMemoryPoolSupportedHandleTypes = 119, /**< Handle types supported with mempool based IPC */
    UPTKDevAttrClusterLaunch                  = 120, /**< Indicates device supports cluster launch */
    UPTKDevAttrDeferredMappingUPTKArraySupported = 121, /**< Device supports deferred mapping UPTK arrays and UPTK mipmapped arrays */
    UPTKDevAttrReserved122                    = 122,
    UPTKDevAttrReserved123                    = 123,
    UPTKDevAttrReserved124                    = 124,
    UPTKDevAttrIpcEventSupport                = 125, /**< Device supports IPC Events. */ 
    UPTKDevAttrMemSyncDomainCount             = 126, /**< Number of memory synchronization domains the device supports. */
    UPTKDevAttrReserved127                    = 127,
    UPTKDevAttrReserved128                    = 128,
    UPTKDevAttrReserved129                    = 129,
    UPTKDevAttrNumaConfig                     = 130, /**< NUMA configuration of a device: value is of type ::UPTKDeviceNumaConfig enum */
    UPTKDevAttrNumaId                         = 131, /**< NUMA node ID of the GPU memory */
    UPTKDevAttrReserved132                    = 132,
    UPTKDevAttrMpsEnabled                     = 133, /**< Contexts created on this device will be shared via MPS */
    UPTKDevAttrHostNumaId                     = 134, /**< NUMA ID of the host node closest to the device. Returns -1 when system does not support NUMA. */
    UPTKDevAttrD3D12CigSupported              = 135, /**< Device supports CIG with D3D12. */
    UPTKDevAttrMax
};

/**
 * UPTK memory pool attributes
 */
enum UPTKMemPoolAttr
{
    /**
     * (value type = int)
     * Allow cuMemAllocAsync to use memory asynchronously freed
     * in another streams as long as a stream ordering dependency
     * of the allocating stream on the free action exists.
     * UPTK events and null stream interactions can create the required
     * stream ordered dependencies. (default enabled)
     */
    UPTKMemPoolReuseFollowEventDependencies   = 0x1,

    /**
     * (value type = int)
     * Allow reuse of already completed frees when there is no dependency
     * between the free and allocation. (default enabled)
     */
    UPTKMemPoolReuseAllowOpportunistic        = 0x2,

    /**
     * (value type = int)
     * Allow cuMemAllocAsync to insert new stream dependencies
     * in order to establish the stream ordering required to reuse
     * a piece of memory released by cuFreeAsync (default enabled).
     */
    UPTKMemPoolReuseAllowInternalDependencies = 0x3,


    /**
     * (value type = cuuint64_t)
     * Amount of reserved memory in bytes to hold onto before trying
     * to release memory back to the OS. When more than the release
     * threshold bytes of memory are held by the memory pool, the
     * allocator will try to release memory back to the OS on the
     * next call to stream, event or context synchronize. (default 0)
     */
    UPTKMemPoolAttrReleaseThreshold           = 0x4,

    /**
     * (value type = cuuint64_t)
     * Amount of backing memory currently allocated for the mempool.
     */
    UPTKMemPoolAttrReservedMemCurrent         = 0x5,

    /**
     * (value type = cuuint64_t)
     * High watermark of backing memory allocated for the mempool since the
     * last time it was reset. High watermark can only be reset to zero.
     */
    UPTKMemPoolAttrReservedMemHigh            = 0x6,

    /**
     * (value type = cuuint64_t)
     * Amount of memory from the pool that is currently in use by the application.
     */
    UPTKMemPoolAttrUsedMemCurrent             = 0x7,

    /**
     * (value type = cuuint64_t)
     * High watermark of the amount of memory from the pool that was in use by the application since
     * the last time it was reset. High watermark can only be reset to zero.
     */
    UPTKMemPoolAttrUsedMemHigh                = 0x8
};

/**
 * Specifies the type of location 
 */
enum UPTKMemLocationType {
    UPTKMemLocationTypeInvalid = 0,
    UPTKMemLocationTypeDevice = 1  /**< Location is a device location, thus id is a device ordinal */
    , UPTKMemLocationTypeHost = 2 /**< Location is host, id is ignored */
    , UPTKMemLocationTypeHostNuma = 3 /**< Location is a host NUMA node, thus id is a host NUMA node id */
    , UPTKMemLocationTypeHostNumaCurrent = 4 /**< Location is the host NUMA node closest to the current thread's CPU, id is ignored */
};

/**
 * Specifies a memory location.
 *
 * To specify a gpu, set type = ::UPTKMemLocationTypeDevice and set id = the gpu's device ordinal.
 * To specify a cpu NUMA node, set type = ::UPTKMemLocationTypeHostNuma and set id = host NUMA node id.
 */
struct UPTKMemLocation {
    enum UPTKMemLocationType type;  /**< Specifies the location type, which modifies the meaning of id. */
    int id;                         /**< identifier for a given this location's ::CUmemLocationType. */
};

/**
 * Specifies the memory protection flags for mapping.
 */
enum UPTKMemAccessFlags {
    UPTKMemAccessFlagsProtNone      = 0,  /**< Default, make the address range not accessible */
    UPTKMemAccessFlagsProtRead      = 1,  /**< Make the address range read accessible */
    UPTKMemAccessFlagsProtReadWrite = 3   /**< Make the address range read-write accessible */
};

/**
 * Memory access descriptor
 */
struct UPTKMemAccessDesc {
    struct UPTKMemLocation  location; /**< Location on which the request is to change it's accessibility */
    enum UPTKMemAccessFlags flags;    /**< ::CUmemProt accessibility flags to set on the request */
};

/**
 * Defines the allocation types available
 */
enum UPTKMemAllocationType {
    UPTKMemAllocationTypeInvalid = 0x0,
    /** This allocation type is 'pinned', i.e. cannot migrate from its current
      * location while the application is actively using it
      */
    UPTKMemAllocationTypePinned  = 0x1,
    UPTKMemAllocationTypeMax     = 0x7FFFFFFF 
};

/**
 * Flags for specifying particular handle types
 */
enum UPTKMemAllocationHandleType {
    UPTKMemHandleTypeNone                    = 0x0,  /**< Does not allow any export mechanism. > */
    UPTKMemHandleTypePosixFileDescriptor     = 0x1,  /**< Allows a file descriptor to be used for exporting. Permitted only on POSIX systems. (int) */
    UPTKMemHandleTypeWin32                   = 0x2,  /**< Allows a Win32 NT handle to be used for exporting. (HANDLE) */
    UPTKMemHandleTypeWin32Kmt                = 0x4,   /**< Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE) */
    UPTKMemHandleTypeFabric                  = 0x8  /**< Allows a fabric handle to be used for exporting. (UPTKMemFabricHandle_t) */
};

/**
 * Specifies the properties of allocations made from the pool.
 */
struct UPTKMemPoolProps {
    enum UPTKMemAllocationType         allocType;   /**< Allocation type. Currently must be specified as UPTKMemAllocationTypePinned */
    enum UPTKMemAllocationHandleType   handleTypes; /**< Handle types that will be supported by allocations from the pool. */
    struct UPTKMemLocation             location;    /**< Location allocations should reside. */
    /**
     * Windows-specific LPSECURITYATTRIBUTES required when
     * ::UPTKMemHandleTypeWin32 is specified.  This security attribute defines
     * the scope of which exported allocations may be tranferred to other
     * processes.  In all other cases, this field is required to be zero.
     */
    void                              *win32SecurityAttributes;
    size_t                             maxSize;     /**< Maximum pool size. When set to 0, defaults to a system dependent value.*/
    unsigned short                     usage;        /**< Bitmask indicating intended usage for the pool. */
    unsigned char                      reserved[54]; /**< reserved for future use, must be 0 */
};

/**
 * Opaque data for exporting a pool allocation
 */
struct UPTKMemPoolPtrExportData {
    unsigned char reserved[64];
};

/**
 * Memory allocation node parameters
 */
struct UPTKMemAllocNodeParams {
    /**
    * in: location where the allocation should reside (specified in ::location).
    * ::handleTypes must be ::UPTKMemHandleTypeNone. IPC is not supported.
    */
    struct UPTKMemPoolProps         poolProps;       /**< in: array of memory access descriptors. Used to describe peer GPU access */
    const struct UPTKMemAccessDesc *accessDescs;     /**< in: number of memory access descriptors.  Must not exceed the number of GPUs. */
    size_t                          accessDescCount; /**< in: Number of `accessDescs`s */
    size_t                          bytesize;        /**< in: size in bytes of the requested allocation */
    void                           *dptr;            /**< out: address of the allocation returned by UPTK */
};

/**
 * Memory allocation node parameters
 */
struct UPTKMemAllocNodeParamsV2 {
    /**
    * in: location where the allocation should reside (specified in ::location).
    * ::handleTypes must be ::UPTKMemHandleTypeNone. IPC is not supported.
    */
    struct UPTKMemPoolProps         poolProps;       /**< in: array of memory access descriptors. Used to describe peer GPU access */
    const struct UPTKMemAccessDesc *accessDescs;     /**< in: number of memory access descriptors.  Must not exceed the number of GPUs. */
    size_t                          accessDescCount; /**< in: Number of `accessDescs`s */
    size_t                          bytesize;        /**< in: size in bytes of the requested allocation */
    void                           *dptr;            /**< out: address of the allocation returned by UPTK */
};

/**
 * Memory free node parameters
 */
struct UPTKMemFreeNodeParams {
    void *dptr; /**< in: the pointer to free */
};

/**
 * Graph memory attributes
 */
enum UPTKGraphMemAttributeType {
    /**
     * (value type = cuuint64_t)
     * Amount of memory, in bytes, currently associated with graphs.
     */
    UPTKGraphMemAttrUsedMemCurrent      = 0x0,

    /**
     * (value type = cuuint64_t)
     * High watermark of memory, in bytes, associated with graphs since the
     * last time it was reset.  High watermark can only be reset to zero.
     */
    UPTKGraphMemAttrUsedMemHigh         = 0x1,

    /**
     * (value type = cuuint64_t)
     * Amount of memory, in bytes, currently allocated for use by
     * the UPTK graphs asynchronous allocator.
     */
    UPTKGraphMemAttrReservedMemCurrent  = 0x2,

    /**
     * (value type = cuuint64_t)
     * High watermark of memory, in bytes, currently allocated for use by
     * the UPTK graphs asynchronous allocator.
     */
    UPTKGraphMemAttrReservedMemHigh     = 0x3
};

/**
 * UPTK device P2P attributes
 */

enum UPTKDeviceP2PAttr {
    UPTKDevP2PAttrPerformanceRank              = 1, /**< A relative value indicating the performance of the link between two devices */
    UPTKDevP2PAttrAccessSupported              = 2, /**< Peer access is enabled */
    UPTKDevP2PAttrNativeAtomicSupported        = 3, /**< Native atomic operation over the link supported */
    UPTKDevP2PAttrCudaArrayAccessSupported     = 4  /**< Accessing UPTK arrays over the link supported */
};

/**
 * UPTK UUID types
 */
#ifndef CU_UUID_HAS_BEEN_DEFINED
#define CU_UUID_HAS_BEEN_DEFINED
struct CUuuid_st {     /**< UPTK definition of UUID */
    char bytes[16];
};
typedef struct CUuuid_st CUuuid;
#endif
typedef struct CUuuid_st UPTKUUID_t;

/**
 * UPTK device properties
 */
struct UPTKDeviceProp
{
    char         name[256];                  /**< ASCII string identifying device */
    UPTKUUID_t   uuid;                       /**< 16-byte unique identifier */
    char         luid[8];                    /**< 8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms */
    unsigned int luidDeviceNodeMask;         /**< LUID device node mask. Value is undefined on TCC and non-Windows platforms */
    size_t       totalGlobalMem;             /**< Global memory available on device in bytes */
    size_t       sharedMemPerBlock;          /**< Shared memory available per block in bytes */
    int          regsPerBlock;               /**< 32-bit registers available per block */
    int          warpSize;                   /**< Warp size in threads */
    size_t       memPitch;                   /**< Maximum pitch in bytes allowed by memory copies */
    int          maxThreadsPerBlock;         /**< Maximum number of threads per block */
    int          maxThreadsDim[3];           /**< Maximum size of each dimension of a block */
    int          maxGridSize[3];             /**< Maximum size of each dimension of a grid */
    int          clockRate;                  /**< Deprecated, Clock frequency in kilohertz */
    size_t       totalConstMem;              /**< Constant memory available on device in bytes */
    int          major;                      /**< Major compute capability */
    int          minor;                      /**< Minor compute capability */
    size_t       textureAlignment;           /**< Alignment requirement for textures */
    size_t       texturePitchAlignment;      /**< Pitch alignment requirement for texture references bound to pitched memory */
    int          deviceOverlap;              /**< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
    int          multiProcessorCount;        /**< Number of multiprocessors on device */
    int          kernelExecTimeoutEnabled;   /**< Deprecated, Specified whether there is a run time limit on kernels */
    int          integrated;                 /**< Device is integrated as opposed to discrete */
    int          canMapHostMemory;           /**< Device can map host memory with UPTKHostAlloc/UPTKHostGetDevicePointer */
    int          computeMode;                /**< Deprecated, Compute mode (See ::UPTKComputeMode) */
    int          maxTexture1D;               /**< Maximum 1D texture size */
    int          maxTexture1DMipmap;         /**< Maximum 1D mipmapped texture size */
    int          maxTexture1DLinear;         /**< Deprecated, do not use. Use UPTKDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead. */
    int          maxTexture2D[2];            /**< Maximum 2D texture dimensions */
    int          maxTexture2DMipmap[2];      /**< Maximum 2D mipmapped texture dimensions */
    int          maxTexture2DLinear[3];      /**< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
    int          maxTexture2DGather[2];      /**< Maximum 2D texture dimensions if texture gather operations have to be performed */
    int          maxTexture3D[3];            /**< Maximum 3D texture dimensions */
    int          maxTexture3DAlt[3];         /**< Maximum alternate 3D texture dimensions */
    int          maxTextureCubemap;          /**< Maximum Cubemap texture dimensions */
    int          maxTexture1DLayered[2];     /**< Maximum 1D layered texture dimensions */
    int          maxTexture2DLayered[3];     /**< Maximum 2D layered texture dimensions */
    int          maxTextureCubemapLayered[2];/**< Maximum Cubemap layered texture dimensions */
    int          maxSurface1D;               /**< Maximum 1D surface size */
    int          maxSurface2D[2];            /**< Maximum 2D surface dimensions */
    int          maxSurface3D[3];            /**< Maximum 3D surface dimensions */
    int          maxSurface1DLayered[2];     /**< Maximum 1D layered surface dimensions */
    int          maxSurface2DLayered[3];     /**< Maximum 2D layered surface dimensions */
    int          maxSurfaceCubemap;          /**< Maximum Cubemap surface dimensions */
    int          maxSurfaceCubemapLayered[2];/**< Maximum Cubemap layered surface dimensions */
    size_t       surfaceAlignment;           /**< Alignment requirements for surfaces */
    int          concurrentKernels;          /**< Device can possibly execute multiple kernels concurrently */
    int          ECCEnabled;                 /**< Device has ECC support enabled */
    int          pciBusID;                   /**< PCI bus ID of the device */
    int          pciDeviceID;                /**< PCI device ID of the device */
    int          pciDomainID;                /**< PCI domain ID of the device */
    int          tccDriver;                  /**< 1 if device is a Tesla device using TCC driver, 0 otherwise */
    int          asyncEngineCount;           /**< Number of asynchronous engines */
    int          unifiedAddressing;          /**< Device shares a unified address space with the host */
    int          memoryClockRate;            /**< Deprecated, Peak memory clock frequency in kilohertz */
    int          memoryBusWidth;             /**< Global memory bus width in bits */
    int          l2CacheSize;                /**< Size of L2 cache in bytes */
    int          persistingL2CacheMaxSize;   /**< Device's maximum l2 persisting lines capacity setting in bytes */
    int          maxThreadsPerMultiProcessor;/**< Maximum resident threads per multiprocessor */
    int          streamPrioritiesSupported;  /**< Device supports stream priorities */
    int          globalL1CacheSupported;     /**< Device supports caching globals in L1 */
    int          localL1CacheSupported;      /**< Device supports caching locals in L1 */
    size_t       sharedMemPerMultiprocessor; /**< Shared memory available per multiprocessor in bytes */
    int          regsPerMultiprocessor;      /**< 32-bit registers available per multiprocessor */
    int          managedMemory;              /**< Device supports allocating managed memory on this system */
    int          isMultiGpuBoard;            /**< Device is on a multi-GPU board */
    int          multiGpuBoardGroupID;       /**< Unique identifier for a group of devices on the same multi-GPU board */
    int          hostNativeAtomicSupported;  /**< Link between the device and the host supports native atomic operations */
    int          singleToDoublePrecisionPerfRatio; /**< Deprecated, Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    int          pageableMemoryAccess;       /**< Device supports coherently accessing pageable memory without calling UPTKHostRegister on it */
    int          concurrentManagedAccess;    /**< Device can coherently access managed memory concurrently with the CPU */
    int          computePreemptionSupported; /**< Device supports Compute Preemption */
    int          canUseHostPointerForRegisteredMem; /**< Device can access host registered memory at the same virtual address as the CPU */
    int          cooperativeLaunch;          /**< Device supports launching cooperative kernels via ::UPTKLaunchCooperativeKernel */
    int          cooperativeMultiDeviceLaunch; /**< Deprecated, UPTKLaunchCooperativeKernelMultiDevice is deprecated. */
    size_t       sharedMemPerBlockOptin;     /**< Per device maximum shared memory per block usable by special opt in */
    int          pageableMemoryAccessUsesHostPageTables; /**< Device accesses pageable memory via the host's page tables */
    int          directManagedMemAccessFromHost; /**< Host can directly access managed memory on the device without migration. */
    int          maxBlocksPerMultiProcessor; /**< Maximum number of resident blocks per multiprocessor */
    int          accessPolicyMaxWindowSize;  /**< The maximum value of ::UPTKAccessPolicyWindow::num_bytes. */
    size_t       reservedSharedMemPerBlock;  /**< Shared memory reserved by UPTK driver per block in bytes */
    int          hostRegisterSupported;      /**< Device supports host memory registration via ::UPTKHostRegister. */
    int          sparseUPTKArraySupported;   /**< 1 if the device supports sparse UPTK arrays and sparse UPTK mipmapped arrays, 0 otherwise */
    int          hostRegisterReadOnlySupported; /**< Device supports using the ::UPTKHostRegister flag UPTKHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU */
    int          timelineSemaphoreInteropSupported; /**< External timeline semaphore interop is supported on the device */
    int          memoryPoolsSupported;       /**< 1 if the device supports using the UPTKMallocAsync and UPTKMemPool family of APIs, 0 otherwise */
    int          gpuDirectRDMASupported;     /**< 1 if the device supports GPUDirect RDMA APIs, 0 otherwise */
    unsigned int gpuDirectRDMAFlushWritesOptions; /**< Bitmask to be interpreted according to the ::UPTKFlushGPUDirectRDMAWritesOptions enum */
    int          gpuDirectRDMAWritesOrdering;/**< See the ::UPTKGPUDirectRDMAWritesOrdering enum for numerical values */
    unsigned int memoryPoolSupportedHandleTypes; /**< Bitmask of handle types supported with mempool-based IPC */
    int          deferredMappingUPTKArraySupported; /**< 1 if the device supports deferred mapping UPTK arrays and UPTK mipmapped arrays */
    int          ipcEventSupported;          /**< Device supports IPC Events. */
    int          clusterLaunch;              /**< Indicates device supports cluster launch */
    int          unifiedFunctionPointers;    /**< Indicates device supports unified pointers */
    int          reserved2[2];
    int          reserved1[1];               /**< Reserved for future use */
    int          reserved[60];               /**< Reserved for future use */
};

/**
 * UPTK IPC Handle Size
 */
#define UPTK_IPC_HANDLE_SIZE 64

/**
 * UPTK IPC event handle
 */
typedef struct UPTKIpcEventHandle_st
{
    char reserved[UPTK_IPC_HANDLE_SIZE];
}UPTKIpcEventHandle_t;

/**
 * UPTK IPC memory handle
 */
typedef struct UPTKIpcMemHandle_st 
{
    char reserved[UPTK_IPC_HANDLE_SIZE];
}UPTKIpcMemHandle_t;

/*
 * UPTK Mem Fabric Handle
 */
typedef struct UPTKMemFabricHandle_st 
{
    char reserved[UPTK_IPC_HANDLE_SIZE];
}UPTKMemFabricHandle_t;

/**
 * External memory handle types
 */
enum UPTKExternalMemoryHandleType {
    /**
     * Handle is an opaque file descriptor
     */
    UPTKExternalMemoryHandleTypeOpaqueFd         = 1,
    /**
     * Handle is an opaque shared NT handle
     */
    UPTKExternalMemoryHandleTypeOpaqueWin32      = 2,
    /**
     * Handle is an opaque, globally shared handle
     */
    UPTKExternalMemoryHandleTypeOpaqueWin32Kmt   = 3,
    /**
     * Handle is a D3D12 heap object
     */
    UPTKExternalMemoryHandleTypeD3D12Heap        = 4,
    /**
     * Handle is a D3D12 committed resource
     */
    UPTKExternalMemoryHandleTypeD3D12Resource    = 5,
    /**
    *  Handle is a shared NT handle to a D3D11 resource
    */
    UPTKExternalMemoryHandleTypeD3D11Resource    = 6,
    /**
    *  Handle is a globally shared handle to a D3D11 resource
    */
    UPTKExternalMemoryHandleTypeD3D11ResourceKmt = 7,
    /**
    *  Handle is an NvSciBuf object
    */
    UPTKExternalMemoryHandleTypeNvSciBuf         = 8
};

/**
 * Indicates that the external memory object is a dedicated resource
 */
#define UPTKExternalMemoryDedicated   0x1

/** When the /p flags parameter of ::UPTKExternalSemaphoreSignalParams
 * contains this flag, it indicates that signaling an external semaphore object
 * should skip performing appropriate memory synchronization operations over all
 * the external memory objects that are imported as ::UPTKExternalMemoryHandleTypeNvSciBuf,
 * which otherwise are performed by default to ensure data coherency with other
 * importers of the same NvSciBuf memory objects.
 */
#define UPTKExternalSemaphoreSignalSkipNvSciBufMemSync     0x01

/** When the /p flags parameter of ::UPTKExternalSemaphoreWaitParams
 * contains this flag, it indicates that waiting an external semaphore object
 * should skip performing appropriate memory synchronization operations over all
 * the external memory objects that are imported as ::UPTKExternalMemoryHandleTypeNvSciBuf,
 * which otherwise are performed by default to ensure data coherency with other
 * importers of the same NvSciBuf memory objects.
 */
#define UPTKExternalSemaphoreWaitSkipNvSciBufMemSync       0x02

/**
 * When /p flags of ::UPTKDeviceGetNvSciSyncAttributes is set to this,
 * it indicates that application need signaler specific NvSciSyncAttr
 * to be filled by ::UPTKDeviceGetNvSciSyncAttributes.
 */
#define UPTKNvSciSyncAttrSignal       0x1

/**
 * When /p flags of ::UPTKDeviceGetNvSciSyncAttributes is set to this,
 * it indicates that application need waiter specific NvSciSyncAttr
 * to be filled by ::UPTKDeviceGetNvSciSyncAttributes.
 */
#define UPTKNvSciSyncAttrWait         0x2

/**
 * External memory handle descriptor
 */
struct UPTKExternalMemoryHandleDesc {
    /**
     * Type of the handle
     */
    enum  UPTKExternalMemoryHandleType type;
    union {
        /**
         * File descriptor referencing the memory object. Valid
         * when type is
         * ::UPTKExternalMemoryHandleTypeOpaqueFd
         */
        int fd;
        /**
         * Win32 handle referencing the semaphore object. Valid when
         * type is one of the following:
         * - ::UPTKExternalMemoryHandleTypeOpaqueWin32
         * - ::UPTKExternalMemoryHandleTypeOpaqueWin32Kmt
         * - ::UPTKExternalMemoryHandleTypeD3D12Heap 
         * - ::UPTKExternalMemoryHandleTypeD3D12Resource
		 * - ::UPTKExternalMemoryHandleTypeD3D11Resource
		 * - ::UPTKExternalMemoryHandleTypeD3D11ResourceKmt
         * Exactly one of 'handle' and 'name' must be non-NULL. If
         * type is one of the following: 
         * ::UPTKExternalMemoryHandleTypeOpaqueWin32Kmt
         * ::UPTKExternalMemoryHandleTypeD3D11ResourceKmt
         * then 'name' must be NULL.
         */
        struct {
            /**
             * Valid NT handle. Must be NULL if 'name' is non-NULL
             */
            void *handle;
            /**
             * Name of a valid memory object.
             * Must be NULL if 'handle' is non-NULL.
             */
            const void *name;
        } win32;
        /**
         * A handle representing NvSciBuf Object. Valid when type
         * is ::UPTKExternalMemoryHandleTypeNvSciBuf
         */
        const void *nvSciBufObject;
    } handle;
    /**
     * Size of the memory allocation
     */
    unsigned long long size;
    /**
     * Flags must either be zero or ::UPTKExternalMemoryDedicated
     */
    unsigned int flags;
};

/**
 * External memory buffer descriptor
 */
struct UPTKExternalMemoryBufferDesc {
    /**
     * Offset into the memory object where the buffer's base is
     */
    unsigned long long offset;
    /**
     * Size of the buffer
     */
    unsigned long long size;
    /**
     * Flags reserved for future use. Must be zero.
     */
    unsigned int flags;
};
 
/**
 * External memory mipmap descriptor
 */
struct UPTKExternalMemoryMipmappedArrayDesc {
    /**
     * Offset into the memory object where the base level of the
     * mipmap chain is.
     */
    unsigned long long offset;
    /**
     * Format of base level of the mipmap chain
     */
    struct UPTKChannelFormatDesc formatDesc;
    /**
     * Dimensions of base level of the mipmap chain
     */
    struct UPTKExtent extent;
    /**
     * Flags associated with UPTK mipmapped arrays.
     * See ::UPTKMallocMipmappedArray
     */
    unsigned int flags;
    /**
     * Total number of levels in the mipmap chain
     */
    unsigned int numLevels;
};
 
/**
 * External semaphore handle types
 */
enum UPTKExternalSemaphoreHandleType {
    /**
     * Handle is an opaque file descriptor
     */
    UPTKExternalSemaphoreHandleTypeOpaqueFd       = 1,
    /**
     * Handle is an opaque shared NT handle
     */
    UPTKExternalSemaphoreHandleTypeOpaqueWin32    = 2,
    /**
     * Handle is an opaque, globally shared handle
     */
    UPTKExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3,
    /**
     * Handle is a shared NT handle referencing a D3D12 fence object
     */
    UPTKExternalSemaphoreHandleTypeD3D12Fence     = 4,
    /**
     * Handle is a shared NT handle referencing a D3D11 fence object
     */
    UPTKExternalSemaphoreHandleTypeD3D11Fence     = 5,
    /**
     * Opaque handle to NvSciSync Object
     */
     UPTKExternalSemaphoreHandleTypeNvSciSync     = 6,
    /**
     * Handle is a shared NT handle referencing a D3D11 keyed mutex object
     */
    UPTKExternalSemaphoreHandleTypeKeyedMutex     = 7,
    /**
     * Handle is a shared KMT handle referencing a D3D11 keyed mutex object
     */
    UPTKExternalSemaphoreHandleTypeKeyedMutexKmt  = 8,
    /**
     * Handle is an opaque handle file descriptor referencing a timeline semaphore
     */
    UPTKExternalSemaphoreHandleTypeTimelineSemaphoreFd  = 9,
    /**
     * Handle is an opaque handle file descriptor referencing a timeline semaphore
     */
    UPTKExternalSemaphoreHandleTypeTimelineSemaphoreWin32  = 10
};

/**
 * External semaphore handle descriptor
 */
struct UPTKExternalSemaphoreHandleDesc {
    /**
     * Type of the handle
     */
    enum UPTKExternalSemaphoreHandleType type;
    union {
        /**
         * File descriptor referencing the semaphore object. Valid when
         * type is one of the following:
         * - ::UPTKExternalSemaphoreHandleTypeOpaqueFd
         * - ::UPTKExternalSemaphoreHandleTypeTimelineSemaphoreFd
         */
        int fd;
        /**
         * Win32 handle referencing the semaphore object. Valid when
         * type is one of the following:
         * - ::UPTKExternalSemaphoreHandleTypeOpaqueWin32
         * - ::UPTKExternalSemaphoreHandleTypeOpaqueWin32Kmt
         * - ::UPTKExternalSemaphoreHandleTypeD3D12Fence
         * - ::UPTKExternalSemaphoreHandleTypeD3D11Fence
         * - ::UPTKExternalSemaphoreHandleTypeKeyedMutex
         * - ::UPTKExternalSemaphoreHandleTypeTimelineSemaphoreWin32
         * Exactly one of 'handle' and 'name' must be non-NULL. If
         * type is one of the following:
         * ::UPTKExternalSemaphoreHandleTypeOpaqueWin32Kmt
         * ::UPTKExternalSemaphoreHandleTypeKeyedMutexKmt
         * then 'name' must be NULL.
         */
        struct {
            /**
             * Valid NT handle. Must be NULL if 'name' is non-NULL
             */
            void *handle;
            /**
             * Name of a valid synchronization primitive.
             * Must be NULL if 'handle' is non-NULL.
             */
            const void *name;
        } win32;
        /**
         * Valid NvSciSyncObj. Must be non NULL
         */
        const void* nvSciSyncObj;
    } handle;
    /**
     * Flags reserved for the future. Must be zero.
     */
    unsigned int flags;
};

/**
 * External semaphore signal parameters(deprecated)
 */
struct UPTKExternalSemaphoreSignalParams_v1 {
    struct {
        /**
         * Parameters for fence objects
         */
        struct {
            /**
             * Value of fence to be signaled
             */
            unsigned long long value;
        } fence;
        union {
            /**
             * Pointer to NvSciSyncFence. Valid if ::UPTKExternalSemaphoreHandleType
             * is of type ::UPTKExternalSemaphoreHandleTypeNvSciSync.
             */
            void *fence;
            unsigned long long reserved;
        } nvSciSync;
        /**
         * Parameters for keyed mutex objects
         */
        struct {
            /*
             * Value of key to release the mutex with
             */
            unsigned long long key;
        } keyedMutex;
    } params;
    /**
     * Only when ::UPTKExternalSemaphoreSignalParams is used to
     * signal a ::UPTKExternalSemaphore_t of type
     * ::UPTKExternalSemaphoreHandleTypeNvSciSync, the valid flag is 
     * ::UPTKExternalSemaphoreSignalSkipNvSciBufMemSync: which indicates
     * that while signaling the ::UPTKExternalSemaphore_t, no memory
     * synchronization operations should be performed for any external memory
     * object imported as ::UPTKExternalMemoryHandleTypeNvSciBuf.
     * For all other types of ::UPTKExternalSemaphore_t, flags must be zero.
     */
    unsigned int flags;
};

/**
* External semaphore wait parameters(deprecated)
*/
struct UPTKExternalSemaphoreWaitParams_v1 {
    struct {
        /**
        * Parameters for fence objects
        */
        struct {
            /**
            * Value of fence to be waited on
            */
            unsigned long long value;
        } fence;
        union {
            /**
             * Pointer to NvSciSyncFence. Valid if ::UPTKExternalSemaphoreHandleType
             * is of type ::UPTKExternalSemaphoreHandleTypeNvSciSync.
             */
            void *fence;
            unsigned long long reserved;
        } nvSciSync;
        /**
         * Parameters for keyed mutex objects
         */
        struct {
            /**
             * Value of key to acquire the mutex with
             */
            unsigned long long key;
            /**
             * Timeout in milliseconds to wait to acquire the mutex
             */
            unsigned int timeoutMs;
        } keyedMutex;
    } params;
    /**
     * Only when ::UPTKExternalSemaphoreSignalParams is used to
     * signal a ::UPTKExternalSemaphore_t of type
     * ::UPTKExternalSemaphoreHandleTypeNvSciSync, the valid flag is 
     * ::UPTKExternalSemaphoreSignalSkipNvSciBufMemSync: which indicates
     * that while waiting for the ::UPTKExternalSemaphore_t, no memory
     * synchronization operations should be performed for any external memory
     * object imported as ::UPTKExternalMemoryHandleTypeNvSciBuf.
     * For all other types of ::UPTKExternalSemaphore_t, flags must be zero.
     */
    unsigned int flags;
};

/**
 * External semaphore signal parameters, compatible with driver type
 */
struct UPTKExternalSemaphoreSignalParams{
    struct {
        /**
         * Parameters for fence objects
         */
        struct {
            /**
             * Value of fence to be signaled
             */
            unsigned long long value;
        } fence;
        union {
            /**
             * Pointer to NvSciSyncFence. Valid if ::UPTKExternalSemaphoreHandleType
             * is of type ::UPTKExternalSemaphoreHandleTypeNvSciSync.
             */
            void *fence;
            unsigned long long reserved;
        } nvSciSync;
        /**
         * Parameters for keyed mutex objects
         */
        struct {
            /*
             * Value of key to release the mutex with
             */
            unsigned long long key;
        } keyedMutex;
        unsigned int reserved[12];
    } params;
    /**
     * Only when ::UPTKExternalSemaphoreSignalParams is used to
     * signal a ::UPTKExternalSemaphore_t of type
     * ::UPTKExternalSemaphoreHandleTypeNvSciSync, the valid flag is 
     * ::UPTKExternalSemaphoreSignalSkipNvSciBufMemSync: which indicates
     * that while signaling the ::UPTKExternalSemaphore_t, no memory
     * synchronization operations should be performed for any external memory
     * object imported as ::UPTKExternalMemoryHandleTypeNvSciBuf.
     * For all other types of ::UPTKExternalSemaphore_t, flags must be zero.
     */
    unsigned int flags;
    unsigned int reserved[16];
};

/**
 * External semaphore wait parameters, compatible with driver type
 */
struct UPTKExternalSemaphoreWaitParams {
    struct {
        /**
        * Parameters for fence objects
        */
        struct {
            /**
            * Value of fence to be waited on
            */
            unsigned long long value;
        } fence;
        union {
            /**
             * Pointer to NvSciSyncFence. Valid if ::UPTKExternalSemaphoreHandleType
             * is of type ::UPTKExternalSemaphoreHandleTypeNvSciSync.
             */
            void *fence;
            unsigned long long reserved;
        } nvSciSync;
        /**
         * Parameters for keyed mutex objects
         */
        struct {
            /**
             * Value of key to acquire the mutex with
             */
            unsigned long long key;
            /**
             * Timeout in milliseconds to wait to acquire the mutex
             */
            unsigned int timeoutMs;
        } keyedMutex;
        unsigned int reserved[10];
    } params;
    /**
     * Only when ::UPTKExternalSemaphoreSignalParams is used to
     * signal a ::UPTKExternalSemaphore_t of type
     * ::UPTKExternalSemaphoreHandleTypeNvSciSync, the valid flag is 
     * ::UPTKExternalSemaphoreSignalSkipNvSciBufMemSync: which indicates
     * that while waiting for the ::UPTKExternalSemaphore_t, no memory
     * synchronization operations should be performed for any external memory
     * object imported as ::UPTKExternalMemoryHandleTypeNvSciBuf.
     * For all other types of ::UPTKExternalSemaphore_t, flags must be zero.
     */
    unsigned int flags;
    unsigned int reserved[16];
};

/*******************************************************************************
*                                                                              *
*  SHORTHAND TYPE DEFINITION USED BY RUNTIME API                               *
*                                                                              *
*******************************************************************************/

/**
 * UPTK Error types
 */
typedef enum UPTKError UPTKError_t;

/**
 * UPTK stream
 */
typedef struct CUstream_st *UPTKStream_t;

/**
 * UPTK event types
 */
typedef struct CUevent_st *UPTKEvent_t;

/**
 * UPTK graphics resource types
 */
typedef struct UPTKGraphicsResource *UPTKGraphicsResource_t;

/**
 * UPTK external memory
 */
typedef struct CUexternalMemory_st *UPTKExternalMemory_t;

/**
 * UPTK external semaphore
 */
typedef struct CUexternalSemaphore_st *UPTKExternalSemaphore_t;

/**
 * UPTK graph
 */
typedef struct CUgraph_st *UPTKGraph_t;

/**
 * UPTK graph node.
 */
typedef struct CUgraphNode_st *UPTKGraphNode_t;

/**
 * UPTK user object for graphs
 */
typedef struct CUuserObject_st *UPTKUserObject_t;

/**
 * UPTK handle for conditional graph nodes
 */
typedef unsigned long long UPTKGraphConditionalHandle;

/**
 * UPTK function
 */
typedef struct CUfunc_st *UPTKFunction_t;

/**
 * UPTK kernel
 */
typedef struct CUkern_st *UPTKKernel_t;

/**
 * UPTK memory pool
 */
typedef struct CUmemPoolHandle_st *UPTKMemPool_t;

/**
 * UPTK cooperative group scope
 */
enum UPTKCGScope {
    UPTKCGScopeInvalid   = 0, /**< Invalid cooperative group scope */
    UPTKCGScopeGrid      = 1, /**< Scope represented by a grid_group */
    UPTKCGScopeMultiGrid = 2  /**< Scope represented by a multi_grid_group */
};

/**
 * UPTK launch parameters
 */
struct UPTKLaunchParams
{
    void *func;          /**< Device function symbol */
    dim3 gridDim;        /**< Grid dimentions */
    dim3 blockDim;       /**< Block dimentions */
    void **args;         /**< Arguments */
    size_t sharedMem;    /**< Shared memory */
    UPTKStream_t stream; /**< Stream identifier */
};

/**
 * UPTK GPU kernel node parameters
 */
struct UPTKKernelNodeParams {
    void* func;                     /**< Kernel to launch */
    dim3 gridDim;                   /**< Grid dimensions */
    dim3 blockDim;                  /**< Block dimensions */
    unsigned int sharedMemBytes;    /**< Dynamic shared-memory size per thread block in bytes */
    void **kernelParams;            /**< Array of pointers to individual kernel arguments*/
    void **extra;                   /**< Pointer to kernel arguments in the "extra" format */
};

/**
 * UPTK GPU kernel node parameters
 */
struct UPTKKernelNodeParamsV2 {
    void* func;                     /**< Kernel to launch */
    #if !defined(__cplusplus) || __cplusplus >= 201103L
        dim3 gridDim;                   /**< Grid dimensions */
        dim3 blockDim;                  /**< Block dimensions */
    #else
        /* Union members cannot have nontrivial constructors until C++11. */
        uint3 gridDim;                  /**< Grid dimensions */
        uint3 blockDim;                 /**< Block dimensions */
    #endif
    unsigned int sharedMemBytes;    /**< Dynamic shared-memory size per thread block in bytes */
    void **kernelParams;            /**< Array of pointers to individual kernel arguments*/
    void **extra;                   /**< Pointer to kernel arguments in the "extra" format */
};

/**
 * External semaphore signal node parameters
 */
struct UPTKExternalSemaphoreSignalNodeParams {
    UPTKExternalSemaphore_t* extSemArray;                        /**< Array of external semaphore handles. */
    const struct UPTKExternalSemaphoreSignalParams* paramsArray; /**< Array of external semaphore signal parameters. */
    unsigned int numExtSems;                                     /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
};

/**
 * External semaphore signal node parameters
 */
struct UPTKExternalSemaphoreSignalNodeParamsV2 {
    UPTKExternalSemaphore_t* extSemArray;                        /**< Array of external semaphore handles. */
    const struct UPTKExternalSemaphoreSignalParams* paramsArray; /**< Array of external semaphore signal parameters. */
    unsigned int numExtSems;                                     /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
};

/**
 * External semaphore wait node parameters
 */
struct UPTKExternalSemaphoreWaitNodeParams {
    UPTKExternalSemaphore_t* extSemArray;                      /**< Array of external semaphore handles. */
    const struct UPTKExternalSemaphoreWaitParams* paramsArray; /**< Array of external semaphore wait parameters. */
    unsigned int numExtSems;                                   /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
};

/**
 * External semaphore wait node parameters
 */
struct UPTKExternalSemaphoreWaitNodeParamsV2 {
    UPTKExternalSemaphore_t* extSemArray;                      /**< Array of external semaphore handles. */
    const struct UPTKExternalSemaphoreWaitParams* paramsArray; /**< Array of external semaphore wait parameters. */
    unsigned int numExtSems;                                   /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
};

enum UPTKGraphConditionalHandleFlags {
    UPTKGraphCondAssignDefault = 1 /**< Apply default handle value when graph is launched. */
};

/**
 * UPTK conditional node types
 */
enum UPTKGraphConditionalNodeType {
     UPTKGraphCondTypeIf = 0,      /**< Conditional 'if' Node. Body executed once if condition value is non-zero. */
     UPTKGraphCondTypeWhile = 1,   /**< Conditional 'while' Node. Body executed repeatedly while condition value is non-zero. */
};

/**
 * UPTK conditional node parameters
 */
struct UPTKConditionalNodeParams {
    UPTKGraphConditionalHandle handle;       /**< Conditional node handle.
                                                  Handles must be created in advance of creating the node
                                                  using ::UPTKGraphConditionalHandleCreate. */
    enum UPTKGraphConditionalNodeType type;  /**< Type of conditional node. */
    unsigned int size;                       /**< Size of graph output array.  Must be 1. */
    UPTKGraph_t *phGraph_out;                /**< UPTK-owned array populated with conditional node child graphs during creation of the node.
                                                  Valid for the lifetime of the conditional node.
                                                  The contents of the graph(s) are subject to the following constraints:
                                                  
                                                  - Allowed node types are kernel nodes, empty nodes, child graphs, memsets,
                                                    memcopies, and conditionals. This applies recursively to child graphs and conditional bodies.
                                                  - All kernels, including kernels in nested conditionals or child graphs at any level,
                                                    must belong to the same UPTK context.
                                                  
                                                  These graphs may be populated using graph node creation APIs or ::UPTKStreamBeginCaptureToGraph. */
};

/**
* UPTK Graph node types
*/
enum UPTKGraphNodeType {
    UPTKGraphNodeTypeKernel      = 0x00, /**< GPU kernel node */
    UPTKGraphNodeTypeMemcpy      = 0x01, /**< Memcpy node */
    UPTKGraphNodeTypeMemset      = 0x02, /**< Memset node */
    UPTKGraphNodeTypeHost        = 0x03, /**< Host (executable) node */
    UPTKGraphNodeTypeGraph       = 0x04, /**< Node which executes an embedded graph */
    UPTKGraphNodeTypeEmpty       = 0x05, /**< Empty (no-op) node */
    UPTKGraphNodeTypeWaitEvent   = 0x06, /**< External event wait node */
    UPTKGraphNodeTypeEventRecord = 0x07, /**< External event record node */
    UPTKGraphNodeTypeExtSemaphoreSignal = 0x08, /**< External semaphore signal node */
    UPTKGraphNodeTypeExtSemaphoreWait = 0x09, /**< External semaphore wait node */
    UPTKGraphNodeTypeMemAlloc    = 0x0a, /**< Memory allocation node */
    UPTKGraphNodeTypeMemFree     = 0x0b, /**< Memory free node */
    UPTKGraphNodeTypeConditional = 0x0d, /**< Conditional node
                                              
                                              May be used to implement a conditional execution path or loop
                                              inside of a graph. The graph(s) contained within the body of the conditional node
                                              can be selectively executed or iterated upon based on the value of a conditional
                                              variable.
                                              
                                              Handles must be created in advance of creating the node
                                              using ::UPTKGraphConditionalHandleCreate.
                                              
                                              The following restrictions apply to graphs which contain conditional nodes:
                                                The graph cannot be used in a child node.
                                                Only one instantiation of the graph may exist at any point in time.
                                                The graph cannot be cloned.
                                              
                                              To set the control value, supply a default value when creating the handle and/or
                                              call ::UPTKGraphSetConditional from device code.*/
    UPTKGraphNodeTypeCount
};

/**
 * Child graph node parameters
 */
struct UPTKChildGraphNodeParams {
    UPTKGraph_t graph; /**< The child graph to clone into the node for node creation, or
                            a handle to the graph owned by the node for node query */
};

/**
 * Event record node parameters
 */
struct UPTKEventRecordNodeParams {
    UPTKEvent_t event; /**< The event to record when the node executes */
};

/**
 * Event wait node parameters
 */
struct UPTKEventWaitNodeParams {
    UPTKEvent_t event; /**< The event to wait on from the node */
};

/**
 * Graph node parameters.  See ::UPTKGraphAddNode.
 */
struct UPTKGraphNodeParams {
    enum UPTKGraphNodeType type; /**< Type of the node */
    int reserved0[3];            /**< Reserved.  Must be zero. */

    union {
        long long                                      reserved1[29]; /**< Padding. Unused bytes must be zero. */
        struct UPTKKernelNodeParamsV2                  kernel;        /**< Kernel node parameters. */
        struct UPTKMemcpyNodeParams                    memcpy;        /**< Memcpy node parameters. */
        struct UPTKMemsetParamsV2                      memset;        /**< Memset node parameters. */
        struct UPTKHostNodeParamsV2                    host;          /**< Host node parameters. */
        struct UPTKChildGraphNodeParams                graph;         /**< Child graph node parameters. */
        struct UPTKEventWaitNodeParams                 eventWait;     /**< Event wait node parameters. */
        struct UPTKEventRecordNodeParams               eventRecord;   /**< Event record node parameters. */
        struct UPTKExternalSemaphoreSignalNodeParamsV2 extSemSignal;  /**< External semaphore signal node parameters. */
        struct UPTKExternalSemaphoreWaitNodeParamsV2   extSemWait;    /**< External semaphore wait node parameters. */
        struct UPTKMemAllocNodeParamsV2                alloc;         /**< Memory allocation node parameters. */
        struct UPTKMemFreeNodeParams                   free;          /**< Memory free node parameters. */
        struct UPTKConditionalNodeParams               conditional;   /**< Conditional node parameters. */
    };

    long long reserved2; /**< Reserved bytes. Must be zero. */
};

/**
 * Type annotations that can be applied to graph edges as part of ::UPTKGraphEdgeData.
 */
typedef enum UPTKGraphDependencyType_enum {
    UPTKGraphDependencyTypeDefault = 0, /**< This is an ordinary dependency. */
    UPTKGraphDependencyTypeProgrammatic = 1  /**< This dependency type allows the downstream node to
                                                  use \c UPTKGridDependencySynchronize(). It may only be used
                                                  between kernel nodes, and must be used with either the
                                                  ::UPTKGraphKernelNodePortProgrammatic or
                                                  ::UPTKGraphKernelNodePortLaunchCompletion outgoing port. */
} UPTKGraphDependencyType;

/**
 * Optional annotation for edges in a UPTK graph. Note, all edges implicitly have annotations and
 * default to a zero-initialized value if not specified. A zero-initialized struct indicates a
 * standard full serialization of two nodes with memory visibility.
 */
typedef struct UPTKGraphEdgeData_st {
    unsigned char from_port; /**< This indicates when the dependency is triggered from the upstream
                                  node on the edge. The meaning is specfic to the node type. A value
                                  of 0 in all cases means full completion of the upstream node, with
                                  memory visibility to the downstream node or portion thereof
                                  (indicated by \c to_port).
                                  <br>
                                  Only kernel nodes define non-zero ports. A kernel node
                                  can use the following output port types:
                                  ::UPTKGraphKernelNodePortDefault, ::UPTKGraphKernelNodePortProgrammatic,
                                  or ::UPTKGraphKernelNodePortLaunchCompletion. */
    unsigned char to_port; /**< This indicates what portion of the downstream node is dependent on
                                the upstream node or portion thereof (indicated by \c from_port). The
                                meaning is specific to the node type. A value of 0 in all cases means
                                the entirety of the downstream node is dependent on the upstream work.
                                <br>
                                Currently no node types define non-zero ports. Accordingly, this field
                                must be set to zero. */
    unsigned char type; /**< This should be populated with a value from ::UPTKGraphDependencyType. (It
                             is typed as char due to compiler-specific layout of bitfields.) See
                             ::UPTKGraphDependencyType. */
    unsigned char reserved[5]; /**< These bytes are unused and must be zeroed. This ensures
                                    compatibility if additional fields are added in the future. */
} UPTKGraphEdgeData;

/**
 * This port activates when the kernel has finished executing.
 */
#define UPTKGraphKernelNodePortDefault 0
/**
 * This port activates when all blocks of the kernel have performed UPTKTriggerProgrammaticLaunchCompletion()
 * or have terminated. It must be used with edge type ::UPTKGraphDependencyTypeProgrammatic. See also
 * ::UPTKLaunchAttributeProgrammaticEvent.
 */
#define UPTKGraphKernelNodePortProgrammatic 1
/**
 * This port activates when all blocks of the kernel have begun execution. See also
 * ::UPTKLaunchAttributeLaunchCompletionEvent.
 */
#define UPTKGraphKernelNodePortLaunchCompletion 2

/**
 * UPTK executable (launchable) graph
 */
typedef struct CUgraphExec_st* UPTKGraphExec_t;

/**
* UPTK Graph Update error types
*/
enum UPTKGraphExecUpdateResult {
    UPTKGraphExecUpdateSuccess                = 0x0, /**< The update succeeded */
    UPTKGraphExecUpdateError                  = 0x1, /**< The update failed for an unexpected reason which is described in the return value of the function */
    UPTKGraphExecUpdateErrorTopologyChanged   = 0x2, /**< The update failed because the topology changed */
    UPTKGraphExecUpdateErrorNodeTypeChanged   = 0x3, /**< The update failed because a node type changed */
    UPTKGraphExecUpdateErrorFunctionChanged   = 0x4, /**< The update failed because the function of a kernel node changed (UPTK driver < 11.2) */
    UPTKGraphExecUpdateErrorParametersChanged = 0x5, /**< The update failed because the parameters changed in a way that is not supported */
    UPTKGraphExecUpdateErrorNotSupported      = 0x6, /**< The update failed because something about the node is not supported */
    UPTKGraphExecUpdateErrorUnsupportedFunctionChange = 0x7, /**< The update failed because the function of a kernel node changed in an unsupported way */
    UPTKGraphExecUpdateErrorAttributesChanged = 0x8 /**< The update failed because the node attributes changed in a way that is not supported */
};

/**
 * Graph instantiation results
*/
typedef enum UPTKGraphInstantiateResult {
    UPTKGraphInstantiateSuccess = 0,                       /**< Instantiation succeeded */
    UPTKGraphInstantiateError = 1,                         /**< Instantiation failed for an unexpected reason which is described in the return value of the function */
    UPTKGraphInstantiateInvalidStructure = 2,              /**< Instantiation failed due to invalid structure, such as cycles */
    UPTKGraphInstantiateNodeOperationNotSupported = 3,     /**< Instantiation for device launch failed because the graph contained an unsupported operation */
    UPTKGraphInstantiateMultipleDevicesNotSupported = 4    /**< Instantiation for device launch failed due to the nodes belonging to different contexts */
} UPTKGraphInstantiateResult;

/**
 * Graph instantiation parameters
 */
typedef struct UPTKGraphInstantiateParams
{
    unsigned long long flags;              /**< Instantiation flags */
    UPTKStream_t uploadStream;             /**< Upload stream */
    UPTKGraphNode_t errNode_out;           /**< The node which caused instantiation to fail, if any */
    UPTKGraphInstantiateResult result_out; /**< Whether instantiation was successful.  If it failed, the reason why */
} UPTKGraphInstantiateParams;

/**
 * Result information returned by UPTKGraphExecUpdate
 */
typedef struct UPTKGraphExecUpdateResultInfo_st {
    /**
     * Gives more specific detail when a UPTK graph update fails. 
     */
    enum UPTKGraphExecUpdateResult result;

    /**
     * The "to node" of the error edge when the topologies do not match.
     * The error node when the error is associated with a specific node.
     * NULL when the error is generic.
     */
    UPTKGraphNode_t errorNode;

    /**
     * The from node of error edge when the topologies do not match. Otherwise NULL.
     */
    UPTKGraphNode_t errorFromNode;
} UPTKGraphExecUpdateResultInfo;

/**
 * UPTK device node handle for device-side node update
 */
typedef struct CUgraphDeviceUpdatableNode_st* UPTKGraphDeviceNode_t;

/**
 * Specifies the field to update when performing multiple node updates from the device
 */
enum UPTKGraphKernelNodeField
{
    UPTKGraphKernelNodeFieldInvalid = 0, /**< Invalid field */
    UPTKGraphKernelNodeFieldGridDim,     /**< Grid dimension update */
    UPTKGraphKernelNodeFieldParam,       /**< Kernel parameter update */
    UPTKGraphKernelNodeFieldEnabled      /**< Node enable/disable */
};

/**
 * Struct to specify a single node update to pass as part of a larger array to ::UPTKGraphKernelNodeUpdatesApply
 */
struct UPTKGraphKernelNodeUpdate {
    UPTKGraphDeviceNode_t node;     /**< Node to update */
    enum UPTKGraphKernelNodeField field; /**< Which type of update to apply. Determines how updateData is interpreted */
    union {
#if !defined(__cplusplus) || __cplusplus >= 201103L
        dim3 gridDim;               /**< Grid dimensions */
#else
        /* Union members cannot have nontrivial constructors until C++11. */
        uint3 gridDim;              /**< Grid dimensions */
#endif
        struct {
            const void *pValue;     /**< Kernel parameter data to write in */
            size_t offset;          /**< Offset into the parameter buffer at which to apply the update */
            size_t size;            /**< Number of bytes to update */
        } param;                    /**< Kernel parameter data */
        unsigned int isEnabled;     /**< Node enable/disable data. Nonzero if the node should be enabled, 0 if it should be disabled */
    } updateData;                   /**< Update data to apply. Which field is used depends on field's value */
};

/**
 * Flags to specify search options to be used with ::UPTKGetDriverEntryPoint
 * For more details see ::cuGetProcAddress
 */ 
enum UPTKGetDriverEntryPointFlags {
    UPTKEnableDefault                = 0x0, /**< Default search mode for driver symbols. */
    UPTKEnableLegacyStream           = 0x1, /**< Search for legacy versions of driver symbols. */
    UPTKEnablePerThreadDefaultStream = 0x2  /**< Search for per-thread versions of driver symbols. */
};

/**
 * Enum for status from obtaining driver entry points, used with ::UPTKApiGetDriverEntryPoint
 */
enum UPTKDriverEntryPointQueryResult {
    UPTKDriverEntryPointSuccess             = 0,  /**< Search for symbol found a match */
    UPTKDriverEntryPointSymbolNotFound      = 1,  /**< Search for symbol was not found */
    UPTKDriverEntryPointVersionNotSufficent = 2   /**< Search for symbol was found but version wasn't great enough */
};

/**
 * UPTK Graph debug write options
 */
enum UPTKGraphDebugDotFlags {
    UPTKGraphDebugDotFlagsVerbose                  = 1<<0,  /**< Output all debug data as if every debug flag is enabled */
    UPTKGraphDebugDotFlagsKernelNodeParams         = 1<<2,  /**< Adds UPTKKernelNodeParams to output */
    UPTKGraphDebugDotFlagsMemcpyNodeParams         = 1<<3,  /**< Adds UPTKMemcpy3DParms to output */
    UPTKGraphDebugDotFlagsMemsetNodeParams         = 1<<4,  /**< Adds UPTKMemsetParams to output */
    UPTKGraphDebugDotFlagsHostNodeParams           = 1<<5,  /**< Adds UPTKHostNodeParams to output */
    UPTKGraphDebugDotFlagsEventNodeParams          = 1<<6,  /**< Adds UPTKEvent_t handle from record and wait nodes to output */
    UPTKGraphDebugDotFlagsExtSemasSignalNodeParams = 1<<7,  /**< Adds UPTKExternalSemaphoreSignalNodeParams values to output */
    UPTKGraphDebugDotFlagsExtSemasWaitNodeParams   = 1<<8,  /**< Adds UPTKExternalSemaphoreWaitNodeParams to output */
    UPTKGraphDebugDotFlagsKernelNodeAttributes     = 1<<9,  /**< Adds UPTKKernelNodeAttrID values to output */
    UPTKGraphDebugDotFlagsHandles                  = 1<<10  /**< Adds node handles and every kernel function handle to output */
    ,UPTKGraphDebugDotFlagsConditionalNodeParams   = 1<<15,  /**< Adds UPTKConditionalNodeParams to output */
};

/**
 * Flags for instantiating a graph
 */
enum UPTKGraphInstantiateFlags {
    UPTKGraphInstantiateFlagAutoFreeOnLaunch = 1 /**< Automatically free memory allocated in a graph before relaunching. */
  , UPTKGraphInstantiateFlagUpload           = 2 /**< Automatically upload the graph after instantiation. Only supported by                                                                                                                                                                                                                                                                                                     
                                                      ::UPTKGraphInstantiateWithParams.  The upload will be performed using the                                                                                                                                                                                                                                                                                                   
                                                      stream provided in \p instantiateParams. */                                                                                                                                                                                                                                                                                                                               
  , UPTKGraphInstantiateFlagDeviceLaunch     = 4 /**< Instantiate the graph to be launchable from the device. This flag can only                                                                                                                                                                                                                                                                                                
                                                      be used on platforms which support unified addressing. This flag cannot be                                                                                                                                                                                                                                                                                                
                                                      used in conjunction with UPTKGraphInstantiateFlagAutoFreeOnLaunch. */                                                                                                                                                                                                                                                                                              
  , UPTKGraphInstantiateFlagUseNodePriority  = 8 /**< Run the graph using the per-node priority attributes rather than the
                                                      priority of the stream it is launched into. */
};

/**
 * Memory Synchronization Domain
 *
 * A kernel can be launched in a specified memory synchronization domain that affects all memory operations issued by
 * that kernel. A memory barrier issued in one domain will only order memory operations in that domain, thus eliminating
 * latency increase from memory barriers ordering unrelated traffic.
 *
 * By default, kernels are launched in domain 0. Kernel launched with ::UPTKLaunchMemSyncDomainRemote will have a
 * different domain ID. User may also alter the domain ID with ::UPTKLaunchMemSyncDomainMap for a specific stream /
 * graph node / kernel launch. See ::UPTKLaunchAttributeMemSyncDomain, ::UPTKStreamSetAttribute, ::UPTKLaunchKernelEx,
 * ::UPTKGraphKernelNodeSetAttribute.
 *
 * Memory operations done in kernels launched in different domains are considered system-scope distanced. In other
 * words, a GPU scoped memory synchronization is not sufficient for memory order to be observed by kernels in another
 * memory synchronization domain even if they are on the same GPU.
 */
typedef enum UPTKLaunchMemSyncDomain {
    UPTKLaunchMemSyncDomainDefault = 0,    /**< Launch kernels in the default domain */
    UPTKLaunchMemSyncDomainRemote  = 1     /**< Launch kernels in the remote domain */
} UPTKLaunchMemSyncDomain;

/**
 * Memory Synchronization Domain map
 *
 * See ::UPTKLaunchMemSyncDomain.
 *
 * By default, kernels are launched in domain 0. Kernel launched with ::UPTKLaunchMemSyncDomainRemote will have a
 * different domain ID. User may also alter the domain ID with ::UPTKLaunchMemSyncDomainMap for a specific stream /
 * graph node / kernel launch. See ::UPTKLaunchAttributeMemSyncDomainMap.
 *
 * Domain ID range is available through ::UPTKDevAttrMemSyncDomainCount.
 */
typedef struct UPTKLaunchMemSyncDomainMap_st {
    unsigned char default_;                /**< The default domain ID to use for designated kernels */
    unsigned char remote;                  /**< The remote domain ID to use for designated kernels */
} UPTKLaunchMemSyncDomainMap;

/**
 * Launch attributes enum; used as id field of ::UPTKLaunchAttribute
 */
typedef enum UPTKLaunchAttributeID {
    UPTKLaunchAttributeIgnore                = 0 /**< Ignored entry, for convenient composition */
  , UPTKLaunchAttributeAccessPolicyWindow    = 1 /**< Valid for streams, graph nodes, launches. See
                                                    ::UPTKLaunchAttributeValue::accessPolicyWindow. */
  , UPTKLaunchAttributeCooperative           = 2 /**< Valid for graph nodes, launches. See
                                                    ::UPTKLaunchAttributeValue::cooperative. */
  , UPTKLaunchAttributeSynchronizationPolicy = 3 /**< Valid for streams. See ::UPTKLaunchAttributeValue::syncPolicy. */
  , UPTKLaunchAttributeClusterDimension                  = 4 /**< Valid for graph nodes, launches. See
                                                                ::UPTKLaunchAttributeValue::clusterDim. */
  , UPTKLaunchAttributeClusterSchedulingPolicyPreference = 5 /**< Valid for graph nodes, launches. See
                                                                ::UPTKLaunchAttributeValue::clusterSchedulingPolicyPreference. */
  , UPTKLaunchAttributeProgrammaticStreamSerialization   = 6 /**< Valid for launches. Setting
                                                                  ::UPTKLaunchAttributeValue::programmaticStreamSerializationAllowed
                                                                  to non-0 signals that the kernel will use programmatic
                                                                  means to resolve its stream dependency, so that the
                                                                  UPTK runtime should opportunistically allow the grid's
                                                                  execution to overlap with the previous kernel in the
                                                                  stream, if that kernel requests the overlap. The
                                                                  dependent launches can choose to wait on the
                                                                  dependency using the programmatic sync
                                                                  (UPTKGridDependencySynchronize() or equivalent PTX
                                                                  instructions). */
  , UPTKLaunchAttributeProgrammaticEvent                 = 7 /**< Valid for launches. Set
                                                                  ::UPTKLaunchAttributeValue::programmaticEvent to
                                                                  record the event. Event recorded through this launch
                                                                  attribute is guaranteed to only trigger after all
                                                                  block in the associated kernel trigger the event.  A
                                                                  block can trigger the event programmatically in a
                                                                  future UPTK release. A trigger can also be inserted at
                                                                  the beginning of each block's execution if
                                                                  triggerAtBlockStart is set to non-0. The dependent
                                                                  launches can choose to wait on the dependency using
                                                                  the programmatic sync (UPTKGridDependencySynchronize()
                                                                  or equivalent PTX instructions). Note that dependents
                                                                  (including the CPU thread calling
                                                                  UPTKEventSynchronize()) are not guaranteed to observe
                                                                  the release precisely when it is released. For
                                                                  example, UPTKEventSynchronize() may only observe the
                                                                  event trigger long after the associated kernel has
                                                                  completed. This recording type is primarily meant for
                                                                  establishing programmatic dependency between device
                                                                  tasks. Note also this type of dependency allows, but
                                                                  does not guarantee, concurrent execution of tasks.
                                                                  <br>
                                                                  The event supplied must not be an interprocess or
                                                                  interop event. The event must disable timing (i.e.
                                                                  must be created with the ::UPTKEventDisableTiming flag
                                                                  set). */
  , UPTKLaunchAttributePriority              = 8 /**< Valid for streams, graph nodes, launches. See
                                                    ::UPTKLaunchAttributeValue::priority. */
  , UPTKLaunchAttributeMemSyncDomainMap                  = 9 /**< Valid for streams, graph nodes, launches. See
                                                                ::UPTKLaunchAttributeValue::memSyncDomainMap. */
  , UPTKLaunchAttributeMemSyncDomain                    = 10 /**< Valid for streams, graph nodes, launches. See
                                                                ::UPTKLaunchAttributeValue::memSyncDomain. */
  , UPTKLaunchAttributeLaunchCompletionEvent = 12 /**< Valid for launches. Set
                                                       ::UPTKLaunchAttributeValue::launchCompletionEvent to record the
                                                       event.
                                                       <br>
                                                       Nominally, the event is triggered once all blocks of the kernel
                                                       have begun execution. Currently this is a best effort. If a kernel
                                                       B has a launch completion dependency on a kernel A, B may wait
                                                       until A is complete. Alternatively, blocks of B may begin before
                                                       all blocks of A have begun, for example if B can claim execution
                                                       resources unavailable to A (e.g. they run on different GPUs) or
                                                       if B is a higher priority than A.
                                                       Exercise caution if such an ordering inversion could lead
                                                       to deadlock.
                                                       <br>
                                                       A launch completion event is nominally similar to a programmatic
                                                       event with \c triggerAtBlockStart set except that it is not
                                                       visible to \c UPTKGridDependencySynchronize() and can be used with
                                                       compute capability less than 9.0.
                                                       <br>
                                                       The event supplied must not be an interprocess or interop event.
                                                       The event must disable timing (i.e. must be created with the
                                                       ::UPTKEventDisableTiming flag set). */
  , UPTKLaunchAttributeDeviceUpdatableKernelNode = 13 /**< Valid for graph nodes, launches. This attribute is graphs-only,
                                                           and passing it to a launch in a non-capturing stream will result
                                                           in an error.
                                                           <br>
                                                           :UPTKLaunchAttributeValue::deviceUpdatableKernelNode::deviceUpdatable can 
                                                           only be set to 0 or 1. Setting the field to 1 indicates that the
                                                           corresponding kernel node should be device-updatable. On success, a handle
                                                           will be returned via
                                                           ::UPTKLaunchAttributeValue::deviceUpdatableKernelNode::devNode which can be
                                                           passed to the various device-side update functions to update the node's
                                                           kernel parameters from within another kernel. For more information on the
                                                           types of device updates that can be made, as well as the relevant limitations
                                                           thereof, see ::UPTKGraphKernelNodeUpdatesApply.
                                                           <br>
                                                           Nodes which are device-updatable have additional restrictions compared to
                                                           regular kernel nodes. Firstly, device-updatable nodes cannot be removed
                                                           from their graph via ::UPTKGraphDestroyNode. Additionally, once opted-in
                                                           to this functionality, a node cannot opt out, and any attempt to set the
                                                           deviceUpdatable attribute to 0 will result in an error. Device-updatable
                                                           kernel nodes also cannot have their attributes copied to/from another kernel
                                                           node via ::UPTKGraphKernelNodeCopyAttributes. Graphs containing one or more
                                                           device-updatable nodes also do not allow multiple instantiation, and neither
                                                           the graph nor its instantiated version can be passed to ::UPTKGraphExecUpdate.
                                                           <br>
                                                           If a graph contains device-updatable nodes and updates those nodes from the device
                                                           from within the graph, the graph must be uploaded with ::cuGraphUpload before it
                                                           is launched. For such a graph, if host-side executable graph updates are made to the
                                                           device-updatable nodes, the graph must be uploaded before it is launched again. */
  , UPTKLaunchAttributePreferredSharedMemoryCarveout = 14 /**< Valid for launches. On devices where the L1 cache and shared memory use the
                                                               same hardware resources, setting ::UPTKLaunchAttributeValue::sharedMemCarveout 
                                                               to a percentage between 0-100 signals sets the shared memory carveout 
                                                               preference in percent of the total shared memory for that kernel launch. 
                                                               This attribute takes precedence over ::UPTKFuncAttributePreferredSharedMemoryCarveout.
                                                               This is only a hint, and the driver can choose a different configuration if
                                                               required for the launch.*/  
} UPTKLaunchAttributeID;

/**
 * Launch attributes union; used as value field of ::UPTKLaunchAttribute
 */
typedef union UPTKLaunchAttributeValue {
    char pad[64]; /* Pad to 64 bytes */
    struct UPTKAccessPolicyWindow accessPolicyWindow; /**< Value of launch attribute ::UPTKLaunchAttributeAccessPolicyWindow. */
    int cooperative; /**< Value of launch attribute ::UPTKLaunchAttributeCooperative. Nonzero indicates a cooperative
                        kernel (see ::UPTKLaunchCooperativeKernel). */
    enum UPTKSynchronizationPolicy syncPolicy; /**< Value of launch attribute
                                                  ::UPTKLaunchAttributeSynchronizationPolicy. ::UPTKSynchronizationPolicy
                                                  for work queued up in this stream. */
    /**
     * Value of launch attribute ::UPTKLaunchAttributeClusterDimension that
     * represents the desired cluster dimensions for the kernel. Opaque type
     * with the following fields:
     *     - \p x - The X dimension of the cluster, in blocks. Must be a divisor
     *              of the grid X dimension.
     *     - \p y - The Y dimension of the cluster, in blocks. Must be a divisor
     *              of the grid Y dimension.
     *     - \p z - The Z dimension of the cluster, in blocks. Must be a divisor
     *              of the grid Z dimension.
     */
    struct {
        unsigned int x;
        unsigned int y;
        unsigned int z;
    } clusterDim;
    enum UPTKClusterSchedulingPolicy clusterSchedulingPolicyPreference; /**< Value of launch attribute
                                                                           ::UPTKLaunchAttributeClusterSchedulingPolicyPreference. Cluster
                                                                           scheduling policy preference for the kernel. */
    int programmaticStreamSerializationAllowed; /**< Value of launch attribute
                                                   ::UPTKLaunchAttributeProgrammaticStreamSerialization. */

    /**
     * Value of launch attribute ::UPTKLaunchAttributeProgrammaticEvent
     * with the following fields:
     *     - \p UPTKEvent_t event - Event to fire when all blocks trigger it.
     *     - \p int flags;        - Event record flags, see ::UPTKEventRecordWithFlags. Does not accept
     *                               ::UPTKEventRecordExternal.
     *     - \p int triggerAtBlockStart - If this is set to non-0, each block launch will automatically trigger the event.
     */
    struct {
        UPTKEvent_t event;
        int flags;
        int triggerAtBlockStart;
    } programmaticEvent;
    int priority; /**< Value of launch attribute ::UPTKLaunchAttributePriority. Execution priority of the kernel. */
    UPTKLaunchMemSyncDomainMap memSyncDomainMap; /**< Value of launch attribute
                                                    ::UPTKLaunchAttributeMemSyncDomainMap. See
                                                    ::UPTKLaunchMemSyncDomainMap. */
    UPTKLaunchMemSyncDomain memSyncDomain;       /**< Value of launch attribute ::UPTKLaunchAttributeMemSyncDomain. See
                                                    ::UPTKLaunchMemSyncDomain. */
    /**
     * Value of launch attribute ::UPTKLaunchAttributeLaunchCompletionEvent
     * with the following fields:
     *     - \p UPTKEvent_t event - Event to fire when the last block launches.
     *     - \p int flags - Event record flags, see ::UPTKEventRecordWithFlags. Does not accept
     *                   ::UPTKEventRecordExternal.
     */
    struct {
        UPTKEvent_t event;
        int flags;
    } launchCompletionEvent;

    /**
     * Value of launch attribute ::UPTKLaunchAttributeDeviceUpdatableKernelNode
     * with the following fields:
     *    - \p int deviceUpdatable - Whether or not the resulting kernel node should be device-updatable.
     *    - \p UPTKGraphDeviceNode_t devNode - Returns a handle to pass to the various device-side update functions.
     */
    struct {
        int deviceUpdatable;
        UPTKGraphDeviceNode_t devNode;
    } deviceUpdatableKernelNode;
    unsigned int sharedMemCarveout; /**< Value of launch attribute ::UPTKLaunchAttributePreferredSharedMemoryCarveout. */
} UPTKLaunchAttributeValue;

/**
 * Launch attribute
 */
typedef struct UPTKLaunchAttribute_st {
    UPTKLaunchAttributeID id; /**< Attribute to set */
    char pad[8 - sizeof(UPTKLaunchAttributeID)];
    UPTKLaunchAttributeValue val; /**< Value of the attribute */
} UPTKLaunchAttribute;

/**
 * UPTK extensible launch configuration
 */
typedef struct UPTKLaunchConfig_st {
    dim3 gridDim;               /**< Grid dimensions */
    dim3 blockDim;              /**< Block dimensions */
    size_t dynamicSmemBytes;    /**< Dynamic shared-memory size per thread block in bytes */
    UPTKStream_t stream;        /**< Stream identifier */
    UPTKLaunchAttribute *attrs; /**< List of attributes; nullable if ::UPTKLaunchConfig_t::numAttrs == 0 */
    unsigned int numAttrs;      /**< Number of attributes populated in ::UPTKLaunchConfig_t::attrs */
} UPTKLaunchConfig_t;

#define UPTKStreamAttrID UPTKLaunchAttributeID
#define UPTKStreamAttributeAccessPolicyWindow    UPTKLaunchAttributeAccessPolicyWindow
#define UPTKStreamAttributeSynchronizationPolicy UPTKLaunchAttributeSynchronizationPolicy
#define UPTKStreamAttributeMemSyncDomainMap      UPTKLaunchAttributeMemSyncDomainMap
#define UPTKStreamAttributeMemSyncDomain         UPTKLaunchAttributeMemSyncDomain
#define UPTKStreamAttributePriority UPTKLaunchAttributePriority

#define UPTKStreamAttrValue UPTKLaunchAttributeValue

#define UPTKKernelNodeAttrID UPTKLaunchAttributeID
#define UPTKKernelNodeAttributeAccessPolicyWindow UPTKLaunchAttributeAccessPolicyWindow
#define UPTKKernelNodeAttributeCooperative        UPTKLaunchAttributeCooperative
#define UPTKKernelNodeAttributePriority           UPTKLaunchAttributePriority
#define UPTKKernelNodeAttributeClusterDimension                     UPTKLaunchAttributeClusterDimension
#define UPTKKernelNodeAttributeClusterSchedulingPolicyPreference    UPTKLaunchAttributeClusterSchedulingPolicyPreference
#define UPTKKernelNodeAttributeMemSyncDomainMap   UPTKLaunchAttributeMemSyncDomainMap
#define UPTKKernelNodeAttributeMemSyncDomain      UPTKLaunchAttributeMemSyncDomain
#define UPTKKernelNodeAttributePreferredSharedMemoryCarveout UPTKLaunchAttributePreferredSharedMemoryCarveout
#define UPTKKernelNodeAttributeDeviceUpdatableKernelNode UPTKLaunchAttributeDeviceUpdatableKernelNode

#define UPTKKernelNodeAttrValue UPTKLaunchAttributeValue

/**
 * UPTK device NUMA config
 */
enum  UPTKDeviceNumaConfig {
    UPTKDeviceNumaConfigNone  = 0, /**< The GPU is not a NUMA node */
    UPTKDeviceNumaConfigNumaNode, /**< The GPU is a NUMA node, UPTKDevAttrNumaId contains its NUMA ID */
};

/**
 * UPTK async callback handle
 */
typedef struct UPTKAsyncCallbackEntry* UPTKAsyncCallbackHandle_t;

struct UPTKAsyncCallbackEntry;

/**
* Types of async notification that can occur
*/
typedef enum UPTKAsyncNotificationType_enum {
    UPTKAsyncNotificationTypeOverBudget = 0x1
} UPTKAsyncNotificationType;

/**
* Information describing an async notification event
*/
typedef struct UPTKAsyncNotificationInfo
{
    UPTKAsyncNotificationType type;
    union {
        struct {
            unsigned long long bytesOverBudget;
        } overBudget;
    } info;
} UPTKAsyncNotificationInfo_t;

typedef void (*UPTKAsyncCallback)(UPTKAsyncNotificationInfo_t*, void*, UPTKAsyncCallbackHandle_t);


/** @} */
/** @} */ /* END UPTKRT_TYPES */

#endif  /* !__CUDACC_RTC_MINIMAL__ */

#if defined(__UNDEF_UPTK_INCLUDE_COMPILER_INTERNAL_HEADERS_DRIVER_TYPES_H__)
#undef __UPTK_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_UPTK_INCLUDE_COMPILER_INTERNAL_HEADERS_DRIVER_TYPES_H__
#endif

#endif /* !__UPTK_DRIVER_TYPES_H__ */

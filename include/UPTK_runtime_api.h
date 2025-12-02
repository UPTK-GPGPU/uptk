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

#if defined(__NVCC__)
#define UPTKStream_t cudaStream_t
#endif 
#if defined(__HIPCC__)
#define UPTKStream_t hipStream_t
#endif 

#if defined(__cplusplus)

#define __dv(v) \
        = v

#else /* __cplusplus */

#define __dv(v)

#endif /* __cplusplus */

#endif /* !__dv */
/** \endcond impl_private */



    /**
     * UPTK memory copy types
     */
    enum UPTKMemcpyKind
    {
        UPTKMemcpyHostToHost = 0,     /**< Host   -> Host */
        UPTKMemcpyHostToDevice = 1,   /**< Host   -> Device */
        UPTKMemcpyDeviceToHost = 2,   /**< Device -> Host */
        UPTKMemcpyDeviceToDevice = 3, /**< Device -> Device */
        UPTKMemcpyDefault = 4         /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
    };


/********************************************************************************************************************************/

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
 // #define UPTKStreamLegacy                    ((UPTKStream_t)0x1)
 /*
 * In versions earlier than UPTK 7, the relationship between the default stream and other streams is synchronous. 
 * In versions later than UPTK 7, the relationship between the default stream and other streams is asynchronous. 
 * As of now, all versions of hip have a synchronous relationship between the default stream and other streams.
 */
 /*UPTKStreamLegacy is used to simulate the synchronization behavior between the default stream and other streams in earlier UPTK.*/
 #define UPTKStreamLegacy                    UPTKStreamDefault
 
 
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
 #define UPTKDeviceMask                      0x1f  /**< Device flags mask */
 
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
 
//  #endif /* !__UPTK_INTERNAL_COMPILATION__ */
 
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

typedef void (UPTKRT_CB *UPTKHostFn_t)(void *userData);

struct  UPTKHostNodeParams {
    UPTKHostFn_t fn;    /**< The function to call when the node executes */
    void* userData; /**< Argument to pass to the function */
};


struct  UPTKHostNodeParamsV2 {
    UPTKHostFn_t fn;    /**< The function to call when the node executes */
    void* userData; /**< Argument to pass to the function */
};

 /**
  * Defines the allocation types available
  */
 enum  UPTKMemAllocationType {
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
 enum  UPTKMemAllocationHandleType {
     UPTKMemHandleTypeNone                    = 0x0,  /**< Does not allow any export mechanism. > */
     UPTKMemHandleTypePosixFileDescriptor     = 0x1,  /**< Allows a file descriptor to be used for exporting. Permitted only on POSIX systems. (int) */
     UPTKMemHandleTypeWin32                   = 0x2,  /**< Allows a Win32 NT handle to be used for exporting. (HANDLE) */
     UPTKMemHandleTypeWin32Kmt                = 0x4   /**< Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE) */
 };
 
 /**
  * Specifies the type of location 
  */
enum  UPTKMemLocationType {
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
  */
 struct  UPTKMemLocation {
     enum UPTKMemLocationType type;  /**< Specifies the location type, which modifies the meaning of id. */
     int id;                         /**< identifier for a given this location's ::UPTKmemLocationType. */
 };

 /**
  * Specifies the properties of allocations made from the pool.
  */
 struct  UPTKMemPoolProps {
     enum UPTKMemAllocationType         allocType;   /**< Allocation type. Currently must be specified as UPTKMemAllocationTypePinned */
     enum UPTKMemAllocationHandleType   handleTypes; /**< Handle types that will be supported by allocations from the pool. */
     struct UPTKMemLocation             location;    /**< Location allocations should reside. */
     /**
      * Windows-specific LPSEUPTKRITYATTRIBUTES required when
      * ::UPTKMemHandleTypeWin32 is specified.  This security attribute defines
      * the scope of which exported allocations may be tranferred to other
      * processes.  In all other cases, this field is required to be zero.
      */
     void                              *win32SecurityAttributes;
     unsigned char                      reserved[64]; /**< reserved for future use, must be 0 */
 };

struct  UPTKMemAllocNodeParamsV2 {
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
  * UPTK error types
  */
 enum  UPTKError
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
      * The API call failed because it was unable to allocate enough memory to
      * perform the requested operation.
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
     UPTKErrorUPTKrtUnloading              =     4,
 
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
      * user allocations.
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
      * interoperability and there is an existing ::UPTKcontext active on the
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
      * This indicates that an unknown internal error has occurred.
      */
     UPTKErrorUnknown                      =    999,
 
     /**
      * Any unhandled UPTK driver error is added to this value and returned via
      * the runtime. Production releases of UPTK should not return such errors.
      * \deprecated
      * This error return is deprecated as of UPTK 4.1.
      */
     UPTKErrorApiFailureBase               =  10000
 };
 
 /**
  * Channel format kind
  */
 enum  UPTKChannelFormatKind
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
 struct  UPTKChannelFormatDesc
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
 
//  struct UPTKArray;
 
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
 struct  UPTKArraySparseProperties {
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
 struct  UPTKArrayMemoryRequirements {
     size_t size;                    /**< Total size of the array. */
     size_t alignment;               /**< Alignment necessary for mapping the array. */
     unsigned int reserved[4];
 };
 
 /**
  * UPTK memory types
  */
 enum  UPTKMemoryType
 {
     UPTKMemoryTypeUnregistered = 0, /**< Unregistered memory */
     UPTKMemoryTypeHost         = 1, /**< Host memory */
     UPTKMemoryTypeDevice       = 2, /**< Device memory */
     UPTKMemoryTypeManaged      = 3  /**< Managed memory */
 };
 

 /**
  * UPTK Pitched memory pointer
  *
  * \sa ::make_UPTKPitchedPtr
  */
 struct  UPTKPitchedPtr
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
 struct  UPTKExtent
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
 struct  UPTKPos
 {
     size_t x;     /**< x */
     size_t y;     /**< y */
     size_t z;     /**< z */
 };
 
 /**
  * UPTK 3D memory copying parameters
  */
 struct  UPTKMemcpy3DParms
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
  * UPTK 3D cross-device memory copying parameters
  */
 struct  UPTKMemcpy3DPeerParms
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
 struct   UPTKMemsetParams {
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
 enum   UPTKAccessProperty {
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
 struct  UPTKAccessPolicyWindow {
     void *base_ptr;                     /**< Starting address of the access policy window. UPTK driver may align it. */
     size_t num_bytes;                   /**< Size in bytes of the window policy. UPTK driver may restrict the maximum size and alignment. */
     float hitRatio;                     /**< hitRatio specifies percentage of lines assigned hitProp, rest are assigned missProp. */
     enum UPTKAccessProperty hitProp;    /**< ::UPTKaccessProperty set for hit. */
     enum UPTKAccessProperty missProp;   /**< ::UPTKaccessProperty set for miss. Must be either NORMAL or STREAMING. */
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
  * Possible stream capture statuses returned by ::UPTKStreamIsCapturing
  */
 enum  UPTKStreamCaptureStatus {
     UPTKStreamCaptureStatusNone        = 0, /**< Stream is not capturing */
     UPTKStreamCaptureStatusActive      = 1, /**< Stream is actively capturing */
     UPTKStreamCaptureStatusInvalidated = 2  /**< Stream is part of a capture sequence that
                                                    has been invalidated, but not terminated */
 };
 
 /**
  * Possible modes for stream capture thread interactions. For more details see
  * ::UPTKStreamBeginCapture and ::UPTKThreadExchangeStreamCaptureMode
  */
 enum  UPTKStreamCaptureMode {
     UPTKStreamCaptureModeGlobal      = 0,
     UPTKStreamCaptureModeThreadLocal = 1,
     UPTKStreamCaptureModeRelaxed     = 2
 };
 
 enum  UPTKSynchronizationPolicy {
     UPTKSyncPolicyAuto = 1,
     UPTKSyncPolicySpin = 2,
     UPTKSyncPolicyYield = 3,
     UPTKSyncPolicyBlockingSync = 4
 };
 
 /**
  * Cluster scheduling policies. These may be passed to ::UPTKFuncSetAttribute
  */
 enum  UPTKClusterSchedulingPolicy {
     UPTKClusterSchedulingPolicyDefault       = 0, /**< the default policy */
     UPTKClusterSchedulingPolicySpread        = 1, /**< spread the blocks within a cluster to the SMs */
     UPTKClusterSchedulingPolicyLoadBalancing = 2  /**< allow the hardware to load-balance the blocks in a cluster to the SMs */
 };
 
 /**
  * Flags for ::UPTKStreamUpdateCaptureDependencies
  */
 enum  UPTKStreamUpdateCaptureDependenciesFlags {
     UPTKStreamAddCaptureDependencies = 0x0, /**< Add new nodes to the dependency set */
     UPTKStreamSetCaptureDependencies = 0x1  /**< Replace the dependency set with the new nodes */
 };
 
 /**
  * Flags for user objects for graphs
  */
 enum  UPTKUserObjectFlags {
     UPTKUserObjectNoDestructorSync = 0x1  /**< Indicates the destructor execution is not synchronized by any UPTK handle. */
 };
 
 /**
  * Flags for retaining user object references for graphs
  */
 enum  UPTKUserObjectRetainFlags {
     UPTKGraphUserObjectMove = 0x1  /**< Transfer references from the caller rather than creating new references. */
 };
 
 /**
  * UPTK graphics interop resource
  */
 struct UPTKGraphicsResource;
 
 /**
  * UPTK graphics interop register flags
  */
 enum  UPTKGraphicsRegisterFlags
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
 enum  UPTKGraphicsMapFlags
 {
     UPTKGraphicsMapFlagsNone         = 0,  /**< Default; Assume resource can be read/written */
     UPTKGraphicsMapFlagsReadOnly     = 1,  /**< UPTK will not write to this resource */
     UPTKGraphicsMapFlagsWriteDiscard = 2   /**< UPTK will only write to and will not read from this resource */
 };
 
 /**
  * UPTK graphics interop array indices for cube maps
  */
 enum  UPTKGraphicsCubeFace 
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
 enum  UPTKResourceType
 {
     UPTKResourceTypeArray          = 0x00, /**< Array resource */
     UPTKResourceTypeMipmappedArray = 0x01, /**< Mipmapped array resource */
     UPTKResourceTypeLinear         = 0x02, /**< Linear resource */
     UPTKResourceTypePitch2D        = 0x03  /**< Pitch 2D resource */
 };
 
 /**
  * UPTK texture resource view formats
  */
 enum  UPTKResourceViewFormat
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
 struct  UPTKResourceDesc {
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
 struct  UPTKResourceViewDesc
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
 struct  UPTKPointerAttributes
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
 struct  UPTKFuncAttributes
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
 };
 
 /**
  * UPTK function attributes that can be set using ::UPTKFuncSetAttribute
  */
 enum  UPTKFuncAttribute
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
 enum  UPTKFuncCache
 {
     UPTKFuncCachePreferNone   = 0,    /**< Default function cache configuration, no preference */
     UPTKFuncCachePreferShared = 1,    /**< Prefer larger shared memory and smaller L1 cache  */
     UPTKFuncCachePreferL1     = 2,    /**< Prefer larger L1 cache and smaller shared memory */
     UPTKFuncCachePreferEqual  = 3     /**< Prefer equal size L1 cache and shared memory */
 };
 
 /**
  * UPTK shared memory configuration
  */
 
 enum  UPTKSharedMemConfig
 {
     UPTKSharedMemBankSizeDefault   = 0,
     UPTKSharedMemBankSizeFourByte  = 1,
     UPTKSharedMemBankSizeEightByte = 2
 };
 
 /** 
  * Shared memory carveout configurations. These may be passed to UPTKFuncSetAttribute
  */
 enum  UPTKSharedCarveout {
     UPTKSharedmemCarveoutDefault      = -1,  /**< No preference for shared memory or L1 (default) */
     UPTKSharedmemCarveoutMaxShared    = 100, /**< Prefer maximum available shared memory, minimum L1 cache */
     UPTKSharedmemCarveoutMaxL1        = 0    /**< Prefer maximum available L1 cache, minimum shared memory */
 };
 
 /**
  * UPTK device compute modes
  */
 enum  UPTKComputeMode
 {
     UPTKComputeModeDefault          = 0,  /**< Default compute mode (Multiple threads can use ::UPTKSetDevice() with this device) */
     UPTKComputeModeExclusive        = 1,  /**< Compute-exclusive-thread mode (Only one thread in one process will be able to use ::UPTKSetDevice() with this device) */
     UPTKComputeModeProhibited       = 2,  /**< Compute-prohibited mode (No threads can use ::UPTKSetDevice() with this device) */
     UPTKComputeModeExclusiveProcess = 3   /**< Compute-exclusive-process mode (Many threads in one process will be able to use ::UPTKSetDevice() with this device) */
 };
 
 /**
  * UPTK Limits
  */
 enum  UPTKLimit
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
 enum  UPTKMemoryAdvise
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
 enum  UPTKMemRangeAttribute
 {
     UPTKMemRangeAttributeReadMostly           = 1, /**< Whether the range will mostly be read and only occassionally be written to */
     UPTKMemRangeAttributePreferredLocation    = 2, /**< The preferred location of the range */
     UPTKMemRangeAttributeAccessedBy           = 3, /**< Memory range has ::UPTKMemAdviseSetAccessedBy set for specified device */
     UPTKMemRangeAttributeLastPrefetchLocation = 4  /**< The last location to which the range was prefetched */
 };
 
 /**
  * UPTK Profiler Output modes
  */
 enum  UPTKOutputMode
 {
     UPTKKeyValuePair    = 0x00, /**< Output mode Key-Value pair format. */
     UPTKCSV             = 0x01  /**< Output mode Comma separated values format. */
 };
 
 /**
  * UPTK GPUDirect RDMA flush writes APIs supported on the device
  */
 enum  UPTKFlushGPUDirectRDMAWritesOptions {
     UPTKFlushGPUDirectRDMAWritesOptionHost   = 1<<0, /**< ::UPTKDeviceFlushGPUDirectRDMAWrites() and its UPTK Driver API counterpart are supported on the device. */
     UPTKFlushGPUDirectRDMAWritesOptionMemOps = 1<<1  /**< The ::UPTK_STREAM_WAIT_VALUE_FLUSH flag and the ::UPTK_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the UPTK device. */
 };
 
 /**
  * UPTK GPUDirect RDMA flush writes ordering features of the device
  */
 enum  UPTKGPUDirectRDMAWritesOrdering {
     UPTKGPUDirectRDMAWritesOrderingNone       = 0,   /**< The device does not natively support ordering of GPUDirect RDMA writes. ::UPTKFlushGPUDirectRDMAWrites() can be leveraged if supported. */
     UPTKGPUDirectRDMAWritesOrderingOwner      = 100, /**< Natively, the device can consistently consume GPUDirect RDMA writes, although other UPTK devices may not. */
     UPTKGPUDirectRDMAWritesOrderingAllDevices = 200  /**< Any UPTK device in the system can consistently consume GPUDirect RDMA writes to this device. */
 };
 
 /**
  * UPTK GPUDirect RDMA flush writes scopes
  */
 enum  UPTKFlushGPUDirectRDMAWritesScope {
     UPTKFlushGPUDirectRDMAWritesToOwner      = 100, /**< Blocks until remote writes are visible to the UPTK device context owning the data. */
     UPTKFlushGPUDirectRDMAWritesToAllDevices = 200  /**< Blocks until remote writes are visible to all UPTK device contexts. */
 };
 
 /**
  * UPTK GPUDirect RDMA flush writes targets
  */
 enum  UPTKFlushGPUDirectRDMAWritesTarget {
     UPTKFlushGPUDirectRDMAWritesTargetCurrentDevice /**< Sets the target for ::UPTKDeviceFlushGPUDirectRDMAWrites() to the currently active UPTK device context. */
 };
 
 
 /**
  * UPTK device attributes
  */
 enum  UPTKDeviceAttr
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
     UPTKDevAttrMax
 };
 
 /**
  * UPTK memory pool attributes
  */
 enum  UPTKMemPoolAttr
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
  * Specifies the memory protection flags for mapping.
  */
 enum  UPTKMemAccessFlags {
     UPTKMemAccessFlagsProtNone      = 0,  /**< Default, make the address range not accessible */
     UPTKMemAccessFlagsProtRead      = 1,  /**< Make the address range read accessible */
     UPTKMemAccessFlagsProtReadWrite = 3   /**< Make the address range read-write accessible */
 };
 
 /**
  * Memory access descriptor
  */
 struct  UPTKMemAccessDesc {
     struct UPTKMemLocation  location; /**< Location on which the request is to change it's accessibility */
     enum UPTKMemAccessFlags flags;    /**< ::UPTKmemProt accessibility flags to set on the request */
 };
 

 

 /**
  * Opaque data for exporting a pool allocation
  */
 struct  UPTKMemPoolPtrExportData {
     unsigned char reserved[64];
 };
 
 /**
  * Memory allocation node parameters
  */
 struct  UPTKMemAllocNodeParams {
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
  * Graph memory attributes
  */
 enum  UPTKGraphMemAttributeType {
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
 
 enum  UPTKDeviceP2PAttr {
     UPTKDevP2PAttrPerformanceRank              = 1, /**< A relative value indicating the performance of the link between two devices */
     UPTKDevP2PAttrAccessSupported              = 2, /**< Peer access is enabled */
     UPTKDevP2PAttrNativeAtomicSupported        = 3, /**< Native atomic operation over the link supported */
     UPTKDevP2PAttrUPTKArrayAccessSupported     = 4  /**< Accessing UPTK arrays over the link supported */
 };
 
 /**
  * UPTK UUID types
  */
 #ifndef UPTK_UUID_HAS_BEEN_DEFINED
 #define UPTK_UUID_HAS_BEEN_DEFINED
 struct  UPTKuuid_st {     /**< UPTK definition of UUID */
     char bytes[16];
 };
 typedef  struct UPTKuuid_st UPTKuuid;
 #endif
 typedef  struct UPTKuuid_st UPTKUUID_t;
 

struct  UPTKDeviceProp
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
 
 /**
  * UPTK IPC event handle
  */
 typedef  struct  UPTKIpcEventHandle_st
 {
     char reserved[UPTK_IPC_HANDLE_SIZE];
 }UPTKIpcEventHandle_t;
 
 /**
  * UPTK IPC memory handle
  */
 typedef  struct  UPTKIpcMemHandle_st 
 {
     char reserved[UPTK_IPC_HANDLE_SIZE];
 }UPTKIpcMemHandle_t;
 
 /**
  * External memory handle types
  */
 enum  UPTKExternalMemoryHandleType {
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
 struct  UPTKExternalMemoryHandleDesc {
     /**
      * Type of the handle
      */
     enum UPTKExternalMemoryHandleType type;
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
 struct  UPTKExternalMemoryBufferDesc {
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
 struct  UPTKExternalMemoryMipmappedArrayDesc {
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
 enum  UPTKExternalSemaphoreHandleType {
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
 struct  UPTKExternalSemaphoreHandleDesc {
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
 struct  UPTKExternalSemaphoreSignalParams_v1 {
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
 struct  UPTKExternalSemaphoreWaitParams_v1 {
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
 struct  UPTKExternalSemaphoreSignalParams{
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
 struct  UPTKExternalSemaphoreWaitParams {
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
 typedef  enum UPTKError UPTKError_t;
 
 /**
  * UPTK stream
  */
// typedef  struct UPTKstream_st *UPTKStream_t;

 
 /**
  * UPTK event types
  */
 typedef  struct UPTKevent_st *UPTKEvent_t;
 
 /**
  * UPTK graphics resource types
  */
 typedef  struct UPTKGraphicsResource *UPTKGraphicsResource_t;
 
 /**
  * UPTK output file modes
  */
 typedef  enum UPTKOutputMode UPTKOutputMode_t;
 
 /**
  * UPTK external memory
  */
 typedef  struct UPTKexternalMemory_st *UPTKExternalMemory_t;
 
 /**
  * UPTK external semaphore
  */
 typedef  struct UPTKexternalSemaphore_st *UPTKExternalSemaphore_t;
 
 /**
  * UPTK graph
  */
 typedef  struct UPTKgraph_st *UPTKGraph_t;
 
 /**
  * UPTK graph node.
  */
 typedef  struct UPTKgraphNode_st *UPTKGraphNode_t;
 
 /**
  * UPTK user object for graphs
  */
 typedef  struct UPTKuserObject_st *UPTKUserObject_t;
 
 /**
  * UPTK function
  */
 typedef  struct UPTKfunc_st *UPTKFunction_t;
 
 /**
  * UPTK memory pool
  */
 typedef  struct UPTKmemPoolHandle_st *UPTKMemPool_t;
 
 /**
  * UPTK cooperative group scope
  */
 enum  UPTKCGScope {
     UPTKCGScopeInvalid   = 0, /**< Invalid cooperative group scope */
     UPTKCGScopeGrid      = 1, /**< Scope represented by a grid_group */
     UPTKCGScopeMultiGrid = 2  /**< Scope represented by a multi_grid_group */
 };
 
 /**
  * UPTK launch parameters
  */
 struct  UPTKLaunchParams
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
 struct  UPTKKernelNodeParams {
     void* func;                     /**< Kernel to launch */
     dim3 gridDim;                   /**< Grid dimensions */
     dim3 blockDim;                  /**< Block dimensions */
     unsigned int sharedMemBytes;    /**< Dynamic shared-memory size per thread block in bytes */
     void **kernelParams;            /**< Array of pointers to individual kernel arguments*/
     void **extra;                   /**< Pointer to kernel arguments in the "extra" format */
 };
 
 /**
  * External semaphore signal node parameters
  */
 struct  UPTKExternalSemaphoreSignalNodeParams {
     UPTKExternalSemaphore_t* extSemArray;                        /**< Array of external semaphore handles. */
     const struct UPTKExternalSemaphoreSignalParams* paramsArray; /**< Array of external semaphore signal parameters. */
     unsigned int numExtSems;                                     /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
 };
 
 /**
  * External semaphore wait node parameters
  */
 struct  UPTKExternalSemaphoreWaitNodeParams {
     UPTKExternalSemaphore_t* extSemArray;                      /**< Array of external semaphore handles. */
     const struct UPTKExternalSemaphoreWaitParams* paramsArray; /**< Array of external semaphore wait parameters. */
     unsigned int numExtSems;                                   /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
 };
 
/**
* UPTK Graph node types
*/
enum  UPTKGraphNodeType {
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
  * UPTK executable (launchable) graph
  */
 typedef struct UPTKgraphExec_st* UPTKGraphExec_t;
 
 /**
 * UPTK Graph Update error types
 */
 enum  UPTKGraphExecUpdateResult {
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
  * Flags to specify search options to be used with ::UPTKGetDriverEntryPoint
  * For more details see ::cuGetProcAddress
  */ 
 enum  UPTKGetDriverEntryPointFlags {
     UPTKEnableDefault                = 0x0, /**< Default search mode for driver symbols. */
     UPTKEnableLegacyStream           = 0x1, /**< Search for legacy versions of driver symbols. */
     UPTKEnablePerThreadDefaultStream = 0x2  /**< Search for per-thread versions of driver symbols. */
 };
 
 /**
  * UPTK Graph debug write options
  */
 enum  UPTKGraphDebugDotFlags {
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
 };
 
 /**
  * Flags for instantiating a graph
  */
 enum  UPTKGraphInstantiateFlags {
     UPTKGraphInstantiateFlagAutoFreeOnLaunch = 1 /**< Automatically free memory allocated in a graph before relaunching. */
   , UPTKGraphInstantiateFlagUseNodePriority  = 8 /**< Run the graph using the per-node priority attributes rather than the
                                                       priority of the stream it is launched into. */
 };
 
 /**
  * Launch attributes enum; used as id field of ::UPTKLaunchAttribute
  */
 typedef  enum UPTKLaunchAttributeID {
     UPTKLaunchAttributeIgnore                = 0 /**< Ignored entry, for convenient composition */
   , UPTKLaunchAttributeAccessPolicyWindow    = 1 /**< Valid for streams, graph nodes, launches. */
   , UPTKLaunchAttributeCooperative           = 2 /**< Valid for graph nodes, launches. */
   , UPTKLaunchAttributeSynchronizationPolicy = 3 /**< Valid for streams. */
   , UPTKLaunchAttributeClusterDimension                  = 4 /**< Valid for graph nodes, launches. */
   , UPTKLaunchAttributeClusterSchedulingPolicyPreference = 5 /**< Valid for graph nodes, launches. */
   , UPTKLaunchAttributeProgrammaticStreamSerialization   = 6 /**< Valid for launches. Setting
                                                                   programmaticStreamSerializationAllowed to non-0
                                                                   signals that the kernel will use programmatic
                                                                   means to resolve its stream dependency, so that
                                                                   the UPTK runtime should opportunistically allow
                                                                   the grid's execution to overlap with the previous
                                                                   kernel in the stream, if that kernel requests the
                                                                   overlap. */
   , UPTKLaunchAttributeProgrammaticEvent                 = 7 /**< Valid for launches. Event recorded through this launch
                                                                   attribute is guaranteed to only trigger after all
                                                                   block in the associated kernel trigger the event. A
                                                                   block can trigger the event through PTX
                                                                   griddepcontrol.launch_dependents. A trigger can also
                                                                   be inserted at the beginning of each block's execution
                                                                   if triggerAtBlockStart is set to non-0. Note that
                                                                   dependents (including the CPU thread calling
                                                                   UPTKEventSynchronize()) are not guaranteed to observe
                                                                   the release precisely when it is released. For
                                                                   example, UPTKEventSynchronize() may only observe the
                                                                   event trigger long after the associated kernel has
                                                                   completed. This recording type is primarily meant for
                                                                   establishing programmatic dependency between device
                                                                   tasks. The event supplied must not be an interprocess
                                                                   or interop event. The event must disable timing
                                                                   (i.e. created with ::UPTKEventDisableTiming flag
                                                                   set). */
   , UPTKLaunchAttributePriority              = 8 /**< Valid for graph nodes. */
 } UPTKLaunchAttributeID;
 
 /**
  * Launch attributes union; used as value field of ::UPTKLaunchAttribute
  */
 typedef  union UPTKLaunchAttributeValue {
     char pad[64]; /* Pad to 64 bytes */
     struct UPTKAccessPolicyWindow accessPolicyWindow;
     int cooperative;
     enum UPTKSynchronizationPolicy syncPolicy;
     struct {
         unsigned int x;
         unsigned int y;
         unsigned int z;
     } clusterDim;
     enum UPTKClusterSchedulingPolicy clusterSchedulingPolicyPreference;
     int programmaticStreamSerializationAllowed;
     struct {
         UPTKEvent_t event;
         int flags;
         int triggerAtBlockStart;
     } programmaticEvent;
     int priority;
 } UPTKLaunchAttributeValue;
 
 /**
  * Launch attribute
  */
 typedef  struct UPTKLaunchAttribute_st {
     UPTKLaunchAttributeID id;
     char pad[8 - sizeof(UPTKLaunchAttributeID)];
     UPTKLaunchAttributeValue val;
 } UPTKLaunchAttribute;
 
 /**
  * UPTK extensible launch configuration
  */
 typedef  struct UPTKLaunchConfig_st {
     dim3 gridDim;               /**< Grid dimentions */
     dim3 blockDim;              /**< Block dimentions */
     size_t dynamicSmemBytes;    /**< Dynamic shared-memory size per thread block in bytes */
     UPTKStream_t stream;        /**< Stream identifier */
     UPTKLaunchAttribute *attrs; /**< nullable if numAttrs == 0 */
     unsigned int numAttrs;      /**< Number of attributes populated in attrs */
 } UPTKLaunchConfig_t;

 struct  UPTKKernelNodeParamsV2 {
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

struct  UPTKMemcpyNodeParams {
    int flags;                            /**< Must be zero */
    int reserved[3];                      /**< Must be zero */
    struct UPTKMemcpy3DParms copyParams;  /**< Parameters for the memory copy */
};

struct   UPTKMemsetParamsV2 {
    void *dst;                              /**< Destination device pointer */
    size_t pitch;                           /**< Pitch of destination device pointer. Unused if height is 1 */
    unsigned int value;                     /**< Value to be set */
    unsigned int elementSize;               /**< Size of each element in bytes. Must be 1, 2, or 4. */
    size_t width;                           /**< Width of the row in elements */
    size_t height;                          /**< Number of rows */
};

struct  UPTKChildGraphNodeParams {
    UPTKGraph_t graph; /**< The child graph to clone into the node for node creation, or
                            a handle to the graph owned by the node for node query */
};

struct  UPTKEventWaitNodeParams {
    UPTKEvent_t event; /**< The event to wait on from the node */
};

struct  UPTKEventRecordNodeParams {
    UPTKEvent_t event; /**< The event to record when the node executes */
};

struct  UPTKExternalSemaphoreSignalNodeParamsV2 {
    UPTKExternalSemaphore_t* extSemArray;                        /**< Array of external semaphore handles. */
    const struct UPTKExternalSemaphoreSignalParams* paramsArray; /**< Array of external semaphore signal parameters. */
    unsigned int numExtSems;                                     /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
};

struct  UPTKExternalSemaphoreWaitNodeParamsV2 {
    UPTKExternalSemaphore_t* extSemArray;                      /**< Array of external semaphore handles. */
    const struct UPTKExternalSemaphoreWaitParams* paramsArray; /**< Array of external semaphore wait parameters. */
    unsigned int numExtSems;                                   /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
};

struct  UPTKMemFreeNodeParams {
    void *dptr; /**< in: the pointer to free */
};

enum  UPTKGraphConditionalNodeType {
     UPTKGraphCondTypeIf = 0,      /**< Conditional 'if' Node. Body executed once if condition value is non-zero. */
     UPTKGraphCondTypeWhile = 1,   /**< Conditional 'while' Node. Body executed repeatedly while condition value is non-zero. */
};

struct  UPTKConditionalNodeParams {
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

struct  UPTKGraphNodeParams {
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
  * Stream Attributes
  */
 #define UPTKStreamAttrID UPTKLaunchAttributeID
 #define UPTKStreamAttributeAccessPolicyWindow    UPTKLaunchAttributeAccessPolicyWindow
 #define UPTKStreamAttributeSynchronizationPolicy UPTKLaunchAttributeSynchronizationPolicy
 
 /**
  * Stream attributes union used with ::UPTKStreamSetAttribute/::UPTKStreamGetAttribute
  */
 #define UPTKStreamAttrValue UPTKLaunchAttributeValue
 
 /**
  * Graph kernel node Attributes
  */
 #define UPTKKernelNodeAttrID UPTKLaunchAttributeID
 #define UPTKKernelNodeAttributeAccessPolicyWindow UPTKLaunchAttributeAccessPolicyWindow
 #define UPTKKernelNodeAttributeCooperative        UPTKLaunchAttributeCooperative
 #define UPTKKernelNodeAttributePriority           UPTKLaunchAttributePriority
 #define UPTKKernelNodeAttributeClusterDimension                     UPTKLaunchAttributeClusterDimension
 #define UPTKKernelNodeAttributeClusterSchedulingPolicyPreference    UPTKLaunchAttributeClusterSchedulingPolicyPreference
 
 /**
  * Graph kernel node attributes union, used with ::UPTKGraphKernelNodeSetAttribute/::UPTKGraphKernelNodeGetAttribute
  */
 #define UPTKKernelNodeAttrValue UPTKLaunchAttributeValue
    /**
     * UPTK Error types
     */
    typedef enum UPTKError UPTKError_t;
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
    extern __host__ UPTKError_t UPTKEventRecord(UPTKEvent_t event, UPTKStream_t stream);
    // extern __host__ UPTKError_t  UPTKMemcpy(void * dst,const void * src,size_t count,enum UPTKMemcpyKind kind);
    extern __host__ UPTKError_t UPTKIpcGetMemHandle(UPTKIpcMemHandle_t *handle, void *devPtr);
    extern __host__ UPTKError_t UPTKIpcOpenMemHandle(void **devPtr, UPTKIpcMemHandle_t handle, unsigned int flags);
    extern __host__ UPTKError_t UPTKIpcCloseMemHandle(void *devPtr);
    extern __host__ UPTKError_t UPTKChooseDevice(int *device, const struct UPTKDeviceProp *prop);
    extern __host__ UPTKError_t UPTKSetDevice(int device);
    extern __host__ UPTKError_t UPTKGetDevice(int *device);
    // extern __host__ UPTKError_t  UPTKDeviceSynchronize(void);
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
    extern __host__ UPTKError_t UPTKMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, UPTKStream_t stream __dv(0));
    extern __host__ UPTKError_t UPTKMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count);
    extern __host__ UPTKError_t UPTKMemcpyAsync(void *dst, const void *src, size_t count, enum UPTKMemcpyKind kind, UPTKStream_t stream __dv(0));

#if defined(__cplusplus)
    extern __host__ UPTKError_t UPTKMallocManaged(void **devPtr, size_t size, unsigned int flags = UPTKMemAttachGlobal);
#else
    extern __host__ UPTKError_t UPTKMallocManaged(void **devPtr, size_t size, unsigned int flags);
#endif

    extern __host__ UPTKError_t UPTKMallocAsync(void **devPtr, size_t size, UPTKStream_t hStream);

    extern __UPTK_DEPRECATED __host__ UPTKError_t UPTKLaunchCooperativeKernelMultiDevice(struct UPTKLaunchParams *launchParamsList, unsigned int numDevices, unsigned int flags __dv(0));
    extern __host__ UPTKError_t UPTKGraphNodeGetType(UPTKGraphNode_t node, enum UPTKGraphNodeType *pType);
    extern __host__ UPTKError_t UPTKGraphGetNodes(UPTKGraph_t graph, UPTKGraphNode_t *nodes, size_t *numNodes);
    extern __host__ UPTKError_t UPTKGraphExecDestroy(UPTKGraphExec_t graphExec);
    extern __host__ UPTKError_t UPTKGraphDestroyNode(UPTKGraphNode_t node);
    extern __host__ UPTKError_t UPTKGraphAddNode(UPTKGraphNode_t *pGraphNode, UPTKGraph_t graph, const UPTKGraphNode_t *pDependencies, size_t numDependencies, struct UPTKGraphNodeParams *nodeParams);
    extern __host__ UPTKError_t UPTKGetLastError(void);
    extern __host__ const char *UPTKGetErrorString(UPTKError_t error);
    extern __host__ const char *UPTKGetErrorName(UPTKError_t error);
    extern __host__ UPTKError_t UPTKGetDeviceCount(int *count);
    extern __host__ UPTKError_t UPTKFreeAsync(void *devPtr, UPTKStream_t hStream);
    extern __host__ UPTKError_t UPTKEventElapsedTime(float *ms, UPTKEvent_t start, UPTKEvent_t end);
    extern __host__ UPTKError_t UPTKDeviceGetAttribute(int *value, enum UPTKDeviceAttr attr, int device);
    extern __host__ UPTKError_t UPTKDeviceEnablePeerAccess(int peerDevice, unsigned int flags);

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
 * Error codes
 */
typedef enum UPTKError_enum {
    /**
     * The API call returned with no errors. In the case of query calls, this
     * also means that the operation being queried is complete (see
     * ::cuEventQuery() and ::cuStreamQuery()).
     */
    UPTK_SUCCESS                              = 0,

    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    UPTK_ERROR_INVALID_VALUE                  = 1,

    /**
     * The API call failed because it was unable to allocate enough memory or
     * other resources to perform the requested operation.
     */
    UPTK_ERROR_OUT_OF_MEMORY                  = 2,

    /**
     * This indicates that the UPTK driver has not been initialized with
     * ::cuInit() or that initialization has failed.
     */
    UPTK_ERROR_NOT_INITIALIZED                = 3,

    /**
     * This indicates that the UPTK driver is in the process of shutting down.
     */
    UPTK_ERROR_DEINITIALIZED                  = 4,

    /**
     * This indicates profiler is not initialized for this run. This can
     * happen when the application is running with external profiling tools
     * like visual profiler.
     */
    UPTK_ERROR_PROFILER_DISABLED              = 5,

    /**
     * \deprecated
     * This error return is deprecated as of UPTK 5.0. It is no longer an error
     * to attempt to enable/disable the profiling via ::cuProfilerStart or
     * ::cuProfilerStop without initialization.
     */
    UPTK_ERROR_PROFILER_NOT_INITIALIZED       = 6,

    /**
     * \deprecated
     * This error return is deprecated as of UPTK 5.0. It is no longer an error
     * to call cuProfilerStart() when profiling is already enabled.
     */
    UPTK_ERROR_PROFILER_ALREADY_STARTED       = 7,

    /**
     * \deprecated
     * This error return is deprecated as of UPTK 5.0. It is no longer an error
     * to call cuProfilerStop() when profiling is already disabled.
     */
    UPTK_ERROR_PROFILER_ALREADY_STOPPED       = 8,

    /**
     * This indicates that the UPTK driver that the application has loaded is a
     * stub library. Applications that run with the stub rather than a real
     * driver loaded will result in UPTK API returning this error.
     */
    UPTK_ERROR_STUB_LIBRARY                   = 34,

    /**  
     * This indicates that requested UPTK device is unavailable at the current
     * time. Devices are often unavailable due to use of
     * ::UPTK_COMPUTEMODE_EXCLUSIVE_PROCESS or ::UPTK_COMPUTEMODE_PROHIBITED.
     */
    UPTK_ERROR_DEVICE_UNAVAILABLE            = 46,

    /**
     * This indicates that no UPTK-capable devices were detected by the installed
     * UPTK driver.
     */
    UPTK_ERROR_NO_DEVICE                      = 100,

    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid UPTK device or that the action requested is
     * invalid for the specified device.
     */
    UPTK_ERROR_INVALID_DEVICE                 = 101,

    /**
     * This error indicates that the Grid license is not applied.
     */
    UPTK_ERROR_DEVICE_NOT_LICENSED            = 102,

    /**
     * This indicates that the device kernel image is invalid. This can also
     * indicate an invalid UPTK module.
     */
    UPTK_ERROR_INVALID_IMAGE                  = 200,

    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::cuCtxDestroy() invoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::cuCtxGetApiVersion() for more details.
     * This can also be returned if the green context passed to an API call
     * was not converted to a ::UPTKcontext using ::cuCtxFromGreenCtx API.
     */
    UPTK_ERROR_INVALID_CONTEXT                = 201,

    /**
     * This indicated that the context being supplied as a parameter to the
     * API call was already the active context.
     * \deprecated
     * This error return is deprecated as of UPTK 3.2. It is no longer an
     * error to attempt to push the active context via ::cuCtxPushCurrent().
     */
    UPTK_ERROR_CONTEXT_ALREADY_CURRENT        = 202,

    /**
     * This indicates that a map or register operation has failed.
     */
    UPTK_ERROR_MAP_FAILED                     = 205,

    /**
     * This indicates that an unmap or unregister operation has failed.
     */
    UPTK_ERROR_UNMAP_FAILED                   = 206,

    /**
     * This indicates that the specified array is currently mapped and thus
     * cannot be destroyed.
     */
    UPTK_ERROR_ARRAY_IS_MAPPED                = 207,

    /**
     * This indicates that the resource is already mapped.
     */
    UPTK_ERROR_ALREADY_MAPPED                 = 208,

    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular UPTK source file that do not include the
     * corresponding device configuration.
     */
    UPTK_ERROR_NO_BINARY_FOR_GPU              = 209,

    /**
     * This indicates that a resource has already been acquired.
     */
    UPTK_ERROR_ALREADY_ACQUIRED               = 210,

    /**
     * This indicates that a resource is not mapped.
     */
    UPTK_ERROR_NOT_MAPPED                     = 211,

    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    UPTK_ERROR_NOT_MAPPED_AS_ARRAY            = 212,

    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    UPTK_ERROR_NOT_MAPPED_AS_POINTER          = 213,

    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    UPTK_ERROR_ECC_UNCORRECTABLE              = 214,

    /**
     * This indicates that the ::UPTKlimit passed to the API call is not
     * supported by the active device.
     */
    UPTK_ERROR_UNSUPPORTED_LIMIT              = 215,

    /**
     * This indicates that the ::UPTKcontext passed to the API call can
     * only be bound to a single CPU thread at a time but is already
     * bound to a CPU thread.
     */
    UPTK_ERROR_CONTEXT_ALREADY_IN_USE         = 216,

    /**
     * This indicates that peer access is not supported across the given
     * devices.
     */
    UPTK_ERROR_PEER_ACCESS_UNSUPPORTED        = 217,

    /**
     * This indicates that a PTX JIT compilation failed.
     */
    UPTK_ERROR_INVALID_PTX                    = 218,

    /**
     * This indicates an error with OpenGL or DirectX context.
     */
    UPTK_ERROR_INVALID_GRAPHICS_CONTEXT       = 219,

    /**
    * This indicates that an uncorrectable NVLink error was detected during the
    * execution.
    */
    UPTK_ERROR_NVLINK_UNCORRECTABLE           = 220,

    /**
    * This indicates that the PTX JIT compiler library was not found.
    */
    UPTK_ERROR_JIT_COMPILER_NOT_FOUND         = 221,

    /**
     * This indicates that the provided PTX was compiled with an unsupported toolchain.
     */

    UPTK_ERROR_UNSUPPORTED_PTX_VERSION        = 222,

    /**
     * This indicates that the PTX JIT compilation was disabled.
     */
    UPTK_ERROR_JIT_COMPILATION_DISABLED       = 223,

    /**
     * This indicates that the ::UPTKexecAffinityType passed to the API call is not
     * supported by the active device.
     */ 
    UPTK_ERROR_UNSUPPORTED_EXEC_AFFINITY      = 224,

    /**
     * This indicates that the code to be compiled by the PTX JIT contains
     * unsupported call to UPTKDeviceSynchronize.
     */
    UPTK_ERROR_UNSUPPORTED_DEVSIDE_SYNC       = 225,

    /**
     * This indicates that the device kernel source is invalid. This includes
     * compilation/linker errors encountered in device code or user error.
     */
    UPTK_ERROR_INVALID_SOURCE                 = 300,

    /**
     * This indicates that the file specified was not found.
     */
    UPTK_ERROR_FILE_NOT_FOUND                 = 301,

    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    UPTK_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,

    /**
     * This indicates that initialization of a shared object failed.
     */
    UPTK_ERROR_SHARED_OBJECT_INIT_FAILED      = 303,

    /**
     * This indicates that an OS call failed.
     */
    UPTK_ERROR_OPERATING_SYSTEM               = 304,

    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::UPTKstream and ::UPTKevent.
     */
    UPTK_ERROR_INVALID_HANDLE                 = 400,

    /**
     * This indicates that a resource required by the API call is not in a
     * valid state to perform the requested operation.
     */
    UPTK_ERROR_ILLEGAL_STATE                  = 401,

    /**
     * This indicates an attempt was made to introspect an object in a way that
     * would discard semantically important information. This is either due to
     * the object using funtionality newer than the API version used to
     * introspect it or omission of optional return arguments.
     */
    UPTK_ERROR_LOSSY_QUERY                    = 402,

    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, driver function names, texture names,
     * and surface names.
     */
    UPTK_ERROR_NOT_FOUND                      = 500,

    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::UPTK_SUCCESS (which indicates completion). Calls that
     * may return this value include ::cuEventQuery() and ::cuStreamQuery().
     */
    UPTK_ERROR_NOT_READY                      = 600,

    /**
     * While executing a kernel, the device encountered a
     * load or store instruction on an invalid memory address.
     * This leaves the process in an inconsistent state and any further UPTK work
     * will return the same error. To continue using UPTK, the process must be terminated
     * and relaunched.
     */
    UPTK_ERROR_ILLEGAL_ADDRESS                = 700,

    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. This error usually indicates that the user has
     * attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register
     * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
     * when a 32-bit int is expected) is equivalent to passing too many
     * arguments and can also result in this error.
     */
    UPTK_ERROR_LAUNCH_OUT_OF_RESOURCES        = 701,

    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device attribute
     * ::UPTK_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information.
     * This leaves the process in an inconsistent state and any further UPTK work
     * will return the same error. To continue using UPTK, the process must be terminated
     * and relaunched.
     */
    UPTK_ERROR_LAUNCH_TIMEOUT                 = 702,

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    UPTK_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = 703,

    /**
     * This error indicates that a call to ::cuCtxEnablePeerAccess() is
     * trying to re-enable peer access to a context which has already
     * had peer access to it enabled.
     */
    UPTK_ERROR_PEER_ACCESS_ALREADY_ENABLED    = 704,

    /**
     * This error indicates that ::cuCtxDisablePeerAccess() is
     * trying to disable peer access which has not been enabled yet
     * via ::cuCtxEnablePeerAccess().
     */
    UPTK_ERROR_PEER_ACCESS_NOT_ENABLED        = 705,

    /**
     * This error indicates that the primary context for the specified device
     * has already been initialized.
     */
    UPTK_ERROR_PRIMARY_CONTEXT_ACTIVE         = 708,

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::cuCtxDestroy, or is a primary context which
     * has not yet been initialized.
     */
    UPTK_ERROR_CONTEXT_IS_DESTROYED           = 709,

    /**
     * A device-side assert triggered during kernel execution. The context
     * cannot be used anymore, and must be destroyed. All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using UPTK.
     */
    UPTK_ERROR_ASSERT                         = 710,

    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices
     * passed to ::cuCtxEnablePeerAccess().
     */
    UPTK_ERROR_TOO_MANY_PEERS                 = 711,

    /**
     * This error indicates that the memory range passed to ::cuMemHostRegister()
     * has already been registered.
     */
    UPTK_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,

    /**
     * This error indicates that the pointer passed to ::cuMemHostUnregister()
     * does not correspond to any currently registered memory region.
     */
    UPTK_ERROR_HOST_MEMORY_NOT_REGISTERED     = 713,

    /**
     * While executing a kernel, the device encountered a stack error.
     * This can be due to stack corruption or exceeding the stack size limit.
     * This leaves the process in an inconsistent state and any further UPTK work
     * will return the same error. To continue using UPTK, the process must be terminated
     * and relaunched.
     */
    UPTK_ERROR_HARDWARE_STACK_ERROR           = 714,

    /**
     * While executing a kernel, the device encountered an illegal instruction.
     * This leaves the process in an inconsistent state and any further UPTK work
     * will return the same error. To continue using UPTK, the process must be terminated
     * and relaunched.
     */
    UPTK_ERROR_ILLEGAL_INSTRUCTION            = 715,

    /**
     * While executing a kernel, the device encountered a load or store instruction
     * on a memory address which is not aligned.
     * This leaves the process in an inconsistent state and any further UPTK work
     * will return the same error. To continue using UPTK, the process must be terminated
     * and relaunched.
     */
    UPTK_ERROR_MISALIGNED_ADDRESS             = 716,

    /**
     * While executing a kernel, the device encountered an instruction
     * which can only operate on memory locations in certain address spaces
     * (global, shared, or local), but was supplied a memory address not
     * belonging to an allowed address space.
     * This leaves the process in an inconsistent state and any further UPTK work
     * will return the same error. To continue using UPTK, the process must be terminated
     * and relaunched.
     */
    UPTK_ERROR_INVALID_ADDRESS_SPACE          = 717,

    /**
     * While executing a kernel, the device program counter wrapped its address space.
     * This leaves the process in an inconsistent state and any further UPTK work
     * will return the same error. To continue using UPTK, the process must be terminated
     * and relaunched.
     */
    UPTK_ERROR_INVALID_PC                     = 718,

    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. Less common cases can be system specific - more
     * information about these cases can be found in the system specific user guide.
     * This leaves the process in an inconsistent state and any further UPTK work
     * will return the same error. To continue using UPTK, the process must be terminated
     * and relaunched.
     */
    UPTK_ERROR_LAUNCH_FAILED                  = 719,

    /**
     * This error indicates that the number of blocks launched per grid for a kernel that was
     * launched via either ::cuLaunchCooperativeKernel or ::cuLaunchCooperativeKernelMultiDevice
     * exceeds the maximum number of blocks as allowed by ::cuOccupancyMaxActiveBlocksPerMultiprocessor
     * or ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
     * as specified by the device attribute ::UPTK_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
     */
    UPTK_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE   = 720,

    /**
     * This error indicates that the attempted operation is not permitted.
     */
    UPTK_ERROR_NOT_PERMITTED                  = 800,

    /**
     * This error indicates that the attempted operation is not supported
     * on the current system or device.
     */
    UPTK_ERROR_NOT_SUPPORTED                  = 801,

    /**
     * This error indicates that the system is not yet ready to start any UPTK
     * work.  To continue using UPTK, verify the system configuration is in a
     * valid state and all required driver daemons are actively running.
     * More information about this error can be found in the system specific
     * user guide.
     */
    UPTK_ERROR_SYSTEM_NOT_READY               = 802,

    /**
     * This error indicates that there is a mismatch between the versions of
     * the display driver and the UPTK driver. Refer to the compatibility documentation
     * for supported versions.
     */
    UPTK_ERROR_SYSTEM_DRIVER_MISMATCH         = 803,

    /**
     * This error indicates that the system was upgraded to run with forward compatibility
     * but the visible hardware detected by UPTK does not support this configuration.
     * Refer to the compatibility documentation for the supported hardware matrix or ensure
     * that only supported hardware is visible during initialization via the UPTK_VISIBLE_DEVICES
     * environment variable.
     */
    UPTK_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,

    /**
     * This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.
     */
    UPTK_ERROR_MPS_CONNECTION_FAILED          = 805,

    /**
     * This error indicates that the remote procedural call between the MPS server and the MPS client failed.
     */
    UPTK_ERROR_MPS_RPC_FAILURE                = 806,

    /**
     * This error indicates that the MPS server is not ready to accept new MPS client requests.
     * This error can be returned when the MPS server is in the process of recovering from a fatal failure.
     */
    UPTK_ERROR_MPS_SERVER_NOT_READY           = 807,

    /**
     * This error indicates that the hardware resources required to create MPS client have been exhausted.
     */
    UPTK_ERROR_MPS_MAX_CLIENTS_REACHED        = 808,

    /**
     * This error indicates the the hardware resources required to support device connections have been exhausted.
     */
    UPTK_ERROR_MPS_MAX_CONNECTIONS_REACHED    = 809,

    /**
     * This error indicates that the MPS client has been terminated by the server. To continue using UPTK, the process must be terminated and relaunched.
     */
    UPTK_ERROR_MPS_CLIENT_TERMINATED          = 810,

    /**
     * This error indicates that the module is using UPTK Dynamic Parallelism, but the current configuration, like MPS, does not support it.
     */
    UPTK_ERROR_CDP_NOT_SUPPORTED              = 811,

    /**
     * This error indicates that a module contains an unsupported interaction between different versions of UPTK Dynamic Parallelism.
     */
    UPTK_ERROR_CDP_VERSION_MISMATCH           = 812,

    /**
     * This error indicates that the operation is not permitted when
     * the stream is capturing.
     */
    UPTK_ERROR_STREAM_CAPTURE_UNSUPPORTED     = 900,

    /**
     * This error indicates that the current capture sequence on the stream
     * has been invalidated due to a previous error.
     */
    UPTK_ERROR_STREAM_CAPTURE_INVALIDATED     = 901,

    /**
     * This error indicates that the operation would have resulted in a merge
     * of two independent capture sequences.
     */
    UPTK_ERROR_STREAM_CAPTURE_MERGE           = 902,

    /**
     * This error indicates that the capture was not initiated in this stream.
     */
    UPTK_ERROR_STREAM_CAPTURE_UNMATCHED       = 903,

    /**
     * This error indicates that the capture sequence contains a fork that was
     * not joined to the primary stream.
     */
    UPTK_ERROR_STREAM_CAPTURE_UNJOINED        = 904,

    /**
     * This error indicates that a dependency would have been created which
     * crosses the capture sequence boundary. Only implicit in-stream ordering
     * dependencies are allowed to cross the boundary.
     */
    UPTK_ERROR_STREAM_CAPTURE_ISOLATION       = 905,

    /**
     * This error indicates a disallowed implicit dependency on a current capture
     * sequence from UPTKStreamLegacy.
     */
    UPTK_ERROR_STREAM_CAPTURE_IMPLICIT        = 906,

    /**
     * This error indicates that the operation is not permitted on an event which
     * was last recorded in a capturing stream.
     */
    UPTK_ERROR_CAPTURED_EVENT                 = 907,

    /**
     * A stream capture sequence not initiated with the ::UPTK_STREAM_CAPTURE_MODE_RELAXED
     * argument to ::cuStreamBeginCapture was passed to ::cuStreamEndCapture in a
     * different thread.
     */
    UPTK_ERROR_STREAM_CAPTURE_WRONG_THREAD    = 908,

    /**
     * This error indicates that the timeout specified for the wait operation has lapsed.
     */
    UPTK_ERROR_TIMEOUT                        = 909,

    /**
     * This error indicates that the graph update was not performed because it included 
     * changes which violated constraints specific to instantiated graph update.
     */
    UPTK_ERROR_GRAPH_EXEC_UPDATE_FAILURE      = 910,

    /**
     * This indicates that an async error has occurred in a device outside of UPTK.
     * If UPTK was waiting for an external device's signal before consuming shared data,
     * the external device signaled an error indicating that the data is not valid for
     * consumption. This leaves the process in an inconsistent state and any further UPTK
     * work will return the same error. To continue using UPTK, the process must be
     * terminated and relaunched.
     */
    UPTK_ERROR_EXTERNAL_DEVICE               = 911,

    /**
     * Indicates a kernel launch error due to cluster misconfiguration.
     */
    UPTK_ERROR_INVALID_CLUSTER_SIZE           = 912,

    /**
     * Indiciates a function handle is not loaded when calling an API that requires
     * a loaded function.
    */
    UPTK_ERROR_FUNCTION_NOT_LOADED            = 913,

    /**
     * This error indicates one or more resources passed in are not valid resource
     * types for the operation.
    */
    UPTK_ERROR_INVALID_RESOURCE_TYPE          = 914,

    /**
     * This error indicates one or more resources are insufficient or non-applicable for
     * the operation.
    */
    UPTK_ERROR_INVALID_RESOURCE_CONFIGURATION = 915,

    /**
     * This indicates that an unknown internal error has occurred.
     */
    UPTK_ERROR_UNKNOWN                        = 999
} UPTKresult;


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

typedef struct UPTKstream_st *UPTKstream;                        /**< UPTK stream */
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
     * device on which the function is currently loaded.
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

UPTKresult  UPTKModuleLoad(UPTKmodule * module,const char * fname);
UPTKresult  UPTKModuleLoadData(UPTKmodule * module,const void * image);
UPTKresult  UPTKModuleUnload(UPTKmodule hmod);
UPTKresult  UPTKModuleGetFunction(UPTKfunction * hfunc,UPTKmodule hmod,const char * name);
UPTKresult  UPTKModuleGetGlobal(UPTKdeviceptr * dptr, size_t * bytes,UPTKmodule hmod,const char * name);
UPTKresult  UPTKLinkCreate(unsigned int numOptions,UPTKjit_option * options,void ** optionValues,UPTKlinkState * stateOut);
UPTKresult  UPTKLinkDestroy(UPTKlinkState state);
UPTKresult  UPTKLinkAddData(UPTKlinkState state,UPTKjitInputType type,void * data,size_t size,const char * name,unsigned int numOptions,UPTKjit_option * options,void ** optionValues);
UPTKresult  UPTKLinkComplete(UPTKlinkState state, void **cubinOut, size_t *sizeOut);
UPTKresult  UPTKInit(unsigned int Flags);
UPTKresult  UPTKDeviceGetName(char *name, int len, UPTKdevice dev);
UPTKresult  UPTKCtxSynchronize(void);
UPTKresult  UPTKCtxSetLimit(UPTKlimit limit, size_t value);
UPTKresult  UPTKCtxSetCurrent(UPTKcontext ctx);
UPTKresult  UPTKCtxGetLimit(size_t *pvalue, UPTKlimit limit);
UPTKresult  UPTKCtxDestroy(UPTKcontext ctx);
UPTKresult  UPTKCtxCreate(UPTKcontext *pctx, unsigned int flags, UPTKdevice dev);
#if defined(__cplusplus)
}

#endif /* __cplusplus */

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
static __inline__ __host__ struct UPTKPitchedPtr make_UPTKPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz) 
{
  struct UPTKPitchedPtr s;

  s.ptr   = d;
  s.pitch = p;
  s.xsize = xsz;
  s.ysize = ysz;

  return s;
}

/**
 * \brief Returns a UPTKPos based on input parameters
 *
 * Returns a ::UPTKPos based on the specified input parameters \p x,
 * \p y, and \p z.
 *
 * \param x - X position
 * \param y - Y position
 * \param z - Z position
 *
 * \return
 * ::UPTKPos specified by \p x, \p y, and \p z
 *
 * \sa make_UPTKExtent, make_UPTKPitchedPtr
 */
static __inline__ __host__ struct UPTKPos  make_UPTKPos(size_t x, size_t y, size_t z) 
{
  struct UPTKPos p;

  p.x = x;
  p.y = y;
  p.z = z;

  return p;
}

/**
 * \brief Returns a UPTKExtent based on input parameters
 *
 * Returns a ::UPTKExtent based on the specified input parameters \p w,
 * \p h, and \p d.
 *
 * \param w - Width in elements when referring to array memory, in bytes when referring to linear memory
 * \param h - Height in elements
 * \param d - Depth in elements
 *
 * \return
 * ::UPTKExtent specified by \p w, \p h, and \p d
 *
 * \sa make_UPTKPitchedPtr, make_UPTKPos
 */
static __inline__ __host__ struct UPTKExtent make_UPTKExtent(size_t w, size_t h, size_t d) 
{
  struct UPTKExtent e;

  e.width  = w;
  e.height = h;
  e.depth  = d;

  return e;
}

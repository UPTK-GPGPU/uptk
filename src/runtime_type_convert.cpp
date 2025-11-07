#include "runtime.hpp"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

enum UPTKError hipErrorToUPTKError(hipError_t para) {
    switch (para) {
//         case hipErrorAddressOfConstant:
//             return UPTKErrorAddressOfConstant;
        case hipErrorAlreadyAcquired:
            return UPTKErrorAlreadyAcquired;
        case hipErrorAlreadyMapped:
            return UPTKErrorAlreadyMapped;
//         case hipErrorApiFailureBase:
//             return UPTKErrorApiFailureBase;
        case hipErrorArrayIsMapped:
            return UPTKErrorArrayIsMapped;
        case hipErrorAssert:
            return UPTKErrorAssert;
        case hipErrorCapturedEvent:
            return UPTKErrorCapturedEvent;
//         case hipErrorCompatNotSupportedOnDevice:
//             return UPTKErrorCompatNotSupportedOnDevice;
        case hipErrorContextIsDestroyed:
            return UPTKErrorContextIsDestroyed;
        case hipErrorCooperativeLaunchTooLarge:
            return UPTKErrorCooperativeLaunchTooLarge;
        case hipErrorDeinitialized:
            return UPTKErrorUPTKrtUnloading;
        case hipErrorContextAlreadyInUse:
            return UPTKErrorDeviceAlreadyInUse;
        case hipErrorInvalidContext:
            return UPTKErrorDeviceUninitialized;
//         case hipErrorDevicesUnavailable:
//             return UPTKErrorDevicesUnavailable;
//         case hipErrorDuplicateSurfaceName:
//             return UPTKErrorDuplicateSurfaceName;
//         case hipErrorDuplicateTextureName:
//             return UPTKErrorDuplicateTextureName;
//         case hipErrorDuplicateVariableName:
//             return UPTKErrorDuplicateVariableName;
        case hipErrorECCNotCorrectable:
            return UPTKErrorECCUncorrectable;
        case hipErrorFileNotFound:
            return UPTKErrorFileNotFound;
        case hipErrorGraphExecUpdateFailure:
            return UPTKErrorGraphExecUpdateFailure;
//         case hipErrorHardwareStackError:
//             return UPTKErrorHardwareStackError;
        case hipErrorHostMemoryAlreadyRegistered:
            return UPTKErrorHostMemoryAlreadyRegistered;
        case hipErrorHostMemoryNotRegistered:
            return UPTKErrorHostMemoryNotRegistered;
        case hipErrorIllegalAddress:
            return UPTKErrorIllegalAddress;
//         case hipErrorIllegalInstruction:
//             return UPTKErrorIllegalInstruction;
        case hipErrorIllegalState:
            return UPTKErrorIllegalState;
//         case hipErrorIncompatibleDriverContext:
//             return UPTKErrorIncompatibleDriverContext;
        case hipErrorNotInitialized:
            return UPTKErrorInitializationError;
        case hipErrorInsufficientDriver:
            return UPTKErrorInsufficientDriver;
//         case hipErrorInvalidAddressSpace:
//             return UPTKErrorInvalidAddressSpace;
//         case hipErrorInvalidChannelDescriptor:
//             return UPTKErrorInvalidChannelDescriptor;
        case hipErrorInvalidConfiguration:
            return UPTKErrorInvalidConfiguration;
        case hipErrorInvalidDevice:
            return UPTKErrorInvalidDevice;
        case hipErrorInvalidDeviceFunction:
            return UPTKErrorInvalidDeviceFunction;
        case hipErrorInvalidDevicePointer:
            return UPTKErrorInvalidDevicePointer;
//         case hipErrorInvalidFilterSetting:
//             return UPTKErrorInvalidFilterSetting;
        case hipErrorInvalidGraphicsContext:
            return UPTKErrorInvalidGraphicsContext;
//         case hipErrorInvalidHostPointer:
//             return UPTKErrorInvalidHostPointer;
        case hipErrorInvalidImage:
            return UPTKErrorInvalidKernelImage;
        case hipErrorInvalidMemcpyDirection:
            return UPTKErrorInvalidMemcpyDirection;
//         case hipErrorInvalidNormSetting:
//             return UPTKErrorInvalidNormSetting;
//         case hipErrorInvalidPc:
//             return UPTKErrorInvalidPc;
        case hipErrorInvalidPitchValue:
            return UPTKErrorInvalidPitchValue;
        case hipErrorInvalidKernelFile:
            return UPTKErrorInvalidPtx;
        case hipErrorInvalidHandle:
            return UPTKErrorInvalidResourceHandle;
        case hipErrorInvalidSource:
            return UPTKErrorInvalidSource;
//         case hipErrorInvalidSurface:
//             return UPTKErrorInvalidSurface;
        case hipErrorInvalidSymbol:
            return UPTKErrorInvalidSymbol;
//         case hipErrorInvalidTexture:
//             return UPTKErrorInvalidTexture;
//         case hipErrorInvalidTextureBinding:
//             return UPTKErrorInvalidTextureBinding;
        case hipErrorInvalidValue:
            return UPTKErrorInvalidValue;
//         case hipErrorJitCompilerNotFound:
//             return UPTKErrorJitCompilerNotFound;
        case hipErrorLaunchFailure:
            return UPTKErrorLaunchFailure;
//         case hipErrorLaunchFileScopedSurf:
//             return UPTKErrorLaunchFileScopedSurf;
//         case hipErrorLaunchFileScopedTex:
//             return UPTKErrorLaunchFileScopedTex;
//         case hipErrorLaunchIncompatibleTexturing:
//             return UPTKErrorLaunchIncompatibleTexturing;
//         case hipErrorLaunchMaxDepthExceeded:
//             return UPTKErrorLaunchMaxDepthExceeded;
        case hipErrorLaunchOutOfResources:
            return UPTKErrorLaunchOutOfResources;
//         case hipErrorLaunchPendingCountExceeded:
//             return UPTKErrorLaunchPendingCountExceeded;
        case hipErrorLaunchTimeOut:
            return UPTKErrorLaunchTimeout;
        case hipErrorMapFailed:
            return UPTKErrorMapBufferObjectFailed;
        case hipErrorOutOfMemory:
            return UPTKErrorMemoryAllocation;
//         case hipErrorMemoryValueTooLarge:
//             return UPTKErrorMemoryValueTooLarge;
//         case hipErrorMisalignedAddress:
//             return UPTKErrorMisalignedAddress;
        case hipErrorMissingConfiguration:
            return UPTKErrorMissingConfiguration;
//         case hipErrorMixedDeviceExecution:
//             return UPTKErrorMixedDeviceExecution;
        case hipErrorNoDevice:
            return UPTKErrorNoDevice;
        case hipErrorNoBinaryForGpu:
            return UPTKErrorNoKernelImageForDevice;
        case hipErrorNotMapped:
            return UPTKErrorNotMapped;
        case hipErrorNotMappedAsArray:
            return UPTKErrorNotMappedAsArray;
        case hipErrorNotMappedAsPointer:
            return UPTKErrorNotMappedAsPointer;
//         case hipErrorNotPermitted:
//             return UPTKErrorNotPermitted;
        case hipErrorNotReady:
            return UPTKErrorNotReady;
        case hipErrorNotSupported:
            return UPTKErrorNotSupported;
//         case hipErrorNotYetImplemented:
//             return UPTKErrorNotYetImplemented;
//         case hipErrorNvlinkUncorrectable:
//             return UPTKErrorNvlinkUncorrectable;
        case hipErrorOperatingSystem:
            return UPTKErrorOperatingSystem;
        case hipErrorPeerAccessAlreadyEnabled:
            return UPTKErrorPeerAccessAlreadyEnabled;
        case hipErrorPeerAccessNotEnabled:
            return UPTKErrorPeerAccessNotEnabled;
        case hipErrorPeerAccessUnsupported:
            return UPTKErrorPeerAccessUnsupported;
        case hipErrorPriorLaunchFailure:
            return UPTKErrorPriorLaunchFailure;
        case hipErrorProfilerAlreadyStarted:
            return UPTKErrorProfilerAlreadyStarted;
        case hipErrorProfilerAlreadyStopped:
            return UPTKErrorProfilerAlreadyStopped;
        case hipErrorProfilerDisabled:
            return UPTKErrorProfilerDisabled;
        case hipErrorProfilerNotInitialized:
            return UPTKErrorProfilerNotInitialized;
        case hipErrorSetOnActiveProcess:
            return UPTKErrorSetOnActiveProcess;
        case hipErrorSharedObjectInitFailed:
            return UPTKErrorSharedObjectInitFailed;
        case hipErrorSharedObjectSymbolNotFound:
            return UPTKErrorSharedObjectSymbolNotFound;
//         case hipErrorStartupFailure:
//             return UPTKErrorStartupFailure;
        case hipErrorStreamCaptureImplicit:
            return UPTKErrorStreamCaptureImplicit;
        case hipErrorStreamCaptureInvalidated:
            return UPTKErrorStreamCaptureInvalidated;
        case hipErrorStreamCaptureIsolation:
            return UPTKErrorStreamCaptureIsolation;
        case hipErrorStreamCaptureMerge:
            return UPTKErrorStreamCaptureMerge;
        case hipErrorStreamCaptureUnjoined:
            return UPTKErrorStreamCaptureUnjoined;
        case hipErrorStreamCaptureUnmatched:
            return UPTKErrorStreamCaptureUnmatched;
        case hipErrorStreamCaptureUnsupported:
            return UPTKErrorStreamCaptureUnsupported;
        case hipErrorStreamCaptureWrongThread:
            return UPTKErrorStreamCaptureWrongThread;
        case hipErrorNotFound:
            return UPTKErrorSymbolNotFound;
//         case hipErrorSyncDepthExceeded:
//             return UPTKErrorSyncDepthExceeded;
//         case hipErrorSynchronizationError:
//             return UPTKErrorSynchronizationError;
//         case hipErrorSystemDriverMismatch:
//             return UPTKErrorSystemDriverMismatch;
//         case hipErrorSystemNotReady:
//             return UPTKErrorSystemNotReady;
//         case hipErrorTextureFetchFailed:
//             return UPTKErrorTextureFetchFailed;
//         case hipErrorTextureNotBound:
//             return UPTKErrorTextureNotBound;
//         case hipErrorTimeout:
//             return UPTKErrorTimeout;
//         case hipErrorTooManyPeers:
//             return UPTKErrorTooManyPeers;
        case hipErrorUnknown:
        case hipErrorRuntimeOther:
        case hipErrorTbd:
            return UPTKErrorUnknown;
        case hipErrorUnmapFailed:
            return UPTKErrorUnmapBufferObjectFailed;
        case hipErrorUnsupportedLimit:
            return UPTKErrorUnsupportedLimit;
        // case hipErrorInvalidResourcetype:
        //     return UPTKErrorInvalidResourceType;    
        case hipSuccess:
            return UPTKSuccess;
        default:
            ERROR_INVALID_ENUM();
    }
}

hipMemcpyKind UPTKMemcpyKindTohipMemcpyKind(enum UPTKMemcpyKind para) {
    switch (para) {
        case UPTKMemcpyDefault:
            return hipMemcpyDefault;
        case UPTKMemcpyDeviceToDevice:
            return hipMemcpyDeviceToDevice;
        case UPTKMemcpyDeviceToHost:
            return hipMemcpyDeviceToHost;
        case UPTKMemcpyHostToDevice:
            return hipMemcpyHostToDevice;
        case UPTKMemcpyHostToHost:
            return hipMemcpyHostToHost;
        default:
            ERROR_INVALID_ENUM();
    }
}

hipStreamCaptureMode UPTKStreamCaptureModeTohipStreamCaptureMode(enum UPTKStreamCaptureMode para) {
    switch (para) {
        case UPTKStreamCaptureModeGlobal:
            return hipStreamCaptureModeGlobal;
        case UPTKStreamCaptureModeRelaxed:
            return hipStreamCaptureModeRelaxed;
        case UPTKStreamCaptureModeThreadLocal:
            return hipStreamCaptureModeThreadLocal;
        default:
            ERROR_INVALID_ENUM();
    }
}


enum UPTKStreamCaptureStatus hipStreamCaptureStatusToUPTKStreamCaptureStatus(hipStreamCaptureStatus para) {
    switch (para) {
        case hipStreamCaptureStatusActive:
            return UPTKStreamCaptureStatusActive;
        case hipStreamCaptureStatusInvalidated:
            return UPTKStreamCaptureStatusInvalidated;
        case hipStreamCaptureStatusNone:
            return UPTKStreamCaptureStatusNone;
        default:
            ERROR_INVALID_ENUM();
    }
}

void UPTKIpcMemHandleTohipIpcMemHandle(const UPTKIpcMemHandle_t * UPTK_para, hipIpcMemHandle_t * hip_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    int len = min(UPTK_IPC_HANDLE_SIZE, HIP_IPC_HANDLE_SIZE);
    for (int i = 0; i < len; ++i) {
        hip_para->reserved[i] = UPTK_para->reserved[i];
    }
}

void UPTKDevicePropTohipDeviceProp(const struct UPTKDeviceProp * UPTK_para, hipDeviceProp_t * hip_para) {
    if (nullptr == UPTK_para || nullptr == hip_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    strncpy(hip_para->name, UPTK_para->name, 256);
    hip_para->totalGlobalMem = UPTK_para->totalGlobalMem;
    hip_para->sharedMemPerBlock = UPTK_para->sharedMemPerBlock;
    hip_para->regsPerBlock = UPTK_para->regsPerBlock;
    hip_para->warpSize = UPTK_para->warpSize;
    hip_para->maxThreadsPerBlock = UPTK_para->maxThreadsPerBlock;
    for (int i = 0; i < 3; i++) {
        hip_para->maxThreadsDim[i] = UPTK_para->maxThreadsDim[i];
        hip_para->maxGridSize[i] = UPTK_para->maxGridSize[i];
    }
    hip_para->clockRate = UPTK_para->clockRate;
    hip_para->memoryClockRate = UPTK_para->memoryClockRate;
    hip_para->memoryBusWidth = UPTK_para->memoryBusWidth;
    hip_para->totalConstMem = UPTK_para->totalConstMem;
    hip_para->major = UPTK_para->major;
    hip_para->minor = UPTK_para->minor;
    hip_para->multiProcessorCount = UPTK_para->multiProcessorCount;
    hip_para->l2CacheSize = UPTK_para->l2CacheSize;
    hip_para->maxThreadsPerMultiProcessor = UPTK_para->maxThreadsPerMultiProcessor;
    hip_para->computeMode = UPTK_para->computeMode;
    hip_para->clockInstructionRate = UPTK_para->clockRate; // Same as clock-rate:

    int ccVers = hip_para->major * 100 + hip_para->minor * 10;
    hip_para->arch.hasGlobalInt32Atomics = (ccVers >= 110);
    hip_para->arch.hasGlobalFloatAtomicExch = (ccVers >= 110);
    hip_para->arch.hasSharedInt32Atomics = (ccVers >= 120);
    hip_para->arch.hasSharedFloatAtomicExch = (ccVers >= 120);
    hip_para->arch.hasFloatAtomicAdd = (ccVers >= 200);
    hip_para->arch.hasGlobalInt64Atomics = (ccVers >= 120);
    hip_para->arch.hasSharedInt64Atomics = (ccVers >= 110);
    hip_para->arch.hasDoubles = (ccVers >= 130);
    hip_para->arch.hasWarpVote = (ccVers >= 120);
    hip_para->arch.hasWarpBallot = (ccVers >= 200);
    hip_para->arch.hasWarpShuffle = (ccVers >= 300);
    hip_para->arch.hasFunnelShift = (ccVers >= 350);
    hip_para->arch.hasThreadFenceSystem = (ccVers >= 200);
    hip_para->arch.hasSyncThreadsExt = (ccVers >= 200);
    hip_para->arch.hasSurfaceFuncs = (ccVers >= 200);
    hip_para->arch.has3dGrid = (ccVers >= 200);
    hip_para->arch.hasDynamicParallelism = (ccVers >= 350);

    hip_para->concurrentKernels = UPTK_para->concurrentKernels;
    hip_para->pciDomainID = UPTK_para->pciDomainID;
    hip_para->pciBusID = UPTK_para->pciBusID;
    hip_para->pciDeviceID = UPTK_para->pciDeviceID;
    hip_para->maxSharedMemoryPerMultiProcessor = UPTK_para->sharedMemPerMultiprocessor;
    hip_para->isMultiGpuBoard = UPTK_para->isMultiGpuBoard;
    hip_para->canMapHostMemory = UPTK_para->canMapHostMemory;
    hip_para->gcnArch = 0; // Not a GCN arch
    hip_para->integrated = UPTK_para->integrated;
    hip_para->cooperativeLaunch = UPTK_para->cooperativeLaunch;
    hip_para->cooperativeMultiDeviceLaunch = UPTK_para->cooperativeMultiDeviceLaunch;
    hip_para->cooperativeMultiDeviceUnmatchedFunc = 0;
    hip_para->cooperativeMultiDeviceUnmatchedGridDim = 0;
    hip_para->cooperativeMultiDeviceUnmatchedBlockDim = 0;
    hip_para->cooperativeMultiDeviceUnmatchedSharedMem = 0;

    hip_para->maxTexture1D    = UPTK_para->maxTexture1D;
    hip_para->maxTexture2D[0] = UPTK_para->maxTexture2D[0];
    hip_para->maxTexture2D[1] = UPTK_para->maxTexture2D[1];
    hip_para->maxTexture3D[0] = UPTK_para->maxTexture3D[0];
    hip_para->maxTexture3D[1] = UPTK_para->maxTexture3D[1];
    hip_para->maxTexture3D[2] = UPTK_para->maxTexture3D[2];

    hip_para->memPitch                 = UPTK_para->memPitch;
    hip_para->textureAlignment         = UPTK_para->textureAlignment;
    hip_para->texturePitchAlignment    = UPTK_para->texturePitchAlignment;
    hip_para->kernelExecTimeoutEnabled = UPTK_para->kernelExecTimeoutEnabled;
    hip_para->ECCEnabled               = UPTK_para->ECCEnabled;
    hip_para->tccDriver                = UPTK_para->tccDriver;
}

void hipDeviceProp_v2ToUPTKDeviceProp(const hipDeviceProp_t_v2 * hip_para, struct UPTKDeviceProp * UPTK_para) {
    if (nullptr == UPTK_para || nullptr == hip_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    // Add the gcnArchName of hip to the device name
    strncpy(UPTK_para->name, hip_para->name, sizeof(UPTK_para->name) - 1);
    UPTK_para->name[sizeof(UPTK_para->name) - 1] = '\0'; 
    if (strlen(UPTK_para->name) + strlen(hip_para->gcnArchName) < sizeof(UPTK_para->name) - 1){
        strncat(UPTK_para->name, " ", sizeof(UPTK_para->name) - strlen(UPTK_para->name) - 1); 
        strncat(UPTK_para->name, hip_para->gcnArchName, sizeof(UPTK_para->name) - strlen(UPTK_para->name) - 1); 
    }
    else {
        fprintf(stderr, "Error: UPTK_para->name array is not large enough to hold the result.\n");
    }
    UPTK_para->totalGlobalMem = hip_para->totalGlobalMem;
    UPTK_para->sharedMemPerBlock = hip_para->sharedMemPerBlock;
    UPTK_para->regsPerBlock = hip_para->regsPerBlock;
    UPTK_para->warpSize = hip_para->warpSize;
    UPTK_para->maxThreadsPerBlock = hip_para->maxThreadsPerBlock;
    for (int i = 0; i < 3; i++) {
        UPTK_para->maxThreadsDim[i] = hip_para->maxThreadsDim[i];
        UPTK_para->maxGridSize[i] = hip_para->maxGridSize[i];
    }
    UPTK_para->clockRate = hip_para->clockRate;
    UPTK_para->memoryClockRate = hip_para->memoryClockRate;
    UPTK_para->memoryBusWidth = hip_para->memoryBusWidth;
    UPTK_para->totalConstMem = hip_para->totalConstMem;
    UPTK_para->major = SM_VERSION_MAJOR;   // not equal to hip
    UPTK_para->minor = SM_VERSION_MINOR;   // not equal to hip
    UPTK_para->multiProcessorCount = hip_para->multiProcessorCount;
    UPTK_para->l2CacheSize = hip_para->l2CacheSize;
    UPTK_para->maxThreadsPerMultiProcessor = hip_para->maxThreadsPerMultiProcessor;
    UPTK_para->computeMode = hip_para->computeMode;

    UPTK_para->concurrentKernels = hip_para->concurrentKernels;
    UPTK_para->pciDomainID = hip_para->pciDomainID;
    UPTK_para->pciBusID = hip_para->pciBusID;
    UPTK_para->pciDeviceID = hip_para->pciDeviceID;
    
    // Only sharedMemPerMultiprocessor UPTKDeviceProp members
    // Two members of the hipDeviceProp_t_v2 are sharedMemPerMultiprocessor, maxSharedMemoryPerMultiProcessor
    // In the verification to the environment in P100 NV, UPTK sharedMemPerMultiprocessor maxSharedMemoryPerMultiProcessor should correspond to hip
    UPTK_para->sharedMemPerMultiprocessor = hip_para->maxSharedMemoryPerMultiProcessor;

    UPTK_para->isMultiGpuBoard = hip_para->isMultiGpuBoard;
    UPTK_para->canMapHostMemory = hip_para->canMapHostMemory;
    UPTK_para->integrated = hip_para->integrated;
    UPTK_para->cooperativeLaunch = hip_para->cooperativeLaunch;
    UPTK_para->cooperativeMultiDeviceLaunch = hip_para->cooperativeMultiDeviceLaunch;

    UPTK_para->maxTexture1D    = hip_para->maxTexture1D;
    UPTK_para->maxTexture2D[0] = hip_para->maxTexture2D[0];
    UPTK_para->maxTexture2D[1] = hip_para->maxTexture2D[1];
    UPTK_para->maxTexture3D[0] = hip_para->maxTexture3D[0];
    UPTK_para->maxTexture3D[1] = hip_para->maxTexture3D[1];
    UPTK_para->maxTexture3D[2] = hip_para->maxTexture3D[2];

    UPTK_para->memPitch                 = hip_para->memPitch;
    UPTK_para->textureAlignment         = hip_para->textureAlignment;
    UPTK_para->texturePitchAlignment    = hip_para->texturePitchAlignment;
    UPTK_para->kernelExecTimeoutEnabled = hip_para->kernelExecTimeoutEnabled;
    UPTK_para->ECCEnabled               = hip_para->ECCEnabled;
    UPTK_para->tccDriver                = hip_para->tccDriver;

    UPTK_para->regsPerMultiprocessor = hip_para->regsPerMultiprocessor;
    UPTK_para->maxBlocksPerMultiProcessor = hip_para->maxBlocksPerMultiProcessor;

    memcpy(&(UPTK_para->uuid), &(hip_para->uuid), sizeof(UPTKUUID_t));
    for(int i = 0; i < 8; i++){
        UPTK_para->luid[i] = hip_para->luid[i];
    }
    UPTK_para->luidDeviceNodeMask = hip_para->luidDeviceNodeMask;
    UPTK_para->deviceOverlap = hip_para->deviceOverlap;
    UPTK_para->maxTexture1DMipmap = hip_para->maxTexture1DMipmap;
    UPTK_para->maxTexture1DLinear = hip_para->maxTexture1DLinear;
    
    for(int i = 0; i < 2; i++){
    UPTK_para->maxTexture2DMipmap[i] = hip_para->maxTexture2DMipmap[i];
    UPTK_para->maxTexture2DGather[i] = hip_para->maxTexture2DGather[i];
    UPTK_para->maxTexture1DLayered[i] = hip_para->maxTexture1DLayered[i];
    UPTK_para->maxTextureCubemapLayered[i] = hip_para->maxTextureCubemapLayered[i];
    UPTK_para->maxSurface2D[i] = hip_para->maxSurface2D[i];
    UPTK_para->maxSurface1DLayered[i] = hip_para->maxSurface1DLayered[i];
    UPTK_para->maxSurfaceCubemapLayered[i] = hip_para->maxSurfaceCubemapLayered[i];
        
    }

    UPTK_para->maxSurface1D = hip_para->maxSurface1D;

    for(int i = 0; i < 3; i++){
    UPTK_para->maxTexture2DLinear[i] = hip_para->maxTexture2DLinear[i];
    UPTK_para->maxTexture3DAlt[i] = hip_para->maxTexture3DAlt[i];
    UPTK_para->maxTextureCubemap = hip_para->maxTextureCubemap;
    UPTK_para->maxTexture2DLayered[i] = hip_para->maxTexture2DLayered[i];
    UPTK_para->maxSurface3D[i] = hip_para->maxSurface3D[i];
    UPTK_para->maxSurface2DLayered[i] = hip_para->maxSurface2DLayered[i];
    }

    UPTK_para->maxSurfaceCubemap = hip_para->maxSurfaceCubemap;
    UPTK_para->surfaceAlignment = hip_para->surfaceAlignment;
    UPTK_para->asyncEngineCount = hip_para->asyncEngineCount;
    // UPTK_para->unifiedAddressing = hip_para->unifiedAddressing;
    //  UPTK sample: simpleIPC determines whether the device supports this attribute in the code. 
    //  The underlying hardware does not support unifiedAddressing, but setting unifiedAddressing to 1 does not affect the sample
    UPTK_para->unifiedAddressing = 1;
    UPTK_para->persistingL2CacheMaxSize = hip_para->persistingL2CacheMaxSize;
    UPTK_para->streamPrioritiesSupported = hip_para->streamPrioritiesSupported;
    UPTK_para->globalL1CacheSupported = hip_para->globalL1CacheSupported;
    UPTK_para->localL1CacheSupported = hip_para->localL1CacheSupported;
    UPTK_para->managedMemory = hip_para->managedMemory;
    UPTK_para->multiGpuBoardGroupID = hip_para->multiGpuBoardGroupID;
    UPTK_para->hostNativeAtomicSupported = hip_para->hostNativeAtomicSupported;
    UPTK_para->singleToDoublePrecisionPerfRatio = hip_para->singleToDoublePrecisionPerfRatio;
    UPTK_para->pageableMemoryAccess = hip_para->pageableMemoryAccess;
    UPTK_para->concurrentManagedAccess = hip_para->concurrentManagedAccess;
    UPTK_para->computePreemptionSupported = hip_para->computePreemptionSupported;
    UPTK_para->canUseHostPointerForRegisteredMem = hip_para->canUseHostPointerForRegisteredMem;
    // UPTK_para->sharedMemPerBlockOptin = hip_para->sharedMemPerBlockOptin;
    // hip sharedMemPerBlockOptin member returns 0, the actual function is not supported. Replace with the close attribute sharedMemPerBlock.
    UPTK_para->sharedMemPerBlockOptin = hip_para->sharedMemPerBlock;
    UPTK_para->pageableMemoryAccessUsesHostPageTables = hip_para->pageableMemoryAccessUsesHostPageTables;
    UPTK_para->directManagedMemAccessFromHost = hip_para->directManagedMemAccessFromHost;
    UPTK_para->accessPolicyMaxWindowSize = hip_para->accessPolicyMaxWindowSize;
    UPTK_para->reservedSharedMemPerBlock = hip_para->reservedSharedMemPerBlock;
    // UPTK 12.6.2:added some members to the UPTKDeviceProp structure.
    UPTK_para->hostRegisterSupported = hip_para->hostRegisterSupported;
    UPTK_para->sparseUPTKArraySupported = hip_para->sparseHipArraySupported;
    UPTK_para->hostRegisterReadOnlySupported = hip_para->hostRegisterReadOnlySupported;
    UPTK_para->timelineSemaphoreInteropSupported = hip_para->timelineSemaphoreInteropSupported;
    UPTK_para->memoryPoolsSupported = hip_para->memoryPoolsSupported;
    UPTK_para->gpuDirectRDMASupported = hip_para->gpuDirectRDMASupported;
    UPTK_para->gpuDirectRDMAFlushWritesOptions = hip_para->gpuDirectRDMAFlushWritesOptions;
    UPTK_para->gpuDirectRDMAWritesOrdering = hip_para->gpuDirectRDMAWritesOrdering;
    UPTK_para->memoryPoolSupportedHandleTypes = hip_para->memoryPoolSupportedHandleTypes;
    UPTK_para->deferredMappingUPTKArraySupported = hip_para->deferredMappingHipArraySupported;
    UPTK_para->ipcEventSupported = hip_para->ipcEventSupported;
    UPTK_para->clusterLaunch = hip_para->clusterLaunch;
    UPTK_para->unifiedFunctionPointers = hip_para->unifiedFunctionPointers;
    memcpy(UPTK_para->reserved2, hip_para->reserved, 2 * sizeof(int));       
    memcpy(UPTK_para->reserved1, hip_para->reserved + 2, 1 * sizeof(int));           
    memcpy(UPTK_para->reserved, hip_para->reserved + 3, 60 * sizeof(int));         
}

enum UPTKFuncCache hipFuncCacheToUPTKFuncCache(hipFuncCache_t para) {
    switch (para) {
        case hipFuncCachePreferEqual:
            return UPTKFuncCachePreferEqual;
        case hipFuncCachePreferL1:
            return UPTKFuncCachePreferL1;
        case hipFuncCachePreferNone:
            return UPTKFuncCachePreferNone;
        case hipFuncCachePreferShared:
            return UPTKFuncCachePreferShared;
        default:
            ERROR_INVALID_ENUM();
    }
}


enum hipLimit_t UPTKlimitTohipLimit(UPTKlimit para) {
    switch (para) {
        case UPTK_LIMIT_MAX:
            return hipLimitRange;
        case UPTK_LIMIT_MALLOC_HEAP_SIZE:
            return hipLimitMallocHeapSize;
        case UPTK_LIMIT_PRINTF_FIFO_SIZE:
            return hipLimitPrintfFifoSize;
        case UPTK_LIMIT_STACK_SIZE:
            return hipLimitStackSize;
        default:
            ERROR_INVALID_ENUM();
    }
}

void UPTKKernelNodeParamsTohipKernelNodeParams(const struct UPTKKernelNodeParams * UPTK_para, hipKernelNodeParams * hip_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    hip_para->blockDim = UPTK_para->blockDim;
    hip_para->extra = UPTK_para->extra;
    hip_para->func = UPTK_para->func;
    hip_para->gridDim = UPTK_para->gridDim;
    hip_para->kernelParams = UPTK_para->kernelParams;
    hip_para->sharedMemBytes = UPTK_para->sharedMemBytes;
}


void UPTKMemcpy3DParmsTohipMemcpy3DParms(const struct UPTKMemcpy3DParms * UPTK_para, hipMemcpy3DParms * hip_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    hip_para->dstArray = (hipArray_t)UPTK_para->dstArray;
    UPTKPosTohipPos(&(UPTK_para->dstPos), &(hip_para->dstPos));
    UPTKPitchedPtrTohipPitchedPtr(&(UPTK_para->dstPtr), &(hip_para->dstPtr));
    UPTKExtentTohipExtent(&(UPTK_para->extent), &(hip_para->extent));
    hip_para->kind = UPTKMemcpyKindTohipMemcpyKind(UPTK_para->kind);
    hip_para->srcArray = (hipArray_t)UPTK_para->srcArray;
    UPTKPosTohipPos(&(UPTK_para->srcPos), &(hip_para->srcPos));
    UPTKPitchedPtrTohipPitchedPtr(&(UPTK_para->srcPtr), &(hip_para->srcPtr));
}


void UPTKPosTohipPos(const struct UPTKPos * UPTK_para, hipPos * hip_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    hip_para->x = UPTK_para->x;
    hip_para->y = UPTK_para->y;
    hip_para->z = UPTK_para->z;
}


void UPTKPitchedPtrTohipPitchedPtr(const struct UPTKPitchedPtr * UPTK_para, hipPitchedPtr * hip_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    hip_para->pitch = UPTK_para->pitch;
    hip_para->ptr = UPTK_para->ptr;
    hip_para->xsize = UPTK_para->xsize;
    hip_para->ysize = UPTK_para->ysize;
}

void UPTKExtentTohipExtent(const struct UPTKExtent * UPTK_para, hipExtent * hip_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    hip_para->depth = UPTK_para->depth;
    hip_para->height = UPTK_para->height;
    hip_para->width = UPTK_para->width;
}

enum UPTKGraphExecUpdateResult hipGraphExecUpdateResultToUPTKGraphExecUpdateResult(hipGraphExecUpdateResult para) {
    switch (para) {
        case hipGraphExecUpdateError:
            return UPTKGraphExecUpdateError;
        case hipGraphExecUpdateErrorFunctionChanged:
            return UPTKGraphExecUpdateErrorFunctionChanged;
        case hipGraphExecUpdateErrorNodeTypeChanged:
            return UPTKGraphExecUpdateErrorNodeTypeChanged;
        case hipGraphExecUpdateErrorNotSupported:
            return UPTKGraphExecUpdateErrorNotSupported;
        case hipGraphExecUpdateErrorParametersChanged:
            return UPTKGraphExecUpdateErrorParametersChanged;
        case hipGraphExecUpdateErrorTopologyChanged:
            return UPTKGraphExecUpdateErrorTopologyChanged;
        case hipGraphExecUpdateErrorUnsupportedFunctionChange:
            return UPTKGraphExecUpdateErrorUnsupportedFunctionChange;
        case hipGraphExecUpdateSuccess:
            return UPTKGraphExecUpdateSuccess;
        default:
            ERROR_INVALID_ENUM();
    }
}

void UPTKLaunchConfig_tTohipLaunchConfig_t(const UPTKLaunchConfig_t * UPTK_para, hipLaunchConfig_t * hip_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    hip_para->gridDim = UPTK_para->gridDim;
    hip_para->blockDim = UPTK_para->blockDim;
    hip_para->dynamicSmemBytes = UPTK_para->dynamicSmemBytes;
    hip_para->stream = (hipStream_t)UPTK_para->stream;

    hip_para->numAttrs = UPTK_para->numAttrs;
    hipLaunchAttribute* hip_attrs = (hipLaunchAttribute*)malloc(sizeof(hipLaunchAttribute) * hip_para->numAttrs);
    // attrs is a nullable pointer. if numAttrs == 0, attrs can be nullptr.
    if (UPTK_para->attrs != nullptr)
    {
        for (int i = 0; i < hip_para->numAttrs; i++)
        {
            hip_attrs[i] = UPTKLaunchAttributeTohipLaunchAttribute(UPTK_para->attrs[i]);
        }
    }
    hip_para->attrs = hip_attrs;
}

hipClusterSchedulingPolicy UPTKClusterSchedulingPolicyTohipClusterSchedulingPolicy(UPTKClusterSchedulingPolicy para)
{
    switch(para){
        case UPTKClusterSchedulingPolicyDefault:
            return hipClusterSchedulingPolicyDefault;
        case UPTKClusterSchedulingPolicySpread:
            return hipClusterSchedulingPolicySpread;
        case UPTKClusterSchedulingPolicyLoadBalancing:
            return hipClusterSchedulingPolicyLoadBalancing;
        default:
            ERROR_INVALID_ENUM();
    }
}

enum hipSynchronizationPolicy UPTKSynchronizationPolicyTohipSynchronizationPolicy(enum UPTKSynchronizationPolicy para) {
    switch (para) {
        case UPTKSyncPolicyAuto:
            return hipSyncPolicyAuto;
        case UPTKSyncPolicyBlockingSync:
            return hipSyncPolicyBlockingSync;
        case UPTKSyncPolicySpin:
            return hipSyncPolicySpin;
        case UPTKSyncPolicyYield:
            return hipSyncPolicyYield;
        default:
            ERROR_INVALID_ENUM();
    }
}

hipAccessProperty UPTKAccessPropertyTohipAccessProperty(enum UPTKAccessProperty para) {
    switch (para) {
        case UPTKAccessPropertyNormal:
            return hipAccessPropertyNormal;
        case UPTKAccessPropertyPersisting:
            return hipAccessPropertyPersisting;
        case UPTKAccessPropertyStreaming:
            return hipAccessPropertyStreaming;
        default:
              ERROR_INVALID_ENUM();
    }
}

void UPTKAccessPolicyWindowTohipAccessPolicyWindow(const struct UPTKAccessPolicyWindow * UPTK_para, hipAccessPolicyWindow * hip_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    hip_para->base_ptr = UPTK_para->base_ptr;
    hip_para->hitProp = UPTKAccessPropertyTohipAccessProperty(UPTK_para->hitProp);
    hip_para->hitRatio = UPTK_para->hitRatio;
    hip_para->missProp = UPTKAccessPropertyTohipAccessProperty(UPTK_para->missProp);
    hip_para->num_bytes = UPTK_para->num_bytes;
}

void UPTKLaunchAttributeValueTohipLaunchAttributeValue(const UPTKLaunchAttributeValue *UPTK_para, hipLaunchAttributeValue *hip_para, UPTKLaunchAttributeID para)
{
    if (nullptr == UPTK_para || nullptr == hip_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    switch (para)
    {
    case UPTKLaunchAttributeCooperative: // hip underlying hardware support
        hip_para->cooperative = UPTK_para->cooperative;
        break;
    case UPTKLaunchAttributeAccessPolicyWindow: // hip underlying hardware support
        UPTKAccessPolicyWindowTohipAccessPolicyWindow(&(UPTK_para->accessPolicyWindow), &(hip_para->accessPolicyWindow));
        break;
    case UPTKLaunchAttributeIgnore:
        memcpy(hip_para->pad, UPTK_para->pad, sizeof(hip_para->pad));
        break;
    case UPTKLaunchAttributeSynchronizationPolicy:
        hip_para->syncPolicy = UPTKSynchronizationPolicyTohipSynchronizationPolicy(UPTK_para->syncPolicy);
        break;
    case UPTKLaunchAttributeClusterDimension:
        memcpy(&(hip_para->clusterDim), &(UPTK_para->clusterDim), sizeof(hip_para->clusterDim));
        break;
    case UPTKLaunchAttributeClusterSchedulingPolicyPreference:
        hip_para->clusterSchedulingPolicyPreference = UPTKClusterSchedulingPolicyTohipClusterSchedulingPolicy(UPTK_para->clusterSchedulingPolicyPreference);
        break;
    case UPTKLaunchAttributeProgrammaticStreamSerialization:
        hip_para->programmaticStreamSerializationAllowed = UPTK_para->programmaticStreamSerializationAllowed;
        break;
    case UPTKLaunchAttributeProgrammaticEvent:
        hip_para->programmaticEvent.event = (hipEvent_t)UPTK_para->programmaticEvent.event;
        hip_para->programmaticEvent.flags = UPTK_para->programmaticEvent.flags;
        hip_para->programmaticEvent.triggerAtBlockStart = UPTK_para->programmaticEvent.triggerAtBlockStart;
        break;
    case UPTKLaunchAttributePriority:
        hip_para->priority = UPTK_para->priority;
        break;
    default:
        ERROR_INVALID_ENUM();
        break;
    }
}

// TODO: UPTKLaunchAttribute generally appears in the form of an array, so special conversion is performed here.
hipLaunchAttribute UPTKLaunchAttributeTohipLaunchAttribute(UPTKLaunchAttribute UPTK_para)
{
    // if (nullptr == UPTK_para || nullptr == hip_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    hipLaunchAttribute hip_para;
    hip_para.id = UPTKLaunchAttributeIDTohipLaunchAttributeID(UPTK_para.id);
    UPTKLaunchAttributeValueTohipLaunchAttributeValue(&(UPTK_para.val), &(hip_para.val), UPTK_para.id);
    // Note: Padding in UPTKLaunchAttribute is ignored during conversion
    return hip_para;
}

hipLaunchAttributeID UPTKLaunchAttributeIDTohipLaunchAttributeID(enum UPTKLaunchAttributeID para)
{
    switch(para) {
        case UPTKLaunchAttributeCooperative:
            return hipLaunchAttributeCooperative;                  // hip underlying hardware support
        case UPTKLaunchAttributeSynchronizationPolicy:
            return hipLaunchAttributeSynchronizationPolicy;        // hip underlying hardware support
        case UPTKLaunchAttributeIgnore:
            return hipLaunchAttributeIgnore;
        case UPTKLaunchAttributeAccessPolicyWindow:
            return hipLaunchAttributeAccessPolicyWindow;      
        case UPTKLaunchAttributeClusterDimension:
            return hipLaunchAttributeClusterDimension;
        case UPTKLaunchAttributeClusterSchedulingPolicyPreference:
            return hipLaunchAttributeClusterSchedulingPolicyPreference;
        case UPTKLaunchAttributeProgrammaticStreamSerialization:
            return hipLaunchAttributeProgrammaticStreamSerialization;
        case UPTKLaunchAttributeProgrammaticEvent:
            return hipLaunchAttributeProgrammaticEvent;
        case UPTKLaunchAttributePriority:
            return hipLaunchAttributePriority;
        default:
            ERROR_INVALID_ENUM();                 
    }
}

void hipFuncAttributesToUPTKFuncAttributes(const hipFuncAttributes * hip_para, struct UPTKFuncAttributes * UPTK_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    UPTK_para->binaryVersion = hip_para->binaryVersion;
    UPTK_para->cacheModeCA = hip_para->cacheModeCA;
    UPTK_para->constSizeBytes = hip_para->constSizeBytes;
    UPTK_para->localSizeBytes = hip_para->localSizeBytes;
    UPTK_para->maxDynamicSharedSizeBytes = hip_para->maxDynamicSharedSizeBytes;
    UPTK_para->maxThreadsPerBlock = hip_para->maxThreadsPerBlock;
    UPTK_para->numRegs = hip_para->numRegs;
    UPTK_para->preferredShmemCarveout = hip_para->preferredShmemCarveout;
    UPTK_para->ptxVersion = hip_para->ptxVersion;
    UPTK_para->sharedSizeBytes = hip_para->sharedSizeBytes;
}

UPTKresult hipErrorToUPTKresult(hipError_t para) {
    switch (para) {
        case hipErrorAlreadyAcquired:
            return UPTK_ERROR_ALREADY_ACQUIRED;
        case hipErrorAlreadyMapped:
            return UPTK_ERROR_ALREADY_MAPPED;
        case hipErrorArrayIsMapped:
            return UPTK_ERROR_ARRAY_IS_MAPPED;
        case hipErrorAssert:
            return UPTK_ERROR_ASSERT;
        case hipErrorCapturedEvent:
            return UPTK_ERROR_CAPTURED_EVENT;
//         case hipErrorCompatNotSupportedOnDevice:
//             return UPTK_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE;
        case hipErrorContextAlreadyCurrent:
            return UPTK_ERROR_CONTEXT_ALREADY_UPTKRRENT;
        case hipErrorContextAlreadyInUse:
            return UPTK_ERROR_CONTEXT_ALREADY_IN_USE;
        case hipErrorContextIsDestroyed:
            return UPTK_ERROR_CONTEXT_IS_DESTROYED;
        case hipErrorCooperativeLaunchTooLarge:
            return UPTK_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE;
        case hipErrorDeinitialized:
            return UPTK_ERROR_DEINITIALIZED;
        case hipErrorECCNotCorrectable:
            return UPTK_ERROR_ECC_UNCORRECTABLE;
        case hipErrorFileNotFound:
            return UPTK_ERROR_FILE_NOT_FOUND;
        case hipErrorGraphExecUpdateFailure:
            return UPTK_ERROR_GRAPH_EXEC_UPDATE_FAILURE;
//         case hipErrorHardwareStackError:
//             return UPTK_ERROR_HARDWARE_STACK_ERROR;
        case hipErrorHostMemoryAlreadyRegistered:
            return UPTK_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
        case hipErrorHostMemoryNotRegistered:
            return UPTK_ERROR_HOST_MEMORY_NOT_REGISTERED;
        case hipErrorIllegalAddress:
            return UPTK_ERROR_ILLEGAL_ADDRESS;
//         case hipErrorIllegalInstruction:
//             return UPTK_ERROR_ILLEGAL_INSTRUCTION;
        case hipErrorIllegalState:
            return UPTK_ERROR_ILLEGAL_STATE;
//         case hipErrorInvalidAddressSpace:
//             return UPTK_ERROR_INVALID_ADDRESS_SPACE;
        case hipErrorInvalidContext:
            return UPTK_ERROR_INVALID_CONTEXT;
        case hipErrorInvalidDevice:
            return UPTK_ERROR_INVALID_DEVICE;
        case hipErrorInvalidGraphicsContext:
            return UPTK_ERROR_INVALID_GRAPHICS_CONTEXT;
        case hipErrorInvalidHandle:
            return UPTK_ERROR_INVALID_HANDLE;
        case hipErrorInvalidImage:
            return UPTK_ERROR_INVALID_IMAGE;
//         case hipErrorInvalidPc:
//             return UPTK_ERROR_INVALID_PC;
        case hipErrorInvalidKernelFile:
            return UPTK_ERROR_INVALID_PTX;
        case hipErrorInvalidSource:
            return UPTK_ERROR_INVALID_SOURCE;
        case hipErrorInvalidValue:
            return UPTK_ERROR_INVALID_VALUE;
//         case hipErrorJitCompilerNotFound:
//             return UPTK_ERROR_JIT_COMPILER_NOT_FOUND;
        case hipErrorLaunchFailure:
            return UPTK_ERROR_LAUNCH_FAILED;
//         case hipErrorLaunchIncompatibleTexturing:
//             return UPTK_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING;
        case hipErrorLaunchOutOfResources:
            return UPTK_ERROR_LAUNCH_OUT_OF_RESOURCES;
        case hipErrorLaunchTimeOut:
            return UPTK_ERROR_LAUNCH_TIMEOUT;
        case hipErrorMapFailed:
            return UPTK_ERROR_MAP_FAILED;
//         case hipErrorMisalignedAddress:
//             return UPTK_ERROR_MISALIGNED_ADDRESS;
        case hipErrorNotFound:
            return UPTK_ERROR_NOT_FOUND;
        case hipErrorNotInitialized:
            return UPTK_ERROR_NOT_INITIALIZED;
        case hipErrorNotMapped:
            return UPTK_ERROR_NOT_MAPPED;
        case hipErrorNotMappedAsArray:
            return UPTK_ERROR_NOT_MAPPED_AS_ARRAY;
        case hipErrorNotMappedAsPointer:
            return UPTK_ERROR_NOT_MAPPED_AS_POINTER;
//         case hipErrorNotPermitted:
//             return UPTK_ERROR_NOT_PERMITTED;
        case hipErrorNotReady:
            return UPTK_ERROR_NOT_READY;
        case hipErrorNotSupported:
            return UPTK_ERROR_NOT_SUPPORTED;
        case hipErrorNoBinaryForGpu:
            return UPTK_ERROR_NO_BINARY_FOR_GPU;
        case hipErrorNoDevice:
            return UPTK_ERROR_NO_DEVICE;
//         case hipErrorNvlinkUncorrectable:
//             return UPTK_ERROR_NVLINK_UNCORRECTABLE;
        case hipErrorOperatingSystem:
            return UPTK_ERROR_OPERATING_SYSTEM;
        case hipErrorOutOfMemory:
            return UPTK_ERROR_OUT_OF_MEMORY;
        case hipErrorPeerAccessAlreadyEnabled:
            return UPTK_ERROR_PEER_ACCESS_ALREADY_ENABLED;
        case hipErrorPeerAccessNotEnabled:
            return UPTK_ERROR_PEER_ACCESS_NOT_ENABLED;
        case hipErrorPeerAccessUnsupported:
            return UPTK_ERROR_PEER_ACCESS_UNSUPPORTED;
        case hipErrorSetOnActiveProcess:
            return UPTK_ERROR_PRIMARY_CONTEXT_ACTIVE;
        case hipErrorProfilerAlreadyStarted:
            return UPTK_ERROR_PROFILER_ALREADY_STARTED;
        case hipErrorProfilerAlreadyStopped:
            return UPTK_ERROR_PROFILER_ALREADY_STOPPED;
        case hipErrorProfilerDisabled:
            return UPTK_ERROR_PROFILER_DISABLED;
        case hipErrorProfilerNotInitialized:
            return UPTK_ERROR_PROFILER_NOT_INITIALIZED;
        case hipErrorSharedObjectInitFailed:
            return UPTK_ERROR_SHARED_OBJECT_INIT_FAILED;
        case hipErrorSharedObjectSymbolNotFound:
            return UPTK_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
        case hipErrorStreamCaptureImplicit:
            return UPTK_ERROR_STREAM_CAPTURE_IMPLICIT;
        case hipErrorStreamCaptureInvalidated:
            return UPTK_ERROR_STREAM_CAPTURE_INVALIDATED;
        case hipErrorStreamCaptureIsolation:
            return UPTK_ERROR_STREAM_CAPTURE_ISOLATION;
        case hipErrorStreamCaptureMerge:
            return UPTK_ERROR_STREAM_CAPTURE_MERGE;
        case hipErrorStreamCaptureUnjoined:
            return UPTK_ERROR_STREAM_CAPTURE_UNJOINED;
        case hipErrorStreamCaptureUnmatched:
            return UPTK_ERROR_STREAM_CAPTURE_UNMATCHED;
        case hipErrorStreamCaptureUnsupported:
            return UPTK_ERROR_STREAM_CAPTURE_UNSUPPORTED;
        case hipErrorStreamCaptureWrongThread:
            return UPTK_ERROR_STREAM_CAPTURE_WRONG_THREAD;
//         case hipErrorSystemDriverMismatch:
//             return UPTK_ERROR_SYSTEM_DRIVER_MISMATCH;
//         case hipErrorSystemNotReady:
//             return UPTK_ERROR_SYSTEM_NOT_READY;
//         case hipErrorTimeout:
//             return UPTK_ERROR_TIMEOUT;
//         case hipErrorTooManyPeers:
//             return UPTK_ERROR_TOO_MANY_PEERS;
        case hipErrorUnknown:
            return UPTK_ERROR_UNKNOWN;
        case hipErrorUnmapFailed:
            return UPTK_ERROR_UNMAP_FAILED;
        case hipErrorUnsupportedLimit:
            return UPTK_ERROR_UNSUPPORTED_LIMIT;
        case hipErrorInvalidResourcetype:
            return UPTK_ERROR_INVALID_RESOURCE_TYPE;     
        case hipSuccess:
            return UPTK_SUCCESS;
        default:
            ERROR_INVALID_ENUM();
    }
}

hipArray_Format UPTKarray_formatTohipArray_Format(UPTKarray_format para) {
    switch (para) {
        case UPTK_AD_FORMAT_FLOAT:
            return HIP_AD_FORMAT_FLOAT;
        case UPTK_AD_FORMAT_HALF:
            return HIP_AD_FORMAT_HALF;
        case UPTK_AD_FORMAT_SIGNED_INT16:
            return HIP_AD_FORMAT_SIGNED_INT16;
        case UPTK_AD_FORMAT_SIGNED_INT32:
            return HIP_AD_FORMAT_SIGNED_INT32;
        case UPTK_AD_FORMAT_SIGNED_INT8:
            return HIP_AD_FORMAT_SIGNED_INT8;
        case UPTK_AD_FORMAT_UNSIGNED_INT16:
            return HIP_AD_FORMAT_UNSIGNED_INT16;
        case UPTK_AD_FORMAT_UNSIGNED_INT32:
            return HIP_AD_FORMAT_UNSIGNED_INT32;
        case UPTK_AD_FORMAT_UNSIGNED_INT8:
            return HIP_AD_FORMAT_UNSIGNED_INT8;
        default:
            ERROR_INVALID_ENUM();
    }
}

void UPTK_ARRAY_DESCRIPTORToHIP_ARRAY_DESCRIPTOR(const UPTK_ARRAY_DESCRIPTOR * UPTK_para, HIP_ARRAY_DESCRIPTOR * hip_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    hip_para->Format = UPTKarray_formatTohipArray_Format(UPTK_para->Format);
    hip_para->Height = UPTK_para->Height;
    hip_para->NumChannels = UPTK_para->NumChannels;
    hip_para->Width = UPTK_para->Width;
}

hiprtcJIT_option UPTKjit_optionTohiprtcJIT_option(UPTKjit_option para) {
    switch (para) {
        case UPTK_JIT_MAX_REGISTERS:
            return HIPRTC_JIT_MAX_REGISTERS;
        case UPTK_JIT_THREADS_PER_BLOCK:
            return HIPRTC_JIT_THREADS_PER_BLOCK;
        case UPTK_JIT_WALL_TIME:
            return HIPRTC_JIT_WALL_TIME;
        case UPTK_JIT_INFO_LOG_BUFFER:
            return HIPRTC_JIT_INFO_LOG_BUFFER;
        case UPTK_JIT_INFO_LOG_BUFFER_SIZE_BYTES:
            return HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        case UPTK_JIT_ERROR_LOG_BUFFER:
            return HIPRTC_JIT_ERROR_LOG_BUFFER;
        case UPTK_JIT_ERROR_LOG_BUFFER_SIZE_BYTES:
            return HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
        case UPTK_JIT_OPTIMIZATION_LEVEL:
            return HIPRTC_JIT_OPTIMIZATION_LEVEL;
        case UPTK_JIT_TARGET_FROM_UPTKCONTEXT:
            return HIPRTC_JIT_TARGET_FROM_HIPCONTEXT;
        case UPTK_JIT_TARGET:
            return HIPRTC_JIT_TARGET;
        case UPTK_JIT_FALLBACK_STRATEGY:
            return HIPRTC_JIT_FALLBACK_STRATEGY;
        case UPTK_JIT_GENERATE_DEBUG_INFO:
            return HIPRTC_JIT_GENERATE_DEBUG_INFO;
        case UPTK_JIT_LOG_VERBOSE:
            return HIPRTC_JIT_LOG_VERBOSE;
        case UPTK_JIT_GENERATE_LINE_INFO:
            return HIPRTC_JIT_GENERATE_LINE_INFO;
        case UPTK_JIT_CACHE_MODE:
            return HIPRTC_JIT_CACHE_MODE;;
        case UPTK_JIT_NEW_SM3X_OPT:
            return HIPRTC_JIT_NEW_SM3X_OPT;
        case UPTK_JIT_FAST_COMPILE:
            return HIPRTC_JIT_FAST_COMPILE;
        case UPTK_JIT_GLOBAL_SYMBOL_NAMES:
            return HIPRTC_JIT_GLOBAL_SYMBOL_NAMES;
        case UPTK_JIT_GLOBAL_SYMBOL_ADDRESSES:
            return HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS;
        case UPTK_JIT_GLOBAL_SYMBOL_COUNT:
            return HIPRTC_JIT_GLOBAL_SYMBOL_COUNT;
        case UPTK_JIT_LTO:
            return HIPRTC_JIT_LTO;
        case UPTK_JIT_FTZ:
            return HIPRTC_JIT_FTZ;
        case UPTK_JIT_PREC_DIV:
            return HIPRTC_JIT_PREC_DIV;
        case UPTK_JIT_PREC_SQRT:
            return HIPRTC_JIT_PREC_SQRT;
        case UPTK_JIT_FMA:
            return HIPRTC_JIT_FMA;
        // case UPTK_JIT_REFERENCED_KERNEL_NAMES:
        // case UPTK_JIT_REFERENCED_KERNEL_COUNT:
        // case UPTK_JIT_REFERENCED_VARIABLE_NAMES:
        // case UPTK_JIT_REFERENCED_VARIABLE_COUNT:
        // case UPTK_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES:
        case UPTK_JIT_NUM_OPTIONS:
            return HIPRTC_JIT_NUM_OPTIONS;
        default:
            ERROR_INVALID_ENUM();
    }
}

UPTKresult hiprtcResultToUPTKresult(hiprtcResult para) {
    switch (para) {
        case HIPRTC_SUCCESS:
            return UPTK_SUCCESS;
        case HIPRTC_ERROR_LINKING:
            return UPTK_ERROR_INVALID_HANDLE;
        case HIPRTC_ERROR_INVALID_INPUT:
            return UPTK_ERROR_INVALID_VALUE;
        case HIPRTC_ERROR_PROGRAM_CREATION_FAILURE:
            return UPTK_ERROR_INVALID_IMAGE;
        case HIPRTC_ERROR_INVALID_OPTION:
            return UPTK_ERROR_NOT_INITIALIZED;
        default:
            ERROR_INVALID_ENUM();
    }
}

hiprtcJITInputType UPTKjitInputTypeTohiprtcJITInputType(UPTKjitInputType para) {
    switch (para) {
        case UPTK_JIT_INPUT_UPTKBIN:
            return HIPRTC_JIT_INPUT_CUBIN;
        case UPTK_JIT_INPUT_PTX:
            return HIPRTC_JIT_INPUT_LLVM_BITCODE;
        case UPTK_JIT_INPUT_FATBINARY:
            return HIPRTC_JIT_INPUT_FATBINARY;
        case UPTK_JIT_INPUT_OBJECT:
            return HIPRTC_JIT_INPUT_OBJECT;
        case UPTK_JIT_INPUT_LIBRARY:
            return HIPRTC_JIT_INPUT_LIBRARY;
        case UPTK_JIT_INPUT_NVVM:
            return HIPRTC_JIT_INPUT_NVVM;
        case UPTK_JIT_NUM_INPUT_TYPES:
            return HIPRTC_JIT_NUM_INPUT_TYPES;
        default:
            ERROR_INVALID_ENUM();
    }

}

hipDeviceAttribute_t UPTKDeviceAttrTohipDeviceAttribute(enum UPTKDeviceAttr para) {
    switch (para) {
        case UPTKDevAttrAsyncEngineCount:
            return hipDeviceAttributeAsyncEngineCount;
//         case UPTKDevAttrCanFlushRemoteWrites:
//             return hipDeviceAttributeCanFlushRemoteWrites;
        case UPTKDevAttrCanMapHostMemory:
            return hipDeviceAttributeCanMapHostMemory;
        case UPTKDevAttrCanUseHostPointerForRegisteredMem:
            return hipDeviceAttributeCanUseHostPointerForRegisteredMem;
        case UPTKDevAttrClockRate:
            return hipDeviceAttributeClockRate;
        case UPTKDevAttrComputeCapabilityMajor:
            return hipDeviceAttributeComputeCapabilityMajor;
        case UPTKDevAttrComputeCapabilityMinor:
            return hipDeviceAttributeComputeCapabilityMinor;
        case UPTKDevAttrComputeMode:
            return hipDeviceAttributeComputeMode;
        case UPTKDevAttrComputePreemptionSupported:
            return hipDeviceAttributeComputePreemptionSupported;
        case UPTKDevAttrConcurrentKernels:
            return hipDeviceAttributeConcurrentKernels;
        case UPTKDevAttrConcurrentManagedAccess:
            return hipDeviceAttributeConcurrentManagedAccess;
        case UPTKDevAttrCooperativeLaunch:
            return hipDeviceAttributeCooperativeLaunch;
        case UPTKDevAttrCooperativeMultiDeviceLaunch:
            return hipDeviceAttributeCooperativeMultiDeviceLaunch;
        case UPTKDevAttrDirectManagedMemAccessFromHost:
            return hipDeviceAttributeDirectManagedMemAccessFromHost;
        case UPTKDevAttrEccEnabled:
            return hipDeviceAttributeEccEnabled;
        case UPTKDevAttrGlobalL1CacheSupported:
            return hipDeviceAttributeGlobalL1CacheSupported;
        case UPTKDevAttrGlobalMemoryBusWidth:
            return hipDeviceAttributeMemoryBusWidth;
        case UPTKDevAttrGpuOverlap:
            return hipDeviceAttributeAsyncEngineCount;
        case UPTKDevAttrHostNativeAtomicSupported:
            return hipDeviceAttributeHostNativeAtomicSupported;
        case UPTKDevAttrHostRegisterSupported:
            return hipDeviceAttributeHostRegisterSupported;
        case UPTKDevAttrIntegrated:
            return hipDeviceAttributeIntegrated;
        case UPTKDevAttrIsMultiGpuBoard:
            return hipDeviceAttributeIsMultiGpuBoard;
        case UPTKDevAttrKernelExecTimeout:
            return hipDeviceAttributeKernelExecTimeout;
        case UPTKDevAttrL2CacheSize:
            return hipDeviceAttributeL2CacheSize;
        case UPTKDevAttrLocalL1CacheSupported:
            return hipDeviceAttributeLocalL1CacheSupported;
        case UPTKDevAttrManagedMemory:
            return hipDeviceAttributeManagedMemory;
        case UPTKDevAttrMaxBlockDimX:
            return hipDeviceAttributeMaxBlockDimX;
        case UPTKDevAttrMaxBlockDimY:
            return hipDeviceAttributeMaxBlockDimY;
        case UPTKDevAttrMaxBlockDimZ:
            return hipDeviceAttributeMaxBlockDimZ;
        case UPTKDevAttrMaxGridDimX:
            return hipDeviceAttributeMaxGridDimX;
        case UPTKDevAttrMaxGridDimY:
            return hipDeviceAttributeMaxGridDimY;
        case UPTKDevAttrMaxGridDimZ:
            return hipDeviceAttributeMaxGridDimZ;
        case UPTKDevAttrMaxPitch:
            return hipDeviceAttributeMaxPitch;
        case UPTKDevAttrMaxRegistersPerBlock:
            return hipDeviceAttributeMaxRegistersPerBlock;
        case UPTKDevAttrMaxRegistersPerMultiprocessor:
            return hipDeviceAttributeMaxRegistersPerMultiprocessor;
        case UPTKDevAttrMaxSharedMemoryPerBlock:
            return hipDeviceAttributeMaxSharedMemoryPerBlock;
        // hipDeviceAttributeSharedMemPerBlockOptin actually returns 0.
        // Replace with hipDeviceAttributeMaxSharedMemoryPerBlock.
        case UPTKDevAttrMaxSharedMemoryPerBlockOptin:
            return hipDeviceAttributeMaxSharedMemoryPerBlock;   
        case UPTKDevAttrMaxSharedMemoryPerMultiprocessor:
            return hipDeviceAttributeMaxSharedMemoryPerMultiprocessor;
//         case UPTKDevAttrMaxSurface1DLayeredLayers:
//             return hipDeviceAttributeMaxSurface1DLayeredLayers;
        // case UPTKDevAttrMaxSurface1DLayeredWidth:
        //     return hipDeviceAttributeMaxSurface1DLayered;
        case UPTKDevAttrMaxSurface1DWidth:
            return hipDeviceAttributeMaxSurface1D;
        // case UPTKDevAttrMaxSurface2DHeight:
        //     return hipDeviceAttributeMaxSurface2D;
        // case UPTKDevAttrMaxSurface2DLayeredHeight:
        //     return hipDeviceAttributeMaxSurface2DLayered;
//         case UPTKDevAttrMaxSurface2DLayeredLayers:
//             return hipDeviceAttributeMaxSurface2DLayeredLayers;
        // case UPTKDevAttrMaxSurface2DLayeredWidth:
        //     return hipDeviceAttributeMaxSurface2DLayered;
        // case UPTKDevAttrMaxSurface2DWidth:
        //     return hipDeviceAttributeMaxSurface2D;
        // case UPTKDevAttrMaxSurface3DDepth:
        //     return hipDeviceAttributeMaxSurface3D;
        // case UPTKDevAttrMaxSurface3DHeight:
        //     return hipDeviceAttributeMaxSurface3D;
        // case UPTKDevAttrMaxSurface3DWidth:
        //     return hipDeviceAttributeMaxSurface3D;
//         case UPTKDevAttrMaxSurfaceUPTKbemapLayeredLayers:
//             return hipDeviceAttributeMaxSurfaceUPTKbemapLayeredLayers;
        // case UPTKDevAttrMaxSurfaceUPTKbemapLayeredWidth:
        //     return hipDeviceAttributeMaxSurfaceUPTKbemapLayered;
        // case UPTKDevAttrMaxSurfaceUPTKbemapWidth:
        //     return hipDeviceAttributeMaxSurfaceUPTKbemap;
//         case UPTKDevAttrMaxTexture1DLayeredLayers:
//             return hipDeviceAttributeMaxTexture1DLayeredLayers;
        // case UPTKDevAttrMaxTexture1DLayeredWidth:
        //     return hipDeviceAttributeMaxTexture1DLayered;
        case UPTKDevAttrMaxTexture1DLinearWidth:
            return hipDeviceAttributeMaxTexture1DLinear;
        case UPTKDevAttrMaxTexture1DMipmappedWidth:
            return hipDeviceAttributeMaxTexture1DMipmap;
        case UPTKDevAttrMaxTexture1DWidth:
            return hipDeviceAttributeMaxTexture1DWidth;
        // case UPTKDevAttrMaxTexture2DGatherHeight:
        //     return hipDeviceAttributeMaxTexture2DGather;
        // case UPTKDevAttrMaxTexture2DGatherWidth:
        //     return hipDeviceAttributeMaxTexture2DGather;
        case UPTKDevAttrMaxTexture2DHeight:
            return hipDeviceAttributeMaxTexture2DHeight;
        // case UPTKDevAttrMaxTexture2DLayeredHeight:
        //     return hipDeviceAttributeMaxTexture2DLayered;
//         case UPTKDevAttrMaxTexture2DLayeredLayers:
//             return hipDeviceAttributeMaxTexture2DLayeredLayers;
        // case UPTKDevAttrMaxTexture2DLayeredWidth:
        //     return hipDeviceAttributeMaxTexture2DLayered;
        // case UPTKDevAttrMaxTexture2DLinearHeight:
        //     return hipDeviceAttributeMaxTexture2DLinear;
        // case UPTKDevAttrMaxTexture2DLinearPitch:
        //     return hipDeviceAttributeMaxTexture2DLinear;
        // case UPTKDevAttrMaxTexture2DLinearWidth:
        //     return hipDeviceAttributeMaxTexture2DLinear;
        // case UPTKDevAttrMaxTexture2DMipmappedHeight:
        //     return hipDeviceAttributeMaxTexture2DMipmap;
        // case UPTKDevAttrMaxTexture2DMipmappedWidth:
        //     return hipDeviceAttributeMaxTexture2DMipmap;
        case UPTKDevAttrMaxTexture2DWidth:
            return hipDeviceAttributeMaxTexture2DWidth;
        case UPTKDevAttrMaxTexture3DDepth:
            return hipDeviceAttributeMaxTexture3DDepth;
        // case UPTKDevAttrMaxTexture3DDepthAlt:
        //     return hipDeviceAttributeMaxTexture3DAlt;
        case UPTKDevAttrMaxTexture3DHeight:
            return hipDeviceAttributeMaxTexture3DHeight;
        // case UPTKDevAttrMaxTexture3DHeightAlt:
        //     return hipDeviceAttributeMaxTexture3DAlt;
        case UPTKDevAttrMaxTexture3DWidth:
            return hipDeviceAttributeMaxTexture3DWidth;
        // case UPTKDevAttrMaxTexture3DWidthAlt:
        //     return hipDeviceAttributeMaxTexture3DAlt;
//         case UPTKDevAttrMaxTextureUPTKbemapLayeredLayers:
//             return hipDeviceAttributeMaxTextureUPTKbemapLayeredLayers;
        // case UPTKDevAttrMaxTextureUPTKbemapLayeredWidth:
        //     return hipDeviceAttributeMaxTextureUPTKbemapLayered;
        // case UPTKDevAttrMaxTextureCubemapWidth:
        //     return hipDeviceAttributeMaxTexturecubemap;
        case UPTKDevAttrMaxThreadsPerBlock:
            return hipDeviceAttributeMaxThreadsPerBlock;
        case UPTKDevAttrMaxThreadsPerMultiProcessor:
            return hipDeviceAttributeMaxThreadsPerMultiProcessor;
        case UPTKDevAttrMemoryClockRate:
            return hipDeviceAttributeMemoryClockRate;
        case UPTKDevAttrMultiGpuBoardGroupID:
            return hipDeviceAttributeMultiGpuBoardGroupID;
        case UPTKDevAttrMultiProcessorCount:
            return hipDeviceAttributeMultiprocessorCount;
        case UPTKDevAttrPageableMemoryAccess:
            return hipDeviceAttributePageableMemoryAccess;
        case UPTKDevAttrPageableMemoryAccessUsesHostPageTables:
            return hipDeviceAttributePageableMemoryAccessUsesHostPageTables;
        case UPTKDevAttrPciBusId:
            return hipDeviceAttributePciBusId;
        case UPTKDevAttrPciDeviceId:
            return hipDeviceAttributePciDeviceId;
        case UPTKDevAttrPciDomainId:
            return hipDeviceAttributePciDomainID;
//         case UPTKDevAttrReserved92:
//             return hipDeviceAttributeCanUseStreamMemOps;
//         case UPTKDevAttrReserved93:
//             return hipDeviceAttributeCanUse64BitStreamMemOps;
        case UPTKDevAttrReserved94:
            return hipDeviceAttributeCanUseStreamWaitValue;
        // case UPTKDevAttrSingleToDoublePrecisionPerfRatio:
        //     return hipDeviceAttributeSingleToDoublePrecisionPerfRatio;
        case UPTKDevAttrStreamPrioritiesSupported:
            return hipDeviceAttributeStreamPrioritiesSupported;
        case UPTKDevAttrSurfaceAlignment:
            return hipDeviceAttributeSurfaceAlignment;
        // case UPTKDevAttrTccDriver:
        //     return hipDeviceAttributeTccDriver;
        case UPTKDevAttrTextureAlignment:
            return hipDeviceAttributeTextureAlignment;
        case UPTKDevAttrTexturePitchAlignment:
            return hipDeviceAttributeTexturePitchAlignment;
        case UPTKDevAttrTotalConstantMemory:
            return hipDeviceAttributeTotalConstantMemory;
        case UPTKDevAttrUnifiedAddressing:
            return hipDeviceAttributeUnifiedAddressing;
        case UPTKDevAttrWarpSize:
            return hipDeviceAttributeWarpSize;
        case UPTKDevAttrMemoryPoolsSupported:
            return hipDeviceAttributeMemoryPoolsSupported;
        case UPTKDevAttrReservedSharedMemoryPerBlock:
            return hipDeviceAttributeReservedSharedMemPerBlock;
        case UPTKDevAttrMaxBlocksPerMultiprocessor:
            return hipDeviceAttributeMaxBlocksPerMultiProcessor;
        default:
            ERROR_INVALID_OR_UNSUPPORTED_ENUM();
    }
}

enum UPTKGraphNodeType hipGraphNodeTypeToUPTKGraphNodeType(hipGraphNodeType para) {
    switch (para) {
        case hipGraphNodeTypeCount:
            return UPTKGraphNodeTypeCount;
        case hipGraphNodeTypeEmpty:
            return UPTKGraphNodeTypeEmpty;
        case hipGraphNodeTypeEventRecord:
            return UPTKGraphNodeTypeEventRecord;
        case hipGraphNodeTypeGraph:
            return UPTKGraphNodeTypeGraph;
        case hipGraphNodeTypeHost:
            return UPTKGraphNodeTypeHost;
        case hipGraphNodeTypeKernel:
            return UPTKGraphNodeTypeKernel;
        case hipGraphNodeTypeMemcpy:
            return UPTKGraphNodeTypeMemcpy;
        case hipGraphNodeTypeMemset:
            return UPTKGraphNodeTypeMemset;
        case hipGraphNodeTypeWaitEvent:
            return UPTKGraphNodeTypeWaitEvent;
        case hipGraphNodeTypeExtSemaphoreSignal:
            return UPTKGraphNodeTypeExtSemaphoreSignal;
        case hipGraphNodeTypeExtSemaphoreWait:
            return UPTKGraphNodeTypeExtSemaphoreWait;
        case hipGraphNodeTypeMemAlloc:
            return UPTKGraphNodeTypeMemAlloc;
        case hipGraphNodeTypeMemFree:
            return UPTKGraphNodeTypeMemFree;
        case hipGraphNodeTypeConditional:
            return UPTKGraphNodeTypeConditional;
        default: 
            ERROR_INVALID_ENUM();
    }
}

void UPTKKernelNodeParamsV2TohipKernelNodeParams(const UPTKKernelNodeParamsV2 *UPTK_para, hipKernelNodeParams *hip_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    hip_para->func = UPTK_para->func;
    hip_para->sharedMemBytes = UPTK_para->sharedMemBytes;
    hip_para->kernelParams = UPTK_para->kernelParams;
    hip_para->extra = UPTK_para->extra;
#if !defined(__cplusplus) || __cplusplus >= 201103L
    hip_para->gridDim = UPTK_para->gridDim;
    hip_para->blockDim = UPTK_para->blockDim;
#else
    hip_para->gridDim.x = UPTK_para->gridDim.x;
    hip_para->gridDim.y = UPTK_para->gridDim.y;
    hip_para->gridDim.z = UPTK_para->gridDim.z;
    hip_para->blockDim.x = UPTK_para->blockDim.x;
    hip_para->blockDim.y = UPTK_para->blockDim.y;
    hip_para->blockDim.z = UPTK_para->blockDim.z;
#endif
}

hipGraphNodeType UPTKGraphNodeTypeTohipGraphNodeType(enum UPTKGraphNodeType para) {
    switch (para) {
        case UPTKGraphNodeTypeKernel:
            return hipGraphNodeTypeKernel;
        case UPTKGraphNodeTypeMemcpy:
            return hipGraphNodeTypeMemcpy;
        case UPTKGraphNodeTypeMemset:
            return hipGraphNodeTypeMemset;
        case UPTKGraphNodeTypeHost:
            return hipGraphNodeTypeHost;
        case UPTKGraphNodeTypeGraph:
            return hipGraphNodeTypeGraph;
        case UPTKGraphNodeTypeEmpty:
            return hipGraphNodeTypeEmpty;
        case UPTKGraphNodeTypeWaitEvent:
            return hipGraphNodeTypeWaitEvent;
        case UPTKGraphNodeTypeEventRecord:
            return hipGraphNodeTypeEventRecord;
        case UPTKGraphNodeTypeExtSemaphoreSignal:
            return hipGraphNodeTypeExtSemaphoreSignal;
        case UPTKGraphNodeTypeExtSemaphoreWait:
            return hipGraphNodeTypeExtSemaphoreWait;
        case UPTKGraphNodeTypeCount:
            return hipGraphNodeTypeCount;
        case UPTKGraphNodeTypeMemAlloc:
            return hipGraphNodeTypeMemAlloc;
        case UPTKGraphNodeTypeMemFree:
            return hipGraphNodeTypeMemFree;
        case UPTKGraphNodeTypeConditional:
            return hipGraphNodeTypeConditional;
        default: 
            ERROR_INVALID_ENUM();
    }
}

void UPTKMemcpyNodeParamsTohipMemcpyNodeParams(const UPTKMemcpyNodeParams *UPTK_para, hipMemcpyNodeParams *hip_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    hip_para->flags = UPTK_para->flags;
    memcpy(&(hip_para->reserved), &(UPTK_para->reserved), sizeof(int[3]));
    UPTKMemcpy3DParmsTohipMemcpy3DParms(&(UPTK_para->copyParams), &(hip_para->copyParams));
}

void UPTKMemsetParamsV2TohipMemsetParams(const UPTKMemsetParamsV2 *UPTK_para, hipMemsetParams *hip_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    hip_para->dst = UPTK_para->dst;
    hip_para->elementSize = UPTK_para->elementSize;
    hip_para->height = UPTK_para->height;
    hip_para->pitch = UPTK_para->pitch;
    hip_para->value = UPTK_para->value;
    hip_para->width = UPTK_para->width;
}

void UPTKHostNodeParamsV2TohipHostNodeParams(const UPTKHostNodeParamsV2 *UPTK_para, hipHostNodeParams *hip_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    hip_para->fn = (hipHostFn_t)UPTK_para->fn;
    hip_para->userData = UPTK_para->userData;
}

void UPTKChildGraphNodeParamsTohipChildGraphNodeParams(const UPTKChildGraphNodeParams *UPTK_para, hipChildGraphNodeParams *hip_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    hip_para->graph = (hipGraph_t)UPTK_para->graph;
}

void UPTKEventWaitNodeParamsTohipEventWaitNodeParams(const UPTKEventWaitNodeParams *UPTK_para, hipEventWaitNodeParams *hip_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    hip_para->event = (hipEvent_t)UPTK_para->event;
}

void UPTKEventRecordNodeParamsTohipEventRecordNodeParams(const UPTKEventRecordNodeParams *UPTK_para, hipEventRecordNodeParams *hip_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    hip_para->event = (hipEvent_t)UPTK_para->event;
}

// TODO: UPTKExternalSemaphoreSignalParams generally appears in the form of an array, so special conversion is performed here.
hipExternalSemaphoreSignalParams UPTKExternalSemaphoreSignalParamsTohipExternalSemaphoreSignalParams(struct UPTKExternalSemaphoreSignalParams UPTK_para)
{
    // if (nullptr == UPTK_para || nullptr == hip_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    hipExternalSemaphoreSignalParams hip_para;
    hip_para.flags = UPTK_para.flags;
    memcpy(hip_para.reserved, UPTK_para.reserved, sizeof(hip_para.reserved));
    //Copy the members of the params structure
    hip_para.params.fence.value = UPTK_para.params.fence.value;
    hip_para.params.keyedMutex.key = UPTK_para.params.keyedMutex.key;
    memcpy(hip_para.params.reserved, UPTK_para.params.reserved, sizeof(hip_para.params.reserved));
    memcpy(&hip_para.params.nvSciSync, &UPTK_para.params.nvSciSync, sizeof(hip_para.params.nvSciSync));
    return hip_para;
}

hipExternalSemaphoreWaitParams UPTKExternalSemaphoreWaitParamsTohipExternalSemaphoreWaitParams(struct UPTKExternalSemaphoreWaitParams UPTK_para)
{
    // if (nullptr == UPTK_para || nullptr == hip_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    hipExternalSemaphoreWaitParams hip_para;
    hip_para.flags = UPTK_para.flags;
    memcpy(hip_para.reserved, UPTK_para.reserved, sizeof(hip_para.reserved));
    //Copy the members of the params structure
    hip_para.params.fence.value = UPTK_para.params.fence.value;
    hip_para.params.keyedMutex.key = UPTK_para.params.keyedMutex.key;
    hip_para.params.keyedMutex.timeoutMs = UPTK_para.params.keyedMutex.timeoutMs;
    memcpy(hip_para.params.reserved, UPTK_para.params.reserved, sizeof(hip_para.params.reserved));
    memcpy(&hip_para.params.nvSciSync, &UPTK_para.params.nvSciSync, sizeof(hip_para.params.nvSciSync));
    return hip_para;
}

hipExternalSemaphoreSignalNodeParams UPTKExternalSemaphoreSignalNodeParamsV2TohipExternalSemaphoreSignalNodeParams(UPTKExternalSemaphoreSignalNodeParamsV2 UPTK_para)
{
    // if (nullptr == UPTK_para || nullptr == hip_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    hipExternalSemaphoreSignalNodeParams hip_para;
    hip_para.extSemArray = (hipExternalSemaphore_t *)UPTK_para.extSemArray;
    hip_para.numExtSems = UPTK_para.numExtSems;
    hipExternalSemaphoreSignalParams *hip_paramsArray = (hipExternalSemaphoreSignalParams *)malloc(sizeof(hipExternalSemaphoreSignalParams) * hip_para.numExtSems);
    for (int i = 0; i < hip_para.numExtSems; i++)
    {
        hip_paramsArray[i] = UPTKExternalSemaphoreSignalParamsTohipExternalSemaphoreSignalParams(UPTK_para.paramsArray[i]);
    }
    hip_para.paramsArray = hip_paramsArray;
    return hip_para;
}

hipMemAllocationType UPTKMemAllocationTypeTohipMemAllocationType(enum UPTKMemAllocationType para) {
    switch (para) {
        case UPTKMemAllocationTypeInvalid:
            return hipMemAllocationTypeInvalid;
        case UPTKMemAllocationTypeMax:
            return hipMemAllocationTypeMax;
        case UPTKMemAllocationTypePinned:
            return hipMemAllocationTypePinned;
        default:
            ERROR_INVALID_ENUM();
    }
}

hipMemAllocationHandleType UPTKMemAllocationHandleTypeTohipMemAllocationHandleType(enum UPTKMemAllocationHandleType para) {
    switch (para) {
        case UPTKMemHandleTypeNone:
            return hipMemHandleTypeNone;
        case UPTKMemHandleTypePosixFileDescriptor:
            return hipMemHandleTypePosixFileDescriptor;
        case UPTKMemHandleTypeWin32:
            return hipMemHandleTypeWin32;
        case UPTKMemHandleTypeWin32Kmt:
            return hipMemHandleTypeWin32Kmt;
        default:
            ERROR_INVALID_ENUM();
    }
}

hipMemLocationType UPTKMemLocationTypeTohipMemLocationType(enum UPTKMemLocationType para) {
    switch (para) {
        case UPTKMemLocationTypeDevice:
            return hipMemLocationTypeDevice;
        case UPTKMemLocationTypeInvalid:
            return hipMemLocationTypeInvalid;
        case UPTKMemLocationTypeHost:
            return hipMemLocationTypeHost;
        case UPTKMemLocationTypeHostNuma:   
            return hipMemLocationTypeHostNuma;
        case UPTKMemLocationTypeHostNumaCurrent:
            return hipMemLocationTypeHostNumaCurrent;
        default:
            ERROR_INVALID_ENUM();
    }
}

void UPTKMemLocationTohipMemLocation(const struct UPTKMemLocation * UPTK_para, hipMemLocation * hip_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    hip_para->id = UPTK_para->id;
    hip_para->type = UPTKMemLocationTypeTohipMemLocationType(UPTK_para->type);
}

hipExternalSemaphoreWaitNodeParams UPTKExternalSemaphoreWaitNodeParamsV2TohipExternalSemaphoreWaitNodeParams(UPTKExternalSemaphoreWaitNodeParamsV2 UPTK_para)
{
    // if (nullptr == UPTK_para || nullptr == hip_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    hipExternalSemaphoreWaitNodeParams hip_para;
    hip_para.extSemArray = (hipExternalSemaphore_t *)UPTK_para.extSemArray;
    hip_para.numExtSems = UPTK_para.numExtSems;
    hipExternalSemaphoreWaitParams *hip_paramsArray = (hipExternalSemaphoreWaitParams *)malloc(sizeof(hipExternalSemaphoreWaitParams) * hip_para.numExtSems);
    for (int i = 0; i < hip_para.numExtSems; i++)
    {
        hip_paramsArray[i] = UPTKExternalSemaphoreWaitParamsTohipExternalSemaphoreWaitParams(UPTK_para.paramsArray[i]);
    }
    hip_para.paramsArray = hip_paramsArray;
    return hip_para;
}

void UPTKMemPoolPropsTohipMemPoolProps(const UPTKMemPoolProps * UPTK_para, hipMemPoolProps * hip_para) {
    if (nullptr == UPTK_para || nullptr == hip_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    memcpy(hip_para->reserved, UPTK_para->reserved, sizeof(hip_para->reserved));
    hip_para->win32SecurityAttributes = UPTK_para->win32SecurityAttributes;
    hip_para->allocType = UPTKMemAllocationTypeTohipMemAllocationType(UPTK_para->allocType);
    hip_para->handleTypes = UPTKMemAllocationHandleTypeTohipMemAllocationHandleType(UPTK_para->handleTypes);
    UPTKMemLocationTohipMemLocation(&(UPTK_para->location), &(hip_para->location));
}

hipMemAccessFlags UPTKMemAccessFlagsTohipMemAccessFlags(UPTKMemAccessFlags para) {
    switch (para) {
        case UPTKMemAccessFlagsProtNone:
            return hipMemAccessFlagsProtNone;
        case UPTKMemAccessFlagsProtRead:
            return hipMemAccessFlagsProtRead;
        case UPTKMemAccessFlagsProtReadWrite:
            return hipMemAccessFlagsProtReadWrite;
        default:
            ERROR_INVALID_ENUM();
    }
}

hipMemAccessDesc UPTKMemAccessDescTohipMemAccessDesc(struct UPTKMemAccessDesc UPTK_para)
{
    // if (nullptr == UPTK_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    hipMemAccessDesc hip_para;
    hip_para.flags = UPTKMemAccessFlagsTohipMemAccessFlags(UPTK_para.flags);
    UPTKMemLocationTohipMemLocation(&(UPTK_para.location), &(hip_para.location));
    return hip_para;
}


void UPTKMemAllocNodeParamsV2TohipMemAllocNodeParams(const struct UPTKMemAllocNodeParamsV2 *UPTK_para, hipMemAllocNodeParams *hip_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    UPTKMemPoolPropsTohipMemPoolProps(&(UPTK_para->poolProps), &(hip_para->poolProps));

    hip_para->accessDescCount = UPTK_para->accessDescCount;
    hipMemAccessDesc *hip_daccessDescs = (hipMemAccessDesc *)malloc(sizeof(hipMemAccessDesc) * hip_para->accessDescCount);
    for (int i = 0; i < hip_para->accessDescCount; i++)
    {
        hip_daccessDescs[i] = UPTKMemAccessDescTohipMemAccessDesc(UPTK_para->accessDescs[i]);
    }
    hip_para->bytesize = UPTK_para->bytesize;
    hip_para->dptr = UPTK_para->dptr;
}

void UPTKMemFreeNodeParamsTohipMemFreeNodeParams(const UPTKMemFreeNodeParams *UPTK_para, hipMemFreeNodeParams *hip_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    hip_para->dptr = UPTK_para->dptr;
}

void UPTKConditionalNodeParamsTohipConditionalNodeParams(const UPTKConditionalNodeParams *UPTK_para, hipConditionalNodeParams *hip_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    hip_para->handle = (hipGraphConditionalHandle)UPTK_para->handle;
    hip_para->type = UPTKConditionalNodeTypeTohipConditionalNodeType(UPTK_para->type);
    hip_para->size = UPTK_para->size;
    hip_para->phGraph_out = (hipGraph_t *)UPTK_para->phGraph_out;

}

hipGraphConditionalNodeType UPTKConditionalNodeTypeTohipConditionalNodeType(enum UPTKGraphConditionalNodeType para)
{
    switch (para)
    {
    case UPTKGraphCondTypeIf:
        return hipGraphCondTypeIf;
    case UPTKGraphCondTypeWhile:
        return hipGraphCondTypeWhile;
    default:
        ERROR_INVALID_ENUM();
    }
}

void UPTKGraphNodeParamsTohipGraphNodeParams(const UPTKGraphNodeParams * UPTK_para, hipGraphNodeParams * hip_para)
{
    if (nullptr == UPTK_para || nullptr == hip_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    hip_para->type = UPTKGraphNodeTypeTohipGraphNodeType(UPTK_para->type);
    memcpy(hip_para->reserved0, UPTK_para->reserved0, sizeof(int[3]));
    hip_para->reserved2 = UPTK_para->reserved2;
    memset(&(hip_para->reserved1), 0, sizeof(long long[29]));
    memcpy(&(hip_para->reserved1), &(UPTK_para->reserved1), sizeof(long long[29]));
    
    switch (hip_para->type)
    {
    case hipGraphNodeTypeKernel:
        UPTKKernelNodeParamsV2TohipKernelNodeParams(&(UPTK_para->kernel), &(hip_para->kernel));
        break;
    case hipGraphNodeTypeMemcpy:
        UPTKMemcpyNodeParamsTohipMemcpyNodeParams(&(UPTK_para->memcpy), &(hip_para->memcpy));
        break;
    case hipGraphNodeTypeMemset:
        UPTKMemsetParamsV2TohipMemsetParams(&(UPTK_para->memset), &(hip_para->memset));
        break;
    case hipGraphNodeTypeHost:
        UPTKHostNodeParamsV2TohipHostNodeParams(&(UPTK_para->host), &(hip_para->host));
        break;
    case hipGraphNodeTypeGraph:
        UPTKChildGraphNodeParamsTohipChildGraphNodeParams(&(UPTK_para->graph), &(hip_para->graph));
        break;
    case hipGraphNodeTypeWaitEvent:
        UPTKEventWaitNodeParamsTohipEventWaitNodeParams(&(UPTK_para->eventWait), &(hip_para->eventWait));
        break;
    case hipGraphNodeTypeEventRecord:
        UPTKEventRecordNodeParamsTohipEventRecordNodeParams(&(UPTK_para->eventRecord), &(hip_para->eventRecord));
        break;
    case hipGraphNodeTypeExtSemaphoreSignal:
        hip_para->extSemSignal = UPTKExternalSemaphoreSignalNodeParamsV2TohipExternalSemaphoreSignalNodeParams(UPTK_para->extSemSignal);
        break;
    case hipGraphNodeTypeExtSemaphoreWait:
        hip_para->extSemWait = UPTKExternalSemaphoreWaitNodeParamsV2TohipExternalSemaphoreWaitNodeParams(UPTK_para->extSemWait);
        break;
    case hipGraphNodeTypeMemAlloc:
        UPTKMemAllocNodeParamsV2TohipMemAllocNodeParams(&(UPTK_para->alloc), &(hip_para->alloc));
        break;
    case hipGraphNodeTypeMemFree:
        UPTKMemFreeNodeParamsTohipMemFreeNodeParams(&(UPTK_para->free), &(hip_para->free));
        break;
    case hipGraphNodeTypeConditional:
        UPTKConditionalNodeParamsTohipConditionalNodeParams(&(UPTK_para->conditional),&(hip_para->conditional));
        break;
    }
}

hipFunction_attribute UPTKfunction_attributeTohipFunction_attribute(UPTKfunction_attribute para) {
    switch (para) {
        case UPTK_FUNC_ATTRIBUTE_BINARY_VERSION:
            return HIP_FUNC_ATTRIBUTE_BINARY_VERSION;
        case UPTK_FUNC_ATTRIBUTE_CACHE_MODE_CA:
            return HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA;
        case UPTK_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:
            return HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES;
        case UPTK_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:
            return HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES;
        case UPTK_FUNC_ATTRIBUTE_MAX:
            return HIP_FUNC_ATTRIBUTE_MAX;
        case UPTK_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
            return HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
        case UPTK_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
            return HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
        case UPTK_FUNC_ATTRIBUTE_NUM_REGS:
            return HIP_FUNC_ATTRIBUTE_NUM_REGS;
        case UPTK_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
            return HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT;
        case UPTK_FUNC_ATTRIBUTE_PTX_VERSION:
            return HIP_FUNC_ATTRIBUTE_PTX_VERSION;
        case UPTK_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:
            return HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES;
        default:
            ERROR_INVALID_ENUM();
    }
}

hipError_t UPTKErrorTohipError(enum UPTKError para) {
    switch (para) {
//         case UPTKErrorAddressOfConstant:
//             return hipErrorAddressOfConstant;
        case UPTKErrorAlreadyAcquired:
            return hipErrorAlreadyAcquired;
        case UPTKErrorAlreadyMapped:
            return hipErrorAlreadyMapped;
//         case UPTKErrorApiFailureBase:
//             return hipErrorApiFailureBase;
        case UPTKErrorArrayIsMapped:
            return hipErrorArrayIsMapped;
        case UPTKErrorAssert:
            return hipErrorAssert;
        case UPTKErrorCapturedEvent:
            return hipErrorCapturedEvent;
//         case UPTKErrorCompatNotSupportedOnDevice:
//             return hipErrorCompatNotSupportedOnDevice;
        case UPTKErrorContextIsDestroyed:
            return hipErrorContextIsDestroyed;
        case UPTKErrorCooperativeLaunchTooLarge:
            return hipErrorCooperativeLaunchTooLarge;
        case UPTKErrorUPTKrtUnloading:
            return hipErrorDeinitialized;
        case UPTKErrorDeviceAlreadyInUse:
            return hipErrorContextAlreadyInUse;
        case UPTKErrorDeviceUninitialized:
            return hipErrorInvalidContext;
//         case UPTKErrorDevicesUnavailable:
//             return hipErrorDevicesUnavailable;
//         case UPTKErrorDuplicateSurfaceName:
//             return hipErrorDuplicateSurfaceName;
//         case UPTKErrorDuplicateTextureName:
//             return hipErrorDuplicateTextureName;
//         case UPTKErrorDuplicateVariableName:
//             return hipErrorDuplicateVariableName;
        case UPTKErrorECCUncorrectable:
            return hipErrorECCNotCorrectable;
        case UPTKErrorFileNotFound:
            return hipErrorFileNotFound;
        case UPTKErrorGraphExecUpdateFailure:
            return hipErrorGraphExecUpdateFailure;
//         case UPTKErrorHardwareStackError:
//             return hipErrorHardwareStackError;
        case UPTKErrorHostMemoryAlreadyRegistered:
            return hipErrorHostMemoryAlreadyRegistered;
        case UPTKErrorHostMemoryNotRegistered:
            return hipErrorHostMemoryNotRegistered;
        case UPTKErrorIllegalAddress:
            return hipErrorIllegalAddress;
//         case UPTKErrorIllegalInstruction:
//             return hipErrorIllegalInstruction;
        case UPTKErrorIllegalState:
            return hipErrorIllegalState;
//         case UPTKErrorIncompatibleDriverContext:
//             return hipErrorIncompatibleDriverContext;
        case UPTKErrorInitializationError:
            return hipErrorNotInitialized;
        case UPTKErrorInsufficientDriver:
            return hipErrorInsufficientDriver;
//         case UPTKErrorInvalidAddressSpace:
//             return hipErrorInvalidAddressSpace;
//         case UPTKErrorInvalidChannelDescriptor:
//             return hipErrorInvalidChannelDescriptor;
        case UPTKErrorInvalidConfiguration:
            return hipErrorInvalidConfiguration;
        case UPTKErrorInvalidDevice:
            return hipErrorInvalidDevice;
        case UPTKErrorInvalidDeviceFunction:
            return hipErrorInvalidDeviceFunction;
        case UPTKErrorInvalidDevicePointer:
            return hipErrorInvalidDevicePointer;
//         case UPTKErrorInvalidFilterSetting:
//             return hipErrorInvalidFilterSetting;
        case UPTKErrorInvalidGraphicsContext:
            return hipErrorInvalidGraphicsContext;
//         case UPTKErrorInvalidHostPointer:
//             return hipErrorInvalidHostPointer;
        case UPTKErrorInvalidKernelImage:
            return hipErrorInvalidImage;
        case UPTKErrorInvalidMemcpyDirection:
            return hipErrorInvalidMemcpyDirection;
//         case UPTKErrorInvalidNormSetting:
//             return hipErrorInvalidNormSetting;
//         case UPTKErrorInvalidPc:
//             return hipErrorInvalidPc;
        case UPTKErrorInvalidPitchValue:
            return hipErrorInvalidPitchValue;
        case UPTKErrorInvalidPtx:
            return hipErrorInvalidKernelFile;
        case UPTKErrorInvalidResourceHandle:
            return hipErrorInvalidHandle;
        case UPTKErrorInvalidSource:
            return hipErrorInvalidSource;
//         case UPTKErrorInvalidSurface:
//             return hipErrorInvalidSurface;
        case UPTKErrorInvalidSymbol:
            return hipErrorInvalidSymbol;
//         case UPTKErrorInvalidTexture:
//             return hipErrorInvalidTexture;
//         case UPTKErrorInvalidTextureBinding:
//             return hipErrorInvalidTextureBinding;
        case UPTKErrorInvalidValue:
            return hipErrorInvalidValue;
//         case UPTKErrorJitCompilerNotFound:
//             return hipErrorJitCompilerNotFound;
        case UPTKErrorLaunchFailure:
            return hipErrorLaunchFailure;
//         case UPTKErrorLaunchFileScopedSurf:
//             return hipErrorLaunchFileScopedSurf;
//         case UPTKErrorLaunchFileScopedTex:
//             return hipErrorLaunchFileScopedTex;
//         case UPTKErrorLaunchIncompatibleTexturing:
//             return hipErrorLaunchIncompatibleTexturing;
//         case UPTKErrorLaunchMaxDepthExceeded:
//             return hipErrorLaunchMaxDepthExceeded;
        case UPTKErrorLaunchOutOfResources:
            return hipErrorLaunchOutOfResources;
//         case UPTKErrorLaunchPendingCountExceeded:
//             return hipErrorLaunchPendingCountExceeded;
        case UPTKErrorLaunchTimeout:
            return hipErrorLaunchTimeOut;
        case UPTKErrorMapBufferObjectFailed:
            return hipErrorMapFailed;
        case UPTKErrorMemoryAllocation:
            return hipErrorOutOfMemory;
//         case UPTKErrorMemoryValueTooLarge:
//             return hipErrorMemoryValueTooLarge;
//         case UPTKErrorMisalignedAddress:
//             return hipErrorMisalignedAddress;
        case UPTKErrorMissingConfiguration:
            return hipErrorMissingConfiguration;
//         case UPTKErrorMixedDeviceExecution:
//             return hipErrorMixedDeviceExecution;
        case UPTKErrorNoDevice:
            return hipErrorNoDevice;
        case UPTKErrorNoKernelImageForDevice:
            return hipErrorNoBinaryForGpu;
        case UPTKErrorNotMapped:
            return hipErrorNotMapped;
        case UPTKErrorNotMappedAsArray:
            return hipErrorNotMappedAsArray;
        case UPTKErrorNotMappedAsPointer:
            return hipErrorNotMappedAsPointer;
//         case UPTKErrorNotPermitted:
//             return hipErrorNotPermitted;
        case UPTKErrorNotReady:
            return hipErrorNotReady;
        case UPTKErrorNotSupported:
            return hipErrorNotSupported;
//         case UPTKErrorNotYetImplemented:
//             return hipErrorNotYetImplemented;
//         case UPTKErrorNvlinkUncorrectable:
//             return hipErrorNvlinkUncorrectable;
        case UPTKErrorOperatingSystem:
            return hipErrorOperatingSystem;
        case UPTKErrorPeerAccessAlreadyEnabled:
            return hipErrorPeerAccessAlreadyEnabled;
        case UPTKErrorPeerAccessNotEnabled:
            return hipErrorPeerAccessNotEnabled;
        case UPTKErrorPeerAccessUnsupported:
            return hipErrorPeerAccessUnsupported;
        case UPTKErrorPriorLaunchFailure:
            return hipErrorPriorLaunchFailure;
        case UPTKErrorProfilerAlreadyStarted:
            return hipErrorProfilerAlreadyStarted;
        case UPTKErrorProfilerAlreadyStopped:
            return hipErrorProfilerAlreadyStopped;
        case UPTKErrorProfilerDisabled:
            return hipErrorProfilerDisabled;
        case UPTKErrorProfilerNotInitialized:
            return hipErrorProfilerNotInitialized;
        case UPTKErrorSetOnActiveProcess:
            return hipErrorSetOnActiveProcess;
        case UPTKErrorSharedObjectInitFailed:
            return hipErrorSharedObjectInitFailed;
        case UPTKErrorSharedObjectSymbolNotFound:
            return hipErrorSharedObjectSymbolNotFound;
//         case UPTKErrorStartupFailure:
//             return hipErrorStartupFailure;
        case UPTKErrorStreamCaptureImplicit:
            return hipErrorStreamCaptureImplicit;
        case UPTKErrorStreamCaptureInvalidated:
            return hipErrorStreamCaptureInvalidated;
        case UPTKErrorStreamCaptureIsolation:
            return hipErrorStreamCaptureIsolation;
        case UPTKErrorStreamCaptureMerge:
            return hipErrorStreamCaptureMerge;
        case UPTKErrorStreamCaptureUnjoined:
            return hipErrorStreamCaptureUnjoined;
        case UPTKErrorStreamCaptureUnmatched:
            return hipErrorStreamCaptureUnmatched;
        case UPTKErrorStreamCaptureUnsupported:
            return hipErrorStreamCaptureUnsupported;
        case UPTKErrorStreamCaptureWrongThread:
            return hipErrorStreamCaptureWrongThread;
        case UPTKErrorSymbolNotFound:
            return hipErrorNotFound;
//         case UPTKErrorSyncDepthExceeded:
//             return hipErrorSyncDepthExceeded;
//         case UPTKErrorSynchronizationError:
//             return hipErrorSynchronizationError;
//         case UPTKErrorSystemDriverMismatch:
//             return hipErrorSystemDriverMismatch;
//         case UPTKErrorSystemNotReady:
//             return hipErrorSystemNotReady;
//         case UPTKErrorTextureFetchFailed:
//             return hipErrorTextureFetchFailed;
//         case UPTKErrorTextureNotBound:
//             return hipErrorTextureNotBound;
//         case UPTKErrorTimeout:
//             return hipErrorTimeout;
//         case UPTKErrorTooManyPeers:
//             return hipErrorTooManyPeers;
        case UPTKErrorUnknown:
            return hipErrorUnknown;
        case UPTKErrorUnmapBufferObjectFailed:
            return hipErrorUnmapFailed;
        case UPTKErrorUnsupportedLimit:
            return hipErrorUnsupportedLimit;
        case UPTKSuccess:
            return hipSuccess;
        default:
            ERROR_INVALID_ENUM();
    }
}

hipFuncAttribute UPTKFuncAttributeTohipFuncAttribute(enum UPTKFuncAttribute para) {
    switch (para) {
        case UPTKFuncAttributeMax:
            return hipFuncAttributeMax;
        case UPTKFuncAttributeMaxDynamicSharedMemorySize:
            return hipFuncAttributeMaxDynamicSharedMemorySize;
        case UPTKFuncAttributePreferredSharedMemoryCarveout:
            return hipFuncAttributePreferredSharedMemoryCarveout;
        default:
            ERROR_INVALID_ENUM();
    }
}


#if defined(__cplusplus)
}
#endif /* __cplusplus */


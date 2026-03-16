#include "runtime.hpp"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

enum UPTKError cudaErrorToUPTKError(cudaError_t para) {
    switch (para) {
//         case cudaErrorAddressOfConstant:
//             return UPTKErrorAddressOfConstant;
        case cudaErrorAlreadyAcquired:
            return UPTKErrorAlreadyAcquired;
        case cudaErrorAlreadyMapped:
            return UPTKErrorAlreadyMapped;
//         case cudaErrorApiFailureBase:
//             return UPTKErrorApiFailureBase;
        case cudaErrorArrayIsMapped:
            return UPTKErrorArrayIsMapped;
        case cudaErrorAssert:
            return UPTKErrorAssert;
        case cudaErrorCapturedEvent:
            return UPTKErrorCapturedEvent;
//         case cudaErrorCompatNotSupportedOnDevice:
//             return UPTKErrorCompatNotSupportedOnDevice;
        case cudaErrorContextIsDestroyed:
            return UPTKErrorContextIsDestroyed;
        case cudaErrorCooperativeLaunchTooLarge:
            return UPTKErrorCooperativeLaunchTooLarge;
        case cudaErrorCudartUnloading://???正确吗???
             return UPTKErrorUPTKrtUnloading;
        case cudaErrorDeviceAlreadyInUse://???正确吗???
             return UPTKErrorDeviceAlreadyInUse;
        case cudaErrorDeviceUninitialized://???正确吗???
             return UPTKErrorDeviceUninitialized;
//         case cudaErrorDevicesUnavailable:
//             return UPTKErrorDevicesUnavailable;
//         case cudaErrorDuplicateSurfaceName:
//             return UPTKErrorDuplicateSurfaceName;
//         case cudaErrorDuplicateTextureName:
//             return UPTKErrorDuplicateTextureName;
//         case cudaErrorDuplicateVariableName:
//             return UPTKErrorDuplicateVariableName;
        case cudaErrorECCUncorrectable://???正确吗???
            return UPTKErrorECCUncorrectable;
        case cudaErrorFileNotFound:
            return UPTKErrorFileNotFound;
        case cudaErrorGraphExecUpdateFailure:
            return UPTKErrorGraphExecUpdateFailure;
//         case cudaErrorHardwareStackError:
//             return UPTKErrorHardwareStackError;
        case cudaErrorHostMemoryAlreadyRegistered:
            return UPTKErrorHostMemoryAlreadyRegistered;
        case cudaErrorHostMemoryNotRegistered:
            return UPTKErrorHostMemoryNotRegistered;
        case cudaErrorIllegalAddress:
            return UPTKErrorIllegalAddress;
//         case cudaErrorIllegalInstruction:
//             return UPTKErrorIllegalInstruction;
        case cudaErrorIllegalState:
            return UPTKErrorIllegalState;
//         case cudaErrorIncompatibleDriverContext:
//             return UPTKErrorIncompatibleDriverContext;
        case cudaErrorInitializationError://???正确吗???
            return UPTKErrorInitializationError;
        case cudaErrorInsufficientDriver:
            return UPTKErrorInsufficientDriver;
//         case cudaErrorInvalidAddressSpace:
//             return UPTKErrorInvalidAddressSpace;
//         case cudaErrorInvalidChannelDescriptor:
//             return UPTKErrorInvalidChannelDescriptor;
        case cudaErrorInvalidConfiguration:
            return UPTKErrorInvalidConfiguration;
        case cudaErrorInvalidDevice:
            return UPTKErrorInvalidDevice;
        case cudaErrorInvalidDeviceFunction:
            return UPTKErrorInvalidDeviceFunction;
        case cudaErrorInvalidDevicePointer:
            return UPTKErrorInvalidDevicePointer;
//         case cudaErrorInvalidFilterSetting:
//             return UPTKErrorInvalidFilterSetting;
        case cudaErrorInvalidGraphicsContext:
            return UPTKErrorInvalidGraphicsContext;
//         case cudaErrorInvalidHostPointer:
//             return UPTKErrorInvalidHostPointer;
        case cudaErrorInvalidKernelImage://???正确吗???
            return UPTKErrorInvalidKernelImage;
        case cudaErrorInvalidMemcpyDirection:
            return UPTKErrorInvalidMemcpyDirection;
//         case cudaErrorInvalidNormSetting:
//             return UPTKErrorInvalidNormSetting;
//         case cudaErrorInvalidPc:
//             return UPTKErrorInvalidPc;
        case cudaErrorInvalidPitchValue:
            return UPTKErrorInvalidPitchValue;
        case cudaErrorInvalidPtx://???正确吗???
            return UPTKErrorInvalidPtx;
        case cudaErrorInvalidResourceHandle://???正确吗???
             return UPTKErrorInvalidResourceHandle;
        case cudaErrorInvalidSource:
            return UPTKErrorInvalidSource;
//         case cudaErrorInvalidSurface:
//             return UPTKErrorInvalidSurface;
        case cudaErrorInvalidSymbol:
            return UPTKErrorInvalidSymbol;
//         case cudaErrorInvalidTexture:
//             return UPTKErrorInvalidTexture;
//         case cudaErrorInvalidTextureBinding:
//             return UPTKErrorInvalidTextureBinding;
        case cudaErrorInvalidValue:
            return UPTKErrorInvalidValue;
//         case cudaErrorJitCompilerNotFound:
//             return UPTKErrorJitCompilerNotFound;
        case cudaErrorLaunchFailure:
            return UPTKErrorLaunchFailure;
//         case cudaErrorLaunchFileScopedSurf:
//             return UPTKErrorLaunchFileScopedSurf;
//         case cudaErrorLaunchFileScopedTex:
//             return UPTKErrorLaunchFileScopedTex;
//         case cudaErrorLaunchIncompatibleTexturing:
//             return UPTKErrorLaunchIncompatibleTexturing;
//         case cudaErrorLaunchMaxDepthExceeded:
//             return UPTKErrorLaunchMaxDepthExceeded;
        case cudaErrorLaunchOutOfResources:
            return UPTKErrorLaunchOutOfResources;
//         case cudaErrorLaunchPendingCountExceeded:
//             return UPTKErrorLaunchPendingCountExceeded;
        case cudaErrorLaunchTimeout://???正确吗???
            return UPTKErrorLaunchTimeout;
        case cudaErrorMapBufferObjectFailed://???正确吗???
            return UPTKErrorMapBufferObjectFailed;
        case cudaErrorMemoryAllocation://???正确吗???
            return UPTKErrorMemoryAllocation;
//         case cudaErrorMemoryValueTooLarge:
//             return UPTKErrorMemoryValueTooLarge;
//         case cudaErrorMisalignedAddress:
//             return UPTKErrorMisalignedAddress;
        case cudaErrorMissingConfiguration:
            return UPTKErrorMissingConfiguration;
//         case cudaErrorMixedDeviceExecution:
//             return UPTKErrorMixedDeviceExecution;
        case cudaErrorNoDevice:
            return UPTKErrorNoDevice;
        case cudaErrorNoKernelImageForDevice://???正确吗???
            return UPTKErrorNoKernelImageForDevice;
        case cudaErrorNotMapped:
            return UPTKErrorNotMapped;
        case cudaErrorNotMappedAsArray:
            return UPTKErrorNotMappedAsArray;
        case cudaErrorNotMappedAsPointer:
            return UPTKErrorNotMappedAsPointer;
//         case cudaErrorNotPermitted:
//             return UPTKErrorNotPermitted;
        case cudaErrorNotReady:
            return UPTKErrorNotReady;
        case cudaErrorNotSupported:
            return UPTKErrorNotSupported;
//         case cudaErrorNotYetImplemented:
//             return UPTKErrorNotYetImplemented;
//         case cudaErrorNvlinkUncorrectable:
//             return UPTKErrorNvlinkUncorrectable;
        case cudaErrorOperatingSystem:
            return UPTKErrorOperatingSystem;
        case cudaErrorPeerAccessAlreadyEnabled:
            return UPTKErrorPeerAccessAlreadyEnabled;
        case cudaErrorPeerAccessNotEnabled:
            return UPTKErrorPeerAccessNotEnabled;
        case cudaErrorPeerAccessUnsupported:
            return UPTKErrorPeerAccessUnsupported;
        case cudaErrorPriorLaunchFailure:
            return UPTKErrorPriorLaunchFailure;
        case cudaErrorProfilerAlreadyStarted:
            return UPTKErrorProfilerAlreadyStarted;
        case cudaErrorProfilerAlreadyStopped:
            return UPTKErrorProfilerAlreadyStopped;
        case cudaErrorProfilerDisabled:
            return UPTKErrorProfilerDisabled;
        case cudaErrorProfilerNotInitialized:
            return UPTKErrorProfilerNotInitialized;
        case cudaErrorSetOnActiveProcess:
            return UPTKErrorSetOnActiveProcess;
        case cudaErrorSharedObjectInitFailed:
            return UPTKErrorSharedObjectInitFailed;
        case cudaErrorSharedObjectSymbolNotFound:
            return UPTKErrorSharedObjectSymbolNotFound;
//         case cudaErrorStartupFailure:
//             return UPTKErrorStartupFailure;
        case cudaErrorStreamCaptureImplicit:
            return UPTKErrorStreamCaptureImplicit;
        case cudaErrorStreamCaptureInvalidated:
            return UPTKErrorStreamCaptureInvalidated;
        case cudaErrorStreamCaptureIsolation:
            return UPTKErrorStreamCaptureIsolation;
        case cudaErrorStreamCaptureMerge:
            return UPTKErrorStreamCaptureMerge;
        case cudaErrorStreamCaptureUnjoined:
            return UPTKErrorStreamCaptureUnjoined;
        case cudaErrorStreamCaptureUnmatched:
            return UPTKErrorStreamCaptureUnmatched;
        case cudaErrorStreamCaptureUnsupported:
            return UPTKErrorStreamCaptureUnsupported;
        case cudaErrorStreamCaptureWrongThread:
            return UPTKErrorStreamCaptureWrongThread;
        case cudaErrorSymbolNotFound://???正确吗???
            return UPTKErrorSymbolNotFound;
//         case cudaErrorSyncDepthExceeded:
//             return UPTKErrorSyncDepthExceeded;
//         case cudaErrorSynchronizationError:
//             return UPTKErrorSynchronizationError;
//         case cudaErrorSystemDriverMismatch:
//             return UPTKErrorSystemDriverMismatch;
//         case cudaErrorSystemNotReady:
//             return UPTKErrorSystemNotReady;
//         case cudaErrorTextureFetchFailed:
//             return UPTKErrorTextureFetchFailed;
//         case cudaErrorTextureNotBound:
//             return UPTKErrorTextureNotBound;
//         case cudaErrorTimeout:
//             return UPTKErrorTimeout;
//         case cudaErrorTooManyPeers:
//             return UPTKErrorTooManyPeers;
        case cudaErrorUnknown:
        // case cudaErrorRuntimeOther://???正确吗???
        // case cudaErrorTbd://???正确吗???
            return UPTKErrorUnknown;
        case cudaErrorUnmapBufferObjectFailed://???正确吗???
            return UPTKErrorUnmapBufferObjectFailed;
        case cudaErrorUnsupportedLimit:
            return UPTKErrorUnsupportedLimit;
        // case cudaErrorInvalidResourcetype:
        //     return UPTKErrorInvalidResourceType;    
        case cudaSuccess:
            return UPTKSuccess;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaMemcpyKind UPTKMemcpyKindTocudaMemcpyKind(enum UPTKMemcpyKind para) {
    switch (para) {
        case UPTKMemcpyDefault:
            return cudaMemcpyDefault;
        case UPTKMemcpyDeviceToDevice:
            return cudaMemcpyDeviceToDevice;
        case UPTKMemcpyDeviceToHost:
            return cudaMemcpyDeviceToHost;
        case UPTKMemcpyHostToDevice:
            return cudaMemcpyHostToDevice;
        case UPTKMemcpyHostToHost:
            return cudaMemcpyHostToHost;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaStreamCaptureMode UPTKStreamCaptureModeTocudaStreamCaptureMode(enum UPTKStreamCaptureMode para) {
    switch (para) {
        case UPTKStreamCaptureModeGlobal:
            return cudaStreamCaptureModeGlobal;
        case UPTKStreamCaptureModeRelaxed:
            return cudaStreamCaptureModeRelaxed;
        case UPTKStreamCaptureModeThreadLocal:
            return cudaStreamCaptureModeThreadLocal;
        default:
            ERROR_INVALID_ENUM();
    }
}


enum UPTKStreamCaptureStatus cudaStreamCaptureStatusToUPTKStreamCaptureStatus(cudaStreamCaptureStatus para) {
    switch (para) {
        case cudaStreamCaptureStatusActive:
            return UPTKStreamCaptureStatusActive;
        case cudaStreamCaptureStatusInvalidated:
            return UPTKStreamCaptureStatusInvalidated;
        case cudaStreamCaptureStatusNone:
            return UPTKStreamCaptureStatusNone;
        default:
            ERROR_INVALID_ENUM();
    }
}

void UPTKIpcMemHandleTocudaIpcMemHandle(const UPTKIpcMemHandle_t * UPTK_para, cudaIpcMemHandle_t * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    int len = std::min(UPTK_IPC_HANDLE_SIZE, CU_IPC_HANDLE_SIZE);
    for (int i = 0; i < len; ++i) {
        cuda_para->reserved[i] = UPTK_para->reserved[i];
    }
}

//部分字段没有，直接删掉了
void UPTKDevicePropTocudaDeviceProp(const struct UPTKDeviceProp *UPTK_para, cudaDeviceProp *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    strncpy(cuda_para->name, UPTK_para->name, 256);
    cuda_para->totalGlobalMem = UPTK_para->totalGlobalMem;
    cuda_para->sharedMemPerBlock = UPTK_para->sharedMemPerBlock;
    cuda_para->regsPerBlock = UPTK_para->regsPerBlock;
    cuda_para->warpSize = UPTK_para->warpSize;
    cuda_para->maxThreadsPerBlock = UPTK_para->maxThreadsPerBlock;
    for (int i = 0; i < 3; i++) {
        cuda_para->maxThreadsDim[i] = UPTK_para->maxThreadsDim[i];
        cuda_para->maxGridSize[i] = UPTK_para->maxGridSize[i];
    }

    cuda_para->clockRate = UPTK_para->clockRate;
    cuda_para->memoryClockRate = UPTK_para->memoryClockRate;
    cuda_para->memoryBusWidth = UPTK_para->memoryBusWidth;
    cuda_para->totalConstMem = UPTK_para->totalConstMem;
    cuda_para->major = UPTK_para->major;
    cuda_para->minor = UPTK_para->minor;
    cuda_para->multiProcessorCount = UPTK_para->multiProcessorCount;
    cuda_para->l2CacheSize = UPTK_para->l2CacheSize;
    cuda_para->maxThreadsPerMultiProcessor = UPTK_para->maxThreadsPerMultiProcessor;
    cuda_para->computeMode = UPTK_para->computeMode;
    cuda_para->concurrentKernels = UPTK_para->concurrentKernels;
    cuda_para->pciDomainID = UPTK_para->pciDomainID;
    cuda_para->pciBusID = UPTK_para->pciBusID;
    cuda_para->pciDeviceID = UPTK_para->pciDeviceID;
    cuda_para->sharedMemPerMultiprocessor = UPTK_para->sharedMemPerMultiprocessor;
    cuda_para->isMultiGpuBoard = UPTK_para->isMultiGpuBoard;
    cuda_para->canMapHostMemory = UPTK_para->canMapHostMemory;
    cuda_para->integrated = UPTK_para->integrated;
    cuda_para->cooperativeLaunch = UPTK_para->cooperativeLaunch;
    cuda_para->cooperativeMultiDeviceLaunch = UPTK_para->cooperativeMultiDeviceLaunch;
    cuda_para->maxTexture1D    = UPTK_para->maxTexture1D;
    cuda_para->maxTexture2D[0] = UPTK_para->maxTexture2D[0];
    cuda_para->maxTexture2D[1] = UPTK_para->maxTexture2D[1];
    cuda_para->maxTexture3D[0] = UPTK_para->maxTexture3D[0];
    cuda_para->maxTexture3D[1] = UPTK_para->maxTexture3D[1];
    cuda_para->maxTexture3D[2] = UPTK_para->maxTexture3D[2];
    cuda_para->memPitch = UPTK_para->memPitch;
    cuda_para->textureAlignment = UPTK_para->textureAlignment;
    cuda_para->texturePitchAlignment = UPTK_para->texturePitchAlignment;
    cuda_para->kernelExecTimeoutEnabled = UPTK_para->kernelExecTimeoutEnabled;
    cuda_para->ECCEnabled = UPTK_para->ECCEnabled;
    cuda_para->tccDriver = UPTK_para->tccDriver;
}


//cudaDeviceProp这个数据结构中没有gcnArchName这个成员
void cudaDeviceProp_v2ToUPTKDeviceProp(const cudaDeviceProp * cuda_para, struct UPTKDeviceProp * UPTK_para) {
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    // Add the gcnArchName of cuda to the device name
    strncpy(UPTK_para->name, cuda_para->name, sizeof(UPTK_para->name) - 1);
    UPTK_para->name[sizeof(UPTK_para->name) - 1] = '\0'; 
    //没有对应的gcnArchName
    // if (strlen(UPTK_para->name) + strlen(cuda_para->gcnArchName) < sizeof(UPTK_para->name) - 1){
    //     strncat(UPTK_para->name, " ", sizeof(UPTK_para->name) - strlen(UPTK_para->name) - 1); 
    //     strncat(UPTK_para->name, cuda_para->gcnArchName, sizeof(UPTK_para->name) - strlen(UPTK_para->name) - 1); 
    // }
    // else {
    //     fprintf(stderr, "Error: UPTK_para->name array is not large enough to hold the result.\n");
    // }
    UPTK_para->totalGlobalMem = cuda_para->totalGlobalMem;
    UPTK_para->sharedMemPerBlock = cuda_para->sharedMemPerBlock;
    UPTK_para->regsPerBlock = cuda_para->regsPerBlock;
    UPTK_para->warpSize = cuda_para->warpSize;
    UPTK_para->maxThreadsPerBlock = cuda_para->maxThreadsPerBlock;
    for (int i = 0; i < 3; i++) {
        UPTK_para->maxThreadsDim[i] = cuda_para->maxThreadsDim[i];
        UPTK_para->maxGridSize[i] = cuda_para->maxGridSize[i];
    }
    UPTK_para->clockRate = cuda_para->clockRate;
    UPTK_para->memoryClockRate = cuda_para->memoryClockRate;
    UPTK_para->memoryBusWidth = cuda_para->memoryBusWidth;
    UPTK_para->totalConstMem = cuda_para->totalConstMem;
    UPTK_para->major = SM_VERSION_MAJOR;   // not equal to hip
    UPTK_para->minor = SM_VERSION_MINOR;   // not equal to hip
    UPTK_para->multiProcessorCount = cuda_para->multiProcessorCount;
    UPTK_para->l2CacheSize = cuda_para->l2CacheSize;
    UPTK_para->maxThreadsPerMultiProcessor = cuda_para->maxThreadsPerMultiProcessor;
    UPTK_para->computeMode = cuda_para->computeMode;

    UPTK_para->concurrentKernels = cuda_para->concurrentKernels;
    UPTK_para->pciDomainID = cuda_para->pciDomainID;
    UPTK_para->pciBusID = cuda_para->pciBusID;
    UPTK_para->pciDeviceID = cuda_para->pciDeviceID;
    
    // Only sharedMemPerMultiprocessor UPTKDeviceProp members
    // Two members of the cudaDeviceProp_t_v2 are sharedMemPerMultiprocessor, maxSharedMemoryPerMultiProcessor
    // In the verification to the environment in P100 NV, UPTK sharedMemPerMultiprocessor maxSharedMemoryPerMultiProcessor should correspond to hip
    UPTK_para->sharedMemPerMultiprocessor = cuda_para->sharedMemPerMultiprocessor;

    UPTK_para->isMultiGpuBoard = cuda_para->isMultiGpuBoard;
    UPTK_para->canMapHostMemory = cuda_para->canMapHostMemory;
    UPTK_para->integrated = cuda_para->integrated;
    UPTK_para->cooperativeLaunch = cuda_para->cooperativeLaunch;
    UPTK_para->cooperativeMultiDeviceLaunch = cuda_para->cooperativeMultiDeviceLaunch;

    UPTK_para->maxTexture1D    = cuda_para->maxTexture1D;
    UPTK_para->maxTexture2D[0] = cuda_para->maxTexture2D[0];
    UPTK_para->maxTexture2D[1] = cuda_para->maxTexture2D[1];
    UPTK_para->maxTexture3D[0] = cuda_para->maxTexture3D[0];
    UPTK_para->maxTexture3D[1] = cuda_para->maxTexture3D[1];
    UPTK_para->maxTexture3D[2] = cuda_para->maxTexture3D[2];

    UPTK_para->memPitch                 = cuda_para->memPitch;
    UPTK_para->textureAlignment         = cuda_para->textureAlignment;
    UPTK_para->texturePitchAlignment    = cuda_para->texturePitchAlignment;
    UPTK_para->kernelExecTimeoutEnabled = cuda_para->kernelExecTimeoutEnabled;
    UPTK_para->ECCEnabled               = cuda_para->ECCEnabled;
    UPTK_para->tccDriver                = cuda_para->tccDriver;

    UPTK_para->regsPerMultiprocessor = cuda_para->regsPerMultiprocessor;
    UPTK_para->maxBlocksPerMultiProcessor = cuda_para->maxBlocksPerMultiProcessor;

    memcpy(&(UPTK_para->uuid), &(cuda_para->uuid), sizeof(UPTKUUID_t));
    for(int i = 0; i < 8; i++){
        UPTK_para->luid[i] = cuda_para->luid[i];
    }
    UPTK_para->luidDeviceNodeMask = cuda_para->luidDeviceNodeMask;
    UPTK_para->deviceOverlap = cuda_para->deviceOverlap;
    UPTK_para->maxTexture1DMipmap = cuda_para->maxTexture1DMipmap;
    UPTK_para->maxTexture1DLinear = cuda_para->maxTexture1DLinear;
    
    for(int i = 0; i < 2; i++){
    UPTK_para->maxTexture2DMipmap[i] = cuda_para->maxTexture2DMipmap[i];
    UPTK_para->maxTexture2DGather[i] = cuda_para->maxTexture2DGather[i];
    UPTK_para->maxTexture1DLayered[i] = cuda_para->maxTexture1DLayered[i];
    UPTK_para->maxTextureCubemapLayered[i] = cuda_para->maxTextureCubemapLayered[i];
    UPTK_para->maxSurface2D[i] = cuda_para->maxSurface2D[i];
    UPTK_para->maxSurface1DLayered[i] = cuda_para->maxSurface1DLayered[i];
    UPTK_para->maxSurfaceCubemapLayered[i] = cuda_para->maxSurfaceCubemapLayered[i];
        
    }

    UPTK_para->maxSurface1D = cuda_para->maxSurface1D;

    for(int i = 0; i < 3; i++){
    UPTK_para->maxTexture2DLinear[i] = cuda_para->maxTexture2DLinear[i];
    UPTK_para->maxTexture3DAlt[i] = cuda_para->maxTexture3DAlt[i];
    UPTK_para->maxTextureCubemap = cuda_para->maxTextureCubemap;
    UPTK_para->maxTexture2DLayered[i] = cuda_para->maxTexture2DLayered[i];
    UPTK_para->maxSurface3D[i] = cuda_para->maxSurface3D[i];
    UPTK_para->maxSurface2DLayered[i] = cuda_para->maxSurface2DLayered[i];
    }

    UPTK_para->maxSurfaceCubemap = cuda_para->maxSurfaceCubemap;
    UPTK_para->surfaceAlignment = cuda_para->surfaceAlignment;
    UPTK_para->asyncEngineCount = cuda_para->asyncEngineCount;
    // UPTK_para->unifiedAddressing = cuda_para->unifiedAddressing;
    //  UPTK sample: simpleIPC determines whether the device supports this attribute in the code. 
    //  The underlying hardware does not support unifiedAddressing, but setting unifiedAddressing to 1 does not affect the sample
    UPTK_para->unifiedAddressing = 1;
    UPTK_para->persistingL2CacheMaxSize = cuda_para->persistingL2CacheMaxSize;
    UPTK_para->streamPrioritiesSupported = cuda_para->streamPrioritiesSupported;
    UPTK_para->globalL1CacheSupported = cuda_para->globalL1CacheSupported;
    UPTK_para->localL1CacheSupported = cuda_para->localL1CacheSupported;
    UPTK_para->managedMemory = cuda_para->managedMemory;
    UPTK_para->multiGpuBoardGroupID = cuda_para->multiGpuBoardGroupID;
    UPTK_para->hostNativeAtomicSupported = cuda_para->hostNativeAtomicSupported;
    UPTK_para->singleToDoublePrecisionPerfRatio = cuda_para->singleToDoublePrecisionPerfRatio;
    UPTK_para->pageableMemoryAccess = cuda_para->pageableMemoryAccess;
    UPTK_para->concurrentManagedAccess = cuda_para->concurrentManagedAccess;
    UPTK_para->computePreemptionSupported = cuda_para->computePreemptionSupported;
    UPTK_para->canUseHostPointerForRegisteredMem = cuda_para->canUseHostPointerForRegisteredMem;
    // UPTK_para->sharedMemPerBlockOptin = cuda_para->sharedMemPerBlockOptin;
    // hip sharedMemPerBlockOptin member returns 0, the actual function is not supported. Replace with the close attribute sharedMemPerBlock.
    UPTK_para->sharedMemPerBlockOptin = cuda_para->sharedMemPerBlock;
    UPTK_para->pageableMemoryAccessUsesHostPageTables = cuda_para->pageableMemoryAccessUsesHostPageTables;
    UPTK_para->directManagedMemAccessFromHost = cuda_para->directManagedMemAccessFromHost;
    UPTK_para->accessPolicyMaxWindowSize = cuda_para->accessPolicyMaxWindowSize;
    UPTK_para->reservedSharedMemPerBlock = cuda_para->reservedSharedMemPerBlock;
    // UPTK 12.6.2:added some members to the UPTKDeviceProp structure.
    /*UPTK_para->hostRegisterSupported = cuda_para->hostRegisterSupported;
    UPTK_para->sparseUPTKArraySupported = cuda_para->sparseCudaArraySupported;
    UPTK_para->hostRegisterReadOnlySupported = cuda_para->hostRegisterReadOnlySupported;
    UPTK_para->timelineSemaphoreInteropSupported = cuda_para->timelineSemaphoreInteropSupported;
    UPTK_para->memoryPoolsSupported = cuda_para->memoryPoolsSupported;
    UPTK_para->gpuDirectRDMASupported = cuda_para->gpuDirectRDMASupported;
    UPTK_para->gpuDirectRDMAFlushWritesOptions = cuda_para->gpuDirectRDMAFlushWritesOptions;
    UPTK_para->gpuDirectRDMAWritesOrdering = cuda_para->gpuDirectRDMAWritesOrdering;
    UPTK_para->memoryPoolSupportedHandleTypes = cuda_para->memoryPoolSupportedHandleTypes;
    UPTK_para->deferredMappingUPTKArraySupported = cuda_para->deferredMappingCudaArraySupported;
    UPTK_para->ipcEventSupported = cuda_para->ipcEventSupported;
    UPTK_para->clusterLaunch = cuda_para->clusterLaunch;
    UPTK_para->unifiedFunctionPointers = cuda_para->unifiedFunctionPointers;
    memcpy(UPTK_para->reserved2, cuda_para->reserved, 2 * sizeof(int));       
    memcpy(UPTK_para->reserved1, cuda_para->reserved + 2, 1 * sizeof(int));           
    memcpy(UPTK_para->reserved, cuda_para->reserved + 3, 60 * sizeof(int));*/
}

//cudaFuncCache_t,这个数据结构不太对应，用的是cudaFuncCache
enum UPTKFuncCache cudaFuncCacheToUPTKFuncCache(cudaFuncCache para) {
    switch (para) {
        case cudaFuncCachePreferEqual:
            return UPTKFuncCachePreferEqual;
        case cudaFuncCachePreferL1:
            return UPTKFuncCachePreferL1;
        case cudaFuncCachePreferNone:
            return UPTKFuncCachePreferNone;
        case cudaFuncCachePreferShared:
            return UPTKFuncCachePreferShared;
        default:
            ERROR_INVALID_ENUM();
    }
}

// 将UPTKlimit->CUlimit来转换
CUlimit UPTKlimitToCUlimit(UPTKlimit para) {
    switch (para) {
        case UPTK_LIMIT_MAX:
            return CU_LIMIT_MAX;
        case UPTK_LIMIT_MALLOC_HEAP_SIZE:
            return CU_LIMIT_MALLOC_HEAP_SIZE;
        case UPTK_LIMIT_PRINTF_FIFO_SIZE:
            return CU_LIMIT_PRINTF_FIFO_SIZE;
        case UPTK_LIMIT_STACK_SIZE:
            return CU_LIMIT_STACK_SIZE;
        default:
            ERROR_INVALID_ENUM();
    }
}

void UPTKKernelNodeParamsTocudaKernelNodeParams(const struct UPTKKernelNodeParams * UPTK_para, cudaKernelNodeParams * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->blockDim = UPTK_para->blockDim;
    cuda_para->extra = UPTK_para->extra;
    cuda_para->func = UPTK_para->func;
    cuda_para->gridDim = UPTK_para->gridDim;
    cuda_para->kernelParams = UPTK_para->kernelParams;
    cuda_para->sharedMemBytes = UPTK_para->sharedMemBytes;
}


void UPTKMemcpy3DParmsTocudaMemcpy3DParms(const struct UPTKMemcpy3DParms * UPTK_para, cudaMemcpy3DParms * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->dstArray = (cudaArray_t)UPTK_para->dstArray;
    UPTKPosTocudaPos(&(UPTK_para->dstPos), &(cuda_para->dstPos));
    UPTKPitchedPtrTocudaPitchedPtr(&(UPTK_para->dstPtr), &(cuda_para->dstPtr));
    UPTKExtentTocudaExtent(&(UPTK_para->extent), &(cuda_para->extent));
    cuda_para->kind = UPTKMemcpyKindTocudaMemcpyKind(UPTK_para->kind);
    cuda_para->srcArray = (cudaArray_t)UPTK_para->srcArray;
    UPTKPosTocudaPos(&(UPTK_para->srcPos), &(cuda_para->srcPos));
    UPTKPitchedPtrTocudaPitchedPtr(&(UPTK_para->srcPtr), &(cuda_para->srcPtr));
}


void UPTKPosTocudaPos(const struct UPTKPos * UPTK_para, cudaPos * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->x = UPTK_para->x;
    cuda_para->y = UPTK_para->y;
    cuda_para->z = UPTK_para->z;
}


void UPTKPitchedPtrTocudaPitchedPtr(const struct UPTKPitchedPtr * UPTK_para, cudaPitchedPtr * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->pitch = UPTK_para->pitch;
    cuda_para->ptr = UPTK_para->ptr;
    cuda_para->xsize = UPTK_para->xsize;
    cuda_para->ysize = UPTK_para->ysize;
}

void UPTKExtentTocudaExtent(const struct UPTKExtent * UPTK_para, cudaExtent * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->depth = UPTK_para->depth;
    cuda_para->height = UPTK_para->height;
    cuda_para->width = UPTK_para->width;
}

enum UPTKGraphExecUpdateResult cudaGraphExecUpdateResultToUPTKGraphExecUpdateResult(cudaGraphExecUpdateResult para) {
    switch (para) {
        case cudaGraphExecUpdateError:
            return UPTKGraphExecUpdateError;
        case cudaGraphExecUpdateErrorFunctionChanged:
            return UPTKGraphExecUpdateErrorFunctionChanged;
        case cudaGraphExecUpdateErrorNodeTypeChanged:
            return UPTKGraphExecUpdateErrorNodeTypeChanged;
        case cudaGraphExecUpdateErrorNotSupported:
            return UPTKGraphExecUpdateErrorNotSupported;
        case cudaGraphExecUpdateErrorParametersChanged:
            return UPTKGraphExecUpdateErrorParametersChanged;
        case cudaGraphExecUpdateErrorTopologyChanged:
            return UPTKGraphExecUpdateErrorTopologyChanged;
        case cudaGraphExecUpdateErrorUnsupportedFunctionChange:
            return UPTKGraphExecUpdateErrorUnsupportedFunctionChange;
        case cudaGraphExecUpdateSuccess:
            return UPTKGraphExecUpdateSuccess;
        default:
            ERROR_INVALID_ENUM();
    }
}

/*void UPTKLaunchConfig_tTocudaLaunchConfig_t(const UPTKLaunchConfig_t * UPTK_para, cudaLaunchConfig_t * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->gridDim = UPTK_para->gridDim;
    cuda_para->blockDim = UPTK_para->blockDim;
    cuda_para->dynamicSmemBytes = UPTK_para->dynamicSmemBytes;
    cuda_para->stream = (cudaStream_t)UPTK_para->stream;

    cuda_para->numAttrs = UPTK_para->numAttrs;
    cudaLaunchAttribute* cuda_attrs = (cudaLaunchAttribute*)malloc(sizeof(cudaLaunchAttribute) * cuda_para->numAttrs);
    // attrs is a nullable pointer. if numAttrs == 0, attrs can be nullptr.
    if (UPTK_para->attrs != nullptr)
    {
        for (int i = 0; i < cuda_para->numAttrs; i++)
        {
            cuda_attrs[i] = UPTKLaunchAttributeTocudaLaunchAttribute(UPTK_para->attrs[i]);
        }
    }
    cuda_para->attrs = cuda_attrs;
}*/

/*cudaClusterSchedulingPolicy UPTKClusterSchedulingPolicyTocudaClusterSchedulingPolicy(UPTKClusterSchedulingPolicy para)
{
    switch(para){
        case UPTKClusterSchedulingPolicyDefault:
            return cudaClusterSchedulingPolicyDefault;
        case UPTKClusterSchedulingPolicySpread:
            return cudaClusterSchedulingPolicySpread;
        case UPTKClusterSchedulingPolicyLoadBalancing:
            return cudaClusterSchedulingPolicyLoadBalancing;
        default:
            ERROR_INVALID_ENUM();
    }
}*/

enum cudaSynchronizationPolicy UPTKSynchronizationPolicyTocudaSynchronizationPolicy(enum UPTKSynchronizationPolicy para) {
    switch (para) {
        case UPTKSyncPolicyAuto:
            return cudaSyncPolicyAuto;
        case UPTKSyncPolicyBlockingSync:
            return cudaSyncPolicyBlockingSync;
        case UPTKSyncPolicySpin:
            return cudaSyncPolicySpin;
        case UPTKSyncPolicyYield:
            return cudaSyncPolicyYield;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaAccessProperty UPTKAccessPropertyTocudaAccessProperty(enum UPTKAccessProperty para) {
    switch (para) {
        case UPTKAccessPropertyNormal:
            return cudaAccessPropertyNormal;
        case UPTKAccessPropertyPersisting:
            return cudaAccessPropertyPersisting;
        case UPTKAccessPropertyStreaming:
            return cudaAccessPropertyStreaming;
        default:
              ERROR_INVALID_ENUM();
    }
}

void UPTKAccessPolicyWindowTocudaAccessPolicyWindow(const struct UPTKAccessPolicyWindow * UPTK_para, cudaAccessPolicyWindow * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->base_ptr = UPTK_para->base_ptr;
    cuda_para->hitProp = UPTKAccessPropertyTocudaAccessProperty(UPTK_para->hitProp);
    cuda_para->hitRatio = UPTK_para->hitRatio;
    cuda_para->missProp = UPTKAccessPropertyTocudaAccessProperty(UPTK_para->missProp);
    cuda_para->num_bytes = UPTK_para->num_bytes;
}

/*void UPTKLaunchAttributeValueTocudaLaunchAttributeValue(const UPTKLaunchAttributeValue *UPTK_para, cudaLaunchAttributeValue *cuda_para, UPTKLaunchAttributeID para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    switch (para)
    {
    case UPTKLaunchAttributeCooperative: // hip underlying hardware support
        cuda_para->cooperative = UPTK_para->cooperative;
        break;
    case UPTKLaunchAttributeAccessPolicyWindow: // hip underlying hardware support
        UPTKAccessPolicyWindowTocudaAccessPolicyWindow(&(UPTK_para->accessPolicyWindow), &(cuda_para->accessPolicyWindow));
        break;
    case UPTKLaunchAttributeIgnore:
        memcpy(cuda_para->pad, UPTK_para->pad, sizeof(cuda_para->pad));
        break;
    case UPTKLaunchAttributeSynchronizationPolicy:
        cuda_para->syncPolicy = UPTKSynchronizationPolicyTocudaSynchronizationPolicy(UPTK_para->syncPolicy);
        break;
    case UPTKLaunchAttributeClusterDimension:
        memcpy(&(cuda_para->clusterDim), &(UPTK_para->clusterDim), sizeof(cuda_para->clusterDim));
        break;
    case UPTKLaunchAttributeClusterSchedulingPolicyPreference:
        cuda_para->clusterSchedulingPolicyPreference = UPTKClusterSchedulingPolicyTocudaClusterSchedulingPolicy(UPTK_para->clusterSchedulingPolicyPreference);
        break;
    case UPTKLaunchAttributeProgrammaticStreamSerialization:
        cuda_para->programmaticStreamSerializationAllowed = UPTK_para->programmaticStreamSerializationAllowed;
        break;
    case UPTKLaunchAttributeProgrammaticEvent:
        cuda_para->programmaticEvent.event = (cudaEvent_t)UPTK_para->programmaticEvent.event;
        cuda_para->programmaticEvent.flags = UPTK_para->programmaticEvent.flags;
        cuda_para->programmaticEvent.triggerAtBlockStart = UPTK_para->programmaticEvent.triggerAtBlockStart;
        break;
    case UPTKLaunchAttributePriority:
        cuda_para->priority = UPTK_para->priority;
        break;
    default:
        ERROR_INVALID_ENUM();
        break;
    }
}*/

// TODO: UPTKLaunchAttribute generally appears in the form of an array, so special conversion is performed here.
/*cudaLaunchAttribute UPTKLaunchAttributeTocudaLaunchAttribute(UPTKLaunchAttribute UPTK_para)
{
    // if (nullptr == UPTK_para || nullptr == cuda_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    cudaLaunchAttribute cuda_para;
    cuda_para.id = UPTKLaunchAttributeIDTocudaLaunchAttributeID(UPTK_para.id);
    UPTKLaunchAttributeValueTocudaLaunchAttributeValue(&(UPTK_para.val), &(cuda_para.val), UPTK_para.id);
    // Note: Padding in UPTKLaunchAttribute is ignored during conversion
    return cuda_para;
}*/

/*cudaLaunchAttributeID UPTKLaunchAttributeIDTocudaLaunchAttributeID(enum UPTKLaunchAttributeID para)
{
    switch(para) {
        case UPTKLaunchAttributeCooperative:
            return cudaLaunchAttributeCooperative;                  // hip underlying hardware support
        case UPTKLaunchAttributeSynchronizationPolicy:
            return cudaLaunchAttributeSynchronizationPolicy;        // hip underlying hardware support
        case UPTKLaunchAttributeIgnore:
            return cudaLaunchAttributeIgnore;
        case UPTKLaunchAttributeAccessPolicyWindow:
            return cudaLaunchAttributeAccessPolicyWindow;      
        case UPTKLaunchAttributeClusterDimension:
            return cudaLaunchAttributeClusterDimension;
        case UPTKLaunchAttributeClusterSchedulingPolicyPreference:
            return cudaLaunchAttributeClusterSchedulingPolicyPreference;
        case UPTKLaunchAttributeProgrammaticStreamSerialization:
            return cudaLaunchAttributeProgrammaticStreamSerialization;
        case UPTKLaunchAttributeProgrammaticEvent:
            return cudaLaunchAttributeProgrammaticEvent;
        case UPTKLaunchAttributePriority:
            return cudaLaunchAttributePriority;
        default:
            ERROR_INVALID_ENUM();                 
    }
}*/

void cudaFuncAttributesToUPTKFuncAttributes(const cudaFuncAttributes * cuda_para, struct UPTKFuncAttributes * UPTK_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    UPTK_para->binaryVersion = cuda_para->binaryVersion;
    UPTK_para->cacheModeCA = cuda_para->cacheModeCA;
    UPTK_para->constSizeBytes = cuda_para->constSizeBytes;
    UPTK_para->localSizeBytes = cuda_para->localSizeBytes;
    UPTK_para->maxDynamicSharedSizeBytes = cuda_para->maxDynamicSharedSizeBytes;
    UPTK_para->maxThreadsPerBlock = cuda_para->maxThreadsPerBlock;
    UPTK_para->numRegs = cuda_para->numRegs;
    UPTK_para->preferredShmemCarveout = cuda_para->preferredShmemCarveout;
    UPTK_para->ptxVersion = cuda_para->ptxVersion;
    UPTK_para->sharedSizeBytes = cuda_para->sharedSizeBytes;
}

UPTKresult cudaErrorToUPTKresult(CUresult para) {
    switch (para) {
        case CUDA_ERROR_ALREADY_ACQUIRED:
            return UPTK_ERROR_ALREADY_ACQUIRED;
        case CUDA_ERROR_ALREADY_MAPPED:
            return UPTK_ERROR_ALREADY_MAPPED;
        case CUDA_ERROR_ARRAY_IS_MAPPED:
            return UPTK_ERROR_ARRAY_IS_MAPPED;
        case CUDA_ERROR_ASSERT:
            return UPTK_ERROR_ASSERT;
        case CUDA_ERROR_CAPTURED_EVENT:
            return UPTK_ERROR_CAPTURED_EVENT;
        case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
            return UPTK_ERROR_CONTEXT_ALREADY_CURRENT;
        case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
            return UPTK_ERROR_CONTEXT_ALREADY_IN_USE;
        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
            return UPTK_ERROR_CONTEXT_IS_DESTROYED;
        case CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE:
            return UPTK_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE;
        case CUDA_ERROR_DEINITIALIZED:
             return UPTK_ERROR_DEINITIALIZED;
        case CUDA_ERROR_ECC_UNCORRECTABLE:
             return UPTK_ERROR_ECC_UNCORRECTABLE;
        case CUDA_ERROR_FILE_NOT_FOUND:
            return UPTK_ERROR_FILE_NOT_FOUND;
        case CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE:
            return UPTK_ERROR_GRAPH_EXEC_UPDATE_FAILURE;
//         case cudaErrorHardwareStackError:
//             return UPTK_ERROR_HARDWARE_STACK_ERROR;
        case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
            return UPTK_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
        case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
            return UPTK_ERROR_HOST_MEMORY_NOT_REGISTERED;
        case CUDA_ERROR_ILLEGAL_ADDRESS:
            return UPTK_ERROR_ILLEGAL_ADDRESS;
        case CUDA_ERROR_ILLEGAL_INSTRUCTION:
            return UPTK_ERROR_ILLEGAL_INSTRUCTION;
        case CUDA_ERROR_ILLEGAL_STATE:
            return UPTK_ERROR_ILLEGAL_STATE;
//         case cudaErrorInvalidAddressSpace:
//             return UPTK_ERROR_INVALID_ADDRESS_SPACE;
        case CUDA_ERROR_INVALID_CONTEXT:
            return UPTK_ERROR_INVALID_CONTEXT;
        case CUDA_ERROR_INVALID_DEVICE:
            return UPTK_ERROR_INVALID_DEVICE;
        case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
            return UPTK_ERROR_INVALID_GRAPHICS_CONTEXT;
        case CUDA_ERROR_INVALID_HANDLE:
            return UPTK_ERROR_INVALID_HANDLE;
        case CUDA_ERROR_INVALID_IMAGE:
            return UPTK_ERROR_INVALID_IMAGE;
//         case cudaErrorInvalidPc:
//             return UPTK_ERROR_INVALID_PC;
        case CUDA_ERROR_INVALID_PTX:
            return UPTK_ERROR_INVALID_PTX;
        case CUDA_ERROR_INVALID_SOURCE:
            return UPTK_ERROR_INVALID_SOURCE;
        case CUDA_ERROR_INVALID_VALUE:
            return UPTK_ERROR_INVALID_VALUE;
//         case cudaErrorJitCompilerNotFound:
//             return UPTK_ERROR_JIT_COMPILER_NOT_FOUND;
        case CUDA_ERROR_LAUNCH_FAILED:
            return UPTK_ERROR_LAUNCH_FAILED;
//         case cudaErrorLaunchIncompatibleTexturing:
//             return UPTK_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING;
        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
            return UPTK_ERROR_LAUNCH_OUT_OF_RESOURCES;
        case CUDA_ERROR_LAUNCH_TIMEOUT:
            return UPTK_ERROR_LAUNCH_TIMEOUT;
        case CUDA_ERROR_MAP_FAILED:
            return UPTK_ERROR_MAP_FAILED;
//         case cudaErrorMisalignedAddress:
//             return UPTK_ERROR_MISALIGNED_ADDRESS;
        case CUDA_ERROR_NOT_FOUND:
            return UPTK_ERROR_NOT_FOUND;
        case CUDA_ERROR_NOT_INITIALIZED:
            return UPTK_ERROR_NOT_INITIALIZED;
        case CUDA_ERROR_NOT_MAPPED:
            return UPTK_ERROR_NOT_MAPPED;
        case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
            return UPTK_ERROR_NOT_MAPPED_AS_ARRAY;
        case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
            return UPTK_ERROR_NOT_MAPPED_AS_POINTER;
//         case cudaErrorNotPermitted:
//             return UPTK_ERROR_NOT_PERMITTED;
        case CUDA_ERROR_NOT_READY:
            return UPTK_ERROR_NOT_READY;
        case CUDA_ERROR_NOT_SUPPORTED:
            return UPTK_ERROR_NOT_SUPPORTED;
        case CUDA_ERROR_NO_BINARY_FOR_GPU:
            return UPTK_ERROR_NO_BINARY_FOR_GPU;
        case CUDA_ERROR_NO_DEVICE:
            return UPTK_ERROR_NO_DEVICE;
//         case cudaErrorNvlinkUncorrectable:
//             return UPTK_ERROR_NVLINK_UNCORRECTABLE;
        case CUDA_ERROR_OPERATING_SYSTEM:
            return UPTK_ERROR_OPERATING_SYSTEM;
        case CUDA_ERROR_OUT_OF_MEMORY:
            return UPTK_ERROR_OUT_OF_MEMORY;
        case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
            return UPTK_ERROR_PEER_ACCESS_ALREADY_ENABLED;
        case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
            return UPTK_ERROR_PEER_ACCESS_NOT_ENABLED;
        case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
            return UPTK_ERROR_PEER_ACCESS_UNSUPPORTED;
        case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
            return UPTK_ERROR_PRIMARY_CONTEXT_ACTIVE;
        case CUDA_ERROR_PROFILER_ALREADY_STARTED:
            return UPTK_ERROR_PROFILER_ALREADY_STARTED;
        case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
            return UPTK_ERROR_PROFILER_ALREADY_STOPPED;
        case CUDA_ERROR_PROFILER_DISABLED:
            return UPTK_ERROR_PROFILER_DISABLED;
        case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
            return UPTK_ERROR_PROFILER_NOT_INITIALIZED;
        case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
            return UPTK_ERROR_SHARED_OBJECT_INIT_FAILED;
        case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
            return UPTK_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
        case CUDA_ERROR_STREAM_CAPTURE_IMPLICIT:
            return UPTK_ERROR_STREAM_CAPTURE_IMPLICIT;
        case CUDA_ERROR_STREAM_CAPTURE_INVALIDATED:
            return UPTK_ERROR_STREAM_CAPTURE_INVALIDATED;
        case CUDA_ERROR_STREAM_CAPTURE_ISOLATION:
            return UPTK_ERROR_STREAM_CAPTURE_ISOLATION;
        case CUDA_ERROR_STREAM_CAPTURE_MERGE:
            return UPTK_ERROR_STREAM_CAPTURE_MERGE;
        case CUDA_ERROR_STREAM_CAPTURE_UNJOINED:
            return UPTK_ERROR_STREAM_CAPTURE_UNJOINED;
        case CUDA_ERROR_STREAM_CAPTURE_UNMATCHED:
            return UPTK_ERROR_STREAM_CAPTURE_UNMATCHED;
        case CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED:
            return UPTK_ERROR_STREAM_CAPTURE_UNSUPPORTED;
        case CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD:
            return UPTK_ERROR_STREAM_CAPTURE_WRONG_THREAD;
//         case cudaErrorSystemDriverMismatch:
//             return UPTK_ERROR_SYSTEM_DRIVER_MISMATCH;
//         case cudaErrorSystemNotReady:
//             return UPTK_ERROR_SYSTEM_NOT_READY;
//         case cudaErrorTimeout:
//             return UPTK_ERROR_TIMEOUT;
//         case cudaErrorTooManyPeers:
//             return UPTK_ERROR_TOO_MANY_PEERS;
        case CUDA_ERROR_UNKNOWN:
            return UPTK_ERROR_UNKNOWN;
        case CUDA_ERROR_UNMAP_FAILED:
            return UPTK_ERROR_UNMAP_FAILED;
        case CUDA_ERROR_UNSUPPORTED_LIMIT:
            return UPTK_ERROR_UNSUPPORTED_LIMIT;
        //case CUDA_ERROR_INVALID_RESOURCE_TYPE:
        //    return UPTK_ERROR_INVALID_RESOURCE_TYPE;     
        case CUDA_SUCCESS:
            return UPTK_SUCCESS;
        default:
            ERROR_INVALID_ENUM();
    }
}

CUarray_format UPTKarray_formatToCUarray_format(UPTKarray_format para) {
    switch (para) {
        case UPTK_AD_FORMAT_FLOAT:
            return CU_AD_FORMAT_FLOAT;
        case UPTK_AD_FORMAT_HALF:
            return CU_AD_FORMAT_HALF;
        case UPTK_AD_FORMAT_SIGNED_INT8:
            return CU_AD_FORMAT_SIGNED_INT8;
        case UPTK_AD_FORMAT_SIGNED_INT16:
            return CU_AD_FORMAT_SIGNED_INT16;
        case UPTK_AD_FORMAT_SIGNED_INT32:
            return CU_AD_FORMAT_SIGNED_INT32;
        case UPTK_AD_FORMAT_UNSIGNED_INT8:
            return CU_AD_FORMAT_UNSIGNED_INT8;
        case UPTK_AD_FORMAT_UNSIGNED_INT16:
            return CU_AD_FORMAT_UNSIGNED_INT16;
        case UPTK_AD_FORMAT_UNSIGNED_INT32:
            return CU_AD_FORMAT_UNSIGNED_INT32;
        default:
            ERROR_INVALID_ENUM();
    }
}

void UPTK_ARRAY_DESCRIPTORToCUDA_ARRAY_DESCRIPTOR(const UPTK_ARRAY_DESCRIPTOR * UPTK_para, CUDA_ARRAY_DESCRIPTOR * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->Format = UPTKarray_formatToCUarray_format(UPTK_para->Format);
    cuda_para->Height = UPTK_para->Height;
    cuda_para->NumChannels = UPTK_para->NumChannels;
    cuda_para->Width = UPTK_para->Width;
}

//转变为CUjit_option
CUjit_option UPTKjit_optionTonvrtcJIT_option(UPTKjit_option para) {
    switch (para) {
        case UPTK_JIT_MAX_REGISTERS:
            return CU_JIT_MAX_REGISTERS;
        case UPTK_JIT_THREADS_PER_BLOCK:
            return CU_JIT_THREADS_PER_BLOCK;
        case UPTK_JIT_WALL_TIME:
            return CU_JIT_WALL_TIME;
        case UPTK_JIT_INFO_LOG_BUFFER:
            return CU_JIT_INFO_LOG_BUFFER;
        case UPTK_JIT_INFO_LOG_BUFFER_SIZE_BYTES:
            return CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        case UPTK_JIT_ERROR_LOG_BUFFER:
            return CU_JIT_ERROR_LOG_BUFFER;
        case UPTK_JIT_ERROR_LOG_BUFFER_SIZE_BYTES:
            return CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
        case UPTK_JIT_OPTIMIZATION_LEVEL:
            return CU_JIT_OPTIMIZATION_LEVEL;
        case UPTK_JIT_TARGET_FROM_UPTKCONTEXT:
            return CU_JIT_TARGET_FROM_CUCONTEXT;
        case UPTK_JIT_TARGET:
            return CU_JIT_TARGET;
        case UPTK_JIT_FALLBACK_STRATEGY:
            return CU_JIT_FALLBACK_STRATEGY;
        case UPTK_JIT_GENERATE_DEBUG_INFO:
            return CU_JIT_GENERATE_DEBUG_INFO;
        case UPTK_JIT_LOG_VERBOSE:
            return CU_JIT_LOG_VERBOSE;
        case UPTK_JIT_GENERATE_LINE_INFO:
            return CU_JIT_GENERATE_LINE_INFO;
        case UPTK_JIT_CACHE_MODE:
            return CU_JIT_CACHE_MODE;;
        case UPTK_JIT_NEW_SM3X_OPT:
            return CU_JIT_NEW_SM3X_OPT;
        case UPTK_JIT_FAST_COMPILE:
            return CU_JIT_FAST_COMPILE;
        case UPTK_JIT_GLOBAL_SYMBOL_NAMES:
            return CU_JIT_GLOBAL_SYMBOL_NAMES;
        case UPTK_JIT_GLOBAL_SYMBOL_ADDRESSES:
            return CU_JIT_GLOBAL_SYMBOL_ADDRESSES;
        case UPTK_JIT_GLOBAL_SYMBOL_COUNT:
            return CU_JIT_GLOBAL_SYMBOL_COUNT;
        case UPTK_JIT_LTO:
            return CU_JIT_LTO;
        case UPTK_JIT_FTZ:
            return CU_JIT_FTZ;
        case UPTK_JIT_PREC_DIV:
            return CU_JIT_PREC_DIV;
        case UPTK_JIT_PREC_SQRT:
            return CU_JIT_PREC_SQRT;
        case UPTK_JIT_FMA:
            return CU_JIT_FMA;
        case UPTK_JIT_NUM_OPTIONS:
            return CU_JIT_NUM_OPTIONS;
        default:
            ERROR_INVALID_ENUM();
    }
}

//改动多次
UPTKresult cudaResultToUPTKresult(CUresult para) {
    switch (para) {
        case CUDA_SUCCESS:
            return UPTK_SUCCESS;
        case CUDA_ERROR_INVALID_HANDLE:
            return UPTK_ERROR_INVALID_HANDLE;
        case CUDA_ERROR_INVALID_VALUE:
            return UPTK_ERROR_INVALID_VALUE;
        case CUDA_ERROR_INVALID_IMAGE:
            return UPTK_ERROR_INVALID_IMAGE;
        case CUDA_ERROR_NOT_INITIALIZED:
            return UPTK_ERROR_NOT_INITIALIZED;
        default:
            ERROR_INVALID_ENUM();
    }
}
//nvrtcJITInputType有问题
CUjitInputType UPTKjitInputTypeTonvrtcJITInputType(UPTKjitInputType para) {
    switch (para) {
        case UPTK_JIT_INPUT_UPTKBIN:
            return CU_JIT_INPUT_CUBIN;
        case UPTK_JIT_INPUT_PTX:
            return CU_JIT_INPUT_PTX;
        case UPTK_JIT_INPUT_FATBINARY:
            return CU_JIT_INPUT_FATBINARY;
        case UPTK_JIT_INPUT_OBJECT:
            return CU_JIT_INPUT_OBJECT;
        case UPTK_JIT_INPUT_LIBRARY:
            return CU_JIT_INPUT_LIBRARY;
        case UPTK_JIT_INPUT_NVVM:
            return CU_JIT_INPUT_NVVM;
        case UPTK_JIT_NUM_INPUT_TYPES:
            return CU_JIT_NUM_INPUT_TYPES;
        default:
            ERROR_INVALID_ENUM();
    }

}

//hipDeviceAttribute_t换为cudaDeviceAttr
cudaDeviceAttr UPTKDeviceAttrTocudaDeviceAttribute(enum UPTKDeviceAttr para) {
    switch (para) {
        case UPTKDevAttrAsyncEngineCount:
            return cudaDevAttrAsyncEngineCount;
        case UPTKDevAttrCanMapHostMemory:
            return cudaDevAttrCanMapHostMemory;
        case UPTKDevAttrCanUseHostPointerForRegisteredMem:
            return cudaDevAttrCanUseHostPointerForRegisteredMem;
        case UPTKDevAttrClockRate:
            return cudaDevAttrClockRate;
        case UPTKDevAttrComputeCapabilityMajor:
            return cudaDevAttrComputeCapabilityMajor;
        case UPTKDevAttrComputeCapabilityMinor:
            return cudaDevAttrComputeCapabilityMinor;
        case UPTKDevAttrComputeMode:
            return cudaDevAttrComputeMode;
        case UPTKDevAttrComputePreemptionSupported:
            return cudaDevAttrComputePreemptionSupported;
        case UPTKDevAttrConcurrentKernels:
            return cudaDevAttrConcurrentKernels;
        case UPTKDevAttrConcurrentManagedAccess:
            return cudaDevAttrConcurrentManagedAccess;
        case UPTKDevAttrCooperativeLaunch:
            return cudaDevAttrCooperativeLaunch;
        case UPTKDevAttrCooperativeMultiDeviceLaunch:
            return cudaDevAttrCooperativeMultiDeviceLaunch;
        case UPTKDevAttrDirectManagedMemAccessFromHost:
            return cudaDevAttrDirectManagedMemAccessFromHost;
        case UPTKDevAttrEccEnabled:
            return cudaDevAttrEccEnabled;
        case UPTKDevAttrGlobalL1CacheSupported:
            return cudaDevAttrGlobalL1CacheSupported;
        case UPTKDevAttrGlobalMemoryBusWidth:
            return cudaDevAttrGlobalMemoryBusWidth;
        case UPTKDevAttrGpuOverlap:
            return cudaDevAttrAsyncEngineCount;
        case UPTKDevAttrHostNativeAtomicSupported:
            return cudaDevAttrHostNativeAtomicSupported;
        case UPTKDevAttrHostRegisterSupported:
            return cudaDevAttrHostRegisterSupported;
        case UPTKDevAttrIntegrated:
            return cudaDevAttrIntegrated;
        case UPTKDevAttrIsMultiGpuBoard:
            return cudaDevAttrIsMultiGpuBoard;
        case UPTKDevAttrKernelExecTimeout:
            return cudaDevAttrKernelExecTimeout;
        case UPTKDevAttrL2CacheSize:
            return cudaDevAttrL2CacheSize;
        case UPTKDevAttrLocalL1CacheSupported:
            return cudaDevAttrLocalL1CacheSupported;
        case UPTKDevAttrManagedMemory:
            return cudaDevAttrManagedMemory;
        case UPTKDevAttrMaxBlockDimX:
            return cudaDevAttrMaxBlockDimX;
        case UPTKDevAttrMaxBlockDimY:
            return cudaDevAttrMaxBlockDimY;
        case UPTKDevAttrMaxBlockDimZ:
            return cudaDevAttrMaxBlockDimZ;
        case UPTKDevAttrMaxGridDimX:
            return cudaDevAttrMaxGridDimX;
        case UPTKDevAttrMaxGridDimY:
            return cudaDevAttrMaxGridDimY;
        case UPTKDevAttrMaxGridDimZ:
            return cudaDevAttrMaxGridDimZ;
        case UPTKDevAttrMaxPitch:
            return cudaDevAttrMaxPitch;
        case UPTKDevAttrMaxRegistersPerBlock:
            return cudaDevAttrMaxRegistersPerBlock;
        case UPTKDevAttrMaxRegistersPerMultiprocessor:
            return cudaDevAttrMaxRegistersPerMultiprocessor;
        case UPTKDevAttrMaxSharedMemoryPerBlock:
            return cudaDevAttrMaxSharedMemoryPerBlock;
        case UPTKDevAttrMaxSharedMemoryPerBlockOptin:
            return cudaDevAttrMaxSharedMemoryPerBlock;   
        case UPTKDevAttrMaxSharedMemoryPerMultiprocessor:
            return cudaDevAttrMaxSharedMemoryPerMultiprocessor;
        case UPTKDevAttrMaxSurface1DWidth:
            return cudaDevAttrMaxSurface1DWidth;
        case UPTKDevAttrMaxTexture1DLinearWidth:
            return cudaDevAttrMaxTexture1DLinearWidth;
        case UPTKDevAttrMaxTexture1DMipmappedWidth:
            return cudaDevAttrMaxTexture1DMipmappedWidth;
        case UPTKDevAttrMaxTexture1DWidth:
            return cudaDevAttrMaxTexture1DWidth;
        case UPTKDevAttrMaxTexture2DHeight:
            return cudaDevAttrMaxTexture2DHeight;
        case UPTKDevAttrMaxTexture2DWidth:
            return cudaDevAttrMaxTexture2DWidth;
        case UPTKDevAttrMaxTexture3DDepth:
            return cudaDevAttrMaxTexture3DDepth;
        case UPTKDevAttrMaxTexture3DHeight:
            return cudaDevAttrMaxTexture3DHeight;
        case UPTKDevAttrMaxTexture3DWidth:
            return cudaDevAttrMaxTexture3DWidth;
        case UPTKDevAttrMaxThreadsPerBlock:
            return cudaDevAttrMaxThreadsPerBlock;
        case UPTKDevAttrMaxThreadsPerMultiProcessor:
            return cudaDevAttrMaxThreadsPerMultiProcessor;
        case UPTKDevAttrMemoryClockRate:
            return cudaDevAttrMemoryClockRate;
        case UPTKDevAttrMultiGpuBoardGroupID:
            return cudaDevAttrMultiGpuBoardGroupID;
        case UPTKDevAttrMultiProcessorCount:
            return cudaDevAttrMultiProcessorCount;
        case UPTKDevAttrPageableMemoryAccess:
            return cudaDevAttrPageableMemoryAccess;
        case UPTKDevAttrPageableMemoryAccessUsesHostPageTables:
            return cudaDevAttrPageableMemoryAccessUsesHostPageTables;
        case UPTKDevAttrPciBusId:
            return cudaDevAttrPciBusId;
        case UPTKDevAttrPciDeviceId:
            return cudaDevAttrPciDeviceId;
        case UPTKDevAttrPciDomainId:
            return cudaDevAttrPciDomainId;
        case UPTKDevAttrReserved94:
            return cudaDevAttrReserved94;
        case UPTKDevAttrStreamPrioritiesSupported:
            return cudaDevAttrStreamPrioritiesSupported;
        case UPTKDevAttrSurfaceAlignment:
            return cudaDevAttrSurfaceAlignment;
        case UPTKDevAttrTextureAlignment:
            return cudaDevAttrTextureAlignment;
        case UPTKDevAttrTexturePitchAlignment:
            return cudaDevAttrTexturePitchAlignment;
        case UPTKDevAttrTotalConstantMemory:
            return cudaDevAttrTotalConstantMemory;
        case UPTKDevAttrUnifiedAddressing:
            return cudaDevAttrUnifiedAddressing;
        case UPTKDevAttrWarpSize:
            return cudaDevAttrWarpSize;
        case UPTKDevAttrMemoryPoolsSupported:
            return cudaDevAttrMemoryPoolsSupported;
        case UPTKDevAttrReservedSharedMemoryPerBlock:
            return cudaDevAttrReservedSharedMemoryPerBlock;
        case UPTKDevAttrMaxBlocksPerMultiprocessor:
            return cudaDevAttrMaxBlocksPerMultiprocessor;
        default:
            ERROR_INVALID_OR_UNSUPPORTED_ENUM();
    }
}

enum UPTKGraphNodeType cudaGraphNodeTypeToUPTKGraphNodeType(cudaGraphNodeType para) {
    switch (para) {
        case cudaGraphNodeTypeCount:
            return UPTKGraphNodeTypeCount;
        case cudaGraphNodeTypeEmpty:
            return UPTKGraphNodeTypeEmpty;
        case cudaGraphNodeTypeEventRecord:
            return UPTKGraphNodeTypeEventRecord;
        case cudaGraphNodeTypeGraph:
            return UPTKGraphNodeTypeGraph;
        case cudaGraphNodeTypeHost:
            return UPTKGraphNodeTypeHost;
        case cudaGraphNodeTypeKernel:
            return UPTKGraphNodeTypeKernel;
        case cudaGraphNodeTypeMemcpy:
            return UPTKGraphNodeTypeMemcpy;
        case cudaGraphNodeTypeMemset:
            return UPTKGraphNodeTypeMemset;
        case cudaGraphNodeTypeWaitEvent:
            return UPTKGraphNodeTypeWaitEvent;
        case cudaGraphNodeTypeExtSemaphoreSignal:
            return UPTKGraphNodeTypeExtSemaphoreSignal;
        case cudaGraphNodeTypeExtSemaphoreWait:
            return UPTKGraphNodeTypeExtSemaphoreWait;
        case cudaGraphNodeTypeMemAlloc:
            return UPTKGraphNodeTypeMemAlloc;
        case cudaGraphNodeTypeMemFree:
            return UPTKGraphNodeTypeMemFree;
        //    return UPTKGraphNodeTypeConditional;
        default: 
            ERROR_INVALID_ENUM();
    }
}

/*void UPTKKernelNodeParamsV2TocudaKernelNodeParams(const UPTKKernelNodeParamsV2 *UPTK_para, cudaKernelNodeParamsV2 *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->func = UPTK_para->func;
    cuda_para->sharedMemBytes = UPTK_para->sharedMemBytes;
    cuda_para->kernelParams = UPTK_para->kernelParams;
    cuda_para->extra = UPTK_para->extra;
#if !defined(__cplusplus) || __cplusplus >= 201103L
    cuda_para->gridDim = UPTK_para->gridDim;
    cuda_para->blockDim = UPTK_para->blockDim;
#else
    cuda_para->gridDim.x = UPTK_para->gridDim.x;
    cuda_para->gridDim.y = UPTK_para->gridDim.y;
    cuda_para->gridDim.z = UPTK_para->gridDim.z;
    cuda_para->blockDim.x = UPTK_para->blockDim.x;
    cuda_para->blockDim.y = UPTK_para->blockDim.y;
    cuda_para->blockDim.z = UPTK_para->blockDim.z;
#endif
}*/

cudaGraphNodeType UPTKGraphNodeTypeTocudaGraphNodeType(enum UPTKGraphNodeType para) {
    switch (para) {
        case UPTKGraphNodeTypeKernel:
            return cudaGraphNodeTypeKernel;
        case UPTKGraphNodeTypeMemcpy:
            return cudaGraphNodeTypeMemcpy;
        case UPTKGraphNodeTypeMemset:
            return cudaGraphNodeTypeMemset;
        case UPTKGraphNodeTypeHost:
            return cudaGraphNodeTypeHost;
        case UPTKGraphNodeTypeGraph:
            return cudaGraphNodeTypeGraph;
        case UPTKGraphNodeTypeEmpty:
            return cudaGraphNodeTypeEmpty;
        case UPTKGraphNodeTypeWaitEvent:
            return cudaGraphNodeTypeWaitEvent;
        case UPTKGraphNodeTypeEventRecord:
            return cudaGraphNodeTypeEventRecord;
        case UPTKGraphNodeTypeExtSemaphoreSignal:
            return cudaGraphNodeTypeExtSemaphoreSignal;
        case UPTKGraphNodeTypeExtSemaphoreWait:
            return cudaGraphNodeTypeExtSemaphoreWait;
        case UPTKGraphNodeTypeCount:
            return cudaGraphNodeTypeCount;
        case UPTKGraphNodeTypeMemAlloc:
            return cudaGraphNodeTypeMemAlloc;
        case UPTKGraphNodeTypeMemFree:
            return cudaGraphNodeTypeMemFree;
        //case UPTKGraphNodeTypeConditional:
        //    return cudaGraphNodeTypeConditional;
        default: 
            ERROR_INVALID_ENUM();
    }
}

/*void UPTKMemcpyNodeParamsTocudaMemcpyNodeParams(const UPTKMemcpyNodeParams *UPTK_para, cudaMemcpyNodeParams *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->flags = UPTK_para->flags;
    memcpy(&(cuda_para->reserved), &(UPTK_para->reserved), sizeof(int[3]));
    UPTKMemcpy3DParmsTocudaMemcpy3DParms(&(UPTK_para->copyParams), &(cuda_para->copyParams));
}*/

/*void UPTKMemsetParamsV2TocudaMemsetParams(const UPTKMemsetParamsV2 *UPTK_para, cudaMemsetParamsV2 *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->dst = UPTK_para->dst;
    cuda_para->elementSize = UPTK_para->elementSize;
    cuda_para->height = UPTK_para->height;
    cuda_para->pitch = UPTK_para->pitch;
    cuda_para->value = UPTK_para->value;
    cuda_para->width = UPTK_para->width;
}*/

/*void UPTKHostNodeParamsV2TocudaHostNodeParams(const UPTKHostNodeParamsV2 *UPTK_para, cudaHostNodeParamsV2 *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->fn = (cudaHostFn_t)UPTK_para->fn;
    cuda_para->userData = UPTK_para->userData;
}*/

/*void UPTKChildGraphNodeParamsTocudaChildGraphNodeParams(const UPTKChildGraphNodeParams *UPTK_para, cudaChildGraphNodeParams *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->graph = (cudaGraph_t)UPTK_para->graph;
}*/

/*void UPTKEventWaitNodeParamsTocudaEventWaitNodeParams(const UPTKEventWaitNodeParams *UPTK_para, cudaEventWaitNodeParams *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->event = (cudaEvent_t)UPTK_para->event;
}*/

/*void UPTKEventRecordNodeParamsTocudaEventRecordNodeParams(const UPTKEventRecordNodeParams *UPTK_para, cudaEventRecordNodeParams *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->event = (cudaEvent_t)UPTK_para->event;
}*/

// TODO: UPTKExternalSemaphoreSignalParams generally appears in the form of an array, so special conversion is performed here.
cudaExternalSemaphoreSignalParams UPTKExternalSemaphoreSignalParamsTocudaExternalSemaphoreSignalParams(struct UPTKExternalSemaphoreSignalParams UPTK_para)
{
    // if (nullptr == UPTK_para || nullptr == cuda_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    cudaExternalSemaphoreSignalParams cuda_para;
    cuda_para.flags = UPTK_para.flags;
    memcpy(cuda_para.reserved, UPTK_para.reserved, sizeof(cuda_para.reserved));
    //Copy the members of the params structure
    cuda_para.params.fence.value = UPTK_para.params.fence.value;
    cuda_para.params.keyedMutex.key = UPTK_para.params.keyedMutex.key;
    memcpy(cuda_para.params.reserved, UPTK_para.params.reserved, sizeof(cuda_para.params.reserved));
    memcpy(&cuda_para.params.nvSciSync, &UPTK_para.params.nvSciSync, sizeof(cuda_para.params.nvSciSync));
    return cuda_para;
}

cudaExternalSemaphoreWaitParams UPTKExternalSemaphoreWaitParamsTocudaExternalSemaphoreWaitParams(struct UPTKExternalSemaphoreWaitParams UPTK_para)
{
    // if (nullptr == UPTK_para || nullptr == cuda_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    cudaExternalSemaphoreWaitParams cuda_para;
    cuda_para.flags = UPTK_para.flags;
    memcpy(cuda_para.reserved, UPTK_para.reserved, sizeof(cuda_para.reserved));
    //Copy the members of the params structure
    cuda_para.params.fence.value = UPTK_para.params.fence.value;
    cuda_para.params.keyedMutex.key = UPTK_para.params.keyedMutex.key;
    cuda_para.params.keyedMutex.timeoutMs = UPTK_para.params.keyedMutex.timeoutMs;
    memcpy(cuda_para.params.reserved, UPTK_para.params.reserved, sizeof(cuda_para.params.reserved));
    memcpy(&cuda_para.params.nvSciSync, &UPTK_para.params.nvSciSync, sizeof(cuda_para.params.nvSciSync));
    return cuda_para;
}
//cudaExternalSemaphoreSignalNodeParams变成了V2版本
/*cudaExternalSemaphoreSignalNodeParamsV2 UPTKExternalSemaphoreSignalNodeParamsV2TocudaExternalSemaphoreSignalNodeParams(UPTKExternalSemaphoreSignalNodeParamsV2 UPTK_para)
{
// 函数声明返回 V2 版，内部变量也必须是 V2 版
    // 1. 变量类型改为 V2 版（关键修复）
    cudaExternalSemaphoreSignalNodeParamsV2 cuda_para = {}; // 零初始化，避免垃圾值

    // 2. 字段映射逻辑不变（V2 兼容旧版核心字段）
    cuda_para.extSemArray = (cudaExternalSemaphore_t *)UPTK_para.extSemArray;
    cuda_para.numExtSems = UPTK_para.numExtSems;

    // 3. 内存分配和参数数组转换逻辑不变（paramsArray 字段类型在 V2 中未变）
    cudaExternalSemaphoreSignalParams *cuda_paramsArray = (cudaExternalSemaphoreSignalParams *)malloc(
        sizeof(cudaExternalSemaphoreSignalParams) * cuda_para.numExtSems
    );
    if (cuda_paramsArray == nullptr) { // 可选：添加内存分配失败检查
        fprintf(stderr, "%s: malloc failed for paramsArray\n", __FUNCTION__);
        abort();
    }

    for (int i = 0; i < cuda_para.numExtSems; i++)
    {
        cuda_paramsArray[i] = UPTKExternalSemaphoreSignalParamsTocudaExternalSemaphoreSignalParams(UPTK_para.paramsArray[i]);
    }
    cuda_para.paramsArray = cuda_paramsArray;

    // 4. 返回 V2 版变量（类型完全匹配）
    return cuda_para;

}*/

cudaMemAllocationType UPTKMemAllocationTypeTocudaMemAllocationType(enum UPTKMemAllocationType para) {
    switch (para) {
        case UPTKMemAllocationTypeInvalid:
            return cudaMemAllocationTypeInvalid;
        case UPTKMemAllocationTypeMax:
            return cudaMemAllocationTypeMax;
        case UPTKMemAllocationTypePinned:
            return cudaMemAllocationTypePinned;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaMemAllocationHandleType UPTKMemAllocationHandleTypeTocudaMemAllocationHandleType(enum UPTKMemAllocationHandleType para) {
    switch (para) {
        case UPTKMemHandleTypeNone:
            return cudaMemHandleTypeNone;
        case UPTKMemHandleTypePosixFileDescriptor:
            return cudaMemHandleTypePosixFileDescriptor;
        case UPTKMemHandleTypeWin32:
            return cudaMemHandleTypeWin32;
        case UPTKMemHandleTypeWin32Kmt:
            return cudaMemHandleTypeWin32Kmt;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaMemLocationType UPTKMemLocationTypeTocudaMemLocationType(enum UPTKMemLocationType para) {
    switch (para) {
        case UPTKMemLocationTypeDevice:
            return cudaMemLocationTypeDevice;
        case UPTKMemLocationTypeInvalid:
            return cudaMemLocationTypeInvalid;
        //case UPTKMemLocationTypeHost:
        //    return cudaMemLocationTypeHost;
        //case UPTKMemLocationTypeHostNuma:   
        //    return cudaMemLocationTypeHostNuma;
        //case UPTKMemLocationTypeHostNumaCurrent:
        //    return cudaMemLocationTypeHostNumaCurrent;
        default:
            ERROR_INVALID_ENUM();
    }
}

void UPTKMemLocationTocudaMemLocation(const struct UPTKMemLocation * UPTK_para, cudaMemLocation * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->id = UPTK_para->id;
    cuda_para->type = UPTKMemLocationTypeTocudaMemLocationType(UPTK_para->type);
}
//cudaExternalSemaphoreWaitNodeParams变成了V2版本
/*cudaExternalSemaphoreWaitNodeParamsV2 UPTKExternalSemaphoreWaitNodeParamsV2TocudaExternalSemaphoreWaitNodeParams(UPTKExternalSemaphoreWaitNodeParamsV2 UPTK_para)
{
// 函数声明返回 V2 版，内部变量也必须是 V2 版

    // 1. 关键修复：变量类型改成 V2 版 + 零初始化（避免新增字段垃圾值）
    cudaExternalSemaphoreWaitNodeParamsV2 cuda_para = {};

    // 2. 字段映射逻辑完全不变（V2 兼容旧版核心字段）
    cuda_para.extSemArray = (cudaExternalSemaphore_t *)UPTK_para.extSemArray;
    cuda_para.numExtSems = UPTK_para.numExtSems;

    // 3. 内存分配和参数数组转换逻辑不变（paramsArray 字段类型未变）
    cudaExternalSemaphoreWaitParams *cuda_paramsArray = (cudaExternalSemaphoreWaitParams *)malloc(
        sizeof(cudaExternalSemaphoreWaitParams) * cuda_para.numExtSems
    );
    if (cuda_paramsArray == nullptr) { // 可选：添加内存分配失败检查（推荐）
        fprintf(stderr, "%s: malloc failed for paramsArray\n", __FUNCTION__);
        abort();
    }

    for (int i = 0; i < cuda_para.numExtSems; i++)
    {
        cuda_paramsArray[i] = UPTKExternalSemaphoreWaitParamsTocudaExternalSemaphoreWaitParams(UPTK_para.paramsArray[i]);
    }
    cuda_para.paramsArray = cuda_paramsArray;

    // 4. 返回 V2 版变量（类型完全匹配）
    return cuda_para;

}*/

void UPTKMemPoolPropsTocudaMemPoolProps(const UPTKMemPoolProps * UPTK_para, cudaMemPoolProps * cuda_para) {
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    memcpy(cuda_para->reserved, UPTK_para->reserved, sizeof(cuda_para->reserved));
    cuda_para->win32SecurityAttributes = UPTK_para->win32SecurityAttributes;
    cuda_para->allocType = UPTKMemAllocationTypeTocudaMemAllocationType(UPTK_para->allocType);
    cuda_para->handleTypes = UPTKMemAllocationHandleTypeTocudaMemAllocationHandleType(UPTK_para->handleTypes);
    UPTKMemLocationTocudaMemLocation(&(UPTK_para->location), &(cuda_para->location));
}

cudaMemAccessFlags UPTKMemAccessFlagsTocudaMemAccessFlags(UPTKMemAccessFlags para) {
    switch (para) {
        case UPTKMemAccessFlagsProtNone:
            return cudaMemAccessFlagsProtNone;
        case UPTKMemAccessFlagsProtRead:
            return cudaMemAccessFlagsProtRead;
        case UPTKMemAccessFlagsProtReadWrite:
            return cudaMemAccessFlagsProtReadWrite;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaMemAccessDesc UPTKMemAccessDescTocudaMemAccessDesc(struct UPTKMemAccessDesc UPTK_para)
{
    // if (nullptr == UPTK_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    cudaMemAccessDesc cuda_para;
    cuda_para.flags = UPTKMemAccessFlagsTocudaMemAccessFlags(UPTK_para.flags);
    UPTKMemLocationTocudaMemLocation(&(UPTK_para.location), &(cuda_para.location));
    return cuda_para;
}


/*void UPTKMemAllocNodeParamsV2TocudaMemAllocNodeParams(const struct UPTKMemAllocNodeParamsV2 *UPTK_para, cudaMemAllocNodeParamsV2 *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    UPTKMemPoolPropsTocudaMemPoolProps(&(UPTK_para->poolProps), &(cuda_para->poolProps));
    cudaMemAccessDesc *cuda_daccessDescs = (cudaMemAccessDesc *)malloc(sizeof(cudaMemAccessDesc) * cuda_para->accessDescCount);
    cuda_para->accessDescCount = UPTK_para->accessDescCount;
    for (int i = 0; i < cuda_para->accessDescCount; i++)
    {
        cuda_daccessDescs[i] = UPTKMemAccessDescTocudaMemAccessDesc(UPTK_para->accessDescs[i]);
    }
    cuda_para->bytesize = UPTK_para->bytesize;
    cuda_para->dptr = UPTK_para->dptr;
}*/

/*void UPTKMemFreeNodeParamsTocudaMemFreeNodeParams(const UPTKMemFreeNodeParams *UPTK_para, cudaMemFreeNodeParams *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->dptr = UPTK_para->dptr;
}*/

/*void UPTKConditionalNodeParamsTocudaConditionalNodeParams(const UPTKConditionalNodeParams *UPTK_para, cudaConditionalNodeParams *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->handle = (cudaGraphConditionalHandle)UPTK_para->handle;
    cuda_para->type = UPTKConditionalNodeTypeTocudaConditionalNodeType(UPTK_para->type);
    cuda_para->size = UPTK_para->size;
    cuda_para->phGraph_out = (cudaGraph_t *)UPTK_para->phGraph_out;

}*/

/*cudaGraphConditionalNodeType UPTKConditionalNodeTypeTocudaConditionalNodeType(enum UPTKGraphConditionalNodeType para)
{
    switch (para)
    {
    case UPTKGraphCondTypeIf:
        return cudaGraphCondTypeIf;
    case UPTKGraphCondTypeWhile:
        return cudaGraphCondTypeWhile;
    default:
        ERROR_INVALID_ENUM();
    }
}*/

/*void UPTKGraphNodeParamsTomcGraphNodeParams(const UPTKGraphNodeParams * UPTK_para, mcGraphNodeParams * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->type = UPTKGraphNodeTypeTocudaGraphNodeType(UPTK_para->type);
    memcpy(cuda_para->reserved0, UPTK_para->reserved0, sizeof(int[3]));
    cuda_para->reserved2 = UPTK_para->reserved2;
    memset(&(cuda_para->reserved1), 0, sizeof(long long[29]));
    memcpy(&(cuda_para->reserved1), &(UPTK_para->reserved1), sizeof(long long[29]));
    
    switch (cuda_para->type)
    {
    case cudaGraphNodeTypeKernel:
        UPTKKernelNodeParamsV2TocudaKernelNodeParams(&(UPTK_para->kernel), &(cuda_para->kernel));
        break;
    case cudaGraphNodeTypeMemcpy:
        UPTKMemcpyNodeParamsTocudaMemcpyNodeParams(&(UPTK_para->memcpy), &(cuda_para->memcpy));
        break;
    case cudaGraphNodeTypeMemset:
        UPTKMemsetParamsV2TocudaMemsetParams(&(UPTK_para->memset), &(cuda_para->memset));
        break;
    case cudaGraphNodeTypeHost:
        UPTKHostNodeParamsV2TocudaHostNodeParams(&(UPTK_para->host), &(cuda_para->host));
        break;
    case cudaGraphNodeTypeGraph:
        UPTKChildGraphNodeParamsTocudaChildGraphNodeParams(&(UPTK_para->graph), &(cuda_para->graph));
        break;
    case cudaGraphNodeTypeWaitEvent:
        UPTKEventWaitNodeParamsTocudaEventWaitNodeParams(&(UPTK_para->eventWait), &(cuda_para->eventWait));
        break;
    case cudaGraphNodeTypeEventRecord:
        UPTKEventRecordNodeParamsTocudaEventRecordNodeParams(&(UPTK_para->eventRecord), &(cuda_para->eventRecord));
        break;
    case cudaGraphNodeTypeExtSemaphoreSignal:
        cuda_para->extSemSignal = UPTKExternalSemaphoreSignalNodeParamsV2TocudaExternalSemaphoreSignalNodeParams(UPTK_para->extSemSignal);
        break;
    case cudaGraphNodeTypeExtSemaphoreWait:
        cuda_para->extSemWait = UPTKExternalSemaphoreWaitNodeParamsV2TocudaExternalSemaphoreWaitNodeParams(UPTK_para->extSemWait);
        break;
    case cudaGraphNodeTypeMemAlloc:
        UPTKMemAllocNodeParamsV2TocudaMemAllocNodeParams(&(UPTK_para->alloc), &(cuda_para->alloc));
        break;
    case cudaGraphNodeTypeMemFree:
        UPTKMemFreeNodeParamsTocudaMemFreeNodeParams(&(UPTK_para->free), &(cuda_para->free));
        break;
    case cudaGraphNodeTypeConditional:
        UPTKConditionalNodeParamsTocudaConditionalNodeParams(&(UPTK_para->conditional),&(cuda_para->conditional));
        break;
    }
}*/

//hipFunction_attribute换位CUfunction_attribute
CUfunction_attribute UPTKfunction_attributeTocudaFunction_attribute(UPTKfunction_attribute para) {
    switch (para) {
        case UPTK_FUNC_ATTRIBUTE_BINARY_VERSION:
            return CU_FUNC_ATTRIBUTE_BINARY_VERSION;
        case UPTK_FUNC_ATTRIBUTE_CACHE_MODE_CA:
            return CU_FUNC_ATTRIBUTE_CACHE_MODE_CA;
        case UPTK_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:
            return CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES;
        case UPTK_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:
            return CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES;
        case UPTK_FUNC_ATTRIBUTE_MAX:
            return CU_FUNC_ATTRIBUTE_MAX;
        case UPTK_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
            return CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
        case UPTK_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
            return CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
        case UPTK_FUNC_ATTRIBUTE_NUM_REGS:
            return CU_FUNC_ATTRIBUTE_NUM_REGS;
        case UPTK_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
            return CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT;
        case UPTK_FUNC_ATTRIBUTE_PTX_VERSION:
            return CU_FUNC_ATTRIBUTE_PTX_VERSION;
        case UPTK_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:
            return CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES;
        default:
            ERROR_INVALID_ENUM();
    }
}


cudaError_t UPTKErrorTocudaError(enum UPTKError para) {
    switch (para) {
        case UPTKErrorAlreadyAcquired:
            return cudaErrorAlreadyAcquired;
        case UPTKErrorAlreadyMapped:
            return cudaErrorAlreadyMapped;
        case UPTKErrorArrayIsMapped:
            return cudaErrorArrayIsMapped;
        case UPTKErrorAssert:
            return cudaErrorAssert;
        case UPTKErrorCapturedEvent:
            return cudaErrorCapturedEvent;
        case UPTKErrorContextIsDestroyed:
            return cudaErrorContextIsDestroyed;
        case UPTKErrorCooperativeLaunchTooLarge:
            return cudaErrorCooperativeLaunchTooLarge;
        case UPTKErrorUPTKrtUnloading:
             return cudaErrorCudartUnloading;
        case UPTKErrorDeviceAlreadyInUse:
            return cudaErrorDeviceAlreadyInUse;
        case UPTKErrorDeviceUninitialized:
             return cudaErrorDeviceUninitialized;
        case UPTKErrorECCUncorrectable:
             return cudaErrorECCUncorrectable;
        case UPTKErrorFileNotFound:
            return cudaErrorFileNotFound;
        case UPTKErrorGraphExecUpdateFailure:
            return cudaErrorGraphExecUpdateFailure;
        case UPTKErrorHostMemoryAlreadyRegistered:
            return cudaErrorHostMemoryAlreadyRegistered;
        case UPTKErrorHostMemoryNotRegistered:
            return cudaErrorHostMemoryNotRegistered;
        case UPTKErrorIllegalAddress:
            return cudaErrorIllegalAddress;
        case UPTKErrorIllegalState:
            return cudaErrorIllegalState;
        case UPTKErrorInitializationError:
             return cudaErrorInitializationError;
        case UPTKErrorInsufficientDriver:
            return cudaErrorInsufficientDriver;
        case UPTKErrorInvalidConfiguration:
            return cudaErrorInvalidConfiguration;
        case UPTKErrorInvalidDevice:
            return cudaErrorInvalidDevice;
        case UPTKErrorInvalidDeviceFunction:
            return cudaErrorInvalidDeviceFunction;
        case UPTKErrorInvalidDevicePointer:
            return cudaErrorInvalidDevicePointer;
        case UPTKErrorInvalidGraphicsContext:
            return cudaErrorInvalidGraphicsContext;
        case UPTKErrorInvalidKernelImage:
             return cudaErrorInvalidKernelImage;
        case UPTKErrorInvalidMemcpyDirection:
            return cudaErrorInvalidMemcpyDirection;
        case UPTKErrorInvalidPitchValue:
            return cudaErrorInvalidPitchValue;
        case UPTKErrorInvalidPtx:
             return cudaErrorInvalidPtx;
        case UPTKErrorInvalidResourceHandle:
             return cudaErrorInvalidResourceHandle;
        case UPTKErrorInvalidSource:
            return cudaErrorInvalidSource;
        case UPTKErrorInvalidSymbol:
            return cudaErrorInvalidSymbol;
        case UPTKErrorInvalidValue:
            return cudaErrorInvalidValue;
        case UPTKErrorLaunchFailure:
            return cudaErrorLaunchFailure;
        case UPTKErrorLaunchOutOfResources:
            return cudaErrorLaunchOutOfResources;
        case UPTKErrorLaunchTimeout:
             return cudaErrorLaunchTimeout;
        case UPTKErrorMapBufferObjectFailed:
             return cudaErrorMapBufferObjectFailed;
        case UPTKErrorMemoryAllocation:
             return cudaErrorMemoryAllocation;
        case UPTKErrorMissingConfiguration:
            return cudaErrorMissingConfiguration;
        case UPTKErrorNoDevice:
            return cudaErrorNoDevice;
        case UPTKErrorNoKernelImageForDevice:
             return cudaErrorNoKernelImageForDevice;
        case UPTKErrorNotMapped:
            return cudaErrorNotMapped;
        case UPTKErrorNotMappedAsArray:
            return cudaErrorNotMappedAsArray;
        case UPTKErrorNotMappedAsPointer:
            return cudaErrorNotMappedAsPointer;
        case UPTKErrorNotReady:
            return cudaErrorNotReady;
        case UPTKErrorNotSupported:
            return cudaErrorNotSupported;
        case UPTKErrorOperatingSystem:
            return cudaErrorOperatingSystem;
        case UPTKErrorPeerAccessAlreadyEnabled:
            return cudaErrorPeerAccessAlreadyEnabled;
        case UPTKErrorPeerAccessNotEnabled:
            return cudaErrorPeerAccessNotEnabled;
        case UPTKErrorPeerAccessUnsupported:
            return cudaErrorPeerAccessUnsupported;
        case UPTKErrorPriorLaunchFailure:
            return cudaErrorPriorLaunchFailure;
        case UPTKErrorProfilerAlreadyStarted:
            return cudaErrorProfilerAlreadyStarted;
        case UPTKErrorProfilerAlreadyStopped:
            return cudaErrorProfilerAlreadyStopped;
        case UPTKErrorProfilerDisabled:
            return cudaErrorProfilerDisabled;
        case UPTKErrorProfilerNotInitialized:
            return cudaErrorProfilerNotInitialized;
        case UPTKErrorSetOnActiveProcess:
            return cudaErrorSetOnActiveProcess;
        case UPTKErrorSharedObjectInitFailed:
            return cudaErrorSharedObjectInitFailed;
        case UPTKErrorSharedObjectSymbolNotFound:
            return cudaErrorSharedObjectSymbolNotFound;
        case UPTKErrorStreamCaptureImplicit:
            return cudaErrorStreamCaptureImplicit;
        case UPTKErrorStreamCaptureInvalidated:
            return cudaErrorStreamCaptureInvalidated;
        case UPTKErrorStreamCaptureIsolation:
            return cudaErrorStreamCaptureIsolation;
        case UPTKErrorStreamCaptureMerge:
            return cudaErrorStreamCaptureMerge;
        case UPTKErrorStreamCaptureUnjoined:
            return cudaErrorStreamCaptureUnjoined;
        case UPTKErrorStreamCaptureUnmatched:
            return cudaErrorStreamCaptureUnmatched;
        case UPTKErrorStreamCaptureUnsupported:
            return cudaErrorStreamCaptureUnsupported;
        case UPTKErrorStreamCaptureWrongThread:
            return cudaErrorStreamCaptureWrongThread;
        case UPTKErrorSymbolNotFound:
             return cudaErrorSymbolNotFound;
        case UPTKErrorUnknown:
            return cudaErrorUnknown;
        case UPTKErrorUnmapBufferObjectFailed:
            return cudaErrorUnmapBufferObjectFailed;
        case UPTKErrorUnsupportedLimit:
            return cudaErrorUnsupportedLimit;
        case UPTKSuccess:
            return cudaSuccess;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaFuncAttribute UPTKFuncAttributeTocudaFuncAttribute(enum UPTKFuncAttribute para) {
    switch (para) {
        case UPTKFuncAttributeMax:
            return cudaFuncAttributeMax;
        case UPTKFuncAttributeMaxDynamicSharedMemorySize:
            return cudaFuncAttributeMaxDynamicSharedMemorySize;
        case UPTKFuncAttributePreferredSharedMemoryCarveout:
            return cudaFuncAttributePreferredSharedMemoryCarveout;
        default:
            ERROR_INVALID_ENUM();
    }
}

// Based on the newly added application interface
void cudaDevicePropToUPTKDeviceProp(const cudaDeviceProp * cuda_para, struct UPTKDeviceProp * UPTK_para) {
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    strncpy(UPTK_para->name, cuda_para->name, 256);
    UPTK_para->totalGlobalMem = cuda_para->totalGlobalMem;
    UPTK_para->sharedMemPerBlock = cuda_para->sharedMemPerBlock;
    UPTK_para->regsPerBlock = cuda_para->regsPerBlock;
    UPTK_para->warpSize = cuda_para->warpSize;
    UPTK_para->maxThreadsPerBlock = cuda_para->maxThreadsPerBlock;
    for (int i = 0; i < 3; i++) {
        UPTK_para->maxThreadsDim[i] = cuda_para->maxThreadsDim[i];
        UPTK_para->maxGridSize[i] = cuda_para->maxGridSize[i];
    }
    UPTK_para->clockRate = cuda_para->clockRate;
    UPTK_para->memoryClockRate = cuda_para->memoryClockRate;
    UPTK_para->memoryBusWidth = cuda_para->memoryBusWidth;
    UPTK_para->totalConstMem = cuda_para->totalConstMem;
    UPTK_para->major = SM_VERSION_MAJOR;   // not equal to cuda
    UPTK_para->minor = SM_VERSION_MINOR;   // not equal to cuda
    UPTK_para->multiProcessorCount = cuda_para->multiProcessorCount;
    UPTK_para->l2CacheSize = cuda_para->l2CacheSize;
    UPTK_para->maxThreadsPerMultiProcessor = cuda_para->maxThreadsPerMultiProcessor;
    UPTK_para->computeMode = cuda_para->computeMode;

    UPTK_para->concurrentKernels = cuda_para->concurrentKernels;
    UPTK_para->pciDomainID = cuda_para->pciDomainID;
    UPTK_para->pciBusID = cuda_para->pciBusID;
    UPTK_para->pciDeviceID = cuda_para->pciDeviceID;
    
    UPTK_para->sharedMemPerMultiprocessor = cuda_para->sharedMemPerMultiprocessor;

    UPTK_para->isMultiGpuBoard = cuda_para->isMultiGpuBoard;
    UPTK_para->canMapHostMemory = cuda_para->canMapHostMemory;
    UPTK_para->integrated = cuda_para->integrated;
    UPTK_para->cooperativeLaunch = cuda_para->cooperativeLaunch;
    UPTK_para->cooperativeMultiDeviceLaunch = cuda_para->cooperativeMultiDeviceLaunch;

    UPTK_para->maxTexture1D    = cuda_para->maxTexture1D;
    UPTK_para->maxTexture2D[0] = cuda_para->maxTexture2D[0];
    UPTK_para->maxTexture2D[1] = cuda_para->maxTexture2D[1];
    UPTK_para->maxTexture3D[0] = cuda_para->maxTexture3D[0];
    UPTK_para->maxTexture3D[1] = cuda_para->maxTexture3D[1];
    UPTK_para->maxTexture3D[2] = cuda_para->maxTexture3D[2];

    UPTK_para->memPitch                 = cuda_para->memPitch;
    UPTK_para->textureAlignment         = cuda_para->textureAlignment;
    UPTK_para->texturePitchAlignment    = cuda_para->texturePitchAlignment;
    UPTK_para->kernelExecTimeoutEnabled = cuda_para->kernelExecTimeoutEnabled;
    UPTK_para->ECCEnabled               = cuda_para->ECCEnabled;
    UPTK_para->tccDriver                = cuda_para->tccDriver;

    UPTK_para->regsPerMultiprocessor = cuda_para->regsPerMultiprocessor;
    UPTK_para->maxBlocksPerMultiProcessor = cuda_para->maxBlocksPerMultiProcessor;

    memcpy(&(UPTK_para->uuid), &(cuda_para->uuid), sizeof(UPTKUUID_t));
    for(int i = 0; i < 8; i++){
        UPTK_para->luid[i] = cuda_para->luid[i];
    }
    UPTK_para->luidDeviceNodeMask = cuda_para->luidDeviceNodeMask;
    UPTK_para->deviceOverlap = cuda_para->deviceOverlap;
    UPTK_para->maxTexture1DMipmap = cuda_para->maxTexture1DMipmap;
    UPTK_para->maxTexture1DLinear = cuda_para->maxTexture1DLinear;
    
    for(int i = 0; i < 2; i++){
    UPTK_para->maxTexture2DMipmap[i] = cuda_para->maxTexture2DMipmap[i];
    UPTK_para->maxTexture2DGather[i] = cuda_para->maxTexture2DGather[i];
    UPTK_para->maxTexture1DLayered[i] = cuda_para->maxTexture1DLayered[i];
    UPTK_para->maxTextureCubemapLayered[i] = cuda_para->maxTextureCubemapLayered[i];
    UPTK_para->maxSurface2D[i] = cuda_para->maxSurface2D[i];
    UPTK_para->maxSurface1DLayered[i] = cuda_para->maxSurface1DLayered[i];
    UPTK_para->maxSurfaceCubemapLayered[i] = cuda_para->maxSurfaceCubemapLayered[i];
        
    }

    UPTK_para->maxSurface1D = cuda_para->maxSurface1D;

    for(int i = 0; i < 3; i++){
    UPTK_para->maxTexture2DLinear[i] = cuda_para->maxTexture2DLinear[i];
    UPTK_para->maxTexture3DAlt[i] = cuda_para->maxTexture3DAlt[i];
    UPTK_para->maxTextureCubemap = cuda_para->maxTextureCubemap;
    UPTK_para->maxTexture2DLayered[i] = cuda_para->maxTexture2DLayered[i];
    UPTK_para->maxSurface3D[i] = cuda_para->maxSurface3D[i];
    UPTK_para->maxSurface2DLayered[i] = cuda_para->maxSurface2DLayered[i];
    }

    UPTK_para->maxSurfaceCubemap = cuda_para->maxSurfaceCubemap;
    UPTK_para->surfaceAlignment = cuda_para->surfaceAlignment;
    UPTK_para->asyncEngineCount = cuda_para->asyncEngineCount;
    // UPTK_para->unifiedAddressing = cuda_para->unifiedAddressing;
    //  UPTK sample: simpleIPC determines whether the device supports this attribute in the code. 
    //  The underlying hardware does not support unifiedAddressing, but setting unifiedAddressing to 1 does not affect the sample
    UPTK_para->unifiedAddressing = 1;
    UPTK_para->persistingL2CacheMaxSize = cuda_para->persistingL2CacheMaxSize;
    UPTK_para->streamPrioritiesSupported = cuda_para->streamPrioritiesSupported;
    UPTK_para->globalL1CacheSupported = cuda_para->globalL1CacheSupported;
    UPTK_para->localL1CacheSupported = cuda_para->localL1CacheSupported;
    UPTK_para->managedMemory = cuda_para->managedMemory;
    UPTK_para->multiGpuBoardGroupID = cuda_para->multiGpuBoardGroupID;
    UPTK_para->hostNativeAtomicSupported = cuda_para->hostNativeAtomicSupported;
    UPTK_para->singleToDoublePrecisionPerfRatio = cuda_para->singleToDoublePrecisionPerfRatio;
    UPTK_para->pageableMemoryAccess = cuda_para->pageableMemoryAccess;
    UPTK_para->concurrentManagedAccess = cuda_para->concurrentManagedAccess;
    UPTK_para->computePreemptionSupported = cuda_para->computePreemptionSupported;
    UPTK_para->canUseHostPointerForRegisteredMem = cuda_para->canUseHostPointerForRegisteredMem;
    // UPTK_para->sharedMemPerBlockOptin = cuda_para->sharedMemPerBlockOptin;
    // cuda sharedMemPerBlockOptin member returns 0, the actual function is not supported. Replace with the close attribute sharedMemPerBlock.
    UPTK_para->sharedMemPerBlockOptin = cuda_para->sharedMemPerBlockOptin;
    UPTK_para->pageableMemoryAccessUsesHostPageTables = cuda_para->pageableMemoryAccessUsesHostPageTables;
    UPTK_para->directManagedMemAccessFromHost = cuda_para->directManagedMemAccessFromHost;
    UPTK_para->accessPolicyMaxWindowSize = cuda_para->accessPolicyMaxWindowSize;
    UPTK_para->reservedSharedMemPerBlock = cuda_para->reservedSharedMemPerBlock;
    // UPTK 12.6.2:added some members to the UPTKDeviceProp structure.
    /*UPTK_para->hostRegisterSupported = cuda_para->hostRegisterSupported;
    UPTK_para->sparseUPTKArraySupported = cuda_para->sparseCudaArraySupported;
    UPTK_para->hostRegisterReadOnlySupported = cuda_para->hostRegisterReadOnlySupported;
    UPTK_para->timelineSemaphoreInteropSupported = cuda_para->timelineSemaphoreInteropSupported;
    UPTK_para->memoryPoolsSupported = cuda_para->memoryPoolsSupported;
    UPTK_para->gpuDirectRDMASupported = cuda_para->gpuDirectRDMASupported;
    UPTK_para->gpuDirectRDMAFlushWritesOptions = cuda_para->gpuDirectRDMAFlushWritesOptions;
    UPTK_para->gpuDirectRDMAWritesOrdering = cuda_para->gpuDirectRDMAWritesOrdering;
    UPTK_para->memoryPoolSupportedHandleTypes = cuda_para->memoryPoolSupportedHandleTypes;
    UPTK_para->deferredMappingUPTKArraySupported = cuda_para->deferredMappingCudaArraySupported;
    UPTK_para->ipcEventSupported = cuda_para->ipcEventSupported;
    UPTK_para->clusterLaunch = cuda_para->clusterLaunch;
    UPTK_para->unifiedFunctionPointers = cuda_para->unifiedFunctionPointers;
    memcpy(UPTK_para->reserved2, cuda_para->reserved, 2 * sizeof(int));       
    memcpy(UPTK_para->reserved1, cuda_para->reserved + 2, 1 * sizeof(int));           
    memcpy(UPTK_para->reserved, cuda_para->reserved + 3, 60 * sizeof(int));         */
}



#if defined(__cplusplus)
}
#endif /* __cplusplus */


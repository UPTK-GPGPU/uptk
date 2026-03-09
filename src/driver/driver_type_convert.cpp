#include "driver.hpp"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

UPTKError CUresultToUPTKError(CUresult para) {
    switch (para) {
        case CUDA_ERROR_ALREADY_ACQUIRED:
            return UPTKErrorAlreadyAcquiredD;
        case CUDA_ERROR_ALREADY_MAPPED:
            return UPTKErrorAlreadyMapped;
        case CUDA_ERROR_ARRAY_IS_MAPPED:
            return UPTKErrorArrayIsMapped;
        case CUDA_ERROR_ASSERT:
            return UPTKErrorAssert;
        case CUDA_ERROR_CAPTURED_EVENT:
            return UPTKErrorCapturedEvent;
//         case CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:
//             return UPTKErrorCompatNotSupportedOnDevice;
        //case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
        //    return UPTKErrorContextAlreadyCurrent;
        //case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
        //    return UPTKErrorContextAlreadyInUse;
        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
            return UPTKErrorContextIsDestroyed;
        case CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE:
            return UPTKErrorCooperativeLaunchTooLarge;
        //case CUDA_ERROR_DEINITIALIZED:
        //    return UPTKErrorDeinitialized;
        //case CUDA_ERROR_ECC_UNCORRECTABLE:
        //    return UPTKErrorECCNotCorrectable;
       // case CUDA_ERROR_FILE_NOT_FOUND:
       //     return UPTKErrorFileNotFound;
        case CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE:
            return UPTKErrorGraphExecUpdateFailure;
//         case CUDA_ERROR_HARDWARE_STACK_ERROR:
//             return UPTKErrorHardwareStackError;
        case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
            return UPTKErrorHostMemoryAlreadyRegistered;
        case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
            return UPTKErrorHostMemoryNotRegistered;
        case CUDA_ERROR_ILLEGAL_ADDRESS:
            return UPTKErrorIllegalAddress;
//         case CUDA_ERROR_ILLEGAL_INSTRUCTION:
//             return UPTKErrorIllegalInstruction;
        case CUDA_ERROR_ILLEGAL_STATE:
            return UPTKErrorIllegalState;
//         case CUDA_ERROR_INVALID_ADDRESS_SPACE:
//             return UPTKErrorInvalidAddressSpace;
        //case CUDA_ERROR_INVALID_CONTEXT:
        //    return UPTKErrorInvalidContext;
        case CUDA_ERROR_INVALID_DEVICE:
            return UPTKErrorInvalidDevice;
        case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
            return UPTKErrorInvalidGraphicsContext;
        //case CUDA_ERROR_INVALID_HANDLE:
        //    return UPTKErrorInvalidHandle;
        //case CUDA_ERROR_INVALID_IMAGE:
        //    return UPTKErrorInvalidImage;
//         case CUDA_ERROR_INVALID_PC:
//             return UPTKErrorInvalidPc;
        //case CUDA_ERROR_INVALID_PTX:
        //    return UPTKErrorInvalidKernelFile;
        case CUDA_ERROR_INVALID_SOURCE:
            return UPTKErrorInvalidSource;
        case CUDA_ERROR_INVALID_VALUE:
            return UPTKErrorInvalidValue;
//         case CUDA_ERROR_JIT_COMPILER_NOT_FOUND:
//             return UPTKErrorJitCompilerNotFound;
        case CUDA_ERROR_LAUNCH_FAILED:
            return UPTKErrorLaunchFailure;
//         case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
//             return UPTKErrorLaunchIncompatibleTexturing;
        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
            return UPTKErrorLaunchOutOfResources;
        //case CUDA_ERROR_LAUNCH_TIMEOUT:
        //    return UPTKErrorLaunchTimeOut;
        //case CUDA_ERROR_MAP_FAILED:
        //    return UPTKErrorMapFailed;
//         case CUDA_ERROR_MISALIGNED_ADDRESS:
//             return UPTKErrorMisalignedAddress;
        //case CUDA_ERROR_NOT_FOUND:
        //    return UPTKErrorNotFound;
        //case CUDA_ERROR_NOT_INITIALIZED:
        //    return UPTKErrorNotInitialized;
        case CUDA_ERROR_NOT_MAPPED:
            return UPTKErrorNotMapped;
        case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
            return UPTKErrorNotMappedAsArray;
        case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
            return UPTKErrorNotMappedAsPointer;
//         case CUDA_ERROR_NOT_PERMITTED:
//             return UPTKErrorNotPermitted;
        case CUDA_ERROR_NOT_READY:
            return UPTKErrorNotReady;
        case CUDA_ERROR_NOT_SUPPORTED:
            return UPTKErrorNotSupported;
        //case CUDA_ERROR_NO_BINARY_FOR_GPU:
        //    return UPTKErrorNoBinaryForGpu;
        case CUDA_ERROR_NO_DEVICE:
            return UPTKErrorNoDevice;
//         case CUDA_ERROR_NVLINK_UNCORRECTABLE:
//             return UPTKErrorNvlinkUncorrectable;
        case CUDA_ERROR_OPERATING_SYSTEM:
            return UPTKErrorOperatingSystem;
        //case CUDA_ERROR_OUT_OF_MEMORY:
        //    return UPTKErrorOutOfMemory;
        case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
            return UPTKErrorPeerAccessAlreadyEnabled;
        case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
            return UPTKErrorPeerAccessNotEnabled;
        case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
            return UPTKErrorPeerAccessUnsupported;
        case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
            return UPTKErrorSetOnActiveProcess;
        case CUDA_ERROR_PROFILER_ALREADY_STARTED:
            return UPTKErrorProfilerAlreadyStarted;
        case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
            return UPTKErrorProfilerAlreadyStopped;
        case CUDA_ERROR_PROFILER_DISABLED:
            return UPTKErrorProfilerDisabled;
        case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
            return UPTKErrorProfilerNotInitialized;
        case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
            return UPTKErrorSharedObjectInitFailed;
        case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
            return UPTKErrorSharedObjectSymbolNotFound;
        case CUDA_ERROR_STREAM_CAPTURE_IMPLICIT:
            return UPTKErrorStreamCaptureImplicit;
        case CUDA_ERROR_STREAM_CAPTURE_INVALIDATED:
            return UPTKErrorStreamCaptureInvalidated;
        case CUDA_ERROR_STREAM_CAPTURE_ISOLATION:
            return UPTKErrorStreamCaptureIsolation;
        case CUDA_ERROR_STREAM_CAPTURE_MERGE:
            return UPTKErrorStreamCaptureMerge;
        case CUDA_ERROR_STREAM_CAPTURE_UNJOINED:
            return UPTKErrorStreamCaptureUnjoined;
        case CUDA_ERROR_STREAM_CAPTURE_UNMATCHED:
            return UPTKErrorStreamCaptureUnmatched;
        case CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED:
            return UPTKErrorStreamCaptureUnsupported;
        case CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD:
            return UPTKErrorStreamCaptureWrongThread;
//         case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:
//             return UPTKErrorSystemDriverMismatch;
//         case CUDA_ERROR_SYSTEM_NOT_READY:
//             return UPTKErrorSystemNotReady;
//         case CUDA_ERROR_TIMEOUT:
//             return UPTKErrorTimeout;
//         case CUDA_ERROR_TOO_MANY_PEERS:
//             return UPTKErrorTooManyPeers;
        case CUDA_ERROR_UNKNOWN:
            return UPTKErrorUnknown;
        //case CUDA_ERROR_UNMAP_FAILED:
        //    return UPTKErrorUnmapFailed;
        case CUDA_ERROR_UNSUPPORTED_LIMIT:
            return UPTKErrorUnsupportedLimit;
        //case CUDA_ERROR_INVALID_RESOURCE_TYPE:
        //    return UPTKErrorInvalidResourcetype;   
        case CUDA_SUCCESS:
            return UPTKSuccess;
        default:
            ERROR_INVALID_ENUM();
    }
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */

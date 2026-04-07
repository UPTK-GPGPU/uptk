#include "test_common.h"
#include "runtime/runtime.hpp"

int main()
{
    printf("=============================================\n");
    printf("  UPTK Runtime Test Suite\n");
    printf("=============================================\n");

    /* ============================================================
     *  Section 1: Type Converter Tests (no GPU required)
     * ============================================================ */
    TEST_SECTION("Runtime Type Converters");

    /* --- UPTKError <-> cudaError_t --- */
    TEST_ENUM_ROUNDTRIP("Error Success",
        UPTKErrorTocudaError, cudaErrorToUPTKError,
        (UPTKError)UPTKSuccess, cudaSuccess);
    TEST_ENUM_ROUNDTRIP("Error InvalidValue",
        UPTKErrorTocudaError, cudaErrorToUPTKError,
        (UPTKError)UPTKErrorInvalidValue, cudaErrorInvalidValue);
    TEST_ENUM_ROUNDTRIP("Error MemoryAllocation",
        UPTKErrorTocudaError, cudaErrorToUPTKError,
        (UPTKError)UPTKErrorMemoryAllocation, cudaErrorMemoryAllocation);
    TEST_ENUM_ROUNDTRIP("Error InitializationError",
        UPTKErrorTocudaError, cudaErrorToUPTKError,
        (UPTKError)UPTKErrorInitializationError,
        cudaErrorInitializationError);

    /* --- UPTKMemcpyKind <-> cudaMemcpyKind --- */
    TEST_ENUM_ROUNDTRIP("MemcpyKind HostToHost",
        UPTKMemcpyKindTocudaMemcpyKind, cudaMemcpyKindToUPTKMemcpyKind,
        (UPTKMemcpyKind)UPTKMemcpyHostToHost, cudaMemcpyHostToHost);
    TEST_ENUM_ROUNDTRIP("MemcpyKind HostToDevice",
        UPTKMemcpyKindTocudaMemcpyKind, cudaMemcpyKindToUPTKMemcpyKind,
        (UPTKMemcpyKind)UPTKMemcpyHostToDevice, cudaMemcpyHostToDevice);
    TEST_ENUM_ROUNDTRIP("MemcpyKind DeviceToHost",
        UPTKMemcpyKindTocudaMemcpyKind, cudaMemcpyKindToUPTKMemcpyKind,
        (UPTKMemcpyKind)UPTKMemcpyDeviceToHost, cudaMemcpyDeviceToHost);
    TEST_ENUM_ROUNDTRIP("MemcpyKind DeviceToDevice",
        UPTKMemcpyKindTocudaMemcpyKind, cudaMemcpyKindToUPTKMemcpyKind,
        (UPTKMemcpyKind)UPTKMemcpyDeviceToDevice, cudaMemcpyDeviceToDevice);
    TEST_ENUM_ROUNDTRIP("MemcpyKind Default",
        UPTKMemcpyKindTocudaMemcpyKind, cudaMemcpyKindToUPTKMemcpyKind,
        (UPTKMemcpyKind)UPTKMemcpyDefault, cudaMemcpyDefault);

    /* --- UPTKAccessProperty <-> cudaAccessProperty --- */
    TEST_ENUM_ROUNDTRIP("AccessProp Normal",
        UPTKAccessPropertyTocudaAccessProperty,
        cudaAccessPropertyToUPTKAccessProperty,
        (UPTKAccessProperty)UPTKAccessPropertyNormal,
        cudaAccessPropertyNormal);
    TEST_ENUM_ROUNDTRIP("AccessProp Persisting",
        UPTKAccessPropertyTocudaAccessProperty,
        cudaAccessPropertyToUPTKAccessProperty,
        (UPTKAccessProperty)UPTKAccessPropertyPersisting,
        cudaAccessPropertyPersisting);
    TEST_ENUM_ROUNDTRIP("AccessProp Streaming",
        UPTKAccessPropertyTocudaAccessProperty,
        cudaAccessPropertyToUPTKAccessProperty,
        (UPTKAccessProperty)UPTKAccessPropertyStreaming,
        cudaAccessPropertyStreaming);

    /* --- UPTKChannelFormatKind <-> cudaChannelFormatKind --- */
    TEST_ENUM_ROUNDTRIP("ChanFmtKind Float",
        UPTKChannelFormatKindTocudaChannelFormatKind,
        cudaChannelFormatKindToUPTKChannelFormatKind,
        (UPTKChannelFormatKind)UPTKChannelFormatKindFloat,
        cudaChannelFormatKindFloat);
    TEST_ENUM_ROUNDTRIP("ChanFmtKind Signed",
        UPTKChannelFormatKindTocudaChannelFormatKind,
        cudaChannelFormatKindToUPTKChannelFormatKind,
        (UPTKChannelFormatKind)UPTKChannelFormatKindSigned,
        cudaChannelFormatKindSigned);
    TEST_ENUM_ROUNDTRIP("ChanFmtKind Unsigned",
        UPTKChannelFormatKindTocudaChannelFormatKind,
        cudaChannelFormatKindToUPTKChannelFormatKind,
        (UPTKChannelFormatKind)UPTKChannelFormatKindUnsigned,
        cudaChannelFormatKindUnsigned);
    TEST_ENUM_ROUNDTRIP("ChanFmtKind None",
        UPTKChannelFormatKindTocudaChannelFormatKind,
        cudaChannelFormatKindToUPTKChannelFormatKind,
        (UPTKChannelFormatKind)UPTKChannelFormatKindNone,
        cudaChannelFormatKindNone);

    /* --- UPTKComputeMode -> cudaComputeMode --- */
    TEST_ENUM_CONVERT("ComputeMode Default",
        UPTKComputeModeTocudaComputeMode,
        (UPTKComputeMode)UPTKComputeModeDefault, cudaComputeModeDefault);
    TEST_ENUM_CONVERT("ComputeMode Exclusive",
        UPTKComputeModeTocudaComputeMode,
        (UPTKComputeMode)UPTKComputeModeExclusive, cudaComputeModeExclusive);
    TEST_ENUM_CONVERT("ComputeMode Prohibited",
        UPTKComputeModeTocudaComputeMode,
        (UPTKComputeMode)UPTKComputeModeProhibited,
        cudaComputeModeProhibited);
    TEST_ENUM_CONVERT("ComputeMode ExclusiveProcess",
        UPTKComputeModeTocudaComputeMode,
        (UPTKComputeMode)UPTKComputeModeExclusiveProcess,
        cudaComputeModeExclusiveProcess);

    /* --- UPTKFuncCache <-> cudaFuncCache --- */
    TEST_ENUM_ROUNDTRIP("FuncCache PreferNone",
        UPTKFuncCacheTocudaFuncCache, cudaFuncCacheToUPTKFuncCache,
        (UPTKFuncCache)UPTKFuncCachePreferNone, cudaFuncCachePreferNone);
    TEST_ENUM_ROUNDTRIP("FuncCache PreferShared",
        UPTKFuncCacheTocudaFuncCache, cudaFuncCacheToUPTKFuncCache,
        (UPTKFuncCache)UPTKFuncCachePreferShared,
        cudaFuncCachePreferShared);
    TEST_ENUM_ROUNDTRIP("FuncCache PreferL1",
        UPTKFuncCacheTocudaFuncCache, cudaFuncCacheToUPTKFuncCache,
        (UPTKFuncCache)UPTKFuncCachePreferL1, cudaFuncCachePreferL1);
    TEST_ENUM_ROUNDTRIP("FuncCache PreferEqual",
        UPTKFuncCacheTocudaFuncCache, cudaFuncCacheToUPTKFuncCache,
        (UPTKFuncCache)UPTKFuncCachePreferEqual, cudaFuncCachePreferEqual);

    /* --- UPTKSharedMemConfig <-> cudaSharedMemConfig --- */
    TEST_ENUM_ROUNDTRIP("SharedMemCfg Default",
        UPTKSharedMemConfigTocudaSharedMemConfig,
        cudaSharedMemConfigToUPTKSharedMemConfig,
        (UPTKSharedMemConfig)UPTKSharedMemBankSizeDefault,
        cudaSharedMemBankSizeDefault);
    TEST_ENUM_ROUNDTRIP("SharedMemCfg FourByte",
        UPTKSharedMemConfigTocudaSharedMemConfig,
        cudaSharedMemConfigToUPTKSharedMemConfig,
        (UPTKSharedMemConfig)UPTKSharedMemBankSizeFourByte,
        cudaSharedMemBankSizeFourByte);
    TEST_ENUM_ROUNDTRIP("SharedMemCfg EightByte",
        UPTKSharedMemConfigTocudaSharedMemConfig,
        cudaSharedMemConfigToUPTKSharedMemConfig,
        (UPTKSharedMemConfig)UPTKSharedMemBankSizeEightByte,
        cudaSharedMemBankSizeEightByte);

    /* --- UPTKStreamCaptureMode <-> cudaStreamCaptureMode --- */
    TEST_ENUM_ROUNDTRIP("StreamCapMode Global",
        UPTKStreamCaptureModeTocudaStreamCaptureMode,
        cudaStreamCaptureModeToUPTKStreamCaptureMode,
        (UPTKStreamCaptureMode)UPTKStreamCaptureModeGlobal,
        cudaStreamCaptureModeGlobal);
    TEST_ENUM_ROUNDTRIP("StreamCapMode ThreadLocal",
        UPTKStreamCaptureModeTocudaStreamCaptureMode,
        cudaStreamCaptureModeToUPTKStreamCaptureMode,
        (UPTKStreamCaptureMode)UPTKStreamCaptureModeThreadLocal,
        cudaStreamCaptureModeThreadLocal);
    TEST_ENUM_ROUNDTRIP("StreamCapMode Relaxed",
        UPTKStreamCaptureModeTocudaStreamCaptureMode,
        cudaStreamCaptureModeToUPTKStreamCaptureMode,
        (UPTKStreamCaptureMode)UPTKStreamCaptureModeRelaxed,
        cudaStreamCaptureModeRelaxed);

    /* --- UPTKMemRangeAttribute <-> cudaMemRangeAttribute --- */
    TEST_ENUM_ROUNDTRIP("MemRangeAttr ReadMostly",
        UPTKMemRangeAttributeTocudaMemRangeAttribute,
        cudaMemRangeAttributeToUPTKMemRangeAttribute,
        (UPTKMemRangeAttribute)UPTKMemRangeAttributeReadMostly,
        cudaMemRangeAttributeReadMostly);
    TEST_ENUM_ROUNDTRIP("MemRangeAttr PreferredLoc",
        UPTKMemRangeAttributeTocudaMemRangeAttribute,
        cudaMemRangeAttributeToUPTKMemRangeAttribute,
        (UPTKMemRangeAttribute)UPTKMemRangeAttributePreferredLocation,
        cudaMemRangeAttributePreferredLocation);
    TEST_ENUM_ROUNDTRIP("MemRangeAttr AccessedBy",
        UPTKMemRangeAttributeTocudaMemRangeAttribute,
        cudaMemRangeAttributeToUPTKMemRangeAttribute,
        (UPTKMemRangeAttribute)UPTKMemRangeAttributeAccessedBy,
        cudaMemRangeAttributeAccessedBy);
    TEST_ENUM_ROUNDTRIP("MemRangeAttr LastPrefetchLoc",
        UPTKMemRangeAttributeTocudaMemRangeAttribute,
        cudaMemRangeAttributeToUPTKMemRangeAttribute,
        (UPTKMemRangeAttribute)UPTKMemRangeAttributeLastPrefetchLocation,
        cudaMemRangeAttributeLastPrefetchLocation);

    /* --- UPTKGraphNodeType <-> cudaGraphNodeType --- */
    TEST_ENUM_ROUNDTRIP("GraphNodeType Kernel",
        UPTKGraphNodeTypeTocudaGraphNodeType,
        cudaGraphNodeTypeToUPTKGraphNodeType,
        (UPTKGraphNodeType)UPTKGraphNodeTypeKernel,
        cudaGraphNodeTypeKernel);
    TEST_ENUM_ROUNDTRIP("GraphNodeType Memcpy",
        UPTKGraphNodeTypeTocudaGraphNodeType,
        cudaGraphNodeTypeToUPTKGraphNodeType,
        (UPTKGraphNodeType)UPTKGraphNodeTypeMemcpy,
        cudaGraphNodeTypeMemcpy);
    TEST_ENUM_ROUNDTRIP("GraphNodeType Memset",
        UPTKGraphNodeTypeTocudaGraphNodeType,
        cudaGraphNodeTypeToUPTKGraphNodeType,
        (UPTKGraphNodeType)UPTKGraphNodeTypeMemset,
        cudaGraphNodeTypeMemset);
    TEST_ENUM_ROUNDTRIP("GraphNodeType Host",
        UPTKGraphNodeTypeTocudaGraphNodeType,
        cudaGraphNodeTypeToUPTKGraphNodeType,
        (UPTKGraphNodeType)UPTKGraphNodeTypeHost,
        cudaGraphNodeTypeHost);
    TEST_ENUM_ROUNDTRIP("GraphNodeType Empty",
        UPTKGraphNodeTypeTocudaGraphNodeType,
        cudaGraphNodeTypeToUPTKGraphNodeType,
        (UPTKGraphNodeType)UPTKGraphNodeTypeEmpty,
        cudaGraphNodeTypeEmpty);

    /* --- UPTKMemAccessFlags <-> cudaMemAccessFlags --- */
    TEST_ENUM_ROUNDTRIP("MemAccessFlags None",
        UPTKMemAccessFlagsTocudaMemAccessFlags,
        cudaMemAccessFlagsToUPTKMemAccessFlags,
        (UPTKMemAccessFlags)UPTKMemAccessFlagsProtNone,
        cudaMemAccessFlagsProtNone);
    TEST_ENUM_ROUNDTRIP("MemAccessFlags Read",
        UPTKMemAccessFlagsTocudaMemAccessFlags,
        cudaMemAccessFlagsToUPTKMemAccessFlags,
        (UPTKMemAccessFlags)UPTKMemAccessFlagsProtRead,
        cudaMemAccessFlagsProtRead);
    TEST_ENUM_ROUNDTRIP("MemAccessFlags ReadWrite",
        UPTKMemAccessFlagsTocudaMemAccessFlags,
        cudaMemAccessFlagsToUPTKMemAccessFlags,
        (UPTKMemAccessFlags)UPTKMemAccessFlagsProtReadWrite,
        cudaMemAccessFlagsProtReadWrite);

    /* --- UPTKTextureAddressMode -> cudaTextureAddressMode --- */
    TEST_ENUM_CONVERT("TexAddrMode Wrap",
        UPTKTextureAddressModeTocudaTextureAddressMode,
        (UPTKTextureAddressMode)UPTKAddressModeWrap, cudaAddressModeWrap);
    TEST_ENUM_CONVERT("TexAddrMode Clamp",
        UPTKTextureAddressModeTocudaTextureAddressMode,
        (UPTKTextureAddressMode)UPTKAddressModeClamp, cudaAddressModeClamp);
    TEST_ENUM_CONVERT("TexAddrMode Mirror",
        UPTKTextureAddressModeTocudaTextureAddressMode,
        (UPTKTextureAddressMode)UPTKAddressModeMirror,
        cudaAddressModeMirror);
    TEST_ENUM_CONVERT("TexAddrMode Border",
        UPTKTextureAddressModeTocudaTextureAddressMode,
        (UPTKTextureAddressMode)UPTKAddressModeBorder,
        cudaAddressModeBorder);

    /* --- UPTKSurfaceBoundaryMode -> cudaSurfaceBoundaryMode --- */
    TEST_ENUM_CONVERT("SurfBndMode Zero",
        UPTKSurfaceBoundaryModeTocudaSurfaceBoundaryMode,
        (UPTKSurfaceBoundaryMode)UPTKBoundaryModeZero,
        cudaBoundaryModeZero);
    TEST_ENUM_CONVERT("SurfBndMode Clamp",
        UPTKSurfaceBoundaryModeTocudaSurfaceBoundaryMode,
        (UPTKSurfaceBoundaryMode)UPTKBoundaryModeClamp,
        cudaBoundaryModeClamp);
    TEST_ENUM_CONVERT("SurfBndMode Trap",
        UPTKSurfaceBoundaryModeTocudaSurfaceBoundaryMode,
        (UPTKSurfaceBoundaryMode)UPTKBoundaryModeTrap,
        cudaBoundaryModeTrap);

    /* --- UPTKTextureFilterMode -> cudaTextureFilterMode --- */
    TEST_ENUM_CONVERT("TexFilterMode Point",
        UPTKTextureFilterModeTocudaTextureFilterMode,
        (UPTKTextureFilterMode)UPTKFilterModePoint, cudaFilterModePoint);
    TEST_ENUM_CONVERT("TexFilterMode Linear",
        UPTKTextureFilterModeTocudaTextureFilterMode,
        (UPTKTextureFilterMode)UPTKFilterModeLinear, cudaFilterModeLinear);

    /* ============================================================
     *  Section 2: API Function Tests (GPU required)
     * ============================================================ */
    TEST_SECTION("Runtime API Functions");

    /* --- UPTKGetDeviceCount --- */
    {
        int count = -1;
        UPTKError err = UPTKGetDeviceCount(&count);
        TEST_API_STATUS("UPTKGetDeviceCount",
            "UPTKGetDeviceCount(&count)", err, UPTKSuccess);

        char act[64];
        snprintf(act, sizeof(act), "%d", count);
        TEST_CHECK("UPTKGetDeviceCount value",
            "device count", ">= 0", act, count >= 0);
    }

    /* --- UPTKSetDevice / UPTKGetDevice --- */
    {
        UPTKError err = UPTKSetDevice(0);
        TEST_API_STATUS("UPTKSetDevice(0)",
            "device=0", err, UPTKSuccess);

        int dev = -1;
        err = UPTKGetDevice(&dev);
        TEST_API_STATUS("UPTKGetDevice",
            "UPTKGetDevice(&dev)", err, UPTKSuccess);

        char exp[16], act[16];
        snprintf(exp, sizeof(exp), "0");
        snprintf(act, sizeof(act), "%d", dev);
        TEST_CHECK("UPTKGetDevice value",
            "after SetDevice(0)", exp, act, dev == 0);
    }

    /* --- UPTKDeviceSynchronize --- */
    {
        UPTKError err = UPTKDeviceSynchronize();
        TEST_API_STATUS("UPTKDeviceSynchronize",
            "UPTKDeviceSynchronize()", err, UPTKSuccess);
    }

    /* --- UPTKMalloc / UPTKFree --- */
    {
        void *devPtr = nullptr;
        UPTKError err = UPTKMalloc(&devPtr, 1024);
        TEST_API_STATUS("UPTKMalloc(1024)",
            "size=1024", err, UPTKSuccess);

        bool ptr_valid = (devPtr != nullptr);
        TEST_CHECK("UPTKMalloc pointer",
            "devPtr", "non-NULL",
            ptr_valid ? "non-NULL" : "NULL", ptr_valid);

        if (err == UPTKSuccess) {
            err = UPTKFree(devPtr);
            TEST_API_STATUS("UPTKFree",
                "UPTKFree(devPtr)", err, UPTKSuccess);
        }
    }

    /* --- UPTKMallocHost / UPTKFreeHost --- */
    {
        void *hostPtr = nullptr;
        UPTKError err = UPTKMallocHost(&hostPtr, 4096);
        TEST_API_STATUS("UPTKMallocHost(4096)",
            "size=4096", err, UPTKSuccess);

        bool ptr_valid = (hostPtr != nullptr);
        TEST_CHECK("UPTKMallocHost pointer",
            "hostPtr", "non-NULL",
            ptr_valid ? "non-NULL" : "NULL", ptr_valid);

        if (err == UPTKSuccess) {
            err = UPTKFreeHost(hostPtr);
            TEST_API_STATUS("UPTKFreeHost",
                "UPTKFreeHost(hostPtr)", err, UPTKSuccess);
        }
    }

    /* --- UPTKMemcpy H2D and D2H --- */
    {
        const int N = 256;
        float *hSrc = (float*)malloc(N * sizeof(float));
        float *hDst = (float*)malloc(N * sizeof(float));
        for (int i = 0; i < N; i++) hSrc[i] = (float)i;
        memset(hDst, 0, N * sizeof(float));

        float *dBuf = nullptr;
        UPTKMalloc((void**)&dBuf, N * sizeof(float));

        UPTKError err = UPTKMemcpy(dBuf, hSrc, N * sizeof(float),
                                    UPTKMemcpyHostToDevice);
        TEST_API_STATUS("UPTKMemcpy H2D",
            "float[256] host -> device", err, UPTKSuccess);

        err = UPTKMemcpy(hDst, dBuf, N * sizeof(float),
                          UPTKMemcpyDeviceToHost);
        TEST_API_STATUS("UPTKMemcpy D2H",
            "float[256] device -> host", err, UPTKSuccess);

        bool match = true;
        for (int i = 0; i < N; i++)
            if (hSrc[i] != hDst[i]) { match = false; break; }

        char exp_s[128], act_s[128];
        snprintf(exp_s, sizeof(exp_s), "[0,1,2,...,255]");
        snprintf(act_s, sizeof(act_s), "[%.0f,%.0f,%.0f,...,%.0f]",
                 hDst[0], hDst[1], hDst[2], hDst[N-1]);
        TEST_CHECK("UPTKMemcpy data integrity",
            "roundtrip H2D+D2H", exp_s, act_s, match);

        UPTKFree(dBuf);
        free(hSrc);
        free(hDst);
    }

    /* --- UPTKMemset --- */
    {
        void *dPtr = nullptr;
        UPTKMalloc(&dPtr, 256);
        UPTKError err = UPTKMemset(dPtr, 0xAB, 256);
        TEST_API_STATUS("UPTKMemset",
            "memset 256 bytes to 0xAB", err, UPTKSuccess);

        unsigned char hBuf[256];
        UPTKMemcpy(hBuf, dPtr, 256, UPTKMemcpyDeviceToHost);

        bool correct = true;
        for (int i = 0; i < 256; i++)
            if (hBuf[i] != 0xAB) { correct = false; break; }

        TEST_CHECK("UPTKMemset data verify",
            "all bytes", "0xAB",
            correct ? "0xAB" : "mismatch", correct);

        UPTKFree(dPtr);
    }

    /* --- UPTKStreamCreate / UPTKStreamSynchronize / UPTKStreamDestroy --- */
    {
        UPTKStream_t stream = nullptr;
        UPTKError err = UPTKStreamCreate(&stream);
        TEST_API_STATUS("UPTKStreamCreate",
            "UPTKStreamCreate(&stream)", err, UPTKSuccess);

        if (err == UPTKSuccess) {
            err = UPTKStreamSynchronize(stream);
            TEST_API_STATUS("UPTKStreamSynchronize",
                "UPTKStreamSynchronize(stream)", err, UPTKSuccess);

            err = UPTKStreamDestroy(stream);
            TEST_API_STATUS("UPTKStreamDestroy",
                "UPTKStreamDestroy(stream)", err, UPTKSuccess);
        }
    }

    /* --- UPTKStreamCreateWithFlags --- */
    {
        UPTKStream_t stream = nullptr;
        UPTKError err = UPTKStreamCreateWithFlags(&stream,
                                                    UPTKStreamNonBlocking);
        TEST_API_STATUS("UPTKStreamCreateWithFlags(NonBlocking)",
            "flags=UPTKStreamNonBlocking", err, UPTKSuccess);

        if (err == UPTKSuccess) {
            UPTKStreamDestroy(stream);
        }
    }

    /* --- UPTKEventCreate / UPTKEventDestroy --- */
    {
        UPTKEvent_t event = nullptr;
        UPTKError err = UPTKEventCreate(&event);
        TEST_API_STATUS("UPTKEventCreate",
            "UPTKEventCreate(&event)", err, UPTKSuccess);

        if (err == UPTKSuccess) {
            err = UPTKEventDestroy(event);
            TEST_API_STATUS("UPTKEventDestroy",
                "UPTKEventDestroy(event)", err, UPTKSuccess);
        }
    }

    /* --- UPTKEventCreateWithFlags --- */
    {
        UPTKEvent_t event = nullptr;
        UPTKError err = UPTKEventCreateWithFlags(&event,
                                                   UPTKEventDisableTiming);
        TEST_API_STATUS("UPTKEventCreateWithFlags(DisableTiming)",
            "flags=UPTKEventDisableTiming", err, UPTKSuccess);

        if (err == UPTKSuccess) {
            UPTKEventDestroy(event);
        }
    }

    /* --- UPTKMallocManaged --- */
    {
        void *managedPtr = nullptr;
        UPTKError err = UPTKMallocManaged(&managedPtr, 1024,
                                            UPTKMemAttachGlobal);
        TEST_API_STATUS("UPTKMallocManaged",
            "size=1024, flags=Global", err, UPTKSuccess);

        if (err == UPTKSuccess && managedPtr) {
            memset(managedPtr, 0, 1024);
            UPTKDeviceSynchronize();
            UPTKFree(managedPtr);
        }
    }

    /* --- UPTKGetDeviceProperties --- */
    {
        struct UPTKDeviceProp prop;
        memset(&prop, 0, sizeof(prop));
        UPTKError err = UPTKGetDeviceProperties(&prop, 0);
        TEST_API_STATUS("UPTKGetDeviceProperties",
            "device=0", err, UPTKSuccess);

        if (err == UPTKSuccess) {
            bool has_name = (strlen(prop.name) > 0);
            TEST_CHECK("UPTKGetDeviceProperties name",
                "prop.name", "non-empty",
                has_name ? prop.name : "(empty)", has_name);

            char exp_sm[64], act_sm[64];
            snprintf(exp_sm, sizeof(exp_sm), "> 0");
            snprintf(act_sm, sizeof(act_sm), "%d", prop.multiProcessorCount);
            TEST_CHECK("UPTKGetDeviceProperties SM count",
                "multiProcessorCount", exp_sm, act_sm,
                prop.multiProcessorCount > 0);
        }
    }

    /* --- UPTKDeviceReset --- */
    {
        UPTKError err = UPTKDeviceReset();
        TEST_API_STATUS("UPTKDeviceReset",
            "UPTKDeviceReset()", err, UPTKSuccess);
    }

    TEST_SUMMARY("Runtime");
}

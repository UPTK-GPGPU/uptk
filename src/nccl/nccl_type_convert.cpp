#include "nccl.hpp"

#if defined(__cplusplus)
extern "C"
{
#endif /* __cplusplus */

ncclResult_t UPTKncclResultToncclResult(UPTKncclResult_t para)
{
    switch (para)
    {
    case UPTKncclSuccess:
        return ncclSuccess;
    case UPTKncclUnhandledCudaError:
        return ncclUnhandledCudaError;
    case UPTKncclSystemError:
        return ncclSystemError;
    case UPTKncclInternalError:
        return ncclInternalError;
    case UPTKncclInvalidArgument:
        return ncclInvalidArgument;
    case UPTKncclInvalidUsage:
        return ncclInvalidUsage;
    case UPTKncclRemoteError:
        return ncclRemoteError;
    case UPTKncclInProgress:
        return ncclInProgress;
    case UPTKncclNumResults:
        return ncclNumResults;
    default:
        ERROR_INVALID_ENUM();
    }
}
UPTKncclResult_t ncclResultToUPTKncclResult(ncclResult_t para)
{
    switch (para)
    {
    case ncclSuccess:
        return UPTKncclSuccess;
    case ncclUnhandledCudaError:
        return UPTKncclUnhandledCudaError;
    case ncclSystemError:
        return UPTKncclSystemError;
    case ncclInternalError:
        return UPTKncclInternalError;
    case ncclInvalidArgument:
        return UPTKncclInvalidArgument;
    case ncclInvalidUsage:
        return UPTKncclInvalidUsage;
    case ncclRemoteError:
        return UPTKncclRemoteError;
    case ncclInProgress:
        return UPTKncclInProgress;
    case ncclNumResults:
        return UPTKncclNumResults;
    default:
        ERROR_INVALID_ENUM();
    }
}

ncclRedOp_dummy_t UPTKncclRedOp_dummyToncclRedOp_dummy(UPTKncclRedOp_dummy_t para)
{
    switch (para)
    {
    case UPTKncclNumOps_dummy:
        return ncclNumOps_dummy;

    default:
        ERROR_INVALID_ENUM();
    }
}

ncclRedOp_t UPTKncclRedOpToncclRedOp(UPTKncclRedOp_t para)
{
    switch (para)
    {
    case UPTKncclSum:
        return ncclSum;
    case UPTKncclProd:
        return ncclProd;
    case UPTKncclMax:
        return ncclMax;
    case UPTKncclMin:
        return ncclMin;
    case UPTKncclAvg:
        return ncclAvg;
    case UPTKncclNumOps:
        return ncclNumOps;
    case UPTKncclMaxRedOp:
        return ncclMaxRedOp;
    default:
        ERROR_INVALID_ENUM();
    }
}

ncclDataType_t UPTKncclDataTypeToncclDataType(UPTKncclDataType_t para)
{
    switch (para)
    {
    case UPTKncclInt8:
        return ncclInt8;
    case UPTKncclUint8:
        return ncclUint8;
    case UPTKncclInt32:
        return ncclInt32;
    case UPTKncclUint32:
        return ncclUint32;
    case UPTKncclInt64:
        return ncclInt64;
    case UPTKncclUint64:
        return ncclUint64;
    case UPTKncclFloat16:
        return ncclFloat16;
    case UPTKncclFloat32:
        return ncclFloat32;
    case UPTKncclFloat64:
        return ncclFloat64;
    case UPTKncclBfloat16:
        return ncclBfloat16;

#if defined(UPTKRCCL_FLOAT8) && defined(NCCL_HAS_FP8)
    case UPTKncclFp8E4M3:
        return ncclFp8E4M3;
    case UPTKncclFp8E5M2:
        return ncclFp8E5M2;
#endif

    default:
        ERROR_INVALID_ENUM();
    }
}

ncclScalarResidence_t UPTKncclScalarResidenceToncclScalarResidence(UPTKncclScalarResidence_t para)
{
    switch (para)
    {
    case UPTKncclScalarDevice:
        return ncclScalarDevice;
    case UPTKncclScalarHostImmediate:
        return ncclScalarHostImmediate;
    default:
        ERROR_INVALID_ENUM();
    }
}

void UPTKncclUniqueIdToncclUniqueId(const UPTKncclUniqueId * UPTK_para, ncclUniqueId * cuda_para)
{
if (nullptr == UPTK_para || nullptr == cuda_para) {
    fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    abort();
}
int len = std::min(UPTK_NCCL_UNIQUE_ID_BYTES, NCCL_UNIQUE_ID_BYTES);
memcpy(cuda_para, UPTK_para, len);
}

void ncclUniqueIdToUPTKncclUniqueId(const ncclUniqueId * cuda_para, UPTKncclUniqueId * UPTK_para)
{
if (nullptr == UPTK_para || nullptr == cuda_para) {
    fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    abort();
}
int len = std::min(UPTK_NCCL_UNIQUE_ID_BYTES, NCCL_UNIQUE_ID_BYTES);
memcpy(UPTK_para, cuda_para, len);
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */
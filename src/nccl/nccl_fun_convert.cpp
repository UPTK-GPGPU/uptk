#include "nccl.hpp"

#if defined(__cplusplus)
extern "C"
{
#endif /* __cplusplus */

UPTKncclResult_t UPTKNCCLAPI UPTKncclGetVersion(int *version)
{
    if (nullptr == version)
    {
        return UPTKncclInvalidArgument;
    }

    *version = UPTK_NCCL_VERSION(UPTK_NCCL_MAJOR, UPTK_NCCL_MINOR, UPTK_NCCL_PATCH);

    return UPTKncclSuccess;
}

UPTKncclResult_t UPTKNCCLAPI UPTKncclGetUniqueId(UPTKncclUniqueId *uniqueId)
{
    ncclUniqueId cudaId;
    ncclResult_t cudaStatus = ncclGetUniqueId(&cudaId);

    if (ncclSuccess == cudaStatus)
    {
        ncclUniqueIdToUPTKncclUniqueId(&cudaId, uniqueId);
    }

    return ncclResultToUPTKncclResult(cudaStatus);
}

UPTKncclResult_t UPTKNCCLAPI UPTKncclCommInitRank(UPTKncclComm_t *comm, int nranks, UPTKncclUniqueId commId, int rank)
{
    ncclUniqueId cudaId;
    UPTKncclUniqueIdToncclUniqueId(&commId, &cudaId);

    return ncclResultToUPTKncclResult(ncclCommInitRank((ncclComm_t *)comm, nranks, cudaId, rank));
}

UPTKncclResult_t UPTKNCCLAPI UPTKncclCommInitAll(UPTKncclComm_t *comm, int ndev, const int *devlist)
{
    return ncclResultToUPTKncclResult(ncclCommInitAll((ncclComm_t *)comm, ndev, devlist));
}

UPTKncclResult_t UPTKNCCLAPI UPTKncclCommDestroy(UPTKncclComm_t comm)
{
    return ncclResultToUPTKncclResult(ncclCommDestroy((ncclComm_t)comm));
}

UPTKncclResult_t UPTKNCCLAPI UPTKncclCommAbort(UPTKncclComm_t comm)
{
    return ncclResultToUPTKncclResult(ncclCommAbort((ncclComm_t)comm));
}

const char *UPTKNCCLAPI UPTKncclGetErrorString(UPTKncclResult_t result)
{
    return ncclGetErrorString(UPTKncclResultToncclResult(result));
}

UPTKncclResult_t UPTKNCCLAPI UPTKncclCommGetAsyncError(UPTKncclComm_t comm, UPTKncclResult_t *asyncError)
{
    ncclResult_t retStatus = ncclSuccess;
    ncclResult_t cudaStatus = ncclCommGetAsyncError((ncclComm_t)comm, &retStatus);

    if (ncclSuccess == cudaStatus)
    {
        *asyncError = ncclResultToUPTKncclResult(retStatus);
    }
    return ncclResultToUPTKncclResult(cudaStatus);
}

UPTKncclResult_t UPTKNCCLAPI UPTKncclCommCount(const UPTKncclComm_t comm, int *count)
{
    return ncclResultToUPTKncclResult(ncclCommCount((const ncclComm_t)comm, count));
}

UPTKncclResult_t UPTKNCCLAPI UPTKncclCommCuDevice(const UPTKncclComm_t comm, int *device)
{
    return ncclResultToUPTKncclResult(ncclCommCuDevice((const ncclComm_t)comm, device));
}

UPTKncclResult_t UPTKNCCLAPI UPTKncclCommUserRank(const UPTKncclComm_t comm, int *rank)
{
    return ncclResultToUPTKncclResult(ncclCommUserRank((const ncclComm_t)comm, rank));
}

UPTKncclResult_t UPTKNCCLAPI UPTKncclReduce(const void *sendbuff, void *recvbuff, size_t count, UPTKncclDataType_t datatype,
                                            UPTKncclRedOp_t op, int root, UPTKncclComm_t comm, UPTKStream_t stream)
{
    return ncclResultToUPTKncclResult(ncclReduce(sendbuff, recvbuff, count,
                                                 UPTKncclDataTypeToncclDataType(datatype),
                                                 UPTKncclRedOpToncclRedOp(op), root,
                                                 (ncclComm_t)comm, (cudaStream_t)stream));
}

UPTKncclResult_t UPTKNCCLAPI UPTKncclBroadcast(const void *sendbuff, void *recvbuff, size_t count, UPTKncclDataType_t datatype, int root,
                                                UPTKncclComm_t comm, UPTKStream_t stream)
{
    return ncclResultToUPTKncclResult(ncclBroadcast(sendbuff, recvbuff, count,
                                                    UPTKncclDataTypeToncclDataType(datatype), root,
                                                    (ncclComm_t)comm, (cudaStream_t)stream));
}

UPTKncclResult_t UPTKNCCLAPI UPTKncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                                                UPTKncclDataType_t datatype, UPTKncclRedOp_t op, UPTKncclComm_t comm, UPTKStream_t stream)
{
    return ncclResultToUPTKncclResult(ncclAllReduce(sendbuff, recvbuff, count,
                                                    UPTKncclDataTypeToncclDataType(datatype),
                                                    UPTKncclRedOpToncclRedOp(op),
                                                    (ncclComm_t)comm, (cudaStream_t)stream));
}

UPTKncclResult_t UPTKNCCLAPI UPTKncclReduceScatter(const void *sendbuff, void *recvbuff,
                                                    size_t recvcount, UPTKncclDataType_t datatype, UPTKncclRedOp_t op, UPTKncclComm_t comm,
                                                    UPTKStream_t stream)
{
    return ncclResultToUPTKncclResult(ncclReduceScatter(sendbuff, recvbuff, recvcount,
                                                        UPTKncclDataTypeToncclDataType(datatype),
                                                        UPTKncclRedOpToncclRedOp(op),
                                                        (ncclComm_t)comm, (cudaStream_t)stream));
}

UPTKncclResult_t UPTKNCCLAPI UPTKncclAllGather(const void *sendbuff, void *recvbuff, size_t sendcount,
                                                UPTKncclDataType_t datatype, UPTKncclComm_t comm, UPTKStream_t stream)
{
    return ncclResultToUPTKncclResult(ncclAllGather(sendbuff, recvbuff, sendcount,
                                                    UPTKncclDataTypeToncclDataType(datatype),
                                                    (ncclComm_t)comm, (cudaStream_t)stream));
}

UPTKncclResult_t UPTKNCCLAPI UPTKncclSend(const void *sendbuff, size_t count, UPTKncclDataType_t datatype, int peer,
                                            UPTKncclComm_t comm, UPTKStream_t stream)
{
    return ncclResultToUPTKncclResult(ncclSend(sendbuff, count,
                                                UPTKncclDataTypeToncclDataType(datatype), peer,
                                                (ncclComm_t)comm, (cudaStream_t)stream));
}

UPTKncclResult_t UPTKNCCLAPI UPTKncclRecv(void *recvbuff, size_t count, UPTKncclDataType_t datatype, int peer,
                                            UPTKncclComm_t comm, UPTKStream_t stream)
{
    return ncclResultToUPTKncclResult(ncclRecv(recvbuff, count,
                                                UPTKncclDataTypeToncclDataType(datatype), peer,
                                                (ncclComm_t)comm, (cudaStream_t)stream));
}

UPTKncclResult_t UPTKNCCLAPI UPTKncclGroupStart()
{
    return ncclResultToUPTKncclResult(ncclGroupStart());
}

UPTKncclResult_t UPTKNCCLAPI UPTKncclGroupEnd()
{
    return ncclResultToUPTKncclResult(ncclGroupEnd());
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */
#include "runtime.hpp"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

__host__ UPTKError_t UPTKMalloc(void ** devPtr,size_t size)
{
    cudaError_t cuda_res;
    cuda_res = cudaMalloc(devPtr, size);
    return cudaErrorToUPTKError(cuda_res);
}

__host__  UPTKError_t  UPTKFree(void * devPtr)
{
    cudaError_t cuda_res;
    cuda_res = cudaFree(devPtr);
    return cudaErrorToUPTKError(cuda_res);
}

// UPTKError_t __UPTKPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem , UPTKStream_t stream) {
//     cudaError_t res;
//     res = __cudaPushCallConfiguration(gridDim, blockDim, sharedMem, (cudaStream_t)stream);
//     return cudaErrorToUPTKError(res);
// }

// UPTKError_t __UPTKPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, UPTKStream_t *stream) {
//     cudaError_t res;
//     res = __cudaPopCallConfiguration(gridDim, blockDim, sharedMem, (cudaStream_t *) stream);
//     return cudaErrorToUPTKError(res);
// }

__host__ UPTKError_t  UPTKStreamCreate(UPTKStream_t * pStream)
{
    cudaError_t cuda_res;
    cuda_res = cudaStreamCreate((cudaStream_t *) pStream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__  UPTKError_t  UPTKStreamCreateWithFlags(UPTKStream_t * pStream,unsigned int flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaStreamCreateWithFlags((cudaStream_t *) pStream, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__  UPTKError_t  UPTKStreamDestroy(UPTKStream_t stream)
{
    cudaError_t cuda_res;
    cuda_res = cudaStreamDestroy((cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKStreamSynchronize(UPTKStream_t stream)
{
    cudaError_t cuda_res;
    cuda_res = cudaStreamSynchronize((cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__  UPTKError_t  UPTKStreamWaitEvent(UPTKStream_t stream,UPTKEvent_t event,unsigned int flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaStreamWaitEvent((cudaStream_t) stream, (cudaEvent_t) event, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKStreamBeginCapture(UPTKStream_t stream,enum UPTKStreamCaptureMode mode)
{
    if (mode < UPTKStreamCaptureModeGlobal || mode > UPTKStreamCaptureModeRelaxed)
    {
        return UPTKErrorInvalidValue;
    }
    cudaStreamCaptureMode cuda_mode = UPTKStreamCaptureModeTocudaStreamCaptureMode(mode);
    cudaError_t cuda_res;
    cuda_res = cudaStreamBeginCapture((cudaStream_t) stream, cuda_mode);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKStreamEndCapture(UPTKStream_t stream,UPTKGraph_t * pGraph)
{
    cudaError_t cuda_res;
    cuda_res = cudaStreamEndCapture((cudaStream_t) stream, (cudaGraph_t *) pGraph);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKStreamIsCapturing(UPTKStream_t stream,enum UPTKStreamCaptureStatus * pCaptureStatus)
{
    if (nullptr == pCaptureStatus)
        return UPTKErrorInvalidValue;
    cudaStreamCaptureStatus cuda_pCaptureStatus;
    cudaError_t cuda_res;
    cuda_res = cudaStreamIsCapturing((cudaStream_t) stream, &cuda_pCaptureStatus);
    *pCaptureStatus = cudaStreamCaptureStatusToUPTKStreamCaptureStatus(cuda_pCaptureStatus);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKEventCreate(UPTKEvent_t * event)
{
    cudaError_t cuda_res;
    cuda_res = cudaEventCreate((cudaEvent_t *)event);
    return cudaErrorToUPTKError(cuda_res);
}

__host__  UPTKError_t  UPTKEventCreateWithFlags(UPTKEvent_t * event,unsigned int flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaEventCreateWithFlags((cudaEvent_t *)event, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__  UPTKError_t  UPTKEventDestroy(UPTKEvent_t event)
{
    cudaError_t cuda_res;
    cuda_res = cudaEventDestroy((cudaEvent_t) event);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKEventSynchronize(UPTKEvent_t event)
{
    cudaError_t cuda_res;
    cuda_res = cudaEventSynchronize((cudaEvent_t) event);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKEventQuery(UPTKEvent_t event)
{
    cudaError_t cuda_res;
    cuda_res = cudaEventQuery((cudaEvent_t) event);
    return cudaErrorToUPTKError(cuda_res);
}

__host__  UPTKError_t  UPTKEventRecord(UPTKEvent_t event,UPTKStream_t stream)
{
    cudaError_t cuda_res;
    cuda_res = cudaEventRecord((cudaEvent_t) event, (cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKMemcpy(void * dst,const void * src,size_t count,enum UPTKMemcpyKind kind)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpy(dst, src, count, cuda_kind);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKIpcGetMemHandle(UPTKIpcMemHandle_t * handle,void * devPtr)
{
    cudaError_t cuda_res;
    cuda_res = cudaIpcGetMemHandle((cudaIpcMemHandle_t *) handle, devPtr);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKIpcOpenMemHandle(void ** devPtr,UPTKIpcMemHandle_t handle,unsigned int flags)
{
    cudaIpcMemHandle_t cuda_handle;
    UPTKIpcMemHandleTocudaIpcMemHandle(&handle, &cuda_handle);
    if (UPTKIpcMemLazyEnablePeerAccess == flags)
        flags = cudaIpcMemLazyEnablePeerAccess;
    cudaError_t cuda_res;
    cuda_res = cudaIpcOpenMemHandle(devPtr, cuda_handle, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKIpcCloseMemHandle(void * devPtr)
{
    cudaError_t cuda_res;
    cuda_res = cudaIpcCloseMemHandle(devPtr);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKChooseDevice(int * device,const struct UPTKDeviceProp * prop)
{
    if (nullptr == device || nullptr == prop)
        return UPTKErrorInvalidValue;
    cudaDeviceProp cuda_prop;
    UPTKDevicePropTocudaDeviceProp(prop, &cuda_prop);
    cudaError_t cuda_res;
    cuda_res = cudaChooseDevice(device, &cuda_prop);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKSetDevice(int device)
{
    cudaError_t cuda_res;
    cuda_res = cudaSetDevice(device);
    return cudaErrorToUPTKError(cuda_res);
}


__host__  UPTKError_t  UPTKGetDevice(int * device)
{
    cudaError_t cuda_res;
    cuda_res = cudaGetDevice(device);
    return cudaErrorToUPTKError(cuda_res);
}

__host__  UPTKError_t  UPTKDeviceSynchronize(void)
{
    cudaError_t cuda_res;
    cuda_res = cudaDeviceSynchronize();
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKDeviceReset(void)
{
    cudaError_t cuda_res;
    cuda_res = cudaDeviceReset();
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKGraphCreate(UPTKGraph_t * pGraph,unsigned int flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphCreate((cudaGraph_t *) pGraph, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKGraphClone(UPTKGraph_t * pGraphClone,UPTKGraph_t originalGraph)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphClone((cudaGraph_t *) pGraphClone, (cudaGraph_t) originalGraph);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKGraphDestroy(UPTKGraph_t graph)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphDestroy((cudaGraph_t) graph);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKGraphAddDependencies(UPTKGraph_t graph,const UPTKGraphNode_t * from,const UPTKGraphNode_t * to,size_t numDependencies)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphAddDependencies((cudaGraph_t) graph, (const cudaGraphNode_t *) from, (const cudaGraphNode_t *) to, numDependencies);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKGraphRemoveDependencies(UPTKGraph_t graph,const UPTKGraphNode_t * from,const UPTKGraphNode_t * to,size_t numDependencies)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphRemoveDependencies((cudaGraph_t) graph, (const cudaGraphNode_t *) from, (const cudaGraphNode_t *) to, numDependencies);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKGraphInstantiate(UPTKGraphExec_t * pGraphExec,UPTKGraph_t graph,UPTKGraphNode_t * pErrorNode,char * pLogBuffer,size_t bufferSize)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphInstantiate((cudaGraphExec_t *) pGraphExec, (cudaGraph_t) graph, (cudaGraphNode_t *) pErrorNode, pLogBuffer, bufferSize);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKGraphLaunch(UPTKGraphExec_t graphExec,UPTKStream_t stream)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphLaunch((cudaGraphExec_t) graphExec, (cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

/*之前的版本进行修改
__host__ UPTKError_t  UPTKGraphExecUpdate(UPTKGraphExec_t hGraphExec,UPTKGraph_t hGraph,UPTKGraphNode_t * hErrorNode_out,enum UPTKGraphExecUpdateResult * updateResult_out)
{
    if (nullptr == updateResult_out)
        return UPTKErrorInvalidValue;
    cudaGraphExecUpdateResult cuda_updateResult_out;
    cudaError_t cuda_res;
    cuda_res = cudaGraphExecUpdate((cudaGraphExec_t) hGraphExec, (cudaGraph_t) hGraph, (cudaGraphNode_t *) hErrorNode_out, &cuda_updateResult_out);
    // The return value cudaErrorGraphExecUpdateFailure is also a valuable output.
    if (cudaSuccess == cuda_res || cudaErrorGraphExecUpdateFailure == cuda_res)
        *updateResult_out = cudaGraphExecUpdateResultToUPTKGraphExecUpdateResult(cuda_updateResult_out);
    return cudaErrorToUPTKError(cuda_res);
}
*/
//改正的新版本
/*__host__ UPTKError_t UPTKGraphExecUpdate(UPTKGraphExec_t hGraphExec,UPTKGraph_t hGraph,UPTKGraphNode_t *hErrorNode_out, enum UPTKGraphExecUpdateResult *updateResult_out)
{
    if (nullptr == updateResult_out) {
        return UPTKErrorInvalidValue;
    }
    // 1. 使用 CUDA 12.0+ 新结构体
    //cudaGraphExecUpdateResultInfo updateInfo = {}; // 初始化为零
    mcGraphExecUpdateResultInfo updateInfo = {}; // 初始化为零
    // 2. 调用新版函数（仅3个参数！）
    mcError_t cuda_res = mcGraphExecUpdate(
        (cudaGraphExec_t)hGraphExec,
        (cudaGraph_t)hGraph,
        &updateInfo  // ← 唯一输出参数
    );
    // 3. 提取结果到旧版字段（保持 API 兼容）
    if (hErrorNode_out) {
        *hErrorNode_out = (UPTKGraphNode_t)updateInfo.errorNode; // 节点指针
    }
    *updateResult_out = cudaGraphExecUpdateResultToUPTKGraphExecUpdateResult(
        updateInfo.result  // ← 结构体中的 result 字段
    );
    // 4. 返回错误码（注意：cudaErrorGraphExecUpdateFailure 可能被移除）
    return cudaErrorToUPTKError(cuda_res);
}*/

__host__ UPTKError_t  UPTKLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, UPTKStream_t stream)
{
    cudaError_t cuda_res;
    cuda_res = cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, (cudaStream_t)stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__  UPTKError_t  UPTKFuncGetAttributes(struct UPTKFuncAttributes * attr,const void * func)
{
    if(nullptr == attr)
    {
        return UPTKErrorInvalidValue;
    }

    memset(attr, 0, sizeof(UPTKFuncAttributes));

    struct cudaFuncAttributes cuda_attr;
    memset(&cuda_attr, 0, sizeof(cudaFuncAttributes));

    cudaError_t cuda_res;
    cuda_res = cudaFuncGetAttributes(&cuda_attr, func);
    if ( cuda_res == cudaSuccess )
    {
        cudaFuncAttributesToUPTKFuncAttributes(&cuda_attr, attr);
    }
    return cudaErrorToUPTKError(cuda_res);
}

// __host__ UPTKError_t UPTKStreamGetDevice(UPTKStream_t stream, UPTKdevice * device)
// {
//     cudaError_t cuda_res;
//     cuda_res = cudaStreamGetDevice((cudaStream_t) stream, (int*) device);
//     return cudaErrorToUPTKError(cuda_res);
// }
//二次改动
UPTKresult  UPTKModuleLoad(UPTKmodule * module,const char * fname)
{
    CUresult cuda_res;
    cuda_res = cuModuleLoad((CUmodule *)module, fname);
    return cudaErrorToUPTKresult(cuda_res);
}
//二次改动
UPTKresult  UPTKModuleLoadData(UPTKmodule * module,const void * image)
{
    CUresult cuda_res;
    cuda_res = cuModuleLoadData((CUmodule *)module, image);
    return cudaErrorToUPTKresult(cuda_res);
}

UPTKresult  UPTKModuleUnload(UPTKmodule hmod)
{
    CUresult cuda_res;
    cuda_res = cuModuleUnload((CUmodule) hmod);
    return cudaErrorToUPTKresult(cuda_res);
}

UPTKresult  UPTKModuleGetFunction(UPTKfunction * hfunc,UPTKmodule hmod,const char * name)
{
    CUresult cuda_res;
    cuda_res = cuModuleGetFunction((CUfunction *)hfunc, (CUmodule) hmod, name);
    return cudaErrorToUPTKresult(cuda_res);
}

UPTKresult  UPTKModuleGetGlobal(UPTKdeviceptr * dptr, size_t * bytes,UPTKmodule hmod,const char * name)
{
    CUresult cuda_res;
    cuda_res = cuModuleGetGlobal((CUdeviceptr *)dptr, bytes, (CUmodule) hmod, name);
    return cudaErrorToUPTKresult(cuda_res);
}

//改动较多
UPTKresult  UPTKLinkCreate(unsigned int numOptions,UPTKjit_option * options,void ** optionValues,UPTKlinkState * stateOut)
{
    CUresult nvrtc_res;
    CUjit_option* nvrtc_options = (CUjit_option*)malloc(sizeof(CUjit_option) * numOptions);
    if (nullptr != options) {
        if (nullptr == nvrtc_options) {
          return UPTK_ERROR_OUT_OF_MEMORY;
        }
        for (int i = 0; i < numOptions; i++) {
            nvrtc_options[i] = UPTKjit_optionTonvrtcJIT_option(options[i]);
        }
    }
    nvrtc_res = cuLinkCreate(numOptions, nvrtc_options, optionValues, (CUlinkState* )stateOut);
    free(nvrtc_options);
    return cudaResultToUPTKresult(nvrtc_res);
}

//改动了
UPTKresult  UPTKLinkDestroy(UPTKlinkState state)
{
    CUresult nvrtc_res;
    nvrtc_res = cuLinkDestroy((CUlinkState) state);
    return cudaResultToUPTKresult(nvrtc_res);
}

//nvrtcJITInputType多处有问题
UPTKresult  UPTKLinkAddData(UPTKlinkState state,UPTKjitInputType type,void * data,size_t size,const char * name,unsigned int numOptions,UPTKjit_option * options,void ** optionValues)
{
    CUresult nvrtc_res;
    CUjitInputType nvrtc_type = UPTKjitInputTypeTonvrtcJITInputType(type);
    CUjit_option* nvrtc_options = (CUjit_option*)malloc(sizeof(CUjit_option) * numOptions);
    if (nullptr != options) {
        if (nullptr == nvrtc_options) {
          return UPTK_ERROR_OUT_OF_MEMORY;
        }
        for (int i = 0; i < numOptions; i++) {
            nvrtc_options[i] = UPTKjit_optionTonvrtcJIT_option(options[i]);
        }
    }
    nvrtc_res = cuLinkAddData((CUlinkState) state, nvrtc_type, data, size, name, numOptions, nvrtc_options, optionValues);
    free(nvrtc_options);
    return cudaResultToUPTKresult(nvrtc_res);
}

__host__  UPTKError_t  UPTKStreamCreateWithPriority(UPTKStream_t * pStream,unsigned int flags,int priority)
{
    cudaError_t cuda_res;
    cuda_res = cudaStreamCreateWithPriority((cudaStream_t *) pStream, flags, priority);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, UPTKStream_t stream)
{
    cudaError_t cuda_res;
    cuda_res = cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, (cudaStream_t)stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKMemcpyPeer(void * dst,int dstDevice,const void * src,int srcDevice,size_t count)
{
    cudaError_t cuda_res;
    cuda_res = cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
    return cudaErrorToUPTKError(cuda_res);
}

__host__  UPTKError_t  UPTKMemcpyAsync(void *dst, const void *src, size_t count, enum UPTKMemcpyKind kind, UPTKStream_t stream)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpyAsync(dst, src, count, cuda_kind, (cudaStream_t)stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__  UPTKError_t  UPTKMallocManaged(void ** devPtr,size_t size,unsigned int flags)
{
    if(size == 0) {
        if((devPtr == nullptr) ||((flags != UPTKMemAttachGlobal) && (flags != UPTKMemAttachHost)))
        {
            return UPTKErrorInvalidValue;
        }else{
            *devPtr = nullptr;
            return UPTKSuccess;
        }
    }
    cudaError_t cuda_res;
    cuda_res = cudaMallocManaged(devPtr, size, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKMallocAsync(void **devPtr, size_t size, UPTKStream_t hStream)
{
    cudaError_t cuda_res;
    cuda_res = cudaMallocAsync(devPtr, size, (cudaStream_t) hStream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKLaunchCooperativeKernelMultiDevice(struct UPTKLaunchParams *launchParamsList, unsigned int numDevices, unsigned int flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaLaunchCooperativeKernelMultiDevice((cudaLaunchParams *)launchParamsList, numDevices, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKGraphNodeGetType(UPTKGraphNode_t node,enum UPTKGraphNodeType * pType)
{
    if (nullptr == pType)
        return UPTKErrorInvalidValue;
    cudaGraphNodeType cuda_pType;
    cudaError_t cuda_res;
    cuda_res = cudaGraphNodeGetType((cudaGraphNode_t) node, &cuda_pType);
    if (cudaSuccess == cuda_res)
        *pType = cudaGraphNodeTypeToUPTKGraphNodeType(cuda_pType);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKGraphGetNodes(UPTKGraph_t graph,UPTKGraphNode_t * nodes,size_t * numNodes)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphGetNodes((cudaGraph_t) graph, (cudaGraphNode_t *) nodes, numNodes);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKGraphExecDestroy(UPTKGraphExec_t graphExec)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphExecDestroy((cudaGraphExec_t) graphExec);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKGraphDestroyNode(UPTKGraphNode_t node)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphDestroyNode((cudaGraphNode_t) node);
    return cudaErrorToUPTKError(cuda_res);
}

/*__host__ UPTKError_t  UPTKGraphAddNode(UPTKGraphNode_t *pGraphNode, UPTKGraph_t graph, const UPTKGraphNode_t *pDependencies, size_t numDependencies, struct UPTKGraphNodeParams *nodeParams)
{
    if(nodeParams == nullptr){
        return UPTKErrorInvalidValue;
    }
    cudaError_t cuda_res;
    cudaGraphNodeParams cuda_nodeParams = {};
    UPTKGraphNodeParamsTomcGraphNodeParams(nodeParams, &cuda_nodeParams);
    cuda_res = cudaGraphAddNode((cudaGraphNode_t*) pGraphNode, (cudaGraph_t) graph, (const cudaGraphNode_t*) pDependencies, numDependencies, &cuda_nodeParams);
    return cudaErrorToUPTKError(cuda_res);
}*/

__host__  UPTKError_t  UPTKGetLastError(void)
{
    cudaError_t cuda_res;
    cuda_res = cudaGetLastError();
    return cudaErrorToUPTKError(cuda_res);
}

__host__  const char *  UPTKGetErrorString(UPTKError_t error)
{
    cudaError_t cuda_error = UPTKErrorTocudaError(error);
    const char * cuda_res;
    cuda_res = cudaGetErrorString(cuda_error);
    return cuda_res;
}

__host__  const char *  UPTKGetErrorName(UPTKError_t error)
{
    switch (error) {
        case UPTKErrorAlreadyAcquired:
            return "UPTKErrorAlreadyAcquired";
        case UPTKErrorAlreadyMapped:
            return "UPTKErrorAlreadyMapped";
        case UPTKErrorArrayIsMapped:
            return "UPTKErrorArrayIsMapped";
        case UPTKErrorAssert:
            return "UPTKErrorAssert";
        case UPTKErrorCapturedEvent:
            return "UPTKErrorCapturedEvent";
        case UPTKErrorContextIsDestroyed:
            return "UPTKErrorContextIsDestroyed";
        case UPTKErrorCooperativeLaunchTooLarge:
            return "UPTKErrorCooperativeLaunchTooLarge";
        case UPTKErrorUPTKrtUnloading:
            return "UPTKErrorUPTKrtUnloading";
        case UPTKErrorDeviceAlreadyInUse:
            return "UPTKErrorDeviceAlreadyInUse";
        case UPTKErrorDeviceUninitialized:
            return "UPTKErrorDeviceUninitialized";
        case UPTKErrorECCUncorrectable:
            return "UPTKErrorECCUncorrectable";
        case UPTKErrorFileNotFound:
            return "UPTKErrorFileNotFound";
        case UPTKErrorGraphExecUpdateFailure:
            return "UPTKErrorGraphExecUpdateFailure";
        case UPTKErrorHostMemoryAlreadyRegistered:
            return "UPTKErrorHostMemoryAlreadyRegistered";
        case UPTKErrorHostMemoryNotRegistered:
            return "UPTKErrorHostMemoryNotRegistered";
        case UPTKErrorIllegalAddress:
            return "UPTKErrorIllegalAddress";
        case UPTKErrorIllegalState:
            return "UPTKErrorIllegalState";
        case UPTKErrorInitializationError:
            return "UPTKErrorInitializationError";
        case UPTKErrorInsufficientDriver:
            return "UPTKErrorInsufficientDriver";
        case UPTKErrorInvalidConfiguration:
            return "UPTKErrorInvalidConfiguration";
        case UPTKErrorInvalidDevice:
            return "UPTKErrorInvalidDevice";
        case UPTKErrorInvalidDeviceFunction:
            return "UPTKErrorInvalidDeviceFunction";
        case UPTKErrorInvalidDevicePointer:
            return "UPTKErrorInvalidDevicePointer";
        case UPTKErrorInvalidGraphicsContext:
            return "UPTKErrorInvalidGraphicsContext";
        case UPTKErrorInvalidKernelImage:
            return "UPTKErrorInvalidKernelImage";
        case UPTKErrorInvalidMemcpyDirection:
            return "UPTKErrorInvalidMemcpyDirection";
        case UPTKErrorInvalidPitchValue:
            return "UPTKErrorInvalidPitchValue";
        case UPTKErrorInvalidPtx:
            return "UPTKErrorInvalidPtx";
        case UPTKErrorInvalidResourceHandle:
            return "UPTKErrorInvalidResourceHandle";
        case UPTKErrorInvalidSource:
            return "UPTKErrorInvalidSource";
        case UPTKErrorInvalidSymbol:
            return "UPTKErrorInvalidSymbol";
        case UPTKErrorInvalidValue:
            return "UPTKErrorInvalidValue";
        case UPTKErrorLaunchFailure:
            return "UPTKErrorLaunchFailure";
        case UPTKErrorLaunchOutOfResources:
            return "UPTKErrorLaunchOutOfResources";
        case UPTKErrorLaunchTimeout:
            return "UPTKErrorLaunchTimeout";
        case UPTKErrorMapBufferObjectFailed:
            return "UPTKErrorMapBufferObjectFailed";
        case UPTKErrorMemoryAllocation:
            return "UPTKErrorMemoryAllocation";
        case UPTKErrorMissingConfiguration:
            return "UPTKErrorMissingConfiguration";
        case UPTKErrorNoDevice:
            return "UPTKErrorNoDevice";
        case UPTKErrorNoKernelImageForDevice:
            return "UPTKErrorNoKernelImageForDevice";
        case UPTKErrorNotMapped:
            return "UPTKErrorNotMapped";
        case UPTKErrorNotMappedAsArray:
            return "UPTKErrorNotMappedAsArray";
        case UPTKErrorNotMappedAsPointer:
            return "UPTKErrorNotMappedAsPointer";
        case UPTKErrorNotReady:
            return "UPTKErrorNotReady";
        case UPTKErrorNotSupported:
            return "UPTKErrorNotSupported";
        case UPTKErrorOperatingSystem:
            return "UPTKErrorOperatingSystem";
        case UPTKErrorPeerAccessAlreadyEnabled:
            return "UPTKErrorPeerAccessAlreadyEnabled";
        case UPTKErrorPeerAccessNotEnabled:
            return "UPTKErrorPeerAccessNotEnabled";
        case UPTKErrorPeerAccessUnsupported:
            return "UPTKErrorPeerAccessUnsupported";
        case UPTKErrorPriorLaunchFailure:
            return "UPTKErrorPriorLaunchFailure";
        case UPTKErrorProfilerAlreadyStarted:
            return "UPTKErrorProfilerAlreadyStarted";
        case UPTKErrorProfilerAlreadyStopped:
            return "UPTKErrorProfilerAlreadyStopped";
        case UPTKErrorProfilerDisabled:
            return "UPTKErrorProfilerDisabled";
        case UPTKErrorProfilerNotInitialized:
            return "UPTKErrorProfilerNotInitialized";
        case UPTKErrorSetOnActiveProcess:
            return "UPTKErrorSetOnActiveProcess";
        case UPTKErrorSharedObjectInitFailed:
            return "UPTKErrorSharedObjectInitFailed";
        case UPTKErrorSharedObjectSymbolNotFound:
            return "UPTKErrorSharedObjectSymbolNotFound";
        case UPTKErrorStreamCaptureImplicit:
            return "UPTKErrorStreamCaptureImplicit";
        case UPTKErrorStreamCaptureInvalidated:
            return "UPTKErrorStreamCaptureInvalidated";
        case UPTKErrorStreamCaptureIsolation:
            return "UPTKErrorStreamCaptureIsolation";
        case UPTKErrorStreamCaptureMerge:
            return "UPTKErrorStreamCaptureMerge";
        case UPTKErrorStreamCaptureUnjoined:
            return "UPTKErrorStreamCaptureUnjoined";
        case UPTKErrorStreamCaptureUnmatched:
            return "UPTKErrorStreamCaptureUnmatched";
        case UPTKErrorStreamCaptureUnsupported:
            return "UPTKErrorStreamCaptureUnsupported";
        case UPTKErrorStreamCaptureWrongThread:
            return "UPTKErrorStreamCaptureWrongThread";
        case UPTKErrorSymbolNotFound:
            return "UPTKErrorSymbolNotFound";
        case UPTKErrorUnmapBufferObjectFailed:
            return "UPTKErrorUnmapBufferObjectFailed";
        case UPTKErrorUnsupportedLimit:
            return "UPTKErrorUnsupportedLimit";
        case UPTKSuccess:
            return "UPTKSuccess";
        default:
            return "unrecognized error code";
    };
}

__host__  UPTKError_t  UPTKGetDeviceCount(int * count)
{
    cudaError_t cuda_res;
    cuda_res = cudaGetDeviceCount(count);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKFreeAsync(void *devPtr, UPTKStream_t hStream)
{
    if (devPtr == nullptr) {
        return UPTKSuccess;
    }
    cudaError_t cuda_res;
    cuda_res = cudaFreeAsync(devPtr, (cudaStream_t) hStream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKEventElapsedTime(float * ms,UPTKEvent_t start,UPTKEvent_t end)
{
    cudaError_t cuda_res;
    cuda_res = cudaEventElapsedTime(ms, (cudaEvent_t) start, (cudaEvent_t) end);
    return cudaErrorToUPTKError(cuda_res);
}

__host__  UPTKError_t  UPTKDeviceGetAttribute(int* value, enum UPTKDeviceAttr attr, int device) 
{
    if (value == nullptr) {
        return UPTKErrorInvalidValue;
    }
    int device_count = 0;
    cudaError_t cuda_res = cudaGetDeviceCount(&device_count);
    if (cuda_res != cudaSuccess) {
        return cudaErrorToUPTKError(cuda_res);
    }
    if (device < 0 || device >= device_count) {
        return UPTKErrorInvalidDevice;
    }
    cudaDeviceProp prop = {0};
    cuda_res = cudaGetDeviceProperties(&prop, device);
    if (cuda_res != cudaSuccess) {
        return cudaErrorToUPTKError(cuda_res);
    }

    UPTKError_t res = UPTKSuccess;
    bool is_handled = true;

    switch (attr) {
    case UPTKDevAttrMaxSurface2DLayeredWidth:
      *value = prop.maxSurface2DLayered[0];
      break;
    case UPTKDevAttrMaxSurface2DLayeredHeight:
      *value = prop.maxSurface2DLayered[1];
      break;
    case UPTKDevAttrMaxSurface2DLayeredLayers:
      *value = prop.maxSurface2DLayered[2];
      break;
    case UPTKDevAttrMaxTexture2DLinearWidth:
      *value = prop.maxTexture2DLinear[0];
      break;
    case UPTKDevAttrMaxTexture2DLinearHeight:
      *value = prop.maxTexture2DLinear[1];
      break;
    case UPTKDevAttrMaxTexture2DLinearPitch:
      *value = prop.maxTexture2DLinear[2];
      break;
    case UPTKDevAttrMaxSurface2DWidth:
      *value = prop.maxSurface2D[0];
      break;
    case UPTKDevAttrMaxSurface2DHeight:
      *value = prop.maxSurface2D[1];
      break;
    case UPTKDevAttrMaxSurface1DLayeredWidth:
      *value = prop.maxSurface1DLayered[0];
      break;
    case UPTKDevAttrMaxSurface1DLayeredLayers:
      *value = prop.maxSurface1DLayered[1];
      break;
    case UPTKDevAttrMaxTexture3DWidthAlt:
      *value = prop.maxTexture3DAlt[0];
      break;
    case UPTKDevAttrMaxTexture3DHeightAlt:
      *value = prop.maxTexture3DAlt[1];
      break;
    case UPTKDevAttrMaxTexture3DDepthAlt:
      *value = prop.maxTexture3DAlt[2];
      break;
    case UPTKDevAttrTccDriver:
      *value = 1;
      fprintf(stdout, "[Warning] Enabling TCC has no impact on functionality.To ensure compatibility, TCC has enabled simulated return value 1.\n");
      break;
    case UPTKDevAttrMaxTexture1DLayeredWidth:
      *value = prop.maxTexture1DLayered[0];
      break;
    case UPTKDevAttrMaxTexture1DLayeredLayers:
      *value = prop.maxTexture1DLayered[1];
      break;
    case UPTKDevAttrMaxTexture2DMipmappedWidth:
      *value = prop.maxTexture2DMipmap[0];
      break;
    case UPTKDevAttrMaxTexture2DMipmappedHeight:
      *value = prop.maxTexture2DMipmap[1];
      break;
    case UPTKDevAttrMaxTextureCubemapLayeredWidth:
      *value = prop.maxTextureCubemapLayered[0];
      break;
    case UPTKDevAttrMaxTextureCubemapLayeredLayers:
      *value = prop.maxTextureCubemapLayered[1];
      break;
    case UPTKDevAttrMaxTexture2DGatherWidth:
      *value = prop.maxTexture2DGather[0];
      break;
    case UPTKDevAttrMaxTexture2DGatherHeight:
      *value = prop.maxTexture2DGather[1];
      break;
    case UPTKDevAttrMaxTexture2DLayeredWidth:
      *value = prop.maxTexture2DLayered[0];
      break;
    case UPTKDevAttrMaxTexture2DLayeredHeight:
      *value = prop.maxTexture2DLayered[1];
      break;
    case UPTKDevAttrMaxTexture2DLayeredLayers:
      *value = prop.maxTexture2DLayered[2];
      break;
    case UPTKDevAttrMaxSurfaceCubemapLayeredWidth:
      *value = prop.maxSurfaceCubemapLayered[0];
      break;
    case UPTKDevAttrMaxSurfaceCubemapLayeredLayers:
      *value = prop.maxSurfaceCubemapLayered[1];
      break;
    case UPTKDevAttrMaxSurfaceCubemapWidth:
      *value = prop.maxSurfaceCubemap;
      break;
    case UPTKDevAttrSingleToDoublePrecisionPerfRatio:
      *value = prop.singleToDoublePrecisionPerfRatio;
      break;
    case UPTKDevAttrMaxSurface3DWidth:
      *value = prop.maxSurface3D[0];
      break;
    case UPTKDevAttrMaxSurface3DHeight:
      *value = prop.maxSurface3D[1];
      break;
    case UPTKDevAttrMaxSurface3DDepth:
      *value = prop.maxSurface3D[2];
      break;
    // The following three require specific evaluation assignments, temporary comments, and subsequent evaluation assignments.
    // case UPTKDevAttrCanFlushRemoteWrites: 
    //   *value = 0;
    //   break;
    // case UPTKDevAttrReserved92:
    //   *value = 0;
    //   break;
    // case UPTKDevAttrReserved93:
    //   *value = 0;
    //   break;
    default: 
    {
        cudaDeviceAttr cuda_attr = UPTKDeviceAttrTocudaDeviceAttribute(attr);
        cuda_res = cudaDeviceGetAttribute(value, cuda_attr, device);
        res = cudaErrorToUPTKError(cuda_res);
        // Do not change for the time being, the subsequent overall modification to the front of the overall implementation.
        if (res == UPTKSuccess) {
            if (cuda_attr == cudaDevAttrComputeCapabilityMajor) {
                *value = SM_VERSION_MAJOR;
            } else if (cuda_attr == cudaDevAttrComputeCapabilityMinor) {
                *value = SM_VERSION_MINOR;
            }
        }
        is_handled = false;
        break;
    }
    }

    if (is_handled && res == UPTKSuccess) {
        res = UPTKSuccess;
    }

    return res;
}

__host__ UPTKError_t  UPTKDeviceEnablePeerAccess(int peerDevice,unsigned int flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaDeviceEnablePeerAccess(peerDevice, flags);
    return cudaErrorToUPTKError(cuda_res);
}
//改动了
UPTKresult  UPTKLinkComplete(UPTKlinkState state,void ** cubinOut,size_t * sizeOut)
{
    CUresult nvrtc_res;
    nvrtc_res = cuLinkComplete((CUlinkState)state, cubinOut, sizeOut);
    return cudaResultToUPTKresult(nvrtc_res);
}

UPTKresult  UPTKInit(unsigned int Flags)
{
    CUresult cuda_res;
    cuda_res = cuInit(Flags);
    return cudaErrorToUPTKresult(cuda_res);
}

__host__  UPTKError_t  UPTKFuncSetAttribute(const void * func,enum UPTKFuncAttribute attr,int value)
{
    cudaFuncAttribute cuda_attr = UPTKFuncAttributeTocudaFuncAttribute(attr);
    cudaError_t cuda_res;
    cuda_res = cudaFuncSetAttribute(func, cuda_attr, value);
    return cudaErrorToUPTKError(cuda_res);
}
//改动多
UPTKresult  UPTKDeviceGetName(char * name,int len,UPTKdevice dev)
{
    CUresult cuda_res;
    cuda_res = cuDeviceGetName(name, len, (CUdevice) dev);
    /*
    // To enable users to obtain architecture information through the interface, add the architecture information to the device name
    if(cuda_res == cudaSuccess){
        cudaDeviceProp cuda_prop;
        memset(&cuda_prop, 0, sizeof(cudaDeviceProp));
        cudaError_t cuda_res2 = cudaGetDeviceProperties(&cuda_prop, dev);
        if (cuda_res2 == cudaSuccess)
        {
            // Add the gcnArchName of hip to the device name
            strncpy(name, cuda_prop.name, len - 1);
            name[len - 1] = '\0';
            
            if (strlen(name) + strlen(cuda_prop.gcnArchName)  < len - 1)
            {
                strncat(name, " ", len - strlen(name) - 1);
                strncat(name, cuda_prop.gcnArchName, len - strlen(name) - 1);
            }
            else
            {
                return cudaErrorToUPTKresult(cuda_res2);
            }
  
        }
    }
    */
    return cudaErrorToUPTKresult(cuda_res);
}

UPTKresult  UPTKCtxSynchronize(void)
{
    CUresult cuda_res;
    cuda_res = cuCtxSynchronize();
    return cudaErrorToUPTKresult(cuda_res);
}

UPTKresult  UPTKCtxSetLimit(UPTKlimit limit,size_t value)
{
    CUlimit cuda_limit = UPTKlimitToCUlimit(limit);
    CUresult cuda_res;
    cuda_res = cuCtxSetLimit(cuda_limit, value);
    return cudaErrorToUPTKresult(cuda_res);
}

UPTKresult  UPTKCtxSetCurrent(UPTKcontext ctx)
{
    CUresult cuda_res;
    cuda_res = cuCtxSetCurrent((CUcontext) ctx);
    return cudaErrorToUPTKresult(cuda_res);
}

UPTKresult  UPTKCtxGetLimit(size_t * pvalue,UPTKlimit limit)
{
    CUlimit cuda_limit = UPTKlimitToCUlimit(limit);
    CUresult cuda_res;
    cuda_res = cuCtxGetLimit(pvalue, cuda_limit);
    return cudaErrorToUPTKresult(cuda_res);
}

UPTKresult  UPTKCtxDestroy(UPTKcontext ctx)
{
    CUresult cuda_res;
    cuda_res = cuCtxDestroy((CUcontext) ctx);
    return cudaErrorToUPTKresult(cuda_res);
}

UPTKresult  UPTKCtxCreate(UPTKcontext * pctx,unsigned int flags,UPTKdevice dev)
{
    CUresult cuda_res;
    cuda_res = cuCtxCreate((CUcontext *)pctx, flags, (CUdevice) dev);
    return cudaErrorToUPTKresult(cuda_res);
}

// Based on the newly added application interface
__host__ UPTKError_t UPTKMemset(void * devPtr,int value,size_t count)
{
    cudaError_t cuda_res;
    cuda_res = cudaMemset(devPtr, value, count);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t UPTKMallocHost(void ** ptr,size_t size)
{
    cudaError_t cuda_res;
    cuda_res = cudaMallocHost(ptr, size);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t UPTKFreeHost(void * ptr)
{
    cudaError_t cuda_res;
    cuda_res = cudaFreeHost(ptr);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKOccupancyMaxPotentialBlockSize_internal(int* gridSize, int* blockSize, const void* f, size_t dynSharedMemPerBlk, int blockSizeLimit){

    if ((gridSize == nullptr) || (blockSize == nullptr)) {
        return UPTKErrorInvalidValue;
    }
    cudaError_t cuda_res;
    cuda_res = cudaOccupancyMaxPotentialBlockSize(gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, enum UPTKMemcpyKind kind)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpyToSymbol(symbol, src, count, offset, cuda_kind);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError_t  UPTKHostAlloc(void ** pHost,size_t size,unsigned int flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaMallocHost(pHost, size, flags);
    return cudaErrorToUPTKError(cuda_res);
}

/*__host__  UPTKError_t  UPTKGetDeviceProperties_v2(struct UPTKDeviceProp * prop,int device)
{
    if(nullptr == prop)
    {
        return UPTKErrorInvalidValue;
    }

    memset(prop, 0, sizeof(UPTKDeviceProp));

    cudaDeviceProp cuda_prop;
    memset(&cuda_prop, 0, sizeof(UPTKDeviceProp));

    cudaError_t cuda_res;
    cuda_res = cudaGetDeviceProperties_v2(&cuda_prop, device);
    if ( cuda_res == cudaSuccess )
    {
        cudaDevicePropToUPTKDeviceProp(&cuda_prop, prop);
    }
    return cudaErrorToUPTKError(cuda_res);
}*/



#if defined(__cplusplus)
}
#endif /* __cplusplus */

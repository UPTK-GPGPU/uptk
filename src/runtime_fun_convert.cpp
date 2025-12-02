#include "runtime.hpp"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

__host__ UPTKError_t UPTKMalloc(void ** devPtr,size_t size)
{
    hipError_t hip_res;
    hip_res = hipMalloc(devPtr, size);
    return hipErrorToUPTKError(hip_res);
}

__host__  UPTKError_t  UPTKFree(void * devPtr)
{
    hipError_t hip_res;
    hip_res = hipFree(devPtr);
    return hipErrorToUPTKError(hip_res);
}

UPTKError_t __UPTKPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem , UPTKStream_t stream) {
    hipError_t res;
    res = __hipPushCallConfiguration(gridDim, blockDim, sharedMem, (hipStream_t)stream);
    return hipErrorToUPTKError(res);
}

UPTKError_t __UPTKPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, UPTKStream_t *stream) {
    hipError_t res;
    res = __hipPopCallConfiguration(gridDim, blockDim, sharedMem, (hipStream_t *) stream);
    return hipErrorToUPTKError(res);
}

__host__ UPTKError_t  UPTKStreamCreate(UPTKStream_t * pStream)
{
    hipError_t hip_res;
    hip_res = hipStreamCreate((hipStream_t *) pStream);
    return hipErrorToUPTKError(hip_res);
}

__host__  UPTKError_t  UPTKStreamCreateWithFlags(UPTKStream_t * pStream,unsigned int flags)
{
    hipError_t hip_res;
    hip_res = hipStreamCreateWithFlags((hipStream_t *) pStream, flags);
    return hipErrorToUPTKError(hip_res);
}

__host__  UPTKError_t  UPTKStreamDestroy(UPTKStream_t stream)
{
    hipError_t hip_res;
    hip_res = hipStreamDestroy((hipStream_t) stream);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKStreamSynchronize(UPTKStream_t stream)
{
    hipError_t hip_res;
    hip_res = hipStreamSynchronize((hipStream_t) stream);
    return hipErrorToUPTKError(hip_res);
}

__host__  UPTKError_t  UPTKStreamWaitEvent(UPTKStream_t stream,UPTKEvent_t event,unsigned int flags)
{
    hipError_t hip_res;
    hip_res = hipStreamWaitEvent((hipStream_t) stream, (hipEvent_t) event, flags);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKStreamBeginCapture(UPTKStream_t stream,enum UPTKStreamCaptureMode mode)
{
    if (mode < UPTKStreamCaptureModeGlobal || mode > UPTKStreamCaptureModeRelaxed)
    {
        return UPTKErrorInvalidValue;
    }
    hipStreamCaptureMode hip_mode = UPTKStreamCaptureModeTohipStreamCaptureMode(mode);
    hipError_t hip_res;
    hip_res = hipStreamBeginCapture((hipStream_t) stream, hip_mode);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKStreamEndCapture(UPTKStream_t stream,UPTKGraph_t * pGraph)
{
    hipError_t hip_res;
    hip_res = hipStreamEndCapture((hipStream_t) stream, (hipGraph_t *) pGraph);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKStreamIsCapturing(UPTKStream_t stream,enum UPTKStreamCaptureStatus * pCaptureStatus)
{
    if (nullptr == pCaptureStatus)
        return UPTKErrorInvalidValue;
    hipStreamCaptureStatus hip_pCaptureStatus;
    hipError_t hip_res;
    hip_res = hipStreamIsCapturing((hipStream_t) stream, &hip_pCaptureStatus);
    *pCaptureStatus = hipStreamCaptureStatusToUPTKStreamCaptureStatus(hip_pCaptureStatus);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKEventCreate(UPTKEvent_t * event)
{
    hipError_t hip_res;
    hip_res = hipEventCreate((hipEvent_t *)event);
    return hipErrorToUPTKError(hip_res);
}

__host__  UPTKError_t  UPTKEventCreateWithFlags(UPTKEvent_t * event,unsigned int flags)
{
    hipError_t hip_res;
    hip_res = hipEventCreateWithFlags((hipEvent_t *)event, flags);
    return hipErrorToUPTKError(hip_res);
}

__host__  UPTKError_t  UPTKEventDestroy(UPTKEvent_t event)
{
    hipError_t hip_res;
    hip_res = hipEventDestroy((hipEvent_t) event);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKEventSynchronize(UPTKEvent_t event)
{
    hipError_t hip_res;
    hip_res = hipEventSynchronize((hipEvent_t) event);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKEventQuery(UPTKEvent_t event)
{
    hipError_t hip_res;
    hip_res = hipEventQuery((hipEvent_t) event);
    return hipErrorToUPTKError(hip_res);
}

__host__  UPTKError_t  UPTKEventRecord(UPTKEvent_t event,UPTKStream_t stream)
{
    hipError_t hip_res;
    hip_res = hipEventRecord((hipEvent_t) event, (hipStream_t) stream);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKMemcpy(void * dst,const void * src,size_t count,enum UPTKMemcpyKind kind)
{
    hipMemcpyKind hip_kind = UPTKMemcpyKindTohipMemcpyKind(kind);
    hipError_t hip_res;
    hip_res = hipMemcpy(dst, src, count, hip_kind);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKIpcGetMemHandle(UPTKIpcMemHandle_t * handle,void * devPtr)
{
    hipError_t hip_res;
    hip_res = hipIpcGetMemHandle((hipIpcMemHandle_t *) handle, devPtr);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKIpcOpenMemHandle(void ** devPtr,UPTKIpcMemHandle_t handle,unsigned int flags)
{
    hipIpcMemHandle_t hip_handle;
    UPTKIpcMemHandleTohipIpcMemHandle(&handle, &hip_handle);
    if (UPTKIpcMemLazyEnablePeerAccess == flags)
        flags = hipIpcMemLazyEnablePeerAccess;
    hipError_t hip_res;
    hip_res = hipIpcOpenMemHandle(devPtr, hip_handle, flags);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKIpcCloseMemHandle(void * devPtr)
{
    hipError_t hip_res;
    hip_res = hipIpcCloseMemHandle(devPtr);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKChooseDevice(int * device,const struct UPTKDeviceProp * prop)
{
    if (nullptr == device || nullptr == prop)
        return UPTKErrorInvalidValue;
    hipDeviceProp_t hip_prop;
    UPTKDevicePropTohipDeviceProp(prop, &hip_prop);
    hipError_t hip_res;
    hip_res = hipChooseDevice(device, &hip_prop);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKSetDevice(int device)
{
    hipError_t hip_res;
    hip_res = hipSetDevice(device);
    return hipErrorToUPTKError(hip_res);
}


__host__  UPTKError_t  UPTKGetDevice(int * device)
{
    hipError_t hip_res;
    hip_res = hipGetDevice(device);
    return hipErrorToUPTKError(hip_res);
}

__host__  UPTKError_t  UPTKDeviceSynchronize(void)
{
    hipError_t hip_res;
    hip_res = hipDeviceSynchronize();
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKDeviceReset(void)
{
    hipError_t hip_res;
    hip_res = hipDeviceReset();
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKGraphCreate(UPTKGraph_t * pGraph,unsigned int flags)
{
    hipError_t hip_res;
    hip_res = hipGraphCreate((hipGraph_t *) pGraph, flags);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKGraphClone(UPTKGraph_t * pGraphClone,UPTKGraph_t originalGraph)
{
    hipError_t hip_res;
    hip_res = hipGraphClone((hipGraph_t *) pGraphClone, (hipGraph_t) originalGraph);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKGraphDestroy(UPTKGraph_t graph)
{
    hipError_t hip_res;
    hip_res = hipGraphDestroy((hipGraph_t) graph);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKGraphAddDependencies(UPTKGraph_t graph,const UPTKGraphNode_t * from,const UPTKGraphNode_t * to,size_t numDependencies)
{
    hipError_t hip_res;
    hip_res = hipGraphAddDependencies((hipGraph_t) graph, (const hipGraphNode_t *) from, (const hipGraphNode_t *) to, numDependencies);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKGraphRemoveDependencies(UPTKGraph_t graph,const UPTKGraphNode_t * from,const UPTKGraphNode_t * to,size_t numDependencies)
{
    hipError_t hip_res;
    hip_res = hipGraphRemoveDependencies((hipGraph_t) graph, (const hipGraphNode_t *) from, (const hipGraphNode_t *) to, numDependencies);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKGraphInstantiate(UPTKGraphExec_t * pGraphExec,UPTKGraph_t graph,UPTKGraphNode_t * pErrorNode,char * pLogBuffer,size_t bufferSize)
{
    hipError_t hip_res;
    hip_res = hipGraphInstantiate((hipGraphExec_t *) pGraphExec, (hipGraph_t) graph, (hipGraphNode_t *) pErrorNode, pLogBuffer, bufferSize);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKGraphLaunch(UPTKGraphExec_t graphExec,UPTKStream_t stream)
{
    hipError_t hip_res;
    hip_res = hipGraphLaunch((hipGraphExec_t) graphExec, (hipStream_t) stream);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKGraphExecUpdate(UPTKGraphExec_t hGraphExec,UPTKGraph_t hGraph,UPTKGraphNode_t * hErrorNode_out,enum UPTKGraphExecUpdateResult * updateResult_out)
{
    if (nullptr == updateResult_out)
        return UPTKErrorInvalidValue;
    hipGraphExecUpdateResult hip_updateResult_out;
    hipError_t hip_res;
    hip_res = hipGraphExecUpdate((hipGraphExec_t) hGraphExec, (hipGraph_t) hGraph, (hipGraphNode_t *) hErrorNode_out, &hip_updateResult_out);
    // The return value hipErrorGraphExecUpdateFailure is also a valuable output.
    if (hipSuccess == hip_res || hipErrorGraphExecUpdateFailure == hip_res)
        *updateResult_out = hipGraphExecUpdateResultToUPTKGraphExecUpdateResult(hip_updateResult_out);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, UPTKStream_t stream)
{
    hipError_t res;
    res = hipLaunchKernel(func, gridDim, blockDim, args, sharedMem, (hipStream_t)stream);
    return hipErrorToUPTKError(res);
}

__host__  UPTKError_t  UPTKFuncGetAttributes(struct UPTKFuncAttributes * attr,const void * func)
{
    if(nullptr == attr)
    {
        return UPTKErrorInvalidValue;
    }

    memset(attr, 0, sizeof(UPTKFuncAttributes));

    struct hipFuncAttributes hip_attr;
    memset(&hip_attr, 0, sizeof(hipFuncAttributes));

    hipError_t hip_res;
    hip_res = hipFuncGetAttributes(&hip_attr, func);
    if ( hip_res == hipSuccess )
    {
        hipFuncAttributesToUPTKFuncAttributes(&hip_attr, attr);
    }
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t UPTKStreamGetDevice(UPTKStream_t stream, UPTKdevice * device)
{
    hipError_t hip_res;
    hip_res = hipStreamGetDevice((hipStream_t) stream, (hipDevice_t*) device);
    return hipErrorToUPTKError(hip_res);
}

UPTKresult  UPTKModuleLoad(UPTKmodule * module,const char * fname)
{
    hipError_t hip_res;
    hip_res = hipModuleLoad((hipModule_t *)module, fname);
    return hipErrorToUPTKresult(hip_res);
}

UPTKresult  UPTKModuleLoadData(UPTKmodule * module,const void * image)
{
    hipError_t hip_res;
    hip_res = hipModuleLoadData((hipModule_t *)module, image);
    return hipErrorToUPTKresult(hip_res);
}

UPTKresult  UPTKModuleUnload(UPTKmodule hmod)
{
    hipError_t hip_res;
    hip_res = hipModuleUnload((hipModule_t) hmod);
    return hipErrorToUPTKresult(hip_res);
}

UPTKresult  UPTKModuleGetFunction(UPTKfunction * hfunc,UPTKmodule hmod,const char * name)
{
    hipError_t hip_res;
    hip_res = hipModuleGetFunction((hipFunction_t *)hfunc, (hipModule_t) hmod, name);
    return hipErrorToUPTKresult(hip_res);
}

UPTKresult  UPTKModuleGetGlobal(UPTKdeviceptr * dptr, size_t * bytes,UPTKmodule hmod,const char * name)
{
    hipError_t hip_res;
    hip_res = hipModuleGetGlobal((hipDeviceptr_t *)dptr, bytes, (hipModule_t) hmod, name);
    return hipErrorToUPTKresult(hip_res);
}


UPTKresult  UPTKLinkCreate(unsigned int numOptions,UPTKjit_option * options,void ** optionValues,UPTKlinkState * stateOut)
{
    hiprtcResult hip_res;
    hiprtcJIT_option* hiprtc_options = (hiprtcJIT_option*)malloc(sizeof(hiprtcJIT_option) * numOptions);
    if (nullptr != options) {
        if (nullptr == hiprtc_options) {
          return UPTK_ERROR_OUT_OF_MEMORY;
        }
        for (int i = 0; i < numOptions; i++) {
            hiprtc_options[i] = UPTKjit_optionTohiprtcJIT_option(options[i]);
        }
    }
    hip_res = hiprtcLinkCreate(numOptions, hiprtc_options, optionValues, (hiprtcLinkState* )stateOut);
    free(hiprtc_options);
    return hiprtcResultToUPTKresult(hip_res);
}

UPTKresult  UPTKLinkDestroy(UPTKlinkState state)
{
    hiprtcResult hip_res;
    hip_res = hiprtcLinkDestroy((hiprtcLinkState) state);
    return hiprtcResultToUPTKresult(hip_res);
}

UPTKresult  UPTKLinkAddData(UPTKlinkState state,UPTKjitInputType type,void * data,size_t size,const char * name,unsigned int numOptions,UPTKjit_option * options,void ** optionValues)
{
    hiprtcResult hip_res;
    hiprtcJITInputType hiprtc_type = UPTKjitInputTypeTohiprtcJITInputType(type);
    hiprtcJIT_option* hiprtc_options = (hiprtcJIT_option*)malloc(sizeof(hiprtcJIT_option) * numOptions);
    if (nullptr != options) {
        if (nullptr == hiprtc_options) {
          return UPTK_ERROR_OUT_OF_MEMORY;
        }
        for (int i = 0; i < numOptions; i++) {
            hiprtc_options[i] = UPTKjit_optionTohiprtcJIT_option(options[i]);
        }
    }
    hip_res = hiprtcLinkAddData((hiprtcLinkState) state, hiprtc_type, data, size, name, numOptions, hiprtc_options, optionValues);
    free(hiprtc_options);
    return hiprtcResultToUPTKresult(hip_res);
}

__host__  UPTKError_t  UPTKStreamCreateWithPriority(UPTKStream_t * pStream,unsigned int flags,int priority)
{
    hipError_t hip_res;
    hip_res = hipStreamCreateWithPriority((hipStream_t *) pStream, flags, priority);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, UPTKStream_t stream)
{
    hipError_t hip_res;
    hip_res = hipMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, (hipStream_t)stream);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKMemcpyPeer(void * dst,int dstDevice,const void * src,int srcDevice,size_t count)
{
    hipError_t hip_res;
    hip_res = hipMemcpyPeer(dst, dstDevice, src, srcDevice, count);
    return hipErrorToUPTKError(hip_res);
}

__host__  UPTKError_t  UPTKMemcpyAsync(void *dst, const void *src, size_t count, enum UPTKMemcpyKind kind, UPTKStream_t stream)
{
    hipMemcpyKind hip_kind = UPTKMemcpyKindTohipMemcpyKind(kind);
    hipError_t hip_res;
    hip_res = hipMemcpyAsync(dst, src, count, hip_kind, (hipStream_t)stream);
    return hipErrorToUPTKError(hip_res);
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
    hipError_t hip_res;
    hip_res = hipMallocManaged(devPtr, size, flags);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKMallocAsync(void **devPtr, size_t size, UPTKStream_t hStream)
{
    hipError_t hip_res;
    hip_res = hipMallocAsync(devPtr, size, (hipStream_t) hStream);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKLaunchCooperativeKernelMultiDevice(struct UPTKLaunchParams *launchParamsList, unsigned int numDevices, unsigned int flags)
{
    hipError_t hip_res;
    hip_res = hipLaunchCooperativeKernelMultiDevice((hipLaunchParams *)launchParamsList, numDevices, flags);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKGraphNodeGetType(UPTKGraphNode_t node,enum UPTKGraphNodeType * pType)
{
    if (nullptr == pType)
        return UPTKErrorInvalidValue;
    hipGraphNodeType hip_pType;
    hipError_t hip_res;
    hip_res = hipGraphNodeGetType((hipGraphNode_t) node, &hip_pType);
    if (hipSuccess == hip_res)
        *pType = hipGraphNodeTypeToUPTKGraphNodeType(hip_pType);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKGraphGetNodes(UPTKGraph_t graph,UPTKGraphNode_t * nodes,size_t * numNodes)
{
    hipError_t hip_res;
    hip_res = hipGraphGetNodes((hipGraph_t) graph, (hipGraphNode_t *) nodes, numNodes);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKGraphExecDestroy(UPTKGraphExec_t graphExec)
{
    hipError_t hip_res;
    hip_res = hipGraphExecDestroy((hipGraphExec_t) graphExec);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKGraphDestroyNode(UPTKGraphNode_t node)
{
    hipError_t hip_res;
    hip_res = hipGraphDestroyNode((hipGraphNode_t) node);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKGraphAddNode(UPTKGraphNode_t *pGraphNode, UPTKGraph_t graph, const UPTKGraphNode_t *pDependencies, size_t numDependencies, struct UPTKGraphNodeParams *nodeParams)
{
    if(nodeParams == nullptr){
        return UPTKErrorInvalidValue;
    }
    hipError_t hip_res;
    hipGraphNodeParams hip_nodeParams = {};
    UPTKGraphNodeParamsTohipGraphNodeParams(nodeParams, &hip_nodeParams);
    hip_res = hipGraphAddNode((hipGraphNode_t*) pGraphNode, (hipGraph_t) graph, (const hipGraphNode_t*) pDependencies, numDependencies, &hip_nodeParams);
    return hipErrorToUPTKError(hip_res);
}

__host__  UPTKError_t  UPTKGetLastError(void)
{
    hipError_t hip_res;
    hip_res = hipGetLastError();
    return hipErrorToUPTKError(hip_res);
}

__host__  const char *  UPTKGetErrorString(UPTKError_t error)
{
    hipError_t hip_hipError = UPTKErrorTohipError(error);
    const char * hip_res;
    hip_res = hipGetErrorString(hip_hipError);
    return hip_res;
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
    hipError_t hip_res;
    hip_res = hipGetDeviceCount(count);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKFreeAsync(void *devPtr, UPTKStream_t hStream)
{
    if (devPtr == nullptr) {
        return UPTKSuccess;
    }
    hipError_t hip_res;
    hip_res = hipFreeAsync(devPtr, (hipStream_t) hStream);
    return hipErrorToUPTKError(hip_res);
}

__host__ UPTKError_t  UPTKEventElapsedTime(float * ms,UPTKEvent_t start,UPTKEvent_t end)
{
    hipError_t hip_res;
    hip_res = hipEventElapsedTime(ms, (hipEvent_t) start, (hipEvent_t) end);
    return hipErrorToUPTKError(hip_res);
}

__host__  UPTKError_t  UPTKDeviceGetAttribute(int* value, enum UPTKDeviceAttr attr, int device) 
{
    if (value == nullptr) {
        return UPTKErrorInvalidValue;
    }
    int device_count = 0;
    hipError_t hip_res = hipGetDeviceCount(&device_count);
    if (hip_res != hipSuccess) {
        return hipErrorToUPTKError(hip_res);
    }
    if (device < 0 || device >= device_count) {
        return UPTKErrorInvalidDevice;
    }
    hipDeviceProp_t_v2 prop = {0};
    hip_res = hipGetDeviceProperties_v2(&prop, device);
    if (hip_res != hipSuccess) {
        return hipErrorToUPTKError(hip_res);
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
        hipDeviceAttribute_t hip_attr = UPTKDeviceAttrTohipDeviceAttribute(attr);
        hip_res = hipDeviceGetAttribute_v2(value, hip_attr, device);
        res = hipErrorToUPTKError(hip_res);
        // Do not change for the time being, the subsequent overall modification to the front of the overall implementation.
        if (res == UPTKSuccess) {
            if (hip_attr == hipDeviceAttributeComputeCapabilityMajor) {
                *value = SM_VERSION_MAJOR;
            } else if (hip_attr == hipDeviceAttributeComputeCapabilityMinor) {
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
    hipError_t hip_res;
    hip_res = hipDeviceEnablePeerAccess(peerDevice, flags);
    return hipErrorToUPTKError(hip_res);
}

UPTKresult  UPTKLinkComplete(UPTKlinkState state,void ** cubinOut,size_t * sizeOut)
{
    hiprtcResult hip_res;
    hip_res = hiprtcLinkComplete((hiprtcLinkState)state, cubinOut, sizeOut);
    return hiprtcResultToUPTKresult(hip_res);
}

UPTKresult  UPTKInit(unsigned int Flags)
{
    hipError_t hip_res;
    hip_res = hipInit(Flags);
    return hipErrorToUPTKresult(hip_res);
}

__host__  UPTKError_t  UPTKFuncSetAttribute(const void * func,enum UPTKFuncAttribute attr,int value)
{
    hipFuncAttribute hip_attr = UPTKFuncAttributeTohipFuncAttribute(attr);
    hipError_t hip_res;
    hip_res = hipFuncSetAttribute(func, hip_attr, value);
    return hipErrorToUPTKError(hip_res);
}

UPTKresult  UPTKDeviceGetName(char * name,int len,UPTKdevice dev)
{
    hipError_t hip_res;
    hip_res = hipDeviceGetName(name, len, (hipDevice_t) dev);
    // To enable users to obtain architecture information through the interface, add the architecture information to the device name
    if(hip_res == hipSuccess){
        hipDeviceProp_t hip_prop;
        memset(&hip_prop, 0, sizeof(hipDeviceProp_t));
        hipError_t hip_res2 = hipGetDeviceProperties(&hip_prop, dev);
        if (hip_res2 == hipSuccess)
        {
            // Add the gcnArchName of hip to the device name
            strncpy(name, hip_prop.name, len - 1);
            name[len - 1] = '\0';
            if (strlen(name) + strlen(hip_prop.gcnArchName)  < len - 1)
            {
                strncat(name, " ", len - strlen(name) - 1);
                strncat(name, hip_prop.gcnArchName, len - strlen(name) - 1);
            }
            else
            {
                return hipErrorToUPTKresult(hip_res2);
            }
        }
    }
    return hipErrorToUPTKresult(hip_res);
}

UPTKresult  UPTKCtxSynchronize(void)
{
    hipError_t hip_res;
    hip_res = hipDeviceSynchronize();
    return hipErrorToUPTKresult(hip_res);
}

UPTKresult  UPTKCtxSetLimit(UPTKlimit limit,size_t value)
{
    enum hipLimit_t hip_limit = UPTKlimitTohipLimit(limit);
    hipError_t hip_res;
    hip_res = hipDeviceSetLimit(hip_limit, value);
    return hipErrorToUPTKresult(hip_res);
}

UPTKresult  UPTKCtxSetCurrent(UPTKcontext ctx)
{
    hipError_t hip_res;
    hip_res = hipCtxSetCurrent((hipCtx_t) ctx);
    return hipErrorToUPTKresult(hip_res);
}

UPTKresult  UPTKCtxGetLimit(size_t * pvalue,UPTKlimit limit)
{
    enum hipLimit_t hip_limit = UPTKlimitTohipLimit(limit);
    hipError_t hip_res;
    hip_res = hipDeviceGetLimit(pvalue, hip_limit);
    return hipErrorToUPTKresult(hip_res);
}

UPTKresult  UPTKCtxDestroy(UPTKcontext ctx)
{
    hipError_t hip_res;
    hip_res = hipCtxDestroy((hipCtx_t) ctx);
    return hipErrorToUPTKresult(hip_res);
}

UPTKresult  UPTKCtxCreate(UPTKcontext * pctx,unsigned int flags,UPTKdevice dev)
{
    hipError_t hip_res;
    hip_res = hipCtxCreate((hipCtx_t *)pctx, flags, (hipDevice_t) dev);
    return hipErrorToUPTKresult(hip_res);
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */
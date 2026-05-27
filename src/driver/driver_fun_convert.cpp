#include "driver.hpp"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

UPTKError UPArray3DCreate(UPTKarray * pHandle,const UPTK_ARRAY3D_DESCRIPTOR * pAllocateArray)
{
    CUresult cu_res;
    cu_res = cuArray3DCreate((CUarray *)pHandle, (const CUDA_ARRAY3D_DESCRIPTOR *)pAllocateArray);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPArray3DGetDescriptor(UPTK_ARRAY3D_DESCRIPTOR * pArrayDescriptor, UPTKarray hArray)
{
    CUresult cu_res;
    cu_res = cuArray3DGetDescriptor((CUDA_ARRAY3D_DESCRIPTOR *)pArrayDescriptor, (CUarray)hArray);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPArrayCreate(UPTKarray * pHandle,const UPTK_ARRAY_DESCRIPTOR * pAllocateArray)
{
    CUresult cu_res;
    cu_res = cuArrayCreate((CUarray *)pHandle, (const CUDA_ARRAY_DESCRIPTOR *)pAllocateArray);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPArrayDestroy(UPTKarray hArray)
{
    CUresult cu_res;
    cu_res = cuArrayDestroy((CUarray)hArray);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPArrayGetDescriptor(UPTK_ARRAY_DESCRIPTOR * pArrayDescriptor,UPTKarray hArray)
{
    CUresult cu_res;
    cu_res = cuArrayGetDescriptor((CUDA_ARRAY_DESCRIPTOR *)pArrayDescriptor, (CUarray)hArray);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxAttach(UPTKcontext * pctx,unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuCtxAttach((CUcontext *)pctx, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxCreate(UPTKcontext * pctx,unsigned int flags,UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuCtxCreate((CUcontext *)pctx, flags, (CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxDestroy(UPTKcontext ctx)
{
    CUresult cu_res;
    cu_res = cuCtxDestroy((CUcontext)ctx);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxDetach(UPTKcontext ctx)
{
    CUresult cu_res;
    cu_res = cuCtxDetach((CUcontext)ctx);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxDisablePeerAccess(UPTKcontext peerContext)
{
    CUresult cu_res;
    cu_res = cuCtxDisablePeerAccess((CUcontext)peerContext);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxEnablePeerAccess(UPTKcontext peerContext,unsigned int Flags)
{
    CUresult cu_res;
    cu_res = cuCtxEnablePeerAccess((CUcontext)peerContext, Flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxGetApiVersion(UPTKcontext ctx,unsigned int * version)
{
    CUresult cu_res;
    cu_res = cuCtxGetApiVersion((CUcontext)ctx, version);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxGetCacheConfig(UPTKfunc_cache * pconfig)
{
    CUresult cu_res;
    cu_res = cuCtxGetCacheConfig((CUfunc_cache *)pconfig);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxGetCurrent(UPTKcontext * pctx)
{
    CUresult cu_res;
    cu_res = cuCtxGetCurrent((CUcontext *)pctx);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxGetDevice(UPTKdevice * device)
{
    CUresult cu_res;
    cu_res = cuCtxGetDevice((CUdevice *)device);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxGetFlags(unsigned int * flags)
{
    CUresult cu_res;
    cu_res = cuCtxGetFlags(flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxGetLimit(size_t * pvalue,UPTKlimit limit)
{
    CUresult cu_res;
    cu_res = cuCtxGetLimit(pvalue, (CUlimit)limit);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxGetSharedMemConfig(UPTKsharedconfig * pConfig)
{
    CUresult cu_res;
    cu_res = cuCtxGetSharedMemConfig((CUsharedconfig *)pConfig);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxGetStreamPriorityRange(int * leastPriority,int * greatestPriority)
{
    CUresult cu_res;
    cu_res = cuCtxGetStreamPriorityRange(leastPriority, greatestPriority);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxPopCurrent(UPTKcontext * pctx)
{
    CUresult cu_res;
    cu_res = cuCtxPopCurrent((CUcontext *)pctx);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxPushCurrent(UPTKcontext ctx)
{
    CUresult cu_res;
    cu_res = cuCtxPushCurrent((CUcontext)ctx);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxSetCacheConfig(UPTKfunc_cache config)
{
    CUresult cu_res;
    cu_res = cuCtxSetCacheConfig((CUfunc_cache)config);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxSetCurrent(UPTKcontext ctx)
{
    CUresult cu_res;
    cu_res = cuCtxSetCurrent((CUcontext)ctx);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxSetLimit(UPTKlimit limit,size_t value)
{
    CUresult cu_res;
    cu_res = cuCtxSetLimit((CUlimit)limit, value);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxSetSharedMemConfig(UPTKsharedconfig config)
{
    CUresult cu_res;
    cu_res = cuCtxSetSharedMemConfig((CUsharedconfig)config);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxSynchronize(void)
{
    CUresult cu_res;
    cu_res = cuCtxSynchronize();
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDestroyExternalMemory(UPTKExternalMemory_t extMem)
{
    CUresult cu_res;
    cu_res = cuDestroyExternalMemory((CUexternalMemory)extMem);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDestroyExternalSemaphore(UPTKExternalSemaphore_t extSem)
{
    CUresult cu_res;
    cu_res = cuDestroyExternalSemaphore((CUexternalSemaphore)extSem);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceCanAccessPeer(int * canAccessPeer,UPTKdevice dev,UPTKdevice peerDev)
{
    CUresult cu_res;
    cu_res = cuDeviceCanAccessPeer(canAccessPeer, (CUdevice)dev, (CUdevice)peerDev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceComputeCapability(int * major,int * minor,UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuDeviceComputeCapability(major, minor, (CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceGet(UPTKdevice * device,int ordinal)
{
    CUresult cu_res;
    cu_res = cuDeviceGet((CUdevice *)device, ordinal);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceGetAttribute(int * pi,UPTKDeviceAttr attrib,UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuDeviceGetAttribute(pi, (CUdevice_attribute)attrib, (CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceGetByPCIBusId(UPTKdevice * dev,const char * pciBusId)
{
    CUresult cu_res;
    cu_res = cuDeviceGetByPCIBusId((CUdevice *)dev, pciBusId);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceGetCount(int * count)
{
    CUresult cu_res;
    cu_res = cuDeviceGetCount(count);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceGetLuid(char * luid,unsigned int * deviceNodeMask,UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuDeviceGetLuid(luid, deviceNodeMask, (CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceGetName(char * name,int len,UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuDeviceGetName(name, len, (CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceGetNvSciSyncAttributes(void * nvSciSyncAttrList,UPTKdevice dev,int flags)
{
    CUresult cu_res;
    cu_res = cuDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, (CUdevice)dev, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceGetP2PAttribute(int * value,UPTKdevice_P2PAttribute attrib,UPTKdevice srcDevice,UPTKdevice dstDevice)
{
    CUresult cu_res;
    cu_res = cuDeviceGetP2PAttribute(value, (CUdevice_P2PAttribute)attrib, (CUdevice)srcDevice, (CUdevice)dstDevice);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceGetPCIBusId(char * pciBusId,int len,UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuDeviceGetPCIBusId(pciBusId, len, (CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceGetProperties(UPTKdevprop * prop,UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuDeviceGetProperties((CUdevprop *)prop, (CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceGetUuid(UPTKuuid * uuid,UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuDeviceGetUuid((CUuuid *)uuid, (CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDevicePrimaryCtxGetState(UPTKdevice dev,unsigned int * flags,int * active)
{
    CUresult cu_res;
    cu_res = cuDevicePrimaryCtxGetState((CUdevice)dev, flags, active);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDevicePrimaryCtxRelease(UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuDevicePrimaryCtxRelease((CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDevicePrimaryCtxReset(UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuDevicePrimaryCtxReset((CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDevicePrimaryCtxRetain(UPTKcontext * pctx,UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuDevicePrimaryCtxRetain((CUcontext *)pctx, (CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDevicePrimaryCtxSetFlags(UPTKdevice dev,unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuDevicePrimaryCtxSetFlags((CUdevice)dev, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceTotalMem_v2(size_t * bytes,UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuDeviceTotalMem_v2(bytes, (CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDriverGetVersion(int * driverVersion)
{
    CUresult cu_res;
    cu_res = cuDriverGetVersion(driverVersion);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPEventCreate(UPTKEvent_t * phEvent,unsigned int Flags)
{
    CUresult cu_res;
    cu_res = cuEventCreate((CUevent *)phEvent, Flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPEventDestroy(UPTKEvent_t hEvent)
{
    CUresult cu_res;
    cu_res = cuEventDestroy((CUevent)hEvent);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPEventElapsedTime(float * pMilliseconds,UPTKEvent_t hStart,UPTKEvent_t hEnd)
{
    CUresult cu_res;
    cu_res = cuEventElapsedTime(pMilliseconds, (CUevent)hStart, (CUevent)hEnd);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPEventQuery(UPTKEvent_t hEvent)
{
    CUresult cu_res;
    cu_res = cuEventQuery((CUevent)hEvent);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPEventRecord(UPTKEvent_t hEvent,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuEventRecord((CUevent)hEvent, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPEventSynchronize(UPTKEvent_t hEvent)
{
    CUresult cu_res;
    cu_res = cuEventSynchronize((CUevent)hEvent);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPExternalMemoryGetMappedBuffer(UPTKdeviceptr * devPtr,UPTKExternalMemory_t extMem,const UPTK_EXTERNAL_MEMORY_BUFFER_DESC * bufferDesc)
{
    CUresult cu_res;
    cu_res = cuExternalMemoryGetMappedBuffer((CUdeviceptr *)devPtr, (CUexternalMemory)extMem, (const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *)bufferDesc);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPExternalMemoryGetMappedMipmappedArray(UPTKMipmappedArray_t * mipmap,UPTKExternalMemory_t extMem,const UPTK_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC * mipmapDesc)
{
    CUresult cu_res;
    cu_res = cuExternalMemoryGetMappedMipmappedArray((CUmipmappedArray *)mipmap, (CUexternalMemory)extMem, (const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *)mipmapDesc);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPFuncGetAttribute(int * pi,UPTKfunction_attribute attrib,UPTKfunction hfunc)
{
    CUresult cu_res;
    cu_res = cuFuncGetAttribute(pi, (CUfunction_attribute)attrib, (CUfunction)hfunc);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPFuncSetAttribute(UPTKfunction hfunc,UPTKfunction_attribute attrib,int value)
{
    CUresult cu_res;
    cu_res = cuFuncSetAttribute((CUfunction)hfunc, (CUfunction_attribute)attrib, value);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPFuncSetBlockShape(UPTKfunction hfunc,int x,int y,int z)
{
    CUresult cu_res;
    cu_res = cuFuncSetBlockShape((CUfunction)hfunc, x, y, z);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPFuncSetCacheConfig(UPTKfunction hfunc,UPTKfunc_cache config)
{
    CUresult cu_res;
    cu_res = cuFuncSetCacheConfig((CUfunction)hfunc, (CUfunc_cache)config);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPFuncSetSharedMemConfig(UPTKfunction hfunc,UPTKsharedconfig config)
{
    CUresult cu_res;
    cu_res = cuFuncSetSharedMemConfig((CUfunction)hfunc, (CUsharedconfig)config);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPFuncSetSharedSize(UPTKfunction hfunc,unsigned int bytes)
{
    CUresult cu_res;
    cu_res = cuFuncSetSharedSize((CUfunction)hfunc, bytes);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGLGetDevices(unsigned int * pCudaDeviceCount,UPTKdevice * pCudaDevices,unsigned int cudaDeviceCount,UPTKGLDeviceList deviceList)
{
    CUresult cu_res;
    cu_res = cuGLGetDevices(pCudaDeviceCount, (CUdevice *)pCudaDevices, cudaDeviceCount, (CUGLDeviceList)deviceList);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGetErrorName(UPTKError error,const char ** pStr)
{
    CUresult cu_res;
    cu_res = cuGetErrorName((CUresult)error, pStr);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGetErrorString(UPTKError error, const char ** pStr)
{
    CUresult cu_res;
    cu_res = cuGetErrorString((CUresult)error, pStr);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGetExportTable(const void ** ppExportTable,const UPTKuuid * pExportTableId)
{
    CUresult cu_res;
    cu_res = cuGetExportTable(ppExportTable, (const CUuuid *)pExportTableId);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphAddChildGraphNode(UPTKGraphNode_t * phGraphNode,UPTKGraph_t hGraph,const UPTKGraphNode_t * dependencies,size_t numDependencies,UPTKGraph_t childGraph)
{
    CUresult cu_res;
    cu_res = cuGraphAddChildGraphNode((CUgraphNode *)phGraphNode, (CUgraph)hGraph, (const CUgraphNode *)dependencies, numDependencies, (CUgraph)childGraph);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphAddDependencies(UPTKGraph_t hGraph,const UPTKGraphNode_t * from,const UPTKGraphNode_t * to,size_t numDependencies)
{
    CUresult cu_res;
    cu_res = cuGraphAddDependencies((CUgraph)hGraph, (const CUgraphNode *)from, (const CUgraphNode *)to, numDependencies);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphAddEmptyNode(UPTKGraphNode_t * phGraphNode,UPTKGraph_t hGraph,const UPTKGraphNode_t * dependencies,size_t numDependencies)
{
    CUresult cu_res;
    cu_res = cuGraphAddEmptyNode((CUgraphNode *)phGraphNode, (CUgraph)hGraph, (const CUgraphNode *)dependencies, numDependencies);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphAddHostNode(UPTKGraphNode_t * phGraphNode,UPTKGraph_t hGraph,const UPTKGraphNode_t * dependencies,size_t numDependencies,const UPTK_HOST_NODE_PARAMS * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphAddHostNode((CUgraphNode *)phGraphNode, (CUgraph)hGraph, (const CUgraphNode *)dependencies, numDependencies, (const CUDA_HOST_NODE_PARAMS *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphAddMemcpyNode(UPTKGraphNode_t * phGraphNode,UPTKGraph_t hGraph,const UPTKGraphNode_t * dependencies,size_t numDependencies,const UPTK_MEMCPY3D * copyParams,UPTKcontext ctx)
{
    CUresult cu_res;
    cu_res = cuGraphAddMemcpyNode((CUgraphNode *)phGraphNode, (CUgraph)hGraph, (const CUgraphNode *)dependencies, numDependencies, (const CUDA_MEMCPY3D *)copyParams, (CUcontext)ctx);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphAddMemsetNode(UPTKGraphNode_t * phGraphNode,UPTKGraph_t hGraph,const UPTKGraphNode_t * dependencies,size_t numDependencies,const UPTK_MEMSET_NODE_PARAMS * memsetParams,UPTKcontext ctx)
{
    CUresult cu_res;
    cu_res = cuGraphAddMemsetNode((CUgraphNode *)phGraphNode, (CUgraph)hGraph, (const CUgraphNode *)dependencies, numDependencies, (const CUDA_MEMSET_NODE_PARAMS *)memsetParams, (CUcontext)ctx);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphChildGraphNodeGetGraph(UPTKGraphNode_t hNode,UPTKGraph_t * phGraph)
{
    CUresult cu_res;
    cu_res = cuGraphChildGraphNodeGetGraph((CUgraphNode)hNode, (CUgraph *)phGraph);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphClone(UPTKGraph_t * phGraphClone,UPTKGraph_t originalGraph)
{
    CUresult cu_res;
    cu_res = cuGraphClone((CUgraph *)phGraphClone, (CUgraph)originalGraph);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphCreate(UPTKGraph_t * phGraph,unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuGraphCreate((CUgraph *)phGraph, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphDestroy(UPTKGraph_t hGraph)
{
    CUresult cu_res;
    cu_res = cuGraphDestroy((CUgraph)hGraph);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphDestroyNode(UPTKGraphNode_t hNode)
{
    CUresult cu_res;
    cu_res = cuGraphDestroyNode((CUgraphNode)hNode);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphExecDestroy(UPTKGraphExec_t hGraphExec)
{
    CUresult cu_res;
    cu_res = cuGraphExecDestroy((CUgraphExec)hGraphExec);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphExecHostNodeSetParams(UPTKGraphExec_t hGraphExec,UPTKGraphNode_t hNode,const UPTK_HOST_NODE_PARAMS * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphExecHostNodeSetParams((CUgraphExec)hGraphExec, (CUgraphNode)hNode, (const CUDA_HOST_NODE_PARAMS *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphExecMemcpyNodeSetParams(UPTKGraphExec_t hGraphExec,UPTKGraphNode_t hNode,const UPTK_MEMCPY3D * copyParams,UPTKcontext ctx)
{
    CUresult cu_res;
    cu_res = cuGraphExecMemcpyNodeSetParams((CUgraphExec)hGraphExec, (CUgraphNode)hNode, (const CUDA_MEMCPY3D *)copyParams, (CUcontext)ctx);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphExecMemsetNodeSetParams(UPTKGraphExec_t hGraphExec,UPTKGraphNode_t hNode,const UPTK_MEMSET_NODE_PARAMS * memsetParams,UPTKcontext ctx)
{
    CUresult cu_res;
    cu_res = cuGraphExecMemsetNodeSetParams((CUgraphExec)hGraphExec, (CUgraphNode)hNode, (const CUDA_MEMSET_NODE_PARAMS *)memsetParams, (CUcontext)ctx);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphGetEdges(UPTKGraph_t hGraph,UPTKGraphNode_t * from,UPTKGraphNode_t * to,size_t * numEdges)
{
    CUresult cu_res;
    cu_res = cuGraphGetEdges((CUgraph)hGraph, (CUgraphNode *)from, (CUgraphNode *)to, numEdges);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphGetNodes(UPTKGraph_t hGraph,UPTKGraphNode_t * nodes,size_t * numNodes)
{
    CUresult cu_res;
    cu_res = cuGraphGetNodes((CUgraph)hGraph, (CUgraphNode *)nodes, numNodes);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphGetRootNodes(UPTKGraph_t hGraph,UPTKGraphNode_t * rootNodes,size_t * numRootNodes)
{
    CUresult cu_res;
    cu_res = cuGraphGetRootNodes((CUgraph)hGraph, (CUgraphNode *)rootNodes, numRootNodes);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphHostNodeGetParams(UPTKGraphNode_t hNode,UPTK_HOST_NODE_PARAMS * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphHostNodeGetParams((CUgraphNode)hNode, (CUDA_HOST_NODE_PARAMS *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphHostNodeSetParams(UPTKGraphNode_t hNode,const UPTK_HOST_NODE_PARAMS * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphHostNodeSetParams((CUgraphNode)hNode, (const CUDA_HOST_NODE_PARAMS *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphLaunch(UPTKGraphExec_t hGraphExec,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuGraphLaunch((CUgraphExec)hGraphExec, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

/*UPTKError UPGraphMemcpyNodeGetParams(UPTKGraphNode_t hNode,UPTK_MEMCPY3D * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphMemcpyNodeGetParams((CUgraphNode)hNode, (CUDA_MEMCPY3D *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphMemcpyNodeSetParams(UPTKGraphNode_t hNode,const UPTK_MEMCPY3D * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphMemcpyNodeSetParams((CUgraphNode)hNode, (const CUDA_MEMCPY3D *)nodeParams);
    return CUresultToUPTKError(cu_res);
}*/

UPTKError UPGraphMemsetNodeGetParams(UPTKGraphNode_t hNode,UPTK_MEMSET_NODE_PARAMS * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphMemsetNodeGetParams((CUgraphNode)hNode, (CUDA_MEMSET_NODE_PARAMS *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphMemsetNodeSetParams(UPTKGraphNode_t hNode,const UPTK_MEMSET_NODE_PARAMS * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphMemsetNodeSetParams((CUgraphNode)hNode, (const CUDA_MEMSET_NODE_PARAMS *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphNodeFindInClone(UPTKGraphNode_t * phNode,UPTKGraphNode_t hOriginalNode,UPTKGraph_t hClonedGraph)
{
    CUresult cu_res;
    cu_res = cuGraphNodeFindInClone((CUgraphNode *)phNode, (CUgraphNode)hOriginalNode, (CUgraph)hClonedGraph);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphNodeGetDependencies(UPTKGraphNode_t hNode,UPTKGraphNode_t * dependencies,size_t * numDependencies)
{
    CUresult cu_res;
    cu_res = cuGraphNodeGetDependencies((CUgraphNode)hNode, (CUgraphNode *)dependencies, numDependencies);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphNodeGetDependentNodes(UPTKGraphNode_t hNode,UPTKGraphNode_t * dependentNodes,size_t * numDependentNodes)
{
    CUresult cu_res;
    cu_res = cuGraphNodeGetDependentNodes((CUgraphNode)hNode, (CUgraphNode *)dependentNodes, numDependentNodes);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphNodeGetType(UPTKGraphNode_t hNode,UPTKGraphNodeType * type)
{
    CUresult cu_res;
    cu_res = cuGraphNodeGetType((CUgraphNode)hNode, (CUgraphNodeType *)type);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphRemoveDependencies(UPTKGraph_t hGraph,const UPTKGraphNode_t * from,const UPTKGraphNode_t * to,size_t numDependencies)
{
    CUresult cu_res;
    cu_res = cuGraphRemoveDependencies((CUgraph)hGraph, (const CUgraphNode *)from, (const CUgraphNode *)to, numDependencies);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphicsGLRegisterBuffer(UPTKGraphicsResource_t * pCudaResource,GLuint buffer,unsigned int Flags)
{
    CUresult cu_res;
    cu_res = cuGraphicsGLRegisterBuffer((CUgraphicsResource *)pCudaResource, buffer, Flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphicsGLRegisterImage(UPTKGraphicsResource_t * pCudaResource,GLuint image,GLenum target,unsigned int Flags)
{
    CUresult cu_res;
    cu_res = cuGraphicsGLRegisterImage((CUgraphicsResource *)pCudaResource, image, target, Flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphicsMapResources(unsigned int count,UPTKGraphicsResource_t * resources,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuGraphicsMapResources(count, (CUgraphicsResource *)resources, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphicsResourceGetMappedPointer_v2(UPTKdeviceptr * pDevPtr,size_t * pSize,UPTKGraphicsResource_t resource)
{
    CUresult cu_res;
    cu_res = cuGraphicsResourceGetMappedPointer_v2((CUdeviceptr *)pDevPtr, pSize, (CUgraphicsResource)resource);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphicsSubResourceGetMappedArray(UPTKarray * pArray,UPTKGraphicsResource_t resource,unsigned int arrayIndex,unsigned int mipLevel)
{
    CUresult cu_res;
    cu_res = cuGraphicsSubResourceGetMappedArray((CUarray *)pArray, (CUgraphicsResource)resource, arrayIndex, mipLevel);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphicsUnregisterResource(UPTKGraphicsResource_t resource)
{
    CUresult cu_res;
    cu_res = cuGraphicsUnregisterResource((CUgraphicsResource)resource);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphicsUnmapResources(unsigned int count,UPTKGraphicsResource_t * resources,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuGraphicsUnmapResources(count, (CUgraphicsResource *)resources, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphicsResourceGetMappedMipmappedArray(UPTKMipmappedArray_t * pMipmappedArray,UPTKGraphicsResource_t resource)
{
    CUresult cu_res;
    cu_res = cuGraphicsResourceGetMappedMipmappedArray((CUmipmappedArray *)pMipmappedArray, (CUgraphicsResource)resource);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphicsResourceSetMapFlags(UPTKGraphicsResource_t resource,unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuGraphicsResourceSetMapFlags((CUgraphicsResource)resource, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPImportExternalMemory(UPTKExternalMemory_t * extMem_out,const UPTK_EXTERNAL_MEMORY_HANDLE_DESC * memHandleDesc)
{
    if (!extMem_out || !memHandleDesc) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuImportExternalMemory((CUexternalMemory *)extMem_out, (const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *)memHandleDesc);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPImportExternalSemaphore(UPTKExternalSemaphore_t * extSem_out,const UPTK_EXTERNAL_SEMAPHORE_HANDLE_DESC * semHandleDesc)
{
    if (!extSem_out || !semHandleDesc) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuImportExternalSemaphore((CUexternalSemaphore *)extSem_out, (const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *)semHandleDesc);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPInit(unsigned int Flags)
{
    CUresult cu_res;
    cu_res = cuInit(Flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPIpcCloseMemHandle(UPTKdeviceptr dptr)
{
    CUresult cu_res;
    cu_res = cuIpcCloseMemHandle((CUdeviceptr)dptr);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPIpcGetEventHandle(UPTKIpcEventHandle_t * pHandle,UPTKEvent_t event)
{
    if (!pHandle) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuIpcGetEventHandle((CUipcEventHandle *)pHandle, (CUevent)event);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPIpcGetMemHandle(UPTKIpcMemHandle_t * pHandle,UPTKdeviceptr dptr)
{
    CUresult cu_res;
    cu_res = cuIpcGetMemHandle((CUipcMemHandle *)pHandle, (CUdeviceptr)dptr);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPIpcOpenEventHandle(UPTKEvent_t * phEvent,UPTKIpcEventHandle_t handle)
{
    CUresult cu_res;
    //cu_res = cuIpcOpenEventHandle((CUevent *)phEvent, (CUipcEventHandle)handle);
    CUipcEventHandle cu_event_handle;
    memcpy(&cu_event_handle, &handle, sizeof(CUipcEventHandle));
    cu_res = cuIpcOpenEventHandle((CUevent *)phEvent, cu_event_handle);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPIpcOpenMemHandle(UPTKdeviceptr * pdptr,UPTKIpcMemHandle_t handle,unsigned int Flags)
{
    CUresult cu_res;
    CUipcMemHandle cu_handle;
    memcpy(&cu_handle, &handle, sizeof(CUipcMemHandle)); // 内存拷贝
    cu_res = cuIpcOpenMemHandle((CUdeviceptr *)pdptr, cu_handle, Flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLaunch(UPTKfunction f)
{
    CUresult cu_res;
    cu_res = cuLaunch((CUfunction)f);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLaunchCooperativeKernel(UPTKfunction f,unsigned int gridDimX,unsigned int gridDimY,unsigned int gridDimZ,unsigned int blockDimX,unsigned int blockDimY,unsigned int blockDimZ,unsigned int sharedMemBytes,UPTKStream_t hStream,void ** kernelParams)
{
    CUresult cu_res;
    cu_res = cuLaunchCooperativeKernel((CUfunction)f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, (CUstream)hStream, kernelParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLaunchCooperativeKernelMultiDevice(UPTK_LAUNCH_PARAMS * launchParamsList,unsigned int numDevices,unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuLaunchCooperativeKernelMultiDevice((CUDA_LAUNCH_PARAMS *)launchParamsList, numDevices, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLaunchGrid(UPTKfunction f,int grid_width,int grid_height)
{
    CUresult cu_res;
    cu_res = cuLaunchGrid((CUfunction)f, grid_width, grid_height);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLaunchGridAsync(UPTKfunction f,int grid_width,int grid_height,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuLaunchGridAsync((CUfunction)f, grid_width, grid_height, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLaunchHostFunc(UPTKStream_t hStream,UPTKhostFn fn,void * userData)
{
    CUresult cu_res;
    cu_res = cuLaunchHostFunc((CUstream)hStream, (CUhostFn)fn, userData);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLaunchKernel(UPTKfunction f,unsigned int gridDimX,unsigned int gridDimY,unsigned int gridDimZ,unsigned int blockDimX,unsigned int blockDimY,unsigned int blockDimZ,unsigned int sharedMemBytes,UPTKStream_t hStream,void ** kernelParams,void ** extra)
{
    CUresult cu_res;
    cu_res = cuLaunchKernel((CUfunction)f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, (CUstream)hStream, kernelParams, extra);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLinkAddData(UPTKlinkState state,UPTKjitInputType type,void * data,size_t size,const char * name,unsigned int numOptions,UPTKjit_option * options,void ** optionValues)
{
    CUresult cu_res;
    cu_res = cuLinkAddData((CUlinkState)state, (CUjitInputType)type, data, size, name, numOptions, (CUjit_option *)options, optionValues);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLinkAddFile(UPTKlinkState state,UPTKjitInputType type,const char * path,unsigned int numOptions,UPTKjit_option * options,void ** optionValues)
{
    CUresult cu_res;
    cu_res = cuLinkAddFile((CUlinkState)state, (CUjitInputType)type, path, numOptions, (CUjit_option *)options, optionValues);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLinkComplete(UPTKlinkState state,void ** cubinOut,size_t * sizeOut)
{
    CUresult cu_res;
    cu_res = cuLinkComplete((CUlinkState)state, cubinOut, sizeOut);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLinkCreate(unsigned int numOptions,UPTKjit_option * options,void ** optionValues,UPTKlinkState * stateOut)
{
    CUresult cu_res;
    cu_res = cuLinkCreate(numOptions, (CUjit_option *)options, optionValues, (CUlinkState *)stateOut);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLinkDestroy(UPTKlinkState state)
{
    CUresult cu_res;
    cu_res = cuLinkDestroy((CUlinkState)state);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemAddressFree(UPTKdeviceptr ptr,size_t size)
{
    CUresult cu_res;
    cu_res = cuMemAddressFree((CUdeviceptr)ptr, size);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemAddressReserve(UPTKdeviceptr * ptr,size_t size,size_t alignment,UPTKdeviceptr addr,unsigned long long flags)
{
    CUresult cu_res;
    cu_res = cuMemAddressReserve((CUdeviceptr *)ptr, size, alignment, (CUdeviceptr)addr, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemAdvise(UPTKdeviceptr devPtr,size_t count,UPTKmem_advise advice,UPTKdevice device)
{
    if (count == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemAdvise((CUdeviceptr)devPtr, count, (CUmem_advise)advice, (CUdevice)device);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemAlloc_v2(UPTKdeviceptr * dptr,size_t bytesize)
{
    CUresult cu_res;
    cu_res = cuMemAlloc_v2((CUdeviceptr *)dptr, bytesize);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemAllocHost(void ** pp, size_t bytesize)
{
    CUresult cu_res;
    cu_res = cuMemAllocHost(pp, bytesize);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemAllocManaged(UPTKdeviceptr * dptr,size_t bytesize,unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuMemAllocManaged((CUdeviceptr *)dptr, bytesize, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemAllocPitch_v2(UPTKdeviceptr * dptr,size_t * pPitch, size_t WidthInBytes, size_t Height,unsigned int ElementSizeBytes)
{
    CUresult cu_res;
    cu_res = cuMemAllocPitch_v2((CUdeviceptr *)dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemCreate(UPTKmemGenericAllocationHandle * handle,size_t size,const UPTKmemAllocationProp * prop,unsigned long long flags)
{
    CUresult cu_res;
    cu_res = cuMemCreate((CUmemGenericAllocationHandle *)handle, size, (const CUmemAllocationProp *)prop, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemExportToShareableHandle(void * shareableHandle,UPTKmemGenericAllocationHandle handle,UPTKMemAllocationHandleType handleType,unsigned long long flags)
{
    CUresult cu_res;
    cu_res = cuMemExportToShareableHandle(shareableHandle, (CUmemGenericAllocationHandle)handle, (CUmemAllocationHandleType)handleType, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemFree(UPTKdeviceptr dptr)
{
    CUresult cu_res;
    cu_res = cuMemFree((CUdeviceptr)dptr);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemFreeHost(void * p)
{
    CUresult cu_res;
    cu_res = cuMemFreeHost(p);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemGetAccess(unsigned long long * flags,const UPTKMemLocation * location,UPTKdeviceptr ptr)
{
    CUresult cu_res;
    cu_res = cuMemGetAccess(flags, (const CUmemLocation *)location, (CUdeviceptr)ptr);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemGetAddressRange_v2(UPTKdeviceptr * pbase,size_t * psize,UPTKdeviceptr dptr)
{
    CUresult cu_res;
    cu_res = cuMemGetAddressRange_v2((CUdeviceptr *)pbase, psize, (CUdeviceptr)dptr);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemGetAllocationGranularity(size_t * granularity,const UPTKmemAllocationProp * prop,UPTKmemAllocationGranularity_flags option)
{
    CUresult cu_res;
    cu_res = cuMemGetAllocationGranularity(granularity, (const CUmemAllocationProp *)prop, (CUmemAllocationGranularity_flags)option);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemGetAllocationPropertiesFromHandle(UPTKmemAllocationProp * prop,UPTKmemGenericAllocationHandle handle)
{
    CUresult cu_res;
    cu_res = cuMemGetAllocationPropertiesFromHandle((CUmemAllocationProp *)prop, (CUmemGenericAllocationHandle)handle);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemGetInfo_v2(size_t * free, size_t * total)
{
    CUresult cu_res;
    cu_res = cuMemGetInfo_v2(free, total);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemHostAlloc(void ** pp, size_t bytesize, unsigned int Flags)
{
    CUresult cu_res;
    cu_res = cuMemHostAlloc(pp, bytesize, Flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemHostGetDevicePointer(UPTKdeviceptr * pdptr,void * p,unsigned int Flags)
{
    CUresult cu_res;
    cu_res = cuMemHostGetDevicePointer((CUdeviceptr *)pdptr, p, Flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemHostGetFlags(unsigned int * pFlags,void * p)
{
    CUresult cu_res;
    cu_res = cuMemHostGetFlags(pFlags, p);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemHostRegister(void * p,size_t bytesize,unsigned int Flags)
{
    CUresult cu_res;
    cu_res = cuMemHostRegister(p, bytesize, Flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemHostUnregister(void * p)
{
    CUresult cu_res;
    cu_res = cuMemHostUnregister(p);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemImportFromShareableHandle(UPTKmemGenericAllocationHandle * handle,void * osHandle,UPTKMemAllocationHandleType shHandleType)
{
    CUresult cu_res;
    cu_res = cuMemImportFromShareableHandle((CUmemGenericAllocationHandle *)handle, osHandle, (CUmemAllocationHandleType)shHandleType);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemMap(UPTKdeviceptr ptr,size_t size,size_t offset,UPTKmemGenericAllocationHandle handle,unsigned long long flags)
{
    CUresult cu_res;
    cu_res = cuMemMap((CUdeviceptr)ptr, size, offset, (CUmemGenericAllocationHandle)handle, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemPrefetchAsync(UPTKdeviceptr devPtr,size_t count,UPTKdevice dstDevice,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemPrefetchAsync((CUdeviceptr)devPtr, count, (CUdevice)dstDevice, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemRangeGetAttribute(void * data,size_t dataSize,UPTKmem_range_attribute attribute,UPTKdeviceptr devPtr,size_t count)
{
    if (!data || dataSize == 0 || count == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemRangeGetAttribute(data, dataSize, (CUmem_range_attribute)attribute, (CUdeviceptr)devPtr, count);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemRangeGetAttributes(void ** data,size_t * dataSizes,UPTKmem_range_attribute * attributes,size_t numAttributes,UPTKdeviceptr devPtr,size_t count)
{
    CUresult cu_res;
    cu_res = cuMemRangeGetAttributes(data, dataSizes, (CUmem_range_attribute *)attributes, numAttributes, (CUdeviceptr)devPtr, count);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemRelease(UPTKmemGenericAllocationHandle handle)
{
    CUresult cu_res;
    cu_res = cuMemRelease((CUmemGenericAllocationHandle)handle);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemSetAccess(UPTKdeviceptr ptr,size_t size,const UPTKMemAccessDesc * desc,size_t count)
{
    CUresult cu_res;
    cu_res = cuMemSetAccess((CUdeviceptr)ptr, size, (const CUmemAccessDesc *)desc, count);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemUnmap(UPTKdeviceptr ptr,size_t size)
{
    CUresult cu_res;
    cu_res = cuMemUnmap((CUdeviceptr)ptr, size);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpy(UPTKdeviceptr dst,UPTKdeviceptr src,size_t ByteCount)
{
    CUresult cu_res;
    cu_res = cuMemcpy((CUdeviceptr)dst, (CUdeviceptr)src, ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpy2DAsync_v2(const UPTK_MEMCPY2D * pCopy,UPTKStream_t hStream)
{
    if (!pCopy || pCopy->WidthInBytes == 0 || pCopy->Height == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemcpy2DAsync_v2((const CUDA_MEMCPY2D *)pCopy, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpy2DUnaligned_v2(const UPTK_MEMCPY2D * pCopy)
{
    if (!pCopy || pCopy->WidthInBytes == 0 || pCopy->Height == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemcpy2DUnaligned_v2((const CUDA_MEMCPY2D *)pCopy);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpy2D_v2(const UPTK_MEMCPY2D * pCopy)
{
    if (!pCopy || pCopy->WidthInBytes == 0 || pCopy->Height == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemcpy2D_v2((const CUDA_MEMCPY2D *)pCopy);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpy3DAsync_v2(const UPTK_MEMCPY3D * pCopy,UPTKStream_t hStream)
{
    if (!pCopy || pCopy->WidthInBytes == 0 || pCopy->Height == 0 || pCopy->Depth == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemcpy3DAsync_v2((const CUDA_MEMCPY3D *)pCopy, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpy3DPeer(const UPTK_MEMCPY3D_PEER * pCopy)
{
    if (!pCopy || pCopy->WidthInBytes == 0 || pCopy->Height == 0 || pCopy->Depth == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemcpy3DPeer((const CUDA_MEMCPY3D_PEER *)pCopy);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpy3DPeerAsync(const UPTK_MEMCPY3D_PEER * pCopy,UPTKStream_t hStream)
{
    if (!pCopy || pCopy->WidthInBytes == 0 || pCopy->Height == 0 || pCopy->Depth == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemcpy3DPeerAsync((const CUDA_MEMCPY3D_PEER *)pCopy, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpy3D_v2(const UPTK_MEMCPY3D * pCopy)
{
    if (!pCopy || pCopy->WidthInBytes == 0 || pCopy->Height == 0 || pCopy->Depth == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemcpy3D_v2((const CUDA_MEMCPY3D *)pCopy);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyAsync(UPTKdeviceptr dst,UPTKdeviceptr src,size_t ByteCount,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src, ByteCount, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyAtoA_v2(UPTKarray dstArray,size_t dstOffset,UPTKarray srcArray,size_t srcOffset,size_t ByteCount)
{
    CUresult cu_res;
    cu_res = cuMemcpyAtoA_v2((CUarray)dstArray, dstOffset, (CUarray)srcArray, srcOffset, ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyAtoD_v2(UPTKdeviceptr dstDevice,UPTKarray srcArray,size_t srcOffset,size_t ByteCount)
{
    if (ByteCount == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemcpyAtoD_v2((CUdeviceptr)dstDevice, (CUarray)srcArray, srcOffset, ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyAtoHAsync_v2(void * dstHost,UPTKarray srcArray,size_t srcOffset,size_t ByteCount,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemcpyAtoHAsync_v2(dstHost, (CUarray)srcArray, srcOffset, ByteCount, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyAtoH_v2(void * dstHost,UPTKarray srcArray,size_t srcOffset,size_t ByteCount)
{
    CUresult cu_res;
    cu_res = cuMemcpyAtoH_v2(dstHost, (CUarray)srcArray, srcOffset, ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyDtoA_v2(UPTKarray dstArray,size_t dstOffset,UPTKdeviceptr srcDevice,size_t ByteCount)
{
    if (ByteCount == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemcpyDtoA_v2((CUarray)dstArray, dstOffset, (CUdeviceptr)srcDevice, ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyDtoDAsync_v2(UPTKdeviceptr dstDevice,UPTKdeviceptr srcDevice,size_t ByteCount,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemcpyDtoDAsync_v2((CUdeviceptr)dstDevice, (CUdeviceptr)srcDevice, ByteCount, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyDtoD_v2(UPTKdeviceptr dstDevice,UPTKdeviceptr srcDevice,size_t ByteCount)
{
    CUresult cu_res;
    cu_res = cuMemcpyDtoD_v2((CUdeviceptr)dstDevice, (CUdeviceptr)srcDevice, ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyDtoHAsync_v2(void * dstHost,UPTKdeviceptr srcDevice,size_t ByteCount,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemcpyDtoHAsync_v2(dstHost, (CUdeviceptr)srcDevice, ByteCount, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyDtoH_v2(void * dstHost,UPTKdeviceptr srcDevice,size_t ByteCount)
{
    CUresult cu_res;
    cu_res = cuMemcpyDtoH_v2(dstHost, (CUdeviceptr)srcDevice, ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyHtoAAsync_v2(UPTKarray dstArray,size_t dstOffset,const void * srcHost,size_t ByteCount,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemcpyHtoAAsync_v2((CUarray)dstArray, dstOffset, srcHost, ByteCount, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyHtoA_v2(UPTKarray dstArray,size_t dstOffset,const void * srcHost,size_t ByteCount)
{
    CUresult cu_res;
    cu_res = cuMemcpyHtoA_v2((CUarray)dstArray, dstOffset, srcHost, ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyHtoDAsync_v2(UPTKdeviceptr dstDevice,const void * srcHost,size_t ByteCount,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemcpyHtoDAsync_v2((CUdeviceptr)dstDevice, srcHost, ByteCount, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyHtoD_v2(UPTKdeviceptr dstDevice,const void * srcHost,size_t ByteCount)
{
    CUresult cu_res;
    cu_res = cuMemcpyHtoD_v2((CUdeviceptr)dstDevice, srcHost, ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyPeer(UPTKdeviceptr dstDevice,UPTKcontext dstContext,UPTKdeviceptr srcDevice,UPTKcontext srcContext,size_t ByteCount)
{
    CUresult cu_res;
    cu_res = cuMemcpyPeer((CUdeviceptr)dstDevice, (CUcontext)dstContext, (CUdeviceptr)srcDevice, (CUcontext)srcContext, ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyPeerAsync(UPTKdeviceptr dstDevice,UPTKcontext dstContext,UPTKdeviceptr srcDevice,UPTKcontext srcContext,size_t ByteCount,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemcpyPeerAsync((CUdeviceptr)dstDevice, (CUcontext)dstContext, (CUdeviceptr)srcDevice, (CUcontext)srcContext, ByteCount, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD16Async(UPTKdeviceptr dstDevice,unsigned short us,size_t N,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemsetD16Async((CUdeviceptr)dstDevice, us, N, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD16_v2(UPTKdeviceptr dstDevice,unsigned short us,size_t N)
{
    CUresult cu_res;
    cu_res = cuMemsetD16_v2((CUdeviceptr)dstDevice, us, N);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD2D16Async(UPTKdeviceptr dstDevice,size_t dstPitch,unsigned short us,size_t Width,size_t Height,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemsetD2D16Async((CUdeviceptr)dstDevice, dstPitch, us, Width, Height, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD2D16_v2(UPTKdeviceptr dstDevice,size_t dstPitch,unsigned short us,size_t Width,size_t Height)
{
    CUresult cu_res;
    cu_res = cuMemsetD2D16_v2((CUdeviceptr)dstDevice, dstPitch, us, Width, Height);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD2D32Async(UPTKdeviceptr dstDevice,size_t dstPitch,unsigned int ui,size_t Width,size_t Height,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemsetD2D32Async((CUdeviceptr)dstDevice, dstPitch, ui, Width, Height, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD2D32_v2(UPTKdeviceptr dstDevice,size_t dstPitch,unsigned int ui,size_t Width,size_t Height)
{
    CUresult cu_res;
    cu_res = cuMemsetD2D32_v2((CUdeviceptr)dstDevice, dstPitch, ui, Width, Height);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD2D8Async(UPTKdeviceptr dstDevice,size_t dstPitch,unsigned char uc,size_t Width,size_t Height,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemsetD2D8Async((CUdeviceptr)dstDevice, dstPitch, uc, Width, Height, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD2D8_v2(UPTKdeviceptr dstDevice,size_t dstPitch,unsigned char uc,size_t Width,size_t Height)
{
    CUresult cu_res;
    cu_res = cuMemsetD2D8_v2((CUdeviceptr)dstDevice, dstPitch, uc, Width, Height);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD32Async(UPTKdeviceptr dstDevice,unsigned int ui,size_t N,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemsetD32Async((CUdeviceptr)dstDevice, ui, N, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD32_v2(UPTKdeviceptr dstDevice,unsigned int ui,size_t N)
{
    CUresult cu_res;
    cu_res = cuMemsetD32_v2((CUdeviceptr)dstDevice, ui, N);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD8Async(UPTKdeviceptr dstDevice,unsigned char uc,size_t N,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemsetD8Async((CUdeviceptr)dstDevice, uc, N, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD8_v2(UPTKdeviceptr dstDevice,unsigned char uc,size_t N)
{
    CUresult cu_res;
    cu_res = cuMemsetD8_v2((CUdeviceptr)dstDevice, uc, N);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMipmappedArrayCreate(UPTKMipmappedArray_t * pHandle,const UPTK_ARRAY3D_DESCRIPTOR * pMipmappedArrayDesc,unsigned int numMipmapLevels)
{
    if (!pHandle || !pMipmappedArrayDesc) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMipmappedArrayCreate((CUmipmappedArray *)pHandle, (const CUDA_ARRAY3D_DESCRIPTOR *)pMipmappedArrayDesc, numMipmapLevels);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMipmappedArrayDestroy(UPTKMipmappedArray_t hMipmappedArray)
{
    CUresult cu_res;
    cu_res = cuMipmappedArrayDestroy((CUmipmappedArray)hMipmappedArray);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMipmappedArrayGetLevel(UPTKarray * pLevelArray,UPTKMipmappedArray_t hMipmappedArray,unsigned int level)
{
    CUresult cu_res;
    cu_res = cuMipmappedArrayGetLevel((CUarray *)pLevelArray, (CUmipmappedArray)hMipmappedArray, level);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPModuleGetFunction(UPTKfunction * hfunc,UPTKmodule hmod,const char * name)
{
    CUresult cu_res;
    cu_res = cuModuleGetFunction((CUfunction *)hfunc, (CUmodule)hmod, name);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPModuleGetGlobal_v2(UPTKdeviceptr * dptr, size_t * bytes,UPTKmodule hmod,const char * name)
{
    CUresult cu_res;
    cu_res = cuModuleGetGlobal_v2((CUdeviceptr *)dptr, bytes, (CUmodule)hmod, name);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPModuleGetSurfRef(UPTKsurfref * pSurfRef,UPTKmodule hmod,const char * name)
{
    CUresult cu_res;
    cu_res = cuModuleGetSurfRef((CUsurfref *)pSurfRef, (CUmodule)hmod, name);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPModuleGetTexRef(UPTKtexref * pTexRef,UPTKmodule hmod,const char * name)
{
    CUresult cu_res;
    cu_res = cuModuleGetTexRef((CUtexref *)pTexRef, (CUmodule)hmod, name);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPModuleLoad(UPTKmodule * module,const char * fname)
{
    CUresult cu_res;
    cu_res = cuModuleLoad((CUmodule *)module, fname);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPModuleLoadData(UPTKmodule * module,const void * image)
{
    CUresult cu_res;
    cu_res = cuModuleLoadData((CUmodule *)module, image);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPModuleLoadDataEx(UPTKmodule * module,const void * image,unsigned int numOptions,UPTKjit_option * options,void ** optionValues)
{
    CUresult cu_res;
    cu_res = cuModuleLoadDataEx((CUmodule *)module, image, numOptions, (CUjit_option *)options, optionValues);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPModuleLoadFatBinary(UPTKmodule * module,const void * fatCubin)
{
    CUresult cu_res;
    cu_res = cuModuleLoadFatBinary((CUmodule *)module, fatCubin);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPModuleUnload(UPTKmodule hmod)
{
    CUresult cu_res;
    cu_res = cuModuleUnload((CUmodule)hmod);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks,UPTKfunction func,int blockSize,size_t dynamicSMemSize)
{
    CUresult cu_res;
    cu_res = cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, (CUfunction)func, blockSize, dynamicSMemSize);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks,UPTKfunction func,int blockSize,size_t dynamicSMemSize,unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (CUfunction)func, blockSize, dynamicSMemSize, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPOccupancyMaxPotentialBlockSize(int * minGridSize,int * blockSize,UPTKfunction func,UPTKoccupancyB2DSize blockSizeToDynamicSMemSize,size_t dynamicSMemSize,int blockSizeLimit)
{
    CUresult cu_res;
    cu_res = cuOccupancyMaxPotentialBlockSize(minGridSize, blockSize, (CUfunction)func, (CUoccupancyB2DSize)blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPOccupancyMaxPotentialBlockSizeWithFlags(int * minGridSize,int * blockSize,UPTKfunction func,UPTKoccupancyB2DSize blockSizeToDynamicSMemSize,size_t dynamicSMemSize,int blockSizeLimit,unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize, blockSize, (CUfunction)func, (CUoccupancyB2DSize)blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPParamSetSize(UPTKfunction hfunc,unsigned int numbytes)
{
    CUresult cu_res;
    cu_res = cuParamSetSize((CUfunction)hfunc, numbytes);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPParamSetTexRef(UPTKfunction hfunc,int texunit,UPTKtexref hTexRef)
{
    CUresult cu_res;
    cu_res = cuParamSetTexRef((CUfunction)hfunc, texunit, (CUtexref)hTexRef);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPParamSetf(UPTKfunction hfunc,int offset,float value)
{
    CUresult cu_res;
    cu_res = cuParamSetf((CUfunction)hfunc, offset, value);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPParamSeti(UPTKfunction hfunc,int offset,unsigned int value)
{
    CUresult cu_res;
    cu_res = cuParamSeti((CUfunction)hfunc, offset, value);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPParamSetv(UPTKfunction hfunc,int offset,void * ptr,unsigned int numbytes)
{
    CUresult cu_res;
    cu_res = cuParamSetv((CUfunction)hfunc, offset, ptr, numbytes);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPPointerGetAttribute(void * data,UPTKpointer_attribute attribute,UPTKdeviceptr ptr)
{
    if (!data) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuPointerGetAttribute(data, (CUpointer_attribute)attribute, (CUdeviceptr)ptr);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPPointerGetAttributes(unsigned int numAttributes,UPTKpointer_attribute * attributes,void ** data,UPTKdeviceptr ptr)
{
    CUresult cu_res;
    cu_res = cuPointerGetAttributes(numAttributes, (CUpointer_attribute *)attributes, data, (CUdeviceptr)ptr);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPPointerSetAttribute(const void * value,UPTKpointer_attribute attribute,UPTKdeviceptr ptr)
{
    if (!value) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuPointerSetAttribute(value, (CUpointer_attribute)attribute, (CUdeviceptr)ptr);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPProfilerInitialize(const char * configFile,const char * outputFile,UPTKoutput_mode outputMode)
{
    CUresult cu_res;
    cu_res = cuProfilerInitialize(configFile, outputFile, (CUoutput_mode)outputMode);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPProfilerStart(void)
{
    CUresult cu_res;
    cu_res = cuProfilerStart();
    return CUresultToUPTKError(cu_res);
}

UPTKError UPProfilerStop(void)
{
    CUresult cu_res;
    cu_res = cuProfilerStop();
    return CUresultToUPTKError(cu_res);
}

UPTKError UPSignalExternalSemaphoresAsync(const UPTKExternalSemaphore_t * extSemArray,const UPTK_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS * paramsArray,unsigned int numExtSems,UPTKStream_t stream)
{
    CUresult cu_res;
    cu_res = cuSignalExternalSemaphoresAsync((const CUexternalSemaphore *)extSemArray, (const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *)paramsArray, numExtSems, (CUstream)stream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamAddCallback(UPTKStream_t hStream,UPTKstreamCallback callback,void * userData,unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuStreamAddCallback((CUstream)hStream, (CUstreamCallback)callback, userData, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamAttachMemAsync(UPTKStream_t hStream,UPTKdeviceptr dptr,size_t length,unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuStreamAttachMemAsync((CUstream)hStream, (CUdeviceptr)dptr, length, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamBatchMemOp(UPTKStream_t stream,unsigned int count,UPTKstreamBatchMemOpParams * paramArray,unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuStreamBatchMemOp((CUstream)stream, count, (CUstreamBatchMemOpParams *)paramArray, flags);
    return CUresultToUPTKError(cu_res);
}

/*UPTKError UPStreamBeginCapture_ptsz(UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuStreamBeginCapture_ptsz((CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}*/

UPTKError UPStreamBeginCapture_v2(UPTKStream_t hStream,UPTKStreamCaptureMode mode)
{
    CUresult cu_res;
    cu_res = cuStreamBeginCapture_v2((CUstream)hStream, (CUstreamCaptureMode)mode);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamCreate(UPTKStream_t * phStream,unsigned int Flags)
{
    CUresult cu_res;
    cu_res = cuStreamCreate((CUstream *)phStream, Flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamCreateWithPriority(UPTKStream_t * phStream,unsigned int flags,int priority)
{
    CUresult cu_res;
    cu_res = cuStreamCreateWithPriority((CUstream *)phStream, flags, priority);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamDestroy(UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuStreamDestroy((CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamEndCapture(UPTKStream_t hStream,UPTKGraph_t * phGraph)
{
    CUresult cu_res;
    cu_res = cuStreamEndCapture((CUstream)hStream, (CUgraph *)phGraph);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamGetCtx(UPTKStream_t hStream,UPTKcontext * pctx)
{
    CUresult cu_res;
    cu_res = cuStreamGetCtx((CUstream)hStream, (CUcontext *)pctx);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamGetFlags(UPTKStream_t hStream,unsigned int * flags)
{
    CUresult cu_res;
    cu_res = cuStreamGetFlags((CUstream)hStream, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamGetPriority(UPTKStream_t hStream,int * priority)
{
    CUresult cu_res;
    cu_res = cuStreamGetPriority((CUstream)hStream, priority);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamIsCapturing(UPTKStream_t hStream,UPTKStreamCaptureStatus * captureStatus)
{
    CUresult cu_res;
    cu_res = cuStreamIsCapturing((CUstream)hStream, (CUstreamCaptureStatus *)captureStatus);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamQuery(UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuStreamQuery((CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamSynchronize(UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuStreamSynchronize((CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamWaitEvent(UPTKStream_t hStream,UPTKEvent_t hEvent,unsigned int Flags)
{
    CUresult cu_res;
    cu_res = cuStreamWaitEvent((CUstream)hStream, (CUevent)hEvent, Flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamWriteValue32(UPTKStream_t stream,UPTKdeviceptr addr,cuuint32_t value,unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuStreamWriteValue32((CUstream)stream, (CUdeviceptr)addr, value, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamWriteValue64(UPTKStream_t stream,UPTKdeviceptr addr,cuuint64_t value,unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuStreamWriteValue64((CUstream)stream, (CUdeviceptr)addr, value, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPSurfObjectCreate(UPTKSurfaceObject_t * pSurfObject,const UPTK_RESOURCE_DESC * pResDesc)
{
    CUresult cu_res;
    cu_res = cuSurfObjectCreate((CUsurfObject *)pSurfObject, (const CUDA_RESOURCE_DESC *)pResDesc);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPSurfObjectDestroy(UPTKSurfaceObject_t surfObject)
{
    CUresult cu_res;
    cu_res = cuSurfObjectDestroy((CUsurfObject)surfObject);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPSurfObjectGetResourceDesc(UPTK_RESOURCE_DESC * pResDesc,UPTKSurfaceObject_t surfObject)
{
    CUresult cu_res;
    cu_res = cuSurfObjectGetResourceDesc((CUDA_RESOURCE_DESC *)pResDesc, (CUsurfObject)surfObject);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPSurfRefGetArray(UPTKarray * phArray,UPTKsurfref hSurfRef)
{
    CUresult cu_res;
    cu_res = cuSurfRefGetArray((CUarray *)phArray, (CUsurfref)hSurfRef);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPSurfRefSetArray(UPTKsurfref hSurfRef,UPTKarray hArray,unsigned int Flags)
{
    CUresult cu_res;
    cu_res = cuSurfRefSetArray((CUsurfref)hSurfRef, (CUarray)hArray, Flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexObjectCreate(UPTKTextureObject_t * pTexObject,const UPTK_RESOURCE_DESC * pResDesc,const UPTK_TEXTURE_DESC * pTexDesc,const UPTK_RESOURCE_VIEW_DESC * pResViewDesc)
{
    CUresult cu_res;
    cu_res = cuTexObjectCreate((CUtexObject *)pTexObject, (const CUDA_RESOURCE_DESC *)pResDesc, (const CUDA_TEXTURE_DESC *)pTexDesc, (const CUDA_RESOURCE_VIEW_DESC *)pResViewDesc);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexObjectDestroy(UPTKTextureObject_t texObject)
{
    CUresult cu_res;
    cu_res = cuTexObjectDestroy((CUtexObject)texObject);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexObjectGetResourceDesc(UPTK_RESOURCE_DESC * pResDesc,UPTKTextureObject_t texObject)
{
    CUresult cu_res;
    cu_res = cuTexObjectGetResourceDesc((CUDA_RESOURCE_DESC *)pResDesc, (CUtexObject)texObject);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexObjectGetResourceViewDesc(UPTK_RESOURCE_VIEW_DESC * pResViewDesc,UPTKTextureObject_t texObject)
{
    CUresult cu_res;
    cu_res = cuTexObjectGetResourceViewDesc((CUDA_RESOURCE_VIEW_DESC *)pResViewDesc, (CUtexObject)texObject);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexObjectGetTextureDesc(UPTK_TEXTURE_DESC * pTexDesc,UPTKTextureObject_t texObject)
{
    CUresult cu_res;
    cu_res = cuTexObjectGetTextureDesc((CUDA_TEXTURE_DESC *)pTexDesc, (CUtexObject)texObject);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefCreate(UPTKtexref * pTexRef)
{
    CUresult cu_res;
    cu_res = cuTexRefCreate((CUtexref *)pTexRef);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefDestroy(UPTKtexref hTexRef)
{
    CUresult cu_res;
    cu_res = cuTexRefDestroy((CUtexref)hTexRef);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefGetAddress(UPTKdeviceptr * pdptr,UPTKtexref hTexRef)
{
    if (!pdptr) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuTexRefGetAddress((CUdeviceptr *)pdptr, (CUtexref)hTexRef);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefGetAddressMode(UPTKaddress_mode * pam,UPTKtexref hTexRef,int dim)
{
    CUresult cu_res;
    cu_res = cuTexRefGetAddressMode((CUaddress_mode *)pam, (CUtexref)hTexRef, dim);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefGetArray(UPTKarray * phArray, UPTKtexref hTexRef)
{
    if (!phArray) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuTexRefGetArray((CUarray *)phArray, (CUtexref)hTexRef);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefGetBorderColor(float * pBorderColor,UPTKtexref hTexRef)
{
    CUresult cu_res;
    cu_res = cuTexRefGetBorderColor(pBorderColor, (CUtexref)hTexRef);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefGetFilterMode(UPTKfilter_mode * pfm,UPTKtexref hTexRef)
{
    CUresult cu_res;
    cu_res = cuTexRefGetFilterMode((CUfilter_mode *)pfm, (CUtexref)hTexRef);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefGetFlags(unsigned int * pFlags,UPTKtexref hTexRef)
{
    CUresult cu_res;
    cu_res = cuTexRefGetFlags(pFlags, (CUtexref)hTexRef);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefGetFormat(UPTKarray_format * pFormat,int * pNumChannels,UPTKtexref hTexRef)
{
    CUresult cu_res;
    cu_res = cuTexRefGetFormat((CUarray_format *)pFormat, pNumChannels, (CUtexref)hTexRef);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefGetMaxAnisotropy(int * pmaxAniso,UPTKtexref hTexRef)
{
    CUresult cu_res;
    cu_res = cuTexRefGetMaxAnisotropy(pmaxAniso, (CUtexref)hTexRef);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefGetMipmapFilterMode(UPTKfilter_mode * pfm,UPTKtexref hTexRef)
{
    CUresult cu_res;
    cu_res = cuTexRefGetMipmapFilterMode((CUfilter_mode *)pfm, (CUtexref)hTexRef);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefGetMipmapLevelBias(float * pbias,UPTKtexref hTexRef)
{
    CUresult cu_res;
    cu_res = cuTexRefGetMipmapLevelBias(pbias, (CUtexref)hTexRef);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefGetMipmapLevelClamp(float * pminMipmapLevelClamp,float * pmaxMipmapLevelClamp,UPTKtexref hTexRef)
{
    CUresult cu_res;
    cu_res = cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp, pmaxMipmapLevelClamp, (CUtexref)hTexRef);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefGetMipmappedArray(UPTKMipmappedArray_t * phMipmappedArray,UPTKtexref hTexRef)
{
    CUresult cu_res;
    cu_res = cuTexRefGetMipmappedArray((CUmipmappedArray *)phMipmappedArray, (CUtexref)hTexRef);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefSetAddress_v2(size_t * ByteOffset,UPTKtexref hTexRef,UPTKdeviceptr dptr,size_t bytes)
{
    CUresult cu_res;
    cu_res = cuTexRefSetAddress_v2(ByteOffset, (CUtexref)hTexRef, (CUdeviceptr)dptr, bytes);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefSetAddress2D_v3(UPTKtexref hTexRef,const UPTK_ARRAY_DESCRIPTOR * desc,UPTKdeviceptr dptr,size_t Pitch)
{
    if (!desc) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuTexRefSetAddress2D_v3((CUtexref)hTexRef, (const CUDA_ARRAY_DESCRIPTOR *)desc, (CUdeviceptr)dptr, Pitch);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefSetAddressMode(UPTKtexref hTexRef,int dim,UPTKaddress_mode am)
{
    CUresult cu_res;
    cu_res = cuTexRefSetAddressMode((CUtexref)hTexRef, dim, (CUaddress_mode)am);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefSetArray(UPTKtexref hTexRef,UPTKarray hArray,unsigned int Flags)
{
    CUresult cu_res;
    cu_res = cuTexRefSetArray((CUtexref)hTexRef, (CUarray)hArray, Flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefSetBorderColor(UPTKtexref hTexRef,float * pBorderColor)
{
    CUresult cu_res;
    cu_res = cuTexRefSetBorderColor((CUtexref)hTexRef, pBorderColor);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefSetFilterMode(UPTKtexref hTexRef,UPTKfilter_mode fm)
{
    CUresult cu_res;
    cu_res = cuTexRefSetFilterMode((CUtexref)hTexRef, (CUfilter_mode)fm);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefSetFlags(UPTKtexref hTexRef,unsigned int Flags)
{
    CUresult cu_res;
    cu_res = cuTexRefSetFlags((CUtexref)hTexRef, Flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefSetFormat(UPTKtexref hTexRef,UPTKarray_format fmt,int NumPackedComponents)
{
    if (NumPackedComponents <= 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuTexRefSetFormat((CUtexref)hTexRef, (CUarray_format)fmt, NumPackedComponents);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefSetMaxAnisotropy(UPTKtexref hTexRef,unsigned int maxAniso)
{
    CUresult cu_res;
    cu_res = cuTexRefSetMaxAnisotropy((CUtexref)hTexRef, maxAniso);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefSetMipmapFilterMode(UPTKtexref hTexRef,UPTKfilter_mode fm)
{
    CUresult cu_res;
    cu_res = cuTexRefSetMipmapFilterMode((CUtexref)hTexRef, (CUfilter_mode)fm);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefSetMipmapLevelBias(UPTKtexref hTexRef,float bias)
{
    CUresult cu_res;
    cu_res = cuTexRefSetMipmapLevelBias((CUtexref)hTexRef, bias);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefSetMipmapLevelClamp(UPTKtexref hTexRef,float minMipmapLevelClamp,float maxMipmapLevelClamp)
{
    CUresult cu_res;
    cu_res = cuTexRefSetMipmapLevelClamp((CUtexref)hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTexRefSetMipmappedArray(UPTKtexref hTexRef,UPTKMipmappedArray_t hMipmappedArray,unsigned int Flags)
{
    CUresult cu_res;
    cu_res = cuTexRefSetMipmappedArray((CUtexref)hTexRef, (CUmipmappedArray)hMipmappedArray, Flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPThreadExchangeStreamCaptureMode(UPTKStreamCaptureMode * mode)
{
    CUresult cu_res;
    cu_res = cuThreadExchangeStreamCaptureMode((CUstreamCaptureMode *)mode);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPWaitExternalSemaphoresAsync(const UPTKExternalSemaphore_t * extSemArray,const UPTK_EXTERNAL_SEMAPHORE_WAIT_PARAMS * paramsArray,unsigned int numExtSems,UPTKStream_t stream)
{
    CUresult cu_res;
    cu_res = cuWaitExternalSemaphoresAsync((const CUexternalSemaphore *)extSemArray, (const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *)paramsArray, numExtSems, (CUstream)stream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceGetTexture1DLinearMaxWidth(size_t * maxWidthInElements, UPTKarray_format format, unsigned numChannels, UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, (CUarray_format)format, numChannels, (CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceSetMemPool(UPTKdevice dev, UPTKMemPool_t pool)
{
    CUresult cu_res;
    cu_res = cuDeviceSetMemPool((CUdevice)dev, (CUmemoryPool)pool);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceGetMemPool(UPTKMemPool_t * pool, UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuDeviceGetMemPool((CUmemoryPool *)pool, (CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceGetDefaultMemPool(UPTKMemPool_t * pool_out, UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuDeviceGetDefaultMemPool((CUmemoryPool *)pool_out, (CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPFlushGPUDirectRDMAWrites(UPTKFlushGPUDirectRDMAWritesTarget target, UPTKFlushGPUDirectRDMAWritesScope scope)
{
    CUresult cu_res;
    cu_res = cuFlushGPUDirectRDMAWrites((CUflushGPUDirectRDMAWritesTarget)target, (CUflushGPUDirectRDMAWritesScope)scope);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceGetExecAffinitySupport(int * pi, UPTKexecAffinityType type, UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuDeviceGetExecAffinitySupport(pi, (CUexecAffinityType)type, (CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxCreate_v3(UPTKcontext * pctx, UPTKexecAffinityParam * paramsArray, int numParams, unsigned int flags, UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuCtxCreate_v3((CUcontext *)pctx, (CUexecAffinityParam *)paramsArray, numParams, flags, (CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxResetPersistingL2Cache(void)
{
    CUresult cu_res;
    cu_res = cuCtxResetPersistingL2Cache();
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxGetExecAffinity(UPTKexecAffinityParam * pExecAffinity, UPTKexecAffinityType type)
{
    CUresult cu_res;
    cu_res = cuCtxGetExecAffinity((CUexecAffinityParam *)pExecAffinity, (CUexecAffinityType)type);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPModuleGetLoadingMode(UPTKmoduleLoadingMode * mode)
{
    CUresult cu_res;
    cu_res = cuModuleGetLoadingMode((CUmoduleLoadingMode *)mode);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPArrayGetSparseProperties(UPTK_ARRAY_SPARSE_PROPERTIES * sparseProperties, UPTKarray array)
{
    CUresult cu_res;
    cu_res = cuArrayGetSparseProperties((CUDA_ARRAY_SPARSE_PROPERTIES *)sparseProperties, (CUarray)array);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMipmappedArrayGetSparseProperties(UPTK_ARRAY_SPARSE_PROPERTIES * sparseProperties, UPTKMipmappedArray_t mipmap)
{
    CUresult cu_res;
    cu_res = cuMipmappedArrayGetSparseProperties((CUDA_ARRAY_SPARSE_PROPERTIES *)sparseProperties, (CUmipmappedArray)mipmap);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPArrayGetMemoryRequirements(UPTK_ARRAY_MEMORY_REQUIREMENTS * memoryRequirements, UPTKarray array, UPTKdevice device)
{
    CUresult cu_res;
    cu_res = cuArrayGetMemoryRequirements((CUDA_ARRAY_MEMORY_REQUIREMENTS *)memoryRequirements, (CUarray)array, (CUdevice)device);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMipmappedArrayGetMemoryRequirements(UPTK_ARRAY_MEMORY_REQUIREMENTS * memoryRequirements, UPTKMipmappedArray_t mipmap, UPTKdevice device)
{
    CUresult cu_res;
    cu_res = cuMipmappedArrayGetMemoryRequirements((CUDA_ARRAY_MEMORY_REQUIREMENTS *)memoryRequirements, (CUmipmappedArray)mipmap, (CUdevice)device);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPArrayGetPlane(UPTKarray * pPlaneArray, UPTKarray hArray, unsigned int planeIdx)
{
    CUresult cu_res;
    cu_res = cuArrayGetPlane((CUarray *)pPlaneArray, (CUarray)hArray, planeIdx);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemGetHandleForAddressRange(void * handle, UPTKdeviceptr dptr, size_t size, UPTKmemRangeHandleType handleType, unsigned long long flags)
{
    if (!handle || size == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemGetHandleForAddressRange(handle, (CUdeviceptr)dptr, size, (CUmemRangeHandleType)handleType, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemMapArrayAsync(UPTKarrayMapInfo * mapInfoList, unsigned int count, UPTKStream_t hStream)
{
    if (!mapInfoList || count == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemMapArrayAsync((CUarrayMapInfo *)mapInfoList, count, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemRetainAllocationHandle(UPTKmemGenericAllocationHandle * handle, void * addr)
{
    CUresult cu_res;
    cu_res = cuMemRetainAllocationHandle((CUmemGenericAllocationHandle *)handle, addr);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemFreeAsync(UPTKdeviceptr dptr, UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemFreeAsync((CUdeviceptr)dptr, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemAllocAsync(UPTKdeviceptr * dptr, size_t bytesize, UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemAllocAsync((CUdeviceptr *)dptr, bytesize, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemPoolTrimTo(UPTKMemPool_t pool, size_t minBytesToKeep)
{
    CUresult cu_res;
    cu_res = cuMemPoolTrimTo((CUmemoryPool)pool, minBytesToKeep);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemPoolSetAttribute(UPTKMemPool_t pool, UPTKmemPool_attribute attr, void * value)
{
    CUresult cu_res;
    cu_res = cuMemPoolSetAttribute((CUmemoryPool)pool, (CUmemPool_attribute)attr, value);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemPoolGetAttribute(UPTKMemPool_t pool, UPTKmemPool_attribute attr, void * value)
{
    CUresult cu_res;
    cu_res = cuMemPoolGetAttribute((CUmemoryPool)pool, (CUmemPool_attribute)attr, value);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemPoolSetAccess(UPTKMemPool_t pool, const UPTKMemAccessDesc * map, size_t count)
{
    CUresult cu_res;
    cu_res = cuMemPoolSetAccess((CUmemoryPool)pool, (const CUmemAccessDesc *)map, count);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemPoolGetAccess(UPTKmemAccess_flags * flags, UPTKMemPool_t memPool, UPTKMemLocation * location)
{
    CUresult cu_res;
    cu_res = cuMemPoolGetAccess((CUmemAccess_flags *)flags, (CUmemoryPool)memPool, (CUmemLocation *)location);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemPoolCreate(UPTKMemPool_t * pool, const UPTKMemPoolProps * poolProps)
{
    CUresult cu_res;
    cu_res = cuMemPoolCreate((CUmemoryPool *)pool, (const CUmemPoolProps *)poolProps);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemPoolDestroy(UPTKMemPool_t pool)
{
    CUresult cu_res;
    cu_res = cuMemPoolDestroy((CUmemoryPool)pool);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemAllocFromPoolAsync(UPTKdeviceptr * dptr, size_t bytesize, UPTKMemPool_t pool, UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemAllocFromPoolAsync((CUdeviceptr *)dptr, bytesize, (CUmemoryPool)pool, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemPoolExportToShareableHandle(void * handle_out, UPTKMemPool_t pool, UPTKMemAllocationHandleType handleType, unsigned long long flags)
{
    CUresult cu_res;
    cu_res = cuMemPoolExportToShareableHandle(handle_out, (CUmemoryPool)pool, (CUmemAllocationHandleType)handleType, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemPoolExportPointer(UPTKMemPoolPtrExportData * shareData_out, UPTKdeviceptr ptr)
{
    CUresult cu_res;
    cu_res = cuMemPoolExportPointer((CUmemPoolPtrExportData *)shareData_out, (CUdeviceptr)ptr);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemPoolImportFromShareableHandle(UPTKMemPool_t * pool_out, void * handle, UPTKMemAllocationHandleType handleType, unsigned long long flags)
{
    CUresult cu_res;
    cu_res = cuMemPoolImportFromShareableHandle((CUmemoryPool *)pool_out, handle, (CUmemAllocationHandleType)handleType, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemPoolImportPointer(UPTKdeviceptr * ptr_out, UPTKMemPool_t pool, UPTKMemPoolPtrExportData * shareData)
{
    CUresult cu_res;
    cu_res = cuMemPoolImportPointer((CUdeviceptr *)ptr_out, (CUmemoryPool)pool, (CUmemPoolPtrExportData *)shareData);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamGetCaptureInfo_v2(UPTKStream_t hStream, UPTKStreamCaptureStatus * captureStatus_out, cuuint64_t * id_out, UPTKGraph_t * graph_out, const UPTKGraphNode_t ** dependencies_out, size_t * numDependencies_out)
{
    CUresult cu_res;
    cu_res = cuStreamGetCaptureInfo_v2((CUstream)hStream, (CUstreamCaptureStatus *)captureStatus_out, id_out, (CUgraph *)graph_out, (const CUgraphNode **)dependencies_out, numDependencies_out);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamUpdateCaptureDependencies(UPTKStream_t hStream, UPTKGraphNode_t * dependencies, size_t numDependencies, unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuStreamUpdateCaptureDependencies((CUstream)hStream, (CUgraphNode *)dependencies, numDependencies, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamCopyAttributes(UPTKStream_t dst, UPTKStream_t src)
{
    CUresult cu_res;
    cu_res = cuStreamCopyAttributes((CUstream)dst, (CUstream)src);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamGetAttribute(UPTKStream_t hStream, UPTKStreamAttrID attr, UPTKStreamAttrValue * value_out)
{
    CUresult cu_res;
    cu_res = cuStreamGetAttribute((CUstream)hStream, (CUstreamAttrID)attr, (CUstreamAttrValue *)value_out);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamSetAttribute(UPTKStream_t hStream, UPTKStreamAttrID attr, const UPTKStreamAttrValue * value)
{
    CUresult cu_res;
    cu_res = cuStreamSetAttribute((CUstream)hStream, (CUstreamAttrID)attr, (const CUstreamAttrValue *)value);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPEventRecordWithFlags(UPTKEvent_t hEvent, UPTKStream_t hStream, unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuEventRecordWithFlags((CUevent)hEvent, (CUstream)hStream, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamWaitValue32(UPTKStream_t stream,UPTKdeviceptr addr,cuuint32_t value,unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuStreamWaitValue32((CUstream)stream, (CUdeviceptr)addr, value, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamWaitValue64(UPTKStream_t stream,UPTKdeviceptr addr,cuuint64_t value,unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuStreamWaitValue64((CUstream)stream, (CUdeviceptr)addr, value, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPFuncGetModule(UPTKmodule * hmod, UPTKfunction hfunc)
{
    CUresult cu_res;
    cu_res = cuFuncGetModule((CUmodule *)hmod, (CUfunction)hfunc);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLaunchKernelEx(const UPTKLaunchConfig_t * config, UPTKfunction f, void ** kernelParams, void ** extra)
{
    CUresult cu_res;
    cu_res = cuLaunchKernelEx((const CUlaunchConfig *)config, (CUfunction)f, kernelParams, extra);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphAddEventRecordNode(UPTKGraphNode_t * phGraphNode, UPTKGraph_t hGraph, const UPTKGraphNode_t * dependencies, size_t numDependencies, UPTKEvent_t event)
{
    CUresult cu_res;
    cu_res = cuGraphAddEventRecordNode((CUgraphNode *)phGraphNode, (CUgraph)hGraph, (const CUgraphNode *)dependencies, numDependencies, (CUevent)event);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphEventRecordNodeGetEvent(UPTKGraphNode_t hNode, UPTKEvent_t * event_out)
{
    CUresult cu_res;
    cu_res = cuGraphEventRecordNodeGetEvent((CUgraphNode)hNode, (CUevent *)event_out);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphEventRecordNodeSetEvent(UPTKGraphNode_t hNode, UPTKEvent_t event)
{
    CUresult cu_res;
    cu_res = cuGraphEventRecordNodeSetEvent((CUgraphNode)hNode, (CUevent)event);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphAddEventWaitNode(UPTKGraphNode_t * phGraphNode, UPTKGraph_t hGraph, const UPTKGraphNode_t * dependencies, size_t numDependencies, UPTKEvent_t event)
{
    CUresult cu_res;
    cu_res = cuGraphAddEventWaitNode((CUgraphNode *)phGraphNode, (CUgraph)hGraph, (const CUgraphNode *)dependencies, numDependencies, (CUevent)event);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphEventWaitNodeGetEvent(UPTKGraphNode_t hNode, UPTKEvent_t * event_out)
{
    CUresult cu_res;
    cu_res = cuGraphEventWaitNodeGetEvent((CUgraphNode)hNode, (CUevent *)event_out);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphEventWaitNodeSetEvent(UPTKGraphNode_t hNode, UPTKEvent_t event)
{
    CUresult cu_res;
    cu_res = cuGraphEventWaitNodeSetEvent((CUgraphNode)hNode, (CUevent)event);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphAddExternalSemaphoresSignalNode(UPTKGraphNode_t * phGraphNode, UPTKGraph_t hGraph, const UPTKGraphNode_t * dependencies, size_t numDependencies, const UPTK_EXT_SEM_SIGNAL_NODE_PARAMS * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphAddExternalSemaphoresSignalNode((CUgraphNode *)phGraphNode, (CUgraph)hGraph, (const CUgraphNode *)dependencies, numDependencies, (const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphExternalSemaphoresSignalNodeGetParams(UPTKGraphNode_t hNode, UPTK_EXT_SEM_SIGNAL_NODE_PARAMS * params_out)
{
    if (!hNode || !params_out) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuGraphExternalSemaphoresSignalNodeGetParams((CUgraphNode)hNode, (CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *)params_out);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphExternalSemaphoresSignalNodeSetParams(UPTKGraphNode_t hNode, const UPTK_EXT_SEM_SIGNAL_NODE_PARAMS * nodeParams)
{
    if (!hNode || !nodeParams) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuGraphExternalSemaphoresSignalNodeSetParams((CUgraphNode)hNode, (const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphAddExternalSemaphoresWaitNode(UPTKGraphNode_t * phGraphNode, UPTKGraph_t hGraph, const UPTKGraphNode_t * dependencies, size_t numDependencies, const UPTK_EXT_SEM_WAIT_NODE_PARAMS * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphAddExternalSemaphoresWaitNode((CUgraphNode *)phGraphNode, (CUgraph)hGraph, (const CUgraphNode *)dependencies, numDependencies, (const CUDA_EXT_SEM_WAIT_NODE_PARAMS *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphExternalSemaphoresWaitNodeGetParams(UPTKGraphNode_t hNode, UPTK_EXT_SEM_WAIT_NODE_PARAMS * params_out)
{
    if (!hNode || !params_out) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuGraphExternalSemaphoresWaitNodeGetParams((CUgraphNode)hNode, (CUDA_EXT_SEM_WAIT_NODE_PARAMS *)params_out);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphExternalSemaphoresWaitNodeSetParams(UPTKGraphNode_t hNode, const UPTK_EXT_SEM_WAIT_NODE_PARAMS * nodeParams)
{
    if (!hNode || !nodeParams) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuGraphExternalSemaphoresWaitNodeSetParams((CUgraphNode)hNode, (const CUDA_EXT_SEM_WAIT_NODE_PARAMS *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphAddBatchMemOpNode(UPTKGraphNode_t * phGraphNode, UPTKGraph_t hGraph, const UPTKGraphNode_t * dependencies, size_t numDependencies, const UPTK_BATCH_MEM_OP_NODE_PARAMS * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphAddBatchMemOpNode((CUgraphNode *)phGraphNode, (CUgraph)hGraph, (const CUgraphNode *)dependencies, numDependencies, (const CUDA_BATCH_MEM_OP_NODE_PARAMS *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphBatchMemOpNodeGetParams(UPTKGraphNode_t hNode, UPTK_BATCH_MEM_OP_NODE_PARAMS * nodeParams_out)
{
    if (!hNode || !nodeParams_out) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuGraphBatchMemOpNodeGetParams((CUgraphNode)hNode, (CUDA_BATCH_MEM_OP_NODE_PARAMS *)nodeParams_out);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphBatchMemOpNodeSetParams(UPTKGraphNode_t hNode, const UPTK_BATCH_MEM_OP_NODE_PARAMS * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphBatchMemOpNodeSetParams((CUgraphNode)hNode, (const CUDA_BATCH_MEM_OP_NODE_PARAMS *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphExecBatchMemOpNodeSetParams(UPTKGraphExec_t hGraphExec, UPTKGraphNode_t hNode, const UPTK_BATCH_MEM_OP_NODE_PARAMS * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphExecBatchMemOpNodeSetParams((CUgraphExec)hGraphExec, (CUgraphNode)hNode, (const CUDA_BATCH_MEM_OP_NODE_PARAMS *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphAddMemAllocNode(UPTKGraphNode_t * phGraphNode, UPTKGraph_t hGraph, const UPTKGraphNode_t * dependencies, size_t numDependencies, UPTK_MEM_ALLOC_NODE_PARAMS * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphAddMemAllocNode((CUgraphNode *)phGraphNode, (CUgraph)hGraph, (const CUgraphNode *)dependencies, numDependencies, (CUDA_MEM_ALLOC_NODE_PARAMS *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphMemAllocNodeGetParams(UPTKGraphNode_t hNode, UPTK_MEM_ALLOC_NODE_PARAMS * params_out)
{
    CUresult cu_res;
    cu_res = cuGraphMemAllocNodeGetParams((CUgraphNode)hNode, (CUDA_MEM_ALLOC_NODE_PARAMS *)params_out);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphAddMemFreeNode(UPTKGraphNode_t * phGraphNode, UPTKGraph_t hGraph, const UPTKGraphNode_t * dependencies, size_t numDependencies, UPTKdeviceptr dptr)
{
    CUresult cu_res;
    cu_res = cuGraphAddMemFreeNode((CUgraphNode *)phGraphNode, (CUgraph)hGraph, (const CUgraphNode *)dependencies, numDependencies, (CUdeviceptr)dptr);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphMemFreeNodeGetParams(UPTKGraphNode_t hNode, UPTKdeviceptr * dptr_out)
{
    CUresult cu_res;
    cu_res = cuGraphMemFreeNodeGetParams((CUgraphNode)hNode, (CUdeviceptr *)dptr_out);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceGraphMemTrim(UPTKdevice device)
{
    CUresult cu_res;
    cu_res = cuDeviceGraphMemTrim((CUdevice)device);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceGetGraphMemAttribute(UPTKdevice device, UPTKgraphMem_attribute attr, void* value)
{
    CUresult cu_res;
    cu_res = cuDeviceGetGraphMemAttribute((CUdevice)device, (CUgraphMem_attribute)attr, value);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceSetGraphMemAttribute(UPTKdevice device, UPTKgraphMem_attribute attr, void* value)
{
    CUresult cu_res;
    cu_res = cuDeviceSetGraphMemAttribute((CUdevice)device, (CUgraphMem_attribute)attr, value);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphInstantiateWithFlags(UPTKGraphExec_t * phGraphExec, UPTKGraph_t hGraph, unsigned long long flags)
{
    CUresult cu_res;
    cu_res = cuGraphInstantiateWithFlags((CUgraphExec *)phGraphExec, (CUgraph)hGraph, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphExecEventRecordNodeSetEvent(UPTKGraphExec_t hGraphExec, UPTKGraphNode_t hNode, UPTKEvent_t event)
{
    CUresult cu_res;
    cu_res = cuGraphExecEventRecordNodeSetEvent((CUgraphExec)hGraphExec, (CUgraphNode)hNode, (CUevent)event);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphExecEventWaitNodeSetEvent(UPTKGraphExec_t hGraphExec, UPTKGraphNode_t hNode, UPTKEvent_t event)
{
    CUresult cu_res;
    cu_res = cuGraphExecEventWaitNodeSetEvent((CUgraphExec)hGraphExec, (CUgraphNode)hNode, (CUevent)event);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphExecExternalSemaphoresSignalNodeSetParams(UPTKGraphExec_t hGraphExec, UPTKGraphNode_t hNode, const UPTK_EXT_SEM_SIGNAL_NODE_PARAMS * nodeParams)
{
    if (!hGraphExec || !hNode || !nodeParams) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuGraphExecExternalSemaphoresSignalNodeSetParams((CUgraphExec)hGraphExec, (CUgraphNode)hNode, (const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphExecExternalSemaphoresWaitNodeSetParams(UPTKGraphExec_t hGraphExec, UPTKGraphNode_t hNode, const UPTK_EXT_SEM_WAIT_NODE_PARAMS * nodeParams)
{
    if (!hGraphExec || !hNode || !nodeParams) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuGraphExecExternalSemaphoresWaitNodeSetParams((CUgraphExec)hGraphExec, (CUgraphNode)hNode, (const CUDA_EXT_SEM_WAIT_NODE_PARAMS *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphNodeSetEnabled(UPTKGraphExec_t hGraphExec, UPTKGraphNode_t hNode, unsigned int isEnabled)
{
    CUresult cu_res;
    cu_res = cuGraphNodeSetEnabled((CUgraphExec)hGraphExec, (CUgraphNode)hNode, isEnabled);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphNodeGetEnabled(UPTKGraphExec_t hGraphExec, UPTKGraphNode_t hNode, unsigned int * isEnabled)
{
    CUresult cu_res;
    cu_res = cuGraphNodeGetEnabled((CUgraphExec)hGraphExec, (CUgraphNode)hNode, isEnabled);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphUpload(UPTKGraphExec_t hGraphExec, UPTKStream_t hStream)
{
    if (!hGraphExec) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuGraphUpload((CUgraphExec)hGraphExec, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphKernelNodeCopyAttributes(UPTKGraphNode_t dst, UPTKGraphNode_t src)
{
    CUresult cu_res;
    cu_res = cuGraphKernelNodeCopyAttributes((CUgraphNode)dst, (CUgraphNode)src);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphKernelNodeGetAttribute(UPTKGraphNode_t hNode, UPTKKernelNodeAttrID attr, UPTKKernelNodeAttrValue * value_out)
{
    CUresult cu_res;
    cu_res = cuGraphKernelNodeGetAttribute((CUgraphNode)hNode, (CUkernelNodeAttrID)attr, (CUkernelNodeAttrValue *)value_out);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphKernelNodeSetAttribute(UPTKGraphNode_t hNode, UPTKKernelNodeAttrID attr, const UPTKKernelNodeAttrValue * value)
{
    if (!hNode || !value) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuGraphKernelNodeSetAttribute((CUgraphNode)hNode, (CUkernelNodeAttrID)attr, (const CUkernelNodeAttrValue *)value);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphDebugDotPrint(UPTKGraph_t hGraph, const char * path, unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuGraphDebugDotPrint((CUgraph)hGraph, path, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPUserObjectCreate(UPTKUserObject_t * object_out, void * ptr, UPTKhostFn destroy, unsigned int initialRefcount, unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuUserObjectCreate((CUuserObject *)object_out, ptr, (CUhostFn)destroy, initialRefcount, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPUserObjectRetain(UPTKUserObject_t object, unsigned int count)
{
    CUresult cu_res;
    cu_res = cuUserObjectRetain((CUuserObject)object, count);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPUserObjectRelease(UPTKUserObject_t object, unsigned int count)
{
    CUresult cu_res;
    cu_res = cuUserObjectRelease((CUuserObject)object, count);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphRetainUserObject(UPTKGraph_t graph, UPTKUserObject_t object, unsigned int count, unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuGraphRetainUserObject((CUgraph)graph, (CUuserObject)object, count, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphReleaseUserObject(UPTKGraph_t graph, UPTKUserObject_t object, unsigned int count)
{
    CUresult cu_res;
    cu_res = cuGraphReleaseUserObject((CUgraph)graph, (CUuserObject)object, count);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPOccupancyAvailableDynamicSMemPerBlock(size_t * dynamicSmemSize, UPTKfunction func, int numBlocks, int blockSize)
{
    CUresult cu_res;
    cu_res = cuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, (CUfunction)func, numBlocks, blockSize);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPOccupancyMaxPotentialClusterSize(int * clusterSize, UPTKfunction func, const UPTKLaunchConfig_t * config)
{
    CUresult cu_res;
    cu_res = cuOccupancyMaxPotentialClusterSize(clusterSize, (CUfunction)func, (const CUlaunchConfig *)config);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPOccupancyMaxActiveClusters(int * numClusters, UPTKfunction func, const UPTKLaunchConfig_t * config)
{
    CUresult cu_res;
    cu_res = cuOccupancyMaxActiveClusters(numClusters, (CUfunction)func, (const CUlaunchConfig *)config);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceGetUuid_v2(UPTKuuid * uuid, UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuDeviceGetUuid_v2((CUuuid *)uuid, (CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphExecChildGraphNodeSetParams(UPTKGraphExec_t hGraphExec, UPTKGraphNode_t hNode, UPTKGraph_t childGraph)
{
    CUresult cu_res;
    cu_res = cuGraphExecChildGraphNodeSetParams((CUgraphExec)hGraphExec, (CUgraphNode)hNode, (CUgraph)childGraph);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyHtoD_v2_ptds(UPTKdeviceptr dstDevice,const void * srcHost,size_t ByteCount)
{
    CUresult cu_res;
    cu_res = cuMemcpyHtoD_v2((CUdeviceptr)dstDevice, srcHost, ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyDtoH_v2_ptds(void * dstHost,UPTKdeviceptr srcDevice,size_t ByteCount)
{
    CUresult cu_res;
    cu_res = cuMemcpyDtoH_v2(dstHost, (CUdeviceptr)srcDevice, ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyDtoD_v2_ptds(UPTKdeviceptr dstDevice,UPTKdeviceptr srcDevice,size_t ByteCount)
{
    CUresult cu_res;
    cu_res = cuMemcpyDtoD_v2((CUdeviceptr)dstDevice, (CUdeviceptr)srcDevice, ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyDtoA_v2_ptds(UPTKarray dstArray,size_t dstOffset,UPTKdeviceptr srcDevice,size_t ByteCount)
{
    if (ByteCount == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemcpyDtoA_v2((CUarray)dstArray, dstOffset, (CUdeviceptr)srcDevice, ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyAtoD_v2_ptds(UPTKdeviceptr dstDevice,UPTKarray srcArray,size_t srcOffset,size_t ByteCount)
{
    if (ByteCount == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemcpyAtoD_v2((CUdeviceptr)dstDevice, (CUarray)srcArray, srcOffset, ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyHtoA_v2_ptds(UPTKarray dstArray,size_t dstOffset,const void * srcHost,size_t ByteCount)
{
    CUresult cu_res;
    cu_res = cuMemcpyHtoA_v2((CUarray)dstArray, dstOffset, srcHost, ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyAtoH_v2_ptds(void * dstHost,UPTKarray srcArray,size_t srcOffset,size_t ByteCount)
{
    CUresult cu_res;
    cu_res = cuMemcpyAtoH_v2(dstHost, (CUarray)srcArray, srcOffset, ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyAtoA_v2_ptds(UPTKarray dstArray,size_t dstOffset,UPTKarray srcArray,size_t srcOffset,size_t ByteCount)
{
    CUresult cu_res;
    cu_res = cuMemcpyAtoA_v2((CUarray)dstArray, dstOffset, (CUarray)srcArray, srcOffset, ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyHtoAAsync_v2_ptsz(UPTKarray dstArray,size_t dstOffset,const void * srcHost,size_t ByteCount,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemcpyHtoAAsync_v2((CUarray)dstArray, dstOffset, srcHost, ByteCount, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyAtoHAsync_v2_ptsz(void * dstHost,UPTKarray srcArray,size_t srcOffset,size_t ByteCount,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemcpyAtoHAsync_v2(dstHost, (CUarray)srcArray, srcOffset, ByteCount, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpy2D_v2_ptds(const UPTK_MEMCPY2D * pCopy)
{
    if (!pCopy || pCopy->WidthInBytes == 0 || pCopy->Height == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemcpy2D_v2((const CUDA_MEMCPY2D *)pCopy);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpy2DUnaligned_v2_ptds(const UPTK_MEMCPY2D * pCopy)
{
    if (!pCopy || pCopy->WidthInBytes == 0 || pCopy->Height == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemcpy2DUnaligned_v2((const CUDA_MEMCPY2D *)pCopy);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpy3D_v2_ptds(const UPTK_MEMCPY3D * pCopy)
{
    if (!pCopy || pCopy->WidthInBytes == 0 || pCopy->Height == 0 || pCopy->Depth == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemcpy3D_v2((const CUDA_MEMCPY3D *)pCopy);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyHtoDAsync_v2_ptsz(UPTKdeviceptr dstDevice,const void * srcHost,size_t ByteCount,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemcpyHtoDAsync_v2((CUdeviceptr)dstDevice, srcHost, ByteCount, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyDtoHAsync_v2_ptsz(void * dstHost,UPTKdeviceptr srcDevice,size_t ByteCount,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemcpyDtoHAsync_v2(dstHost, (CUdeviceptr)srcDevice, ByteCount, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyDtoDAsync_v2_ptsz(UPTKdeviceptr dstDevice,UPTKdeviceptr srcDevice,size_t ByteCount,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemcpyDtoDAsync_v2((CUdeviceptr)dstDevice, (CUdeviceptr)srcDevice, ByteCount, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpy2DAsync_v2_ptsz(const UPTK_MEMCPY2D * pCopy,UPTKStream_t hStream)
{
    if (!pCopy || pCopy->WidthInBytes == 0 || pCopy->Height == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemcpy2DAsync_v2((const CUDA_MEMCPY2D *)pCopy, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpy3DAsync_v2_ptsz(const UPTK_MEMCPY3D * pCopy,UPTKStream_t hStream)
{
    if (!pCopy || pCopy->WidthInBytes == 0 || pCopy->Height == 0 || pCopy->Depth == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemcpy3DAsync_v2((const CUDA_MEMCPY3D *)pCopy, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD8_v2_ptds(UPTKdeviceptr dstDevice,unsigned char uc,size_t N)
{
    CUresult cu_res;
    cu_res = cuMemsetD8_v2((CUdeviceptr)dstDevice, uc, N);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD16_v2_ptds(UPTKdeviceptr dstDevice,unsigned short us,size_t N)
{
    CUresult cu_res;
    cu_res = cuMemsetD16_v2((CUdeviceptr)dstDevice, us, N);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD32_v2_ptds(UPTKdeviceptr dstDevice,unsigned int ui,size_t N)
{
    CUresult cu_res;
    cu_res = cuMemsetD32_v2((CUdeviceptr)dstDevice, ui, N);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD2D8_v2_ptds(UPTKdeviceptr dstDevice,size_t dstPitch,unsigned char uc,size_t Width,size_t Height)
{
    CUresult cu_res;
    cu_res = cuMemsetD2D8_v2((CUdeviceptr)dstDevice, dstPitch, uc, Width, Height);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD2D16_v2_ptds(UPTKdeviceptr dstDevice,size_t dstPitch,unsigned short us,size_t Width,size_t Height)
{
    CUresult cu_res;
    cu_res = cuMemsetD2D16_v2((CUdeviceptr)dstDevice, dstPitch, us, Width, Height);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD2D32_v2_ptds(UPTKdeviceptr dstDevice,size_t dstPitch,unsigned int ui,size_t Width,size_t Height)
{
    CUresult cu_res;
    cu_res = cuMemsetD2D32_v2((CUdeviceptr)dstDevice, dstPitch, ui, Width, Height);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamBeginCapture_v2_ptsz(UPTKStream_t hStream,UPTKStreamCaptureMode mode)
{
    CUresult cu_res;
    cu_res = cuStreamBeginCapture_v2((CUstream)hStream, (CUstreamCaptureMode)mode);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpy_ptds(UPTKdeviceptr dst, UPTKdeviceptr src, size_t ByteCount)
{
    CUresult cu_res;
    cu_res = cuMemcpy((CUdeviceptr)dst, (CUdeviceptr)src, ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyAsync_ptsz(UPTKdeviceptr dst,UPTKdeviceptr src,size_t ByteCount,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src, ByteCount, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyPeer_ptds(UPTKdeviceptr dstDevice,UPTKcontext dstContext,UPTKdeviceptr srcDevice,UPTKcontext srcContext,size_t ByteCount)
{
    CUresult cu_res;
    cu_res = cuMemcpyPeer((CUdeviceptr)dstDevice, (CUcontext)dstContext, (CUdeviceptr)srcDevice, (CUcontext)srcContext, ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpyPeerAsync_ptsz(UPTKdeviceptr dstDevice,UPTKcontext dstContext,UPTKdeviceptr srcDevice,UPTKcontext srcContext,size_t ByteCount,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemcpyPeerAsync((CUdeviceptr)dstDevice, (CUcontext)dstContext, (CUdeviceptr)srcDevice, (CUcontext)srcContext, ByteCount, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpy3DPeer_ptds(const UPTK_MEMCPY3D_PEER * pCopy)
{
    if (!pCopy || pCopy->WidthInBytes == 0 || pCopy->Height == 0 || pCopy->Depth == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemcpy3DPeer((const CUDA_MEMCPY3D_PEER *)pCopy);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemcpy3DPeerAsync_ptsz(const UPTK_MEMCPY3D_PEER * pCopy,UPTKStream_t hStream)
{
    if (!pCopy || pCopy->WidthInBytes == 0 || pCopy->Height == 0 || pCopy->Depth == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemcpy3DPeerAsync((const CUDA_MEMCPY3D_PEER *)pCopy, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemPrefetchAsync_ptsz(UPTKdeviceptr devPtr,size_t count,UPTKdevice dstDevice,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemPrefetchAsync((CUdeviceptr)devPtr, count, (CUdevice)dstDevice, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD8Async_ptsz(UPTKdeviceptr dstDevice,unsigned char uc,size_t N,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemsetD8Async((CUdeviceptr)dstDevice, uc, N, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD16Async_ptsz(UPTKdeviceptr dstDevice,unsigned short us,size_t N,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemsetD16Async((CUdeviceptr)dstDevice, us, N, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD32Async_ptsz(UPTKdeviceptr dstDevice,unsigned int ui,size_t N,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemsetD32Async((CUdeviceptr)dstDevice, ui, N, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD2D8Async_ptsz(UPTKdeviceptr dstDevice,size_t dstPitch,unsigned char uc,size_t Width,size_t Height,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemsetD2D8Async((CUdeviceptr)dstDevice, dstPitch, uc, Width, Height, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD2D16Async_ptsz(UPTKdeviceptr dstDevice,size_t dstPitch,unsigned short us,size_t Width,size_t Height,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemsetD2D16Async((CUdeviceptr)dstDevice, dstPitch, us, Width, Height, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemsetD2D32Async_ptsz(UPTKdeviceptr dstDevice,size_t dstPitch,unsigned int ui,size_t Width,size_t Height,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemsetD2D32Async((CUdeviceptr)dstDevice, dstPitch, ui, Width, Height, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamGetPriority_ptsz(UPTKStream_t hStream,int * priority)
{
    CUresult cu_res;
    cu_res = cuStreamGetPriority((CUstream)hStream, priority);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamGetFlags_ptsz(UPTKStream_t hStream,unsigned int * flags)
{
    CUresult cu_res;
    cu_res = cuStreamGetFlags((CUstream)hStream, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamGetCtx_ptsz(UPTKStream_t hStream,UPTKcontext * pctx)
{
    CUresult cu_res;
    cu_res = cuStreamGetCtx((CUstream)hStream, (CUcontext *)pctx);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamWaitEvent_ptsz(UPTKStream_t hStream,UPTKEvent_t hEvent,unsigned int Flags)
{
    CUresult cu_res;
    cu_res = cuStreamWaitEvent((CUstream)hStream, (CUevent)hEvent, Flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamEndCapture_ptsz(UPTKStream_t hStream,UPTKGraph_t * phGraph)
{
    CUresult cu_res;
    cu_res = cuStreamEndCapture((CUstream)hStream, (CUgraph *)phGraph);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamIsCapturing_ptsz(UPTKStream_t hStream,UPTKStreamCaptureStatus * captureStatus)
{
    CUresult cu_res;
    cu_res = cuStreamIsCapturing((CUstream)hStream, (CUstreamCaptureStatus *)captureStatus);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamGetCaptureInfo_v2_ptsz(UPTKStream_t hStream, UPTKStreamCaptureStatus * captureStatus_out, cuuint64_t * id_out, UPTKGraph_t * graph_out, const UPTKGraphNode_t ** dependencies_out, size_t * numDependencies_out)
{
    CUresult cu_res;
    cu_res = cuStreamGetCaptureInfo_v2((CUstream)hStream, (CUstreamCaptureStatus *)captureStatus_out, id_out, (CUgraph *)graph_out, (const CUgraphNode **)dependencies_out, numDependencies_out);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamUpdateCaptureDependencies_ptsz(UPTKStream_t hStream, UPTKGraphNode_t * dependencies, size_t numDependencies, unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuStreamUpdateCaptureDependencies((CUstream)hStream, (CUgraphNode *)dependencies, numDependencies, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamAddCallback_ptsz(UPTKStream_t hStream,UPTKstreamCallback callback,void * userData,unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuStreamAddCallback((CUstream)hStream, (CUstreamCallback)callback, userData, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamAttachMemAsync_ptsz(UPTKStream_t hStream,UPTKdeviceptr dptr,size_t length,unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuStreamAttachMemAsync((CUstream)hStream, (CUdeviceptr)dptr, length, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamQuery_ptsz(UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuStreamQuery((CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamSynchronize_ptsz(UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuStreamSynchronize((CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPEventRecord_ptsz(UPTKEvent_t hEvent,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuEventRecord((CUevent)hEvent, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPEventRecordWithFlags_ptsz(UPTKEvent_t hEvent, UPTKStream_t hStream, unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuEventRecordWithFlags((CUevent)hEvent, (CUstream)hStream, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLaunchKernel_ptsz(UPTKfunction f,unsigned int gridDimX,unsigned int gridDimY,unsigned int gridDimZ,unsigned int blockDimX,unsigned int blockDimY,unsigned int blockDimZ,unsigned int sharedMemBytes,UPTKStream_t hStream,void ** kernelParams,void ** extra)
{
    CUresult cu_res;
    cu_res = cuLaunchKernel((CUfunction)f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, (CUstream)hStream, kernelParams, extra);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLaunchKernelEx_ptsz(const UPTKLaunchConfig_t * config, UPTKfunction f, void ** kernelParams, void ** extra)
{
    CUresult cu_res;
    cu_res = cuLaunchKernelEx((const CUlaunchConfig *)config, (CUfunction)f, kernelParams, extra);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLaunchHostFunc_ptsz(UPTKStream_t hStream,UPTKhostFn fn,void * userData)
{
    CUresult cu_res;
    cu_res = cuLaunchHostFunc((CUstream)hStream, (CUhostFn)fn, userData);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphicsMapResources_ptsz(unsigned int count,UPTKGraphicsResource_t * resources,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuGraphicsMapResources(count, (CUgraphicsResource *)resources, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphicsUnmapResources_ptsz(unsigned int count,UPTKGraphicsResource_t * resources,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuGraphicsUnmapResources(count, (CUgraphicsResource *)resources, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamWriteValue32_v2_ptsz(UPTKStream_t stream, UPTKdeviceptr addr, cuuint32_t value, unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuStreamWriteValue32_v2((CUstream)stream, (CUdeviceptr)addr, value, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamWaitValue32_v2_ptsz(UPTKStream_t stream, UPTKdeviceptr addr, cuuint32_t value, unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuStreamWaitValue32_v2((CUstream)stream, (CUdeviceptr)addr, value, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamWriteValue64_v2_ptsz(UPTKStream_t stream, UPTKdeviceptr addr, cuuint64_t value, unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuStreamWriteValue64_v2((CUstream)stream, (CUdeviceptr)addr, value, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamWaitValue64_v2_ptsz(UPTKStream_t stream, UPTKdeviceptr addr, cuuint64_t value, unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuStreamWaitValue64_v2((CUstream)stream, (CUdeviceptr)addr, value, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamBatchMemOp_v2_ptsz(UPTKStream_t stream, unsigned int count, UPTKstreamBatchMemOpParams * paramArray, unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuStreamBatchMemOp_v2((CUstream)stream, count, (CUstreamBatchMemOpParams *)paramArray, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLaunchCooperativeKernel_ptsz(UPTKfunction f,unsigned int gridDimX,unsigned int gridDimY,unsigned int gridDimZ,unsigned int blockDimX,unsigned int blockDimY,unsigned int blockDimZ,unsigned int sharedMemBytes,UPTKStream_t hStream,void ** kernelParams)
{
    CUresult cu_res;
    cu_res = cuLaunchCooperativeKernel((CUfunction)f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, (CUstream)hStream, kernelParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPSignalExternalSemaphoresAsync_ptsz(const UPTKExternalSemaphore_t * extSemArray,const UPTK_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS * paramsArray,unsigned int numExtSems,UPTKStream_t stream)
{
    CUresult cu_res;
    cu_res = cuSignalExternalSemaphoresAsync((const CUexternalSemaphore *)extSemArray, (const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *)paramsArray, numExtSems, (CUstream)stream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPWaitExternalSemaphoresAsync_ptsz(const UPTKExternalSemaphore_t * extSemArray,const UPTK_EXTERNAL_SEMAPHORE_WAIT_PARAMS * paramsArray,unsigned int numExtSems,UPTKStream_t stream)
{
    CUresult cu_res;
    cu_res = cuWaitExternalSemaphoresAsync((const CUexternalSemaphore *)extSemArray, (const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *)paramsArray, numExtSems, (CUstream)stream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphUpload_ptsz(UPTKGraphExec_t hGraphExec, UPTKStream_t hStream)
{
    if (!hGraphExec) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuGraphUpload((CUgraphExec)hGraphExec, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphLaunch_ptsz(UPTKGraphExec_t hGraphExec,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuGraphLaunch((CUgraphExec)hGraphExec, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamCopyAttributes_ptsz(UPTKStream_t dst, UPTKStream_t src)
{
    CUresult cu_res;
    cu_res = cuStreamCopyAttributes((CUstream)dst, (CUstream)src);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamGetAttribute_ptsz(UPTKStream_t hStream, UPTKStreamAttrID attr, UPTKStreamAttrValue * value_out)
{
    CUresult cu_res;
    cu_res = cuStreamGetAttribute((CUstream)hStream, (CUstreamAttrID)attr, (CUstreamAttrValue *)value_out);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamSetAttribute_ptsz(UPTKStream_t hStream, UPTKStreamAttrID attr, const UPTKStreamAttrValue * value)
{
    CUresult cu_res;
    cu_res = cuStreamSetAttribute((CUstream)hStream, (CUstreamAttrID)attr, (const CUstreamAttrValue *)value);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemMapArrayAsync_ptsz(UPTKarrayMapInfo * mapInfoList, unsigned int count, UPTKStream_t hStream)
{
    if (!mapInfoList || count == 0) return UPTKErrorInvalidValue;
    CUresult cu_res;
    cu_res = cuMemMapArrayAsync((CUarrayMapInfo *)mapInfoList, count, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemFreeAsync_ptsz(UPTKdeviceptr dptr, UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemFreeAsync((CUdeviceptr)dptr, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemAllocAsync_ptsz(UPTKdeviceptr * dptr, size_t bytesize, UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemAllocAsync((CUdeviceptr *)dptr, bytesize, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemAllocFromPoolAsync_ptsz(UPTKdeviceptr * dptr, size_t bytesize, UPTKMemPool_t pool, UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemAllocFromPoolAsync((CUdeviceptr *)dptr, bytesize, (CUmemoryPool)pool, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

/* ----------------------------------------------------------------------- *
 * The wrappers below all use CUDA-driver entry points that were introduced
 * in CUDA 12.x (cuCoredumpGetAttribute, cuCtxCreate_v4, cuGreenCtxCreate,
 * cuLibraryLoadData, cuMulticast*, cuTensorMapEncode*, ...). DTK 25.04.1
 * ships CUDA 11 compat headers and does not provide them, so we compile
 * this whole block out for any toolkit older than CUDA 12. The matching
 * UPTK type aliases are guarded the same way in driver.hpp.
 * ----------------------------------------------------------------------- */
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000

UPTKError UPCoredumpGetAttribute(UPTKcoredumpSettings attrib, void* value, size_t * size)
{
    CUresult cu_res;
    cu_res = cuCoredumpGetAttribute((CUcoredumpSettings)attrib, value, size);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCoredumpGetAttributeGlobal(UPTKcoredumpSettings attrib, void * value, size_t * size)
{
    CUresult cu_res;
    cu_res = cuCoredumpGetAttributeGlobal((CUcoredumpSettings)attrib, value, size);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCoredumpSetAttribute(UPTKcoredumpSettings attrib, void* value, size_t * size)
{
    CUresult cu_res;
    cu_res = cuCoredumpSetAttribute((CUcoredumpSettings)attrib, value, size);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCoredumpSetAttributeGlobal(UPTKcoredumpSettings attrib, void * value, size_t * size)
{
    CUresult cu_res;
    cu_res = cuCoredumpSetAttributeGlobal((CUcoredumpSettings)attrib, value, size);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxCreate_v4(UPTKcontext * pctx, UPTKctxCreateParams * ctxCreateParams, unsigned int flags, UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuCtxCreate_v4((CUcontext *)pctx, (CUctxCreateParams *)ctxCreateParams, flags, (CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxGetId(UPTKcontext ctx, unsigned long long * ctxId)
{
    CUresult cu_res;
    cu_res = cuCtxGetId((CUcontext)ctx, ctxId);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxRecordEvent(UPTKcontext hCtx, UPTKEvent_t hEvent)
{
    CUresult cu_res;
    cu_res = cuCtxRecordEvent((CUcontext)hCtx, (CUevent)hEvent);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxSetFlags(unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuCtxSetFlags(flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxWaitEvent(UPTKcontext hCtx, UPTKEvent_t hEvent)
{
    CUresult cu_res;
    cu_res = cuCtxWaitEvent((CUcontext)hCtx, (CUevent)hEvent);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceRegisterAsyncNotification(UPTKdevice device, UPTKAsyncCallback callbackFunc, void * userData, UPTKasyncCallbackHandle * callback)
{
    CUresult cu_res;
    cu_res = cuDeviceRegisterAsyncNotification((CUdevice)device, (CUasyncCallback)callbackFunc, userData, (CUasyncCallbackHandle *)callback);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceUnregisterAsyncNotification(UPTKdevice device, UPTKasyncCallbackHandle callback)
{
    CUresult cu_res;
    cu_res = cuDeviceUnregisterAsyncNotification((CUdevice)device, (CUasyncCallbackHandle)callback);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPFuncGetName(const char ** name, UPTKfunction hfunc)
{
    CUresult cu_res;
    cu_res = cuFuncGetName(name, (CUfunction)hfunc);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPFuncGetParamInfo(UPTKfunction func, size_t paramIndex, size_t * paramOffset, size_t * paramSize)
{
    CUresult cu_res;
    cu_res = cuFuncGetParamInfo((CUfunction)func, paramIndex, paramOffset, paramSize);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPFuncIsLoaded(UPTKfunctionLoadingState * state, UPTKfunction function)
{
    CUresult cu_res;
    cu_res = cuFuncIsLoaded((CUfunctionLoadingState *)state, (CUfunction)function);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPFuncLoad(UPTKfunction function)
{
    CUresult cu_res;
    cu_res = cuFuncLoad((CUfunction)function);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGetProcAddress_v2(const char * symbol, void ** pfn, int cudaVersion, cuuint64_t flags, UPTKdriverProcAddressQueryResult * symbolStatus)
{
    CUresult cu_res;
    cu_res = cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, (CUdriverProcAddressQueryResult *)symbolStatus);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphAddDependencies_v2(UPTKGraph_t hGraph, const UPTKGraphNode_t * from, const UPTKGraphNode_t * to, const UPTKGraphEdgeData * edgeData, size_t numDependencies)
{
    CUresult cu_res;
    cu_res = cuGraphAddDependencies_v2((CUgraph)hGraph, (const CUgraphNode *)from, (const CUgraphNode *)to, (const CUgraphEdgeData *)edgeData, numDependencies);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphAddKernelNode_v2(UPTKGraphNode_t * phGraphNode, UPTKGraph_t hGraph, const UPTKGraphNode_t * dependencies, size_t numDependencies, const UPTK_KERNEL_NODE_PARAMS * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphAddKernelNode_v2((CUgraphNode *)phGraphNode, (CUgraph)hGraph, (const CUgraphNode *)dependencies, numDependencies, (const CUDA_KERNEL_NODE_PARAMS *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphAddNode(UPTKGraphNode_t * phGraphNode, UPTKGraph_t hGraph, const UPTKGraphNode_t * dependencies, size_t numDependencies, UPTKGraphNodeParams * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphAddNode((CUgraphNode *)phGraphNode, (CUgraph)hGraph, (const CUgraphNode *)dependencies, numDependencies, (CUgraphNodeParams *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphAddNode_v2(UPTKGraphNode_t * phGraphNode, UPTKGraph_t hGraph, const UPTKGraphNode_t * dependencies, const UPTKGraphEdgeData * dependencyData, size_t numDependencies, UPTKGraphNodeParams * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphAddNode_v2((CUgraphNode *)phGraphNode, (CUgraph)hGraph, (const CUgraphNode *)dependencies, (const CUgraphEdgeData *)dependencyData, numDependencies, (CUgraphNodeParams *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphConditionalHandleCreate(UPTKGraphConditionalHandle * pHandle_out, UPTKGraph_t hGraph, UPTKcontext ctx, unsigned int defaultLaunchValue, unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuGraphConditionalHandleCreate((CUgraphConditionalHandle *)pHandle_out, (CUgraph)hGraph, (CUcontext)ctx, defaultLaunchValue, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphExecGetFlags(UPTKGraphExec_t hGraphExec, cuuint64_t * flags)
{
    CUresult cu_res;
    cu_res = cuGraphExecGetFlags((CUgraphExec)hGraphExec, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphExecKernelNodeSetParams_v2(UPTKGraphExec_t hGraphExec, UPTKGraphNode_t hNode, const UPTK_KERNEL_NODE_PARAMS * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphExecKernelNodeSetParams_v2((CUgraphExec)hGraphExec, (CUgraphNode)hNode, (const CUDA_KERNEL_NODE_PARAMS *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphExecNodeSetParams(UPTKGraphExec_t hGraphExec, UPTKGraphNode_t hNode, UPTKGraphNodeParams * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphExecNodeSetParams((CUgraphExec)hGraphExec, (CUgraphNode)hNode, (CUgraphNodeParams *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphExecUpdate_v2(UPTKGraphExec_t hGraphExec, UPTKGraph_t hGraph, UPTKGraphExecUpdateResultInfo * resultInfo)
{
    CUresult cu_res;
    cu_res = cuGraphExecUpdate_v2((CUgraphExec)hGraphExec, (CUgraph)hGraph, (CUgraphExecUpdateResultInfo *)resultInfo);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphGetEdges_v2(UPTKGraph_t hGraph, UPTKGraphNode_t * from, UPTKGraphNode_t * to, UPTKGraphEdgeData * edgeData, size_t * numEdges)
{
    CUresult cu_res;
    cu_res = cuGraphGetEdges_v2((CUgraph)hGraph, (CUgraphNode *)from, (CUgraphNode *)to, (CUgraphEdgeData *)edgeData, numEdges);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphInstantiateWithParams(UPTKGraphExec_t * phGraphExec, UPTKGraph_t hGraph, UPTK_GRAPH_INSTANTIATE_PARAMS * instantiateParams)
{
    CUresult cu_res;
    cu_res = cuGraphInstantiateWithParams((CUgraphExec *)phGraphExec, (CUgraph)hGraph, (CUDA_GRAPH_INSTANTIATE_PARAMS *)instantiateParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphInstantiateWithParams_ptsz(UPTKGraphExec_t * phGraphExec, UPTKGraph_t hGraph, UPTK_GRAPH_INSTANTIATE_PARAMS * instantiateParams)
{
    CUresult cu_res;
    cu_res = cuGraphInstantiateWithParams((CUgraphExec *)phGraphExec, (CUgraph)hGraph, (CUDA_GRAPH_INSTANTIATE_PARAMS *)instantiateParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphKernelNodeGetParams_v2(UPTKGraphNode_t hNode,UPTK_KERNEL_NODE_PARAMS * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphKernelNodeGetParams_v2((CUgraphNode)hNode, (CUDA_KERNEL_NODE_PARAMS *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphKernelNodeSetParams_v2(UPTKGraphNode_t hNode,const UPTK_KERNEL_NODE_PARAMS * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphKernelNodeSetParams_v2((CUgraphNode)hNode, (const CUDA_KERNEL_NODE_PARAMS *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphNodeGetDependencies_v2(UPTKGraphNode_t hNode, UPTKGraphNode_t * dependencies, UPTKGraphEdgeData * edgeData, size_t * numDependencies)
{
    CUresult cu_res;
    cu_res = cuGraphNodeGetDependencies_v2((CUgraphNode)hNode, (CUgraphNode *)dependencies, (CUgraphEdgeData *)edgeData, numDependencies);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphNodeGetDependentNodes_v2(UPTKGraphNode_t hNode, UPTKGraphNode_t * dependentNodes, UPTKGraphEdgeData * edgeData, size_t * numDependentNodes)
{
    CUresult cu_res;
    cu_res = cuGraphNodeGetDependentNodes_v2((CUgraphNode)hNode, (CUgraphNode *)dependentNodes, (CUgraphEdgeData *)edgeData, numDependentNodes);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphNodeSetParams(UPTKGraphNode_t hNode, UPTKGraphNodeParams * nodeParams)
{
    CUresult cu_res;
    cu_res = cuGraphNodeSetParams((CUgraphNode)hNode, (CUgraphNodeParams *)nodeParams);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGraphRemoveDependencies_v2(UPTKGraph_t hGraph, const UPTKGraphNode_t * from, const UPTKGraphNode_t * to, const UPTKGraphEdgeData * edgeData, size_t numDependencies)
{
    CUresult cu_res;
    cu_res = cuGraphRemoveDependencies_v2((CUgraph)hGraph, (const CUgraphNode *)from, (const CUgraphNode *)to, (const CUgraphEdgeData *)edgeData, numDependencies);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGreenCtxCreate(UPTKgreenCtx* phCtx, UPTKdevResourceDesc desc, UPTKdevice dev, unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuGreenCtxCreate((CUgreenCtx*)phCtx, (CUdevResourceDesc)desc, (CUdevice)dev, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGreenCtxDestroy(UPTKgreenCtx hCtx)
{
    CUresult cu_res;
    cu_res = cuGreenCtxDestroy((CUgreenCtx)hCtx);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGreenCtxGetDevResource(UPTKgreenCtx hCtx, UPTKdevResource* resource, UPTKdevResourceType type)
{
    CUresult cu_res;
    cu_res = cuGreenCtxGetDevResource((CUgreenCtx)hCtx, (CUdevResource*)resource, (CUdevResourceType)type);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGreenCtxRecordEvent(UPTKgreenCtx hCtx, UPTKEvent_t hEvent)
{
    CUresult cu_res;
    cu_res = cuGreenCtxRecordEvent((CUgreenCtx)hCtx, (CUevent)hEvent);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGreenCtxStreamCreate(UPTKStream_t* phStream, UPTKgreenCtx greenCtx, unsigned int flags, int priority)
{
    CUresult cu_res;
    cu_res = cuGreenCtxStreamCreate((CUstream*)phStream, (CUgreenCtx)greenCtx, flags, priority);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPGreenCtxWaitEvent(UPTKgreenCtx hCtx, UPTKEvent_t hEvent)
{
    CUresult cu_res;
    cu_res = cuGreenCtxWaitEvent((CUgreenCtx)hCtx, (CUevent)hEvent);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDevResourceGenerateDesc(UPTKdevResourceDesc * phDesc, UPTKdevResource * resources, unsigned int nbResources)
{
    CUresult cu_res;
    cu_res = cuDevResourceGenerateDesc((CUdevResourceDesc *)phDesc, (CUdevResource *)resources, nbResources);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDevSmResourceSplitByCount(UPTKdevResource* result, unsigned int* nbGroups, const UPTKdevResource* input, UPTKdevResource* remaining, unsigned int useFlags, unsigned int minCount)
{
    CUresult cu_res;
    cu_res = cuDevSmResourceSplitByCount((CUdevResource*)result, nbGroups, (const CUdevResource*)input, (CUdevResource*)remaining, useFlags, minCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceGetDevResource(UPTKdevice device, UPTKdevResource* resource, UPTKdevResourceType type)
{
    CUresult cu_res;
    cu_res = cuDeviceGetDevResource((CUdevice)device, (CUdevResource*)resource, (CUdevResourceType)type);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxFromGreenCtx(UPTKcontext * pContext, UPTKgreenCtx hCtx)
{
    CUresult cu_res;
    cu_res = cuCtxFromGreenCtx((CUcontext *)pContext, (CUgreenCtx)hCtx);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamGetGreenCtx(UPTKStream_t hStream, UPTKgreenCtx * phCtx)
{
    CUresult cu_res;
    cu_res = cuStreamGetGreenCtx((CUstream)hStream, (CUgreenCtx *)phCtx);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPCtxGetDevResource(UPTKcontext hCtx, UPTKdevResource* resource, UPTKdevResourceType type)
{
    CUresult cu_res;
    cu_res = cuCtxGetDevResource((CUcontext)hCtx, (CUdevResource*)resource, (CUdevResourceType)type);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPKernelGetAttribute(int * pi, UPTKfunction_attribute attrib, UPTKKernel_t kernel, UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuKernelGetAttribute(pi, (CUfunction_attribute)attrib, (CUkernel)kernel, (CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPKernelGetFunction(UPTKfunction * pFunc, UPTKKernel_t kernel)
{
    CUresult cu_res;
    cu_res = cuKernelGetFunction((CUfunction *)pFunc, (CUkernel)kernel);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPKernelGetLibrary(UPTKlibrary * pLib, UPTKKernel_t kernel)
{
    CUresult cu_res;
    cu_res = cuKernelGetLibrary((CUlibrary *)pLib, (CUkernel)kernel);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPKernelGetName(const char ** name, UPTKKernel_t hfunc)
{
    CUresult cu_res;
    cu_res = cuKernelGetName(name, (CUkernel)hfunc);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPKernelGetParamInfo(UPTKKernel_t kernel, size_t paramIndex, size_t * paramOffset, size_t * paramSize)
{
    CUresult cu_res;
    cu_res = cuKernelGetParamInfo((CUkernel)kernel, paramIndex, paramOffset, paramSize);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPKernelSetAttribute(UPTKfunction_attribute attrib, int val, UPTKKernel_t kernel, UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuKernelSetAttribute((CUfunction_attribute)attrib, val, (CUkernel)kernel, (CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPKernelSetCacheConfig(UPTKKernel_t kernel, UPTKfunc_cache config, UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuKernelSetCacheConfig((CUkernel)kernel, (CUfunc_cache)config, (CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLibraryEnumerateKernels(UPTKKernel_t * kernels, unsigned int numKernels, UPTKlibrary lib)
{
    CUresult cu_res;
    cu_res = cuLibraryEnumerateKernels((CUkernel *)kernels, numKernels, (CUlibrary)lib);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLibraryGetGlobal(UPTKdeviceptr * dptr, size_t * bytes, UPTKlibrary library, const char * name)
{
    CUresult cu_res;
    cu_res = cuLibraryGetGlobal((CUdeviceptr *)dptr, bytes, (CUlibrary)library, name);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLibraryGetKernel(UPTKKernel_t * pKernel, UPTKlibrary library, const char * name)
{
    CUresult cu_res;
    cu_res = cuLibraryGetKernel((CUkernel *)pKernel, (CUlibrary)library, name);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLibraryGetKernelCount(unsigned int * count, UPTKlibrary lib)
{
    CUresult cu_res;
    cu_res = cuLibraryGetKernelCount(count, (CUlibrary)lib);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLibraryGetManaged(UPTKdeviceptr * dptr, size_t * bytes, UPTKlibrary library, const char * name)
{
    CUresult cu_res;
    cu_res = cuLibraryGetManaged((CUdeviceptr *)dptr, bytes, (CUlibrary)library, name);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLibraryGetModule(UPTKmodule * pMod, UPTKlibrary library)
{
    CUresult cu_res;
    cu_res = cuLibraryGetModule((CUmodule *)pMod, (CUlibrary)library);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLibraryGetUnifiedFunction(void ** fptr, UPTKlibrary library, const char * symbol)
{
    CUresult cu_res;
    cu_res = cuLibraryGetUnifiedFunction(fptr, (CUlibrary)library, symbol);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLibraryLoadData(UPTKlibrary * library, const void * code, UPTKjit_option * jitOptions, void ** jitOptionsValues, unsigned int numJitOptions, UPTKlibraryOption * libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions)
{
    CUresult cu_res;
    cu_res = cuLibraryLoadData((CUlibrary *)library, code, (CUjit_option *)jitOptions, jitOptionsValues, numJitOptions, (CUlibraryOption *)libraryOptions, libraryOptionValues, numLibraryOptions);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLibraryLoadFromFile(UPTKlibrary * library, const char * fileName, UPTKjit_option * jitOptions, void ** jitOptionsValues, unsigned int numJitOptions, UPTKlibraryOption * libraryOptions, void ** libraryOptionValues, unsigned int numLibraryOptions)
{
    CUresult cu_res;
    cu_res = cuLibraryLoadFromFile((CUlibrary *)library, fileName, (CUjit_option *)jitOptions, jitOptionsValues, numJitOptions, (CUlibraryOption *)libraryOptions, libraryOptionValues, numLibraryOptions);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPLibraryUnload(UPTKlibrary library)
{
    CUresult cu_res;
    cu_res = cuLibraryUnload((CUlibrary)library);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemAdvise_v2(UPTKdeviceptr devPtr, size_t count, UPTKmem_advise advice, UPTKMemLocation location)
{
    CUresult cu_res;
    cu_res = cuMemAdvise_v2((CUdeviceptr)devPtr, count, (CUmem_advise)advice, (CUmemLocation)location);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemPrefetchAsync_v2(UPTKdeviceptr devPtr, size_t count, UPTKMemLocation location, unsigned int flags, UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemPrefetchAsync_v2((CUdeviceptr)devPtr, count, (CUmemLocation)location, flags, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemPrefetchAsync_v2_ptsz(UPTKdeviceptr devPtr, size_t count, UPTKMemLocation location, unsigned int flags, UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemPrefetchAsync_v2((CUdeviceptr)devPtr, count, (CUmemLocation)location, flags, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPModuleEnumerateFunctions(UPTKfunction * functions, unsigned int numFunctions, UPTKmodule mod)
{
    CUresult cu_res;
    cu_res = cuModuleEnumerateFunctions((CUfunction *)functions, numFunctions, (CUmodule)mod);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPModuleGetFunctionCount(unsigned int * count, UPTKmodule mod)
{
    CUresult cu_res;
    cu_res = cuModuleGetFunctionCount(count, (CUmodule)mod);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMulticastAddDevice(UPTKmemGenericAllocationHandle mcHandle, UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuMulticastAddDevice((CUmemGenericAllocationHandle)mcHandle, (CUdevice)dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMulticastBindAddr(UPTKmemGenericAllocationHandle mcHandle, size_t mcOffset, UPTKdeviceptr memptr, size_t size, unsigned long long flags)
{
    CUresult cu_res;
    cu_res = cuMulticastBindAddr((CUmemGenericAllocationHandle)mcHandle, mcOffset, (CUdeviceptr)memptr, size, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMulticastBindMem(UPTKmemGenericAllocationHandle mcHandle, size_t mcOffset, UPTKmemGenericAllocationHandle memHandle, size_t memOffset, size_t size, unsigned long long flags)
{
    CUresult cu_res;
    cu_res = cuMulticastBindMem((CUmemGenericAllocationHandle)mcHandle, mcOffset, (CUmemGenericAllocationHandle)memHandle, memOffset, size, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMulticastCreate(UPTKmemGenericAllocationHandle * mcHandle, const UPTKmulticastObjectProp * prop)
{
    CUresult cu_res;
    cu_res = cuMulticastCreate((CUmemGenericAllocationHandle *)mcHandle, (const CUmulticastObjectProp *)prop);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMulticastGetGranularity(size_t * granularity, const UPTKmulticastObjectProp * prop, UPTKmulticastGranularity_flags option)
{
    CUresult cu_res;
    cu_res = cuMulticastGetGranularity(granularity, (const CUmulticastObjectProp *)prop, (CUmulticastGranularity_flags)option);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMulticastUnbind(UPTKmemGenericAllocationHandle mcHandle, UPTKdevice dev, size_t mcOffset, size_t size)
{
    CUresult cu_res;
    cu_res = cuMulticastUnbind((CUmemGenericAllocationHandle)mcHandle, (CUdevice)dev, mcOffset, size);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamBeginCaptureToGraph(UPTKStream_t hStream, UPTKGraph_t hGraph, const UPTKGraphNode_t * dependencies, const UPTKGraphEdgeData * dependencyData, size_t numDependencies, UPTKStreamCaptureMode mode)
{
    CUresult cu_res;
    cu_res = cuStreamBeginCaptureToGraph((CUstream)hStream, (CUgraph)hGraph, (const CUgraphNode *)dependencies, (const CUgraphEdgeData *)dependencyData, numDependencies, (CUstreamCaptureMode)mode);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamBeginCaptureToGraph_ptsz(UPTKStream_t hStream, UPTKGraph_t hGraph, const UPTKGraphNode_t * dependencies, const UPTKGraphEdgeData * dependencyData, size_t numDependencies, UPTKStreamCaptureMode mode)
{
    CUresult cu_res;
    cu_res = cuStreamBeginCaptureToGraph((CUstream)hStream, (CUgraph)hGraph, (const CUgraphNode *)dependencies, (const CUgraphEdgeData *)dependencyData, numDependencies, (CUstreamCaptureMode)mode);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamGetCaptureInfo_v3(UPTKStream_t hStream, UPTKStreamCaptureStatus * captureStatus_out, cuuint64_t * id_out, UPTKGraph_t * graph_out, const UPTKGraphNode_t ** dependencies_out, const UPTKGraphEdgeData ** edgeData_out, size_t * numDependencies_out)
{
    CUresult cu_res;
    cu_res = cuStreamGetCaptureInfo_v3((CUstream)hStream, (CUstreamCaptureStatus *)captureStatus_out, id_out, (CUgraph *)graph_out, (const CUgraphNode **)dependencies_out, (const CUgraphEdgeData **)edgeData_out, numDependencies_out);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamGetCaptureInfo_v3_ptsz(UPTKStream_t hStream, UPTKStreamCaptureStatus * captureStatus_out, cuuint64_t * id_out, UPTKGraph_t * graph_out, const UPTKGraphNode_t ** dependencies_out, const UPTKGraphEdgeData ** edgeData_out, size_t * numDependencies_out)
{
    CUresult cu_res;
    cu_res = cuStreamGetCaptureInfo_v3((CUstream)hStream, (CUstreamCaptureStatus *)captureStatus_out, id_out, (CUgraph *)graph_out, (const CUgraphNode **)dependencies_out, (const CUgraphEdgeData **)edgeData_out, numDependencies_out);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamGetCtx_v2(UPTKStream_t hStream, UPTKcontext * pCtx, UPTKgreenCtx * pGreenCtx)
{
    CUresult cu_res;
    cu_res = cuStreamGetCtx_v2((CUstream)hStream, (CUcontext *)pCtx, (CUgreenCtx *)pGreenCtx);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamGetCtx_v2_ptsz(UPTKStream_t hStream, UPTKcontext * pCtx, UPTKgreenCtx * pGreenCtx)
{
    CUresult cu_res;
    cu_res = cuStreamGetCtx_v2((CUstream)hStream, (CUcontext *)pCtx, (CUgreenCtx *)pGreenCtx);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamGetId(UPTKStream_t hStream, unsigned long long * streamId)
{
    CUresult cu_res;
    cu_res = cuStreamGetId((CUstream)hStream, streamId);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamGetId_ptsz(UPTKStream_t hStream, unsigned long long * streamId)
{
    CUresult cu_res;
    cu_res = cuStreamGetId((CUstream)hStream, streamId);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamUpdateCaptureDependencies_v2(UPTKStream_t hStream, UPTKGraphNode_t * dependencies, const UPTKGraphEdgeData * dependencyData, size_t numDependencies, unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuStreamUpdateCaptureDependencies_v2((CUstream)hStream, (CUgraphNode *)dependencies, (const CUgraphEdgeData *)dependencyData, numDependencies, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPStreamUpdateCaptureDependencies_v2_ptsz(UPTKStream_t hStream, UPTKGraphNode_t * dependencies, const UPTKGraphEdgeData * dependencyData, size_t numDependencies, unsigned int flags)
{
    CUresult cu_res;
    cu_res = cuStreamUpdateCaptureDependencies_v2((CUstream)hStream, (CUgraphNode *)dependencies, (const CUgraphEdgeData *)dependencyData, numDependencies, flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTensorMapEncodeIm2col(UPTKtensorMap * tensorMap, UPTKtensorMapDataType tensorDataType, cuuint32_t tensorRank, void * globalAddress, const cuuint64_t * globalDim, const cuuint64_t * globalStrides, const int * pixelBoxLowerCorner, const int * pixelBoxUpperCorner, cuuint32_t channelsPerPixel, cuuint32_t pixelsPerColumn, const cuuint32_t * elementStrides, UPTKtensorMapInterleave interleave, UPTKtensorMapSwizzle swizzle, UPTKtensorMapL2promotion l2Promotion, UPTKtensorMapFloatOOBfill oobFill)
{
    CUresult cu_res;
    cu_res = cuTensorMapEncodeIm2col((CUtensorMap *)tensorMap, (CUtensorMapDataType)tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, pixelBoxLowerCorner, pixelBoxUpperCorner, channelsPerPixel, pixelsPerColumn, elementStrides, (CUtensorMapInterleave)interleave, (CUtensorMapSwizzle)swizzle, (CUtensorMapL2promotion)l2Promotion, (CUtensorMapFloatOOBfill)oobFill);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTensorMapEncodeTiled(UPTKtensorMap * tensorMap, UPTKtensorMapDataType tensorDataType, cuuint32_t tensorRank, void * globalAddress, const cuuint64_t * globalDim, const cuuint64_t * globalStrides, const cuuint32_t * boxDim, const cuuint32_t * elementStrides, UPTKtensorMapInterleave interleave, UPTKtensorMapSwizzle swizzle, UPTKtensorMapL2promotion l2Promotion, UPTKtensorMapFloatOOBfill oobFill)
{
    CUresult cu_res;
    cu_res = cuTensorMapEncodeTiled((CUtensorMap *)tensorMap, (CUtensorMapDataType)tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, boxDim, elementStrides, (CUtensorMapInterleave)interleave, (CUtensorMapSwizzle)swizzle, (CUtensorMapL2promotion)l2Promotion, (CUtensorMapFloatOOBfill)oobFill);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPTensorMapReplaceAddress(UPTKtensorMap * tensorMap, void * globalAddress)
{
    CUresult cu_res;
    cu_res = cuTensorMapReplaceAddress((CUtensorMap *)tensorMap, globalAddress);
    return CUresultToUPTKError(cu_res);
}

/*UPTKError UPModuleGetGlobal(UPTKdeviceptr * dptr, size_t * bytes,UPTKmodule hmod,const char * name)
{
    CUresult cu_res;
    cu_res = cuModuleGetGlobal_v2((CUdeviceptr *)dptr, bytes, (CUmodule)hmod, name);
    return CUresultToUPTKError(cu_res);
}*/

#endif /* CUDA_VERSION >= 12000 */

#if defined(__cplusplus)
}
#endif /* __cplusplus */

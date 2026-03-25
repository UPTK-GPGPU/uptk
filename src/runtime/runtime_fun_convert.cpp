#include "runtime.hpp"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

UPTKError __UPTKPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, UPTKStream_t *stream) {
    cudaError_t res;
    res = __cudaPopCallConfiguration(gridDim, blockDim, sharedMem, (cudaStream_t *) stream);
    return cudaErrorToUPTKError(res);
}

void __UPTKPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, UPTKStream_t stream)
{
    __cudaPushCallConfiguration(gridDim, blockDim, sharedMem, (cudaStream_t)stream);
}

void *__UPTKRegisterFatBinary(const void *data) {
    return __cudaRegisterFatBinary(data);
}

void __UPTKRegisterFunction(void * modules, const void *hostFunction, char *deviceFunction, const char *deviceFunctionName, unsigned int threadLimit, uint3 *tid, uint3 *bid, dim3 *blockDim, dim3 *gridDim, int *wSize) {
    __cudaRegisterFunction(modules, hostFunction, deviceFunction, deviceFunctionName, threadLimit, tid, bid, blockDim, gridDim, wSize);
}

void __UPTKRegisterManagedVar(void *modules, void *ManagedPtr, void *variblePtr, const char *hostVarName, size_t varSize, int varAlignment) {
    __cudaRegisterManagedVar(modules, ManagedPtr, variblePtr, hostVarName, varSize, varAlignment);
}

void __UPTKRegisterVar(void *modules, void *hostVar, char *deviceVar, const char *deviceVarName, int isExternal, std::size_t size, int isConstant, int isGlobal) {
    __cudaRegisterVar(modules, hostVar, deviceVar, deviceVarName, isExternal, size, isConstant, isGlobal);
}

void __UPTKUnregisterFatBinary(void *modules) {
    __cudaUnregisterFatBinary(modules);
}

void __UPTKRegisterTexture(void *modules, void *hostVar, char *hostVarName, char *deviceVarName, int texType, int normalized, int isExternal) {
    __cudaRegisterTexture(modules, hostVar, hostVarName, deviceVarName, texType, normalized, isExternal);
}

void __UPTKRegisterSurface(void *modules, void *hostVar, char *hostVarName, char *deviceVarName, int surfType, int isExternal) {
    __cudaRegisterSurface(modules, hostVar, hostVarName, deviceVarName, surfType, isExternal);
}

__host__ UPTKError UPTKBindSurfaceToArray(const struct surfaceReference * surfref,UPTKArray_const_t array,const struct UPTKChannelFormatDesc * desc)
{
    cudaError_t cuda_res;
    cuda_res = cudaBindSurfaceToArray((const surfaceReference *) surfref, (cudaArray_const_t) array, (cudaChannelFormatDesc *)desc);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKBindTexture(size_t * offset,const struct textureReference * texref,const void * devPtr,const struct UPTKChannelFormatDesc * desc,size_t size)
{
    cudaError_t cuda_res;
    cuda_res = cudaBindTexture(offset, texref, devPtr, (cudaChannelFormatDesc *)desc, size);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKBindTexture2D(size_t * offset,const struct textureReference * texref,const void * devPtr,const struct UPTKChannelFormatDesc * desc,size_t width,size_t height,size_t pitch)
{
    cudaError_t cuda_res;
    cuda_res = cudaBindTexture2D(offset, texref, devPtr, (cudaChannelFormatDesc *)desc, width, height, pitch);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKBindTextureToArray(const struct textureReference * texref,UPTKArray_const_t array,const struct UPTKChannelFormatDesc * desc)
{
    cudaError_t cuda_res;;
    cuda_res = cudaBindTextureToArray(texref, (cudaArray_const_t) array, (cudaChannelFormatDesc *)desc);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKBindTextureToMipmappedArray(const struct textureReference * texref,UPTKMipmappedArray_const_t mipmappedArray,const struct UPTKChannelFormatDesc * desc)
{
    cudaError_t cuda_res;
    cuda_res = cudaBindTextureToMipmappedArray(texref, (cudaMipmappedArray_const_t) mipmappedArray, (cudaChannelFormatDesc *)desc);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKCreateTextureObject_v2(UPTKTextureObject_t *pTexObject, const struct UPTKResourceDesc *pResDesc, const struct UPTKTextureDesc_v2 *pTexDesc, const struct UPTKResourceViewDesc *pResViewDesc)
{
    Debug();
    return UPTKErrorNotSupported;
}

__host__ UPTKError UPTKGetDeviceProperties(struct UPTKDeviceProp * prop,int device)
{
    if(nullptr == prop)
    {
        return UPTKErrorInvalidValue;
    }

    cudaError_t cuda_res;
    cuda_res = cudaGetDeviceProperties((cudaDeviceProp * )prop, device);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGetSurfaceReference(const struct surfaceReference ** surfref,const void * symbol)
{
    cudaError_t cuda_res;
    cuda_res = cudaGetSurfaceReference(surfref, symbol);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGetTextureAlignmentOffset(size_t * offset,const struct textureReference * texref)
{
    cudaError_t cuda_res;
    cuda_res = cudaGetTextureAlignmentOffset(offset, texref);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGetTextureObjectTextureDesc_v2(struct UPTKTextureDesc_v2 *pTexDesc, UPTKTextureObject_t texObject)
{
    cudaError_t cuda_res;
    cuda_res = cudaGetTextureObjectTextureDesc_v2((cudaTextureDesc_v2 *)pTexDesc, (cudaTextureObject_t) texObject);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGetTextureReference(const struct textureReference ** texref,const void * symbol)
{
    cudaError_t cuda_res;
    cuda_res = cudaGetTextureReference(texref, symbol);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKProfilerInitialize(const char * configFile,const char * outputFile,UPTKOutputMode_t outputMode)
{
    Debug();
    return UPTKErrorNotSupported;
}

__host__ UPTKError UPTKSignalExternalSemaphoresAsync_v2_ptsz(const UPTKExternalSemaphore_t *extSemArray, const struct UPTKExternalSemaphoreSignalParams *paramsArray, unsigned int numExtSems, UPTKStream_t stream)
{
    cudaError_t cuda_res;
    cuda_res = cudaSignalExternalSemaphoresAsync((cudaExternalSemaphore_t *)extSemArray, (cudaExternalSemaphoreSignalParams *)paramsArray, numExtSems, (cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKStreamGetCaptureInfo(UPTKStream_t stream,enum UPTKStreamCaptureStatus * pCaptureStatus,unsigned long long * pId)
{
    if (nullptr == pCaptureStatus)
        return UPTKErrorInvalidValue;
    cudaError_t cuda_res;
    cuda_res = cudaStreamGetCaptureInfo((cudaStream_t) stream, (cudaStreamCaptureStatus * )pCaptureStatus, pId);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKStreamGetCaptureInfo_ptsz(UPTKStream_t stream,enum UPTKStreamCaptureStatus * pCaptureStatus,unsigned long long * pId)
{
    if (nullptr == pCaptureStatus)
        return UPTKErrorInvalidValue;
    cudaError_t cuda_res;
    cuda_res = cudaStreamGetCaptureInfo((cudaStream_t) stream, (cudaStreamCaptureStatus *)pCaptureStatus, pId);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKUnbindTexture(const struct textureReference * texref)
{
    cudaError_t cuda_res;
    cuda_res = cudaUnbindTexture(texref);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKWaitExternalSemaphoresAsync_v2_ptsz(const UPTKExternalSemaphore_t *extSemArray, const struct UPTKExternalSemaphoreWaitParams *paramsArray, unsigned int numExtSems, UPTKStream_t stream)
{
    cudaError_t cuda_res;
    if (paramsArray == nullptr){
        return UPTKErrorInvalidValue;
    }
    cuda_res = cudaWaitExternalSemaphoresAsync_v2((const cudaExternalSemaphore_t *) extSemArray, (cudaExternalSemaphoreWaitParams *)paramsArray, numExtSems, (cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, UPTKStream_t stream)
{
    cudaError_t res;
    res = cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, (cudaStream_t)stream);
    return cudaErrorToUPTKError(res);
}

__host__ UPTKError UPTKLaunchKernel_ptsz(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, UPTKStream_t stream)
{
    cudaError_t res;
    res = cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, (cudaStream_t)stream);
    return cudaErrorToUPTKError(res);
}

__host__ UPTKError UPTKArrayGetInfo(struct UPTKChannelFormatDesc * desc,struct UPTKExtent * extent,unsigned int * flags,UPTKArray_t array)
{
    if (array == nullptr) {
        return UPTKErrorInvalidResourceHandle;
    }
    if ((desc == nullptr) || (extent == nullptr) || (flags == nullptr)) {
        return UPTKSuccess;
    }
    cudaError_t cuda_res;
    cuda_res = cudaArrayGetInfo((cudaChannelFormatDesc *)desc, (cudaExtent *)extent, flags, (cudaArray_t)array);
    return cudaErrorToUPTKError(cuda_res);
}

// UNSUPPORTED
__host__ unsigned long long UPTKCGGetIntrinsicHandle(enum UPTKCGScope scope)
{
    Debug();
    return UPTKErrorNotSupported;
}

__host__ UPTKError UPTKChooseDevice(int * device,const struct UPTKDeviceProp * prop)
{
    if (nullptr == device || nullptr == prop)
        return UPTKErrorInvalidValue;
    cudaError_t cuda_res;
    cuda_res = cudaChooseDevice(device, (cudaDeviceProp *)prop);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ struct UPTKChannelFormatDesc UPTKCreateChannelDesc(int x,int y,int z,int w,enum UPTKChannelFormatKind f)
{
    enum cudaChannelFormatKind cuda_f = UPTKChannelFormatKindTocudaChannelFormatKind(f);
    struct cudaChannelFormatDesc cuda_res;
    memset(&cuda_res, 0, sizeof(cudaChannelFormatDesc));
    cuda_res = cudaCreateChannelDesc(x, y, z, w, cuda_f);

    struct UPTKChannelFormatDesc res;
    memset(&res, 0, sizeof(UPTKChannelFormatDesc));
    cudaChannelFormatDescToUPTKChannelFormatDesc(&cuda_res, &res);
    return res;
}

__host__ UPTKError UPTKCreateSurfaceObject(UPTKSurfaceObject_t * pSurfObject,const struct UPTKResourceDesc * pResDesc)
{
    cudaResourceDesc cuda_pResDesc;
    UPTKResourceDescTocudaResourceDesc(pResDesc, &cuda_pResDesc);
    cudaError_t cuda_res;
    cuda_res = cudaCreateSurfaceObject((cudaSurfaceObject_t *)pSurfObject, &cuda_pResDesc);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKCreateTextureObject(UPTKTextureObject_t * pTexObject,const struct UPTKResourceDesc * pResDesc,const struct UPTKTextureDesc * pTexDesc,const struct UPTKResourceViewDesc * pResViewDesc)
{
    // Validate input params
    if (pTexObject == nullptr || pResDesc == nullptr || pTexDesc == nullptr) {
        return UPTKErrorInvalidValue;
    }
    cudaResourceDesc cuda_pResDesc;
    UPTKResourceDescTocudaResourceDesc(pResDesc, &cuda_pResDesc);
    cudaTextureDesc cuda_pTexDesc;
    UPTKTextureDescTocudaTextureDesc(pTexDesc, &cuda_pTexDesc);
    cudaError_t cuda_res;
    if (pResViewDesc) {
        struct cudaResourceViewDesc cuda_pResViewDesc;
        UPTKResourceViewDescTocudaResourceViewDesc(pResViewDesc, &cuda_pResViewDesc);
        cuda_res = cudaCreateTextureObject((cudaTextureObject_t *)pTexObject, &cuda_pResDesc, &cuda_pTexDesc, &cuda_pResViewDesc);
    }
    else {
        cuda_res = cudaCreateTextureObject((cudaTextureObject_t *)pTexObject, &cuda_pResDesc, &cuda_pTexDesc, nullptr);
    }
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKDestroyExternalMemory(UPTKExternalMemory_t extMem)
{
    cudaError_t cuda_res;
    cuda_res = cudaDestroyExternalMemory((cudaExternalMemory_t) extMem);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKDestroyExternalSemaphore(UPTKExternalSemaphore_t extSem)
{
    cudaError_t cuda_res;
    cuda_res = cudaDestroyExternalSemaphore((cudaExternalSemaphore_t) extSem);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKDestroySurfaceObject(UPTKSurfaceObject_t surfObject)
{
    cudaError_t cuda_res;
    cuda_res = cudaDestroySurfaceObject((cudaSurfaceObject_t) surfObject);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKDestroyTextureObject(UPTKTextureObject_t texObject)
{
    cudaError_t cuda_res;
    cuda_res = cudaDestroyTextureObject((cudaTextureObject_t) texObject);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKDeviceCanAccessPeer(int * canAccessPeer,int device,int peerDevice)
{
    cudaError_t cuda_res;
    cuda_res = cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKDeviceDisablePeerAccess(int peerDevice)
{
    cudaError_t cuda_res;
    cuda_res = cudaDeviceDisablePeerAccess(peerDevice);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKDeviceEnablePeerAccess(int peerDevice,unsigned int flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaDeviceEnablePeerAccess(peerDevice, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKDeviceGetAttribute(int* value, enum UPTKDeviceAttr attr, int device) 
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

    UPTKError res = UPTKSuccess;
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
        cudaDeviceAttr cuda_attr = UPTKDeviceAttrTocudaDeviceAttr(attr);
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

__host__ UPTKError UPTKDeviceGetByPCIBusId(int * device,const char * pciBusId)
{
    cudaError_t cuda_res;
    cuda_res = cudaDeviceGetByPCIBusId(device, pciBusId);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKDeviceGetCacheConfig(enum UPTKFuncCache * pCacheConfig)
{
    if(nullptr == pCacheConfig)
    {
        return UPTKErrorInvalidValue;
    }

    memset(pCacheConfig, 0, sizeof(UPTKFuncCache));

    cudaFuncCache cuda_cacheConfig;
    cudaError_t cuda_res;
    cuda_res = cudaDeviceGetCacheConfig(&cuda_cacheConfig);
    if ( cuda_res == cudaSuccess )
    {
        *pCacheConfig = cudaFuncCacheToUPTKFuncCache(cuda_cacheConfig);
    }
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKDeviceGetLimit(size_t * pValue,enum UPTKLimit limit)
{
    enum cudaLimit cuda_limit = UPTKLimitTocudaLimit(limit);
    cudaError_t cuda_res;
    cuda_res = cudaDeviceGetLimit(pValue, cuda_limit);
    return cudaErrorToUPTKError(cuda_res);
}

// UNSUPPORTED
__host__ UPTKError UPTKDeviceGetNvSciSyncAttributes(void * nvSciSyncAttrList,int device,int flags)
{
    Debug();
    return UPTKErrorNotSupported;
}

// runtime not implement this api
__host__ UPTKError UPTKDeviceGetP2PAttribute(int * value,enum UPTKDeviceP2PAttr attr,int srcDevice,int dstDevice)
{
    cudaDeviceP2PAttr cuda_attr = UPTKDeviceP2PAttrTocudaDeviceP2PAttr(attr);
    cudaError_t cuda_res;
    cuda_res = cudaDeviceGetP2PAttribute(value, cuda_attr, srcDevice, dstDevice);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKDeviceGetPCIBusId(char * pciBusId,int len,int device)
{
    cudaError_t cuda_res;
    cuda_res = cudaDeviceGetPCIBusId(pciBusId, len, device);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKDeviceGetSharedMemConfig(enum UPTKSharedMemConfig * pConfig)
{
    if(nullptr == pConfig)
    {
        return UPTKErrorInvalidValue;
    }

    memset(pConfig, 0, sizeof(UPTKSharedMemConfig));

    cudaSharedMemConfig cuda_pConfig;
    cudaError_t cuda_res;
    cuda_res = cudaDeviceGetSharedMemConfig(&cuda_pConfig);
    if ( cuda_res == cudaSuccess )
    {
        *pConfig = cudaSharedMemConfigToUPTKSharedMemConfig(cuda_pConfig);
    }
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKDeviceGetStreamPriorityRange(int * leastPriority,int * greatestPriority)
{
    cudaError_t cuda_res;
    cuda_res = cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKDeviceReset(void)
{
    cudaError_t cuda_res;
    cuda_res = cudaDeviceReset();
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKDeviceSetCacheConfig(enum UPTKFuncCache cacheConfig)
{
    cudaFuncCache cuda_cacheConfig = UPTKFuncCacheTocudaFuncCache(cacheConfig);
    cudaError_t cuda_res;
    cuda_res = cudaDeviceSetCacheConfig(cuda_cacheConfig);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKDeviceSetLimit(enum UPTKLimit limit,size_t value)
{
    enum cudaLimit cuda_limit = UPTKLimitTocudaLimit(limit);
    cudaError_t cuda_res;
    cuda_res = cudaDeviceSetLimit(cuda_limit, value);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKDeviceSetSharedMemConfig(enum UPTKSharedMemConfig config)
{
    cudaSharedMemConfig cuda_config = UPTKSharedMemConfigTocudaSharedMemConfig(config);
    cudaError_t cuda_res;
    cuda_res = cudaDeviceSetSharedMemConfig(cuda_config);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKDeviceSynchronize(void)
{
    cudaError_t cuda_res;
    cuda_res = cudaDeviceSynchronize();
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKDriverGetVersion(int * driverVersion)
{
    *driverVersion = CUDA_VERSION;
    return UPTKSuccess;
}

__host__ UPTKError UPTKEventCreate(UPTKEvent_t * event)
{
    cudaError_t cuda_res;
    cuda_res = cudaEventCreate((cudaEvent_t *)event);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKEventCreateWithFlags(UPTKEvent_t * event,unsigned int flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaEventCreateWithFlags((cudaEvent_t *)event, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKEventDestroy(UPTKEvent_t event)
{
    cudaError_t cuda_res;
    cuda_res = cudaEventDestroy((cudaEvent_t) event);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKEventElapsedTime(float * ms,UPTKEvent_t start,UPTKEvent_t end)
{
    cudaError_t cuda_res;
    cuda_res = cudaEventElapsedTime(ms, (cudaEvent_t) start, (cudaEvent_t) end);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKEventQuery(UPTKEvent_t event)
{
    cudaError_t cuda_res;
    cuda_res = cudaEventQuery((cudaEvent_t) event);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKEventRecord(UPTKEvent_t event,UPTKStream_t stream)
{
    cudaError_t cuda_res;
    cuda_res = cudaEventRecord((cudaEvent_t) event, (cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKEventSynchronize(UPTKEvent_t event)
{
    cudaError_t cuda_res;
    cuda_res = cudaEventSynchronize((cudaEvent_t) event);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKExternalMemoryGetMappedBuffer(void ** devPtr,UPTKExternalMemory_t extMem,const struct UPTKExternalMemoryBufferDesc * bufferDesc)
{
    cudaError_t cuda_res;
    cuda_res = cudaExternalMemoryGetMappedBuffer(devPtr, (cudaExternalMemory_t) extMem, (cudaExternalMemoryBufferDesc *)bufferDesc);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKExternalMemoryGetMappedMipmappedArray(UPTKMipmappedArray_t * mipmap,UPTKExternalMemory_t extMem,const struct UPTKExternalMemoryMipmappedArrayDesc * mipmapDesc)
{
    if(mipmapDesc == nullptr)
    {
        return UPTKErrorInvalidValue;
    }
    cudaError_t cuda_res;
    cudaExternalMemoryMipmappedArrayDesc cuda_mipmapDesc;
    UPTKExternalMemoryMipmappedArrayDescTocudaExternalMemoryMipmappedArrayDesc(mipmapDesc, &cuda_mipmapDesc);
    cuda_res = cudaExternalMemoryGetMappedMipmappedArray((cudaMipmappedArray_t *) mipmap, (cudaExternalMemory_t) extMem, &cuda_mipmapDesc);
    return cudaErrorToUPTKError(cuda_res); 
}

__host__ UPTKError UPTKFree(void * devPtr)
{
    cudaError_t cuda_res;
    cuda_res = cudaFree(devPtr);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKFreeArray(UPTKArray_t array)
{
    if (nullptr == array) 
        return UPTKSuccess;
    cudaError_t cuda_res;
    cuda_res = cudaFreeArray((cudaArray *) array);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKFreeHost(void * ptr)
{
    cudaError_t cuda_res;
    cuda_res = cudaFreeHost(ptr);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKFreeMipmappedArray(UPTKMipmappedArray_t mipmappedArray)
{
    cudaError_t cuda_res;
    cuda_res = cudaFreeMipmappedArray((cudaMipmappedArray_t) mipmappedArray);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKFuncGetAttributes(struct UPTKFuncAttributes * attr,const void * func)
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

__host__ UPTKError UPTKFuncSetAttribute(const void * func,enum UPTKFuncAttribute attr,int value)
{
    cudaFuncAttribute cuda_attr = UPTKFuncAttributeTocudaFuncAttribute(attr);
    cudaError_t cuda_res;
    cuda_res = cudaFuncSetAttribute(func, cuda_attr, value);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKFuncSetCacheConfig(const void * func,enum UPTKFuncCache cacheConfig)
{
    cudaFuncCache cuda_config = UPTKFuncCacheTocudaFuncCache(cacheConfig);
    cudaError_t cuda_res;
    cuda_res = cudaFuncSetCacheConfig(func, cuda_config);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKFuncSetSharedMemConfig(const void * func,enum UPTKSharedMemConfig config)
{
    cudaSharedMemConfig cuda_config = UPTKSharedMemConfigTocudaSharedMemConfig(config);
    cudaError_t cuda_res;
    cuda_res = cudaFuncSetSharedMemConfig(func, cuda_config);
    return cudaErrorToUPTKError(cuda_res);
}

#if UPTK_ENABLE_GL==0
__host__ UPTKError UPTKGLGetDevices(unsigned int * pUPTKDeviceCount,int * pUPTKDevices,unsigned int UPTKDeviceCount,enum UPTKGLDeviceList deviceList)
{
    Debug();
    return UPTKErrorNotSupported;
}
#else
__host__ UPTKError UPTKGLGetDevices(unsigned int * pUPTKDeviceCount,int * pUPTKDevices,unsigned int UPTKDeviceCount,enum UPTKGLDeviceList deviceList)
{
    cudaError_t cuda_res;
    cudaGLDeviceList cuda_deviceList;
    cuda_deviceList = UPTKGLDeviceListTocudaGLDeviceList(deviceList);
    cuda_res = cudaGLGetDevices(pUPTKDeviceCount, pUPTKDevices, UPTKDeviceCount, cuda_deviceList);
    return cudaErrorToUPTKError(cuda_res);
}
#endif

__host__ UPTKError UPTKGetChannelDesc(struct UPTKChannelFormatDesc * desc,UPTKArray_const_t array)
{
    if(nullptr == desc)
    {
        return UPTKErrorInvalidValue;
    }
    cudaError_t cuda_res;
    cuda_res = cudaGetChannelDesc((cudaChannelFormatDesc *)desc, (cudaArray_t) array);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGetDevice(int * device)
{
    cudaError_t cuda_res;
    cuda_res = cudaGetDevice(device);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGetDeviceCount(int * count)
{
    cudaError_t cuda_res;
    cuda_res = cudaGetDeviceCount(count);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGetDeviceFlags(unsigned int * flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaGetDeviceFlags(flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ const char * UPTKGetErrorName(UPTKError error)
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
        case UPTKErrorCudartUnloading:
            return "UPTKErrorCudartUnloading";
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

__host__ const char * UPTKGetErrorString(UPTKError error)
{
    cudaError_t cuda_cudaError = UPTKErrorTocudaError(error);
    const char * cuda_res;
    cuda_res = cudaGetErrorString(cuda_cudaError);
    return cuda_res;
}

// UNSUPPORTED
__host__ UPTKError UPTKGetExportTable(const void ** ppExportTable,const UPTKUUID_t * pExportTableId)
{
    Debug();
    return UPTKErrorNotSupported;
}

__host__ UPTKError UPTKGetLastError(void)
{
    cudaError_t cuda_res;
    cuda_res = cudaGetLastError();
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGetMipmappedArrayLevel(UPTKArray_t * levelArray,UPTKMipmappedArray_const_t mipmappedArray,unsigned int level)
{
    cudaError_t cuda_res;
    cuda_res = cudaGetMipmappedArrayLevel((cudaArray_t *)levelArray, (cudaMipmappedArray_const_t) mipmappedArray, level);
    return cudaErrorToUPTKError(cuda_res);
}

// UNSUPPORTED
__host__ void * UPTKGetParameterBuffer(size_t alignment,size_t size)
{
    Debug();
    return nullptr;
}

// UNSUPPORTED
__host__ void * UPTKGetParameterBufferV2(void * func,dim3 gridDimension,dim3 blockDimension,unsigned int sharedMemSize)
{
    Debug();
    return nullptr;
}


__host__ UPTKError UPTKGetSurfaceObjectResourceDesc(struct UPTKResourceDesc * pResDesc,UPTKSurfaceObject_t surfObject)
{
    if (!pResDesc)
    {
        return UPTKErrorInvalidValue;
    }
    cudaError_t cuda_res;
    cudaResourceDesc cuda_pResDesc;
    cuda_res = cudaGetSurfaceObjectResourceDesc(&cuda_pResDesc, (cudaSurfaceObject_t) surfObject);
    if ( cuda_res == cudaSuccess )
    {
        cudaResourceDescToUPTKResourceDesc(&cuda_pResDesc, pResDesc);
    }
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGetSymbolAddress(void ** devPtr,const void * symbol)
{
    cudaError_t cuda_res;
    cuda_res = cudaGetSymbolAddress(devPtr, symbol);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGetSymbolSize(size_t * size,const void * symbol)
{
    cudaError_t cuda_res;
    cuda_res = cudaGetSymbolSize(size, symbol);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGetTextureObjectResourceDesc(struct UPTKResourceDesc * pResDesc,UPTKTextureObject_t texObject)
{
    cudaResourceDesc cuda_pResDesc;
    UPTKResourceDescTocudaResourceDesc(pResDesc, &cuda_pResDesc);
    cudaError_t cuda_res;
    cuda_res = cudaGetTextureObjectResourceDesc(&cuda_pResDesc, (cudaTextureObject_t) texObject);
    if ( cuda_res == cudaSuccess )
    {
        cudaResourceDescToUPTKResourceDesc(&cuda_pResDesc, pResDesc);
    }
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGetTextureObjectResourceViewDesc(struct UPTKResourceViewDesc * pResViewDesc,UPTKTextureObject_t texObject)
{
    struct cudaResourceViewDesc cuda_pResViewDesc;
    UPTKResourceViewDescTocudaResourceViewDesc(pResViewDesc, &cuda_pResViewDesc);
    cudaError_t cuda_res;
    cuda_res = cudaGetTextureObjectResourceViewDesc(&cuda_pResViewDesc, (cudaTextureObject_t) texObject);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGetTextureObjectTextureDesc(struct UPTKTextureDesc * pTexDesc,UPTKTextureObject_t texObject)
{
    cudaTextureDesc cuda_pTexDesc;
    UPTKTextureDescTocudaTextureDesc(pTexDesc, &cuda_pTexDesc);
    cudaError_t cuda_res;
    cuda_res = cudaGetTextureObjectTextureDesc(&cuda_pTexDesc, (cudaTextureObject_t) texObject);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphAddChildGraphNode(UPTKGraphNode_t * pGraphNode,UPTKGraph_t graph,const UPTKGraphNode_t * pDependencies,size_t numDependencies,UPTKGraph_t childGraph)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphAddChildGraphNode((cudaGraphNode_t *) pGraphNode, (cudaGraph_t) graph, (const cudaGraphNode_t *) pDependencies, numDependencies, (cudaGraph_t) childGraph);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphAddDependencies(UPTKGraph_t graph,const UPTKGraphNode_t * from,const UPTKGraphNode_t * to,size_t numDependencies)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphAddDependencies((cudaGraph_t) graph, (const cudaGraphNode_t *) from, (const cudaGraphNode_t *) to, numDependencies);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphAddEmptyNode(UPTKGraphNode_t * pGraphNode,UPTKGraph_t graph,const UPTKGraphNode_t * pDependencies,size_t numDependencies)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphAddEmptyNode((cudaGraphNode_t *) pGraphNode, (cudaGraph_t) graph, (const cudaGraphNode_t *) pDependencies, numDependencies);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphAddEventRecordNode(UPTKGraphNode_t * pGraphNode,UPTKGraph_t graph,const UPTKGraphNode_t * pDependencies,size_t numDependencies,UPTKEvent_t event)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphAddEventRecordNode((cudaGraphNode_t *) pGraphNode, (cudaGraph_t) graph, (const cudaGraphNode_t *) pDependencies, numDependencies, (cudaEvent_t) event);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphAddEventWaitNode(UPTKGraphNode_t * pGraphNode,UPTKGraph_t graph,const UPTKGraphNode_t * pDependencies,size_t numDependencies,UPTKEvent_t event)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphAddEventWaitNode((cudaGraphNode_t *) pGraphNode, (cudaGraph_t) graph, (const cudaGraphNode_t *) pDependencies, numDependencies, (cudaEvent_t) event);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphAddExternalSemaphoresSignalNode(UPTKGraphNode_t * pGraphNode,UPTKGraph_t graph,const UPTKGraphNode_t * pDependencies,size_t numDependencies,const struct UPTKExternalSemaphoreSignalNodeParams * nodeParams)
{
    if (nodeParams == nullptr)
    {
        return UPTKErrorInvalidValue;
    }
    cudaError_t cuda_res;
    cudaExternalSemaphoreSignalNodeParams cuda_nodeParams = UPTKExternalSemaphoreSignalNodeParamsTocudaExternalSemaphoreSignalNodeParams(*nodeParams);
    cuda_res = cudaGraphAddExternalSemaphoresSignalNode((cudaGraphNode_t *) pGraphNode, (cudaGraph_t) graph, (const cudaGraphNode_t *) pDependencies, numDependencies, &cuda_nodeParams);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphAddExternalSemaphoresWaitNode(UPTKGraphNode_t * pGraphNode,UPTKGraph_t graph,const UPTKGraphNode_t * pDependencies,size_t numDependencies,const struct UPTKExternalSemaphoreWaitNodeParams * nodeParams)
{
    if (nodeParams == nullptr)
    {
        return UPTKErrorInvalidValue;
    }
    cudaError_t cuda_res;
    cudaExternalSemaphoreWaitNodeParams cuda_nodeParams = UPTKExternalSemaphoreWaitNodeParamsTocudaExternalSemaphoreWaitNodeParams(*nodeParams);
    cuda_res = cudaGraphAddExternalSemaphoresWaitNode((cudaGraphNode_t *) pGraphNode, (cudaGraph_t) graph, (const cudaGraphNode_t *) pDependencies, numDependencies, &cuda_nodeParams);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphAddHostNode(UPTKGraphNode_t * pGraphNode,UPTKGraph_t graph,const UPTKGraphNode_t * pDependencies,size_t numDependencies,const struct UPTKHostNodeParams * pNodeParams)
{
    if (nullptr == pNodeParams)
        return UPTKErrorInvalidValue;
    cudaHostNodeParams cuda_pNodeParams;
    UPTKHostNodeParamsTocudaHostNodeParams(pNodeParams, &cuda_pNodeParams);
    cudaError_t cuda_res;
    cuda_res = cudaGraphAddHostNode((cudaGraphNode_t *) pGraphNode, (cudaGraph_t) graph, (const cudaGraphNode_t *) pDependencies, numDependencies, &cuda_pNodeParams);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphAddKernelNode(UPTKGraphNode_t * pGraphNode,UPTKGraph_t graph,const UPTKGraphNode_t * pDependencies,size_t numDependencies,const struct UPTKKernelNodeParams * pNodeParams)
{
    if (nullptr == pNodeParams)
        return UPTKErrorInvalidValue;
    cudaKernelNodeParams cuda_pNodeParams;
    UPTKKernelNodeParamsTocudaKernelNodeParams(pNodeParams, &cuda_pNodeParams);
    cudaError_t cuda_res;
    cuda_res = cudaGraphAddKernelNode((cudaGraphNode_t *) pGraphNode, (cudaGraph_t) graph, (const cudaGraphNode_t *) pDependencies, numDependencies, &cuda_pNodeParams);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphAddMemcpyNode(UPTKGraphNode_t * pGraphNode,UPTKGraph_t graph,const UPTKGraphNode_t * pDependencies,size_t numDependencies,const struct UPTKMemcpy3DParms * pCopyParams)
{
    if (nullptr == pCopyParams)
        return UPTKErrorInvalidValue;
    cudaMemcpy3DParms cuda_pCopyParams;
    UPTKMemcpy3DParmsTocudaMemcpy3DParms(pCopyParams, &cuda_pCopyParams);
    cudaError_t cuda_res;
    cuda_res = cudaGraphAddMemcpyNode((cudaGraphNode_t *) pGraphNode, (cudaGraph_t) graph, (const cudaGraphNode_t *) pDependencies, numDependencies, &cuda_pCopyParams);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphAddMemcpyNode1D(UPTKGraphNode_t * pGraphNode,UPTKGraph_t graph,const UPTKGraphNode_t * pDependencies,size_t numDependencies,void * dst,const void * src,size_t count,enum UPTKMemcpyKind kind)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaGraphAddMemcpyNode1D((cudaGraphNode_t *) pGraphNode, (cudaGraph_t) graph, (const cudaGraphNode_t *) pDependencies, numDependencies, dst, src, count, cuda_kind);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphAddMemcpyNodeFromSymbol(UPTKGraphNode_t * pGraphNode,UPTKGraph_t graph,const UPTKGraphNode_t * pDependencies,size_t numDependencies,void * dst,const void * symbol,size_t count,size_t offset,enum UPTKMemcpyKind kind)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaGraphAddMemcpyNodeFromSymbol((cudaGraphNode_t *) pGraphNode, (cudaGraph_t) graph, (const cudaGraphNode_t *) pDependencies, numDependencies, dst, symbol, count, offset, cuda_kind);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphAddMemcpyNodeToSymbol(UPTKGraphNode_t * pGraphNode,UPTKGraph_t graph,const UPTKGraphNode_t * pDependencies,size_t numDependencies,const void * symbol,const void * src,size_t count,size_t offset,enum UPTKMemcpyKind kind)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaGraphAddMemcpyNodeToSymbol((cudaGraphNode_t *) pGraphNode, (cudaGraph_t) graph, (const cudaGraphNode_t *) pDependencies, numDependencies, symbol, src, count, offset, cuda_kind);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphAddMemsetNode(UPTKGraphNode_t * pGraphNode,UPTKGraph_t graph,const UPTKGraphNode_t * pDependencies,size_t numDependencies,const struct UPTKMemsetParams * pMemsetParams)
{
    if (nullptr == pMemsetParams)
        return UPTKErrorInvalidValue;
    cudaMemsetParams cuda_pMemsetParams;
    UPTKMemsetParamsTocudaMemsetParams(pMemsetParams, &cuda_pMemsetParams);
    cudaError_t cuda_res;
    cuda_res = cudaGraphAddMemsetNode((cudaGraphNode_t *) pGraphNode, (cudaGraph_t) graph, (const cudaGraphNode_t *) pDependencies, numDependencies, &cuda_pMemsetParams);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphChildGraphNodeGetGraph(UPTKGraphNode_t node,UPTKGraph_t * pGraph)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphChildGraphNodeGetGraph((cudaGraphNode_t) node, (cudaGraph_t *) pGraph);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphClone(UPTKGraph_t * pGraphClone,UPTKGraph_t originalGraph)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphClone((cudaGraph_t *) pGraphClone, (cudaGraph_t) originalGraph);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphCreate(UPTKGraph_t * pGraph,unsigned int flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphCreate((cudaGraph_t *) pGraph, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphDestroy(UPTKGraph_t graph)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphDestroy((cudaGraph_t) graph);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphDestroyNode(UPTKGraphNode_t node)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphDestroyNode((cudaGraphNode_t) node);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphEventRecordNodeGetEvent(UPTKGraphNode_t node,UPTKEvent_t * event_out)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphEventRecordNodeGetEvent((cudaGraphNode_t) node, (cudaEvent_t *) event_out);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphEventRecordNodeSetEvent(UPTKGraphNode_t node,UPTKEvent_t event)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphEventRecordNodeSetEvent((cudaGraphNode_t) node, (cudaEvent_t) event);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphEventWaitNodeGetEvent(UPTKGraphNode_t node,UPTKEvent_t * event_out)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphEventWaitNodeGetEvent((cudaGraphNode_t) node, (cudaEvent_t *) event_out);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphEventWaitNodeSetEvent(UPTKGraphNode_t node,UPTKEvent_t event)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphEventWaitNodeSetEvent((cudaGraphNode_t) node, (cudaEvent_t) event);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphExecChildGraphNodeSetParams(UPTKGraphExec_t hGraphExec,UPTKGraphNode_t node,UPTKGraph_t childGraph)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphExecChildGraphNodeSetParams((cudaGraphExec_t) hGraphExec, (cudaGraphNode_t) node, (cudaGraph_t) childGraph);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphExecDestroy(UPTKGraphExec_t graphExec)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphExecDestroy((cudaGraphExec_t) graphExec);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphExecEventRecordNodeSetEvent(UPTKGraphExec_t hGraphExec,UPTKGraphNode_t hNode,UPTKEvent_t event)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphExecEventRecordNodeSetEvent((cudaGraphExec_t) hGraphExec, (cudaGraphNode_t) hNode, (cudaEvent_t) event);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphExecEventWaitNodeSetEvent(UPTKGraphExec_t hGraphExec,UPTKGraphNode_t hNode,UPTKEvent_t event)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphExecEventWaitNodeSetEvent((cudaGraphExec_t) hGraphExec, (cudaGraphNode_t) hNode, (cudaEvent_t) event);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphExecExternalSemaphoresSignalNodeSetParams(UPTKGraphExec_t hGraphExec,UPTKGraphNode_t hNode,const struct UPTKExternalSemaphoreSignalNodeParams * nodeParams)
{
    if(nodeParams == nullptr)
    {
        return UPTKErrorInvalidValue;
    }
    cudaError_t cuda_res;
    cudaExternalSemaphoreSignalNodeParams cuda_nodeParams = UPTKExternalSemaphoreSignalNodeParamsTocudaExternalSemaphoreSignalNodeParams(*nodeParams);
    cuda_res = cudaGraphExecExternalSemaphoresSignalNodeSetParams((cudaGraphExec_t) hGraphExec, (cudaGraphNode_t) hNode, &cuda_nodeParams);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphExecExternalSemaphoresWaitNodeSetParams(UPTKGraphExec_t hGraphExec,UPTKGraphNode_t hNode,const struct UPTKExternalSemaphoreWaitNodeParams * nodeParams)
{
    if(nodeParams == nullptr)
    {
        return UPTKErrorInvalidValue;
    }
    cudaError_t cuda_res;
    cudaExternalSemaphoreWaitNodeParams cuda_nodeParams = UPTKExternalSemaphoreWaitNodeParamsTocudaExternalSemaphoreWaitNodeParams(*nodeParams);
    cuda_res = cudaGraphExecExternalSemaphoresWaitNodeSetParams((cudaGraphExec_t) hGraphExec, (cudaGraphNode_t) hNode, &cuda_nodeParams);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphExecHostNodeSetParams(UPTKGraphExec_t hGraphExec,UPTKGraphNode_t node,const struct UPTKHostNodeParams * pNodeParams)
{
    if (nullptr == pNodeParams)
        return UPTKErrorInvalidValue;
    cudaHostNodeParams cuda_pNodeParams;
    UPTKHostNodeParamsTocudaHostNodeParams(pNodeParams, &cuda_pNodeParams);
    cudaError_t cuda_res;
    cuda_res = cudaGraphExecHostNodeSetParams((cudaGraphExec_t) hGraphExec, (cudaGraphNode_t) node, &cuda_pNodeParams);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphExecKernelNodeSetParams(UPTKGraphExec_t hGraphExec,UPTKGraphNode_t node,const struct UPTKKernelNodeParams * pNodeParams)
{
    if (nullptr == pNodeParams)
        return UPTKErrorInvalidValue;
    cudaKernelNodeParams cuda_pNodeParams;
    UPTKKernelNodeParamsTocudaKernelNodeParams(pNodeParams, &cuda_pNodeParams);
    cudaError_t cuda_res;
    cuda_res = cudaGraphExecKernelNodeSetParams((cudaGraphExec_t) hGraphExec, (cudaGraphNode_t) node, &cuda_pNodeParams);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphExecMemcpyNodeSetParams(UPTKGraphExec_t hGraphExec,UPTKGraphNode_t node,const struct UPTKMemcpy3DParms * pNodeParams)
{
    if (nullptr == pNodeParams)
        return UPTKErrorInvalidValue;
    cudaMemcpy3DParms cuda_pNodeParams;
    UPTKMemcpy3DParmsTocudaMemcpy3DParms(pNodeParams, &cuda_pNodeParams);
    cudaError_t cuda_res;
    cuda_res = cudaGraphExecMemcpyNodeSetParams((cudaGraphExec_t) hGraphExec, (cudaGraphNode_t) node, &cuda_pNodeParams);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphExecMemcpyNodeSetParams1D(UPTKGraphExec_t hGraphExec,UPTKGraphNode_t node,void * dst,const void * src,size_t count,enum UPTKMemcpyKind kind)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaGraphExecMemcpyNodeSetParams1D((cudaGraphExec_t) hGraphExec, (cudaGraphNode_t) node, dst, src, count, cuda_kind);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphExecMemcpyNodeSetParamsFromSymbol(UPTKGraphExec_t hGraphExec,UPTKGraphNode_t node,void * dst,const void * symbol,size_t count,size_t offset,enum UPTKMemcpyKind kind)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaGraphExecMemcpyNodeSetParamsFromSymbol((cudaGraphExec_t) hGraphExec, (cudaGraphNode_t) node, dst, symbol, count, offset, cuda_kind);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphExecMemcpyNodeSetParamsToSymbol(UPTKGraphExec_t hGraphExec,UPTKGraphNode_t node,const void * symbol,const void * src,size_t count,size_t offset,enum UPTKMemcpyKind kind)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaGraphExecMemcpyNodeSetParamsToSymbol((cudaGraphExec_t) hGraphExec, (cudaGraphNode_t) node, symbol, src, count, offset, cuda_kind);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphExecMemsetNodeSetParams(UPTKGraphExec_t hGraphExec,UPTKGraphNode_t node,const struct UPTKMemsetParams * pNodeParams)
{
    if (nullptr == pNodeParams)
        return UPTKErrorInvalidValue;
    cudaMemsetParams cuda_pNodeParams;
    UPTKMemsetParamsTocudaMemsetParams(pNodeParams, &cuda_pNodeParams);
    cudaError_t cuda_res;
    cuda_res = cudaGraphExecMemsetNodeSetParams((cudaGraphExec_t) hGraphExec, (cudaGraphNode_t) node, &cuda_pNodeParams);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphExecUpdate(UPTKGraphExec_t hGraphExec,UPTKGraph_t hGraph,UPTKGraphNode_t * hErrorNode_out,enum UPTKGraphExecUpdateResult * updateResult_out)
{
    if (nullptr == updateResult_out)
        return UPTKErrorInvalidValue;
    cudaError_t cuda_res;
    cuda_res = cudaGraphExecUpdate((cudaGraphExec_t) hGraphExec, (cudaGraph_t) hGraph, (cudaGraphNode_t *) hErrorNode_out, (cudaGraphExecUpdateResult *)updateResult_out);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphExternalSemaphoresSignalNodeGetParams(UPTKGraphNode_t hNode,struct UPTKExternalSemaphoreSignalNodeParams * params_out)
{
    if (nullptr == params_out)
    {
        return UPTKErrorInvalidValue;
    }
    cudaError_t cuda_res;
    cudaExternalSemaphoreSignalNodeParams cuda_params_out;
    cuda_res = cudaGraphExternalSemaphoresSignalNodeGetParams((cudaGraphNode_t) hNode, &cuda_params_out);
    if(cuda_res == cudaSuccess)
    {
        *params_out = cudaExternalSemaphoreSignalNodeParamsToUPTKExternalSemaphoreSignalNodeParams(cuda_params_out);
    }
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphExternalSemaphoresSignalNodeSetParams(UPTKGraphNode_t hNode,const struct UPTKExternalSemaphoreSignalNodeParams * nodeParams)
{
    if (nullptr == nodeParams)
    {
        return UPTKErrorInvalidValue;
    }
    cudaError_t cuda_res;
    cudaExternalSemaphoreSignalNodeParams cuda_nodeParams = UPTKExternalSemaphoreSignalNodeParamsTocudaExternalSemaphoreSignalNodeParams(*nodeParams);
    cuda_res = cudaGraphExternalSemaphoresSignalNodeSetParams((cudaGraphNode_t) hNode, &cuda_nodeParams);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphExternalSemaphoresWaitNodeGetParams(UPTKGraphNode_t hNode,struct UPTKExternalSemaphoreWaitNodeParams * params_out)
{
    if (nullptr == params_out)
    {
        return UPTKErrorInvalidValue;
    }
    cudaError_t cuda_res;
    cudaExternalSemaphoreWaitNodeParams cuda_params_out;
    cuda_res = cudaGraphExternalSemaphoresWaitNodeGetParams((cudaGraphNode_t) hNode, &cuda_params_out);
    if(cuda_res == cudaSuccess)
    {
        *params_out = cudaExternalSemaphoreWaitNodeParamsToUPTKExternalSemaphoreWaitNodeParams(cuda_params_out);
    }
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphExternalSemaphoresWaitNodeSetParams(UPTKGraphNode_t hNode,const struct UPTKExternalSemaphoreWaitNodeParams * nodeParams)
{
    if (nullptr == nodeParams)
    {
        return UPTKErrorInvalidValue;
    }
    cudaError_t cuda_res;
    cudaExternalSemaphoreWaitNodeParams cuda_nodeParams = UPTKExternalSemaphoreWaitNodeParamsTocudaExternalSemaphoreWaitNodeParams(*nodeParams);
    cuda_res = cudaGraphExternalSemaphoresWaitNodeSetParams((cudaGraphNode_t) hNode, &cuda_nodeParams);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphGetEdges(UPTKGraph_t graph,UPTKGraphNode_t * from,UPTKGraphNode_t * to,size_t * numEdges)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphGetEdges((cudaGraph_t) graph, (cudaGraphNode_t *) from, (cudaGraphNode_t *) to, numEdges);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphGetNodes(UPTKGraph_t graph,UPTKGraphNode_t * nodes,size_t * numNodes)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphGetNodes((cudaGraph_t) graph, (cudaGraphNode_t *) nodes, numNodes);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphGetRootNodes(UPTKGraph_t graph,UPTKGraphNode_t * pRootNodes,size_t * pNumRootNodes)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphGetRootNodes((cudaGraph_t) graph, (cudaGraphNode_t *) pRootNodes, pNumRootNodes);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphHostNodeGetParams(UPTKGraphNode_t node,struct UPTKHostNodeParams * pNodeParams)
{
    if (nullptr == pNodeParams)
        return UPTKErrorInvalidValue;
    cudaHostNodeParams cuda_pNodeParams;
    UPTKHostNodeParamsTocudaHostNodeParams(pNodeParams, &cuda_pNodeParams);
    cudaError_t cuda_res;
    cuda_res = cudaGraphHostNodeGetParams((cudaGraphNode_t) node, &cuda_pNodeParams);
    if ( cuda_res == cudaSuccess )
    {
        cudaHostNodeParamsToUPTKHostNodeParams(&cuda_pNodeParams, pNodeParams);
    }
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphHostNodeSetParams(UPTKGraphNode_t node,const struct UPTKHostNodeParams * pNodeParams)
{
    if (nullptr == pNodeParams)
        return UPTKErrorInvalidValue;
    cudaHostNodeParams cuda_pNodeParams;
    UPTKHostNodeParamsTocudaHostNodeParams(pNodeParams, &cuda_pNodeParams);
    cudaError_t cuda_res;
    cuda_res = cudaGraphHostNodeSetParams((cudaGraphNode_t) node, &cuda_pNodeParams);
    return cudaErrorToUPTKError(cuda_res);
}
// UPTK 12.6.2: In the driver module,cuGraphInstantiate is equal to cuGraphInstantiateWithFlags interface.
// UPTKGraphInstantiate statement after the change, the number and type and parameters of the corresponding cudaGraphInstantiateWithFlags interface.
__host__ UPTKError UPTKGraphInstantiate(UPTKGraphExec_t *pGraphExec, UPTKGraph_t graph, unsigned long long flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphInstantiateWithFlags((cudaGraphExec_t *) pGraphExec, (cudaGraph_t) graph, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphKernelNodeGetParams(UPTKGraphNode_t node,struct UPTKKernelNodeParams * pNodeParams)
{
    if (nullptr == pNodeParams)
        return UPTKErrorInvalidValue;    
    cudaKernelNodeParams cuda_pNodeParams;
    UPTKKernelNodeParamsTocudaKernelNodeParams(pNodeParams, &cuda_pNodeParams);
    cudaError_t cuda_res;
    cuda_res = cudaGraphKernelNodeGetParams((cudaGraphNode_t) node, &cuda_pNodeParams);
    if (cudaSuccess == cuda_res)
        cudaKernelNodeParamsToUPTKKernelNodeParams(&cuda_pNodeParams, pNodeParams);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphKernelNodeSetParams(UPTKGraphNode_t node,const struct UPTKKernelNodeParams * pNodeParams)
{
    if (nullptr == pNodeParams)
        return UPTKErrorInvalidValue;
    cudaKernelNodeParams cuda_pNodeParams;
    UPTKKernelNodeParamsTocudaKernelNodeParams(pNodeParams, &cuda_pNodeParams);
    cudaError_t cuda_res;
    cuda_res = cudaGraphKernelNodeSetParams((cudaGraphNode_t) node, &cuda_pNodeParams);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphLaunch(UPTKGraphExec_t graphExec,UPTKStream_t stream)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphLaunch((cudaGraphExec_t) graphExec, (cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphMemcpyNodeGetParams(UPTKGraphNode_t node,struct UPTKMemcpy3DParms * pNodeParams)
{
    if (nullptr == pNodeParams)
        return UPTKErrorInvalidValue;
    cudaMemcpy3DParms cuda_pNodeParams;
    cudaError_t cuda_res;
    cuda_res = cudaGraphMemcpyNodeGetParams((cudaGraphNode_t) node, &cuda_pNodeParams);
    if ( cuda_res == cudaSuccess )
    {
        cudaMemcpy3DParmsToUPTKMemcpy3DParms(&cuda_pNodeParams, pNodeParams);
    }
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphMemcpyNodeSetParams(UPTKGraphNode_t node,const struct UPTKMemcpy3DParms * pNodeParams)
{
    if (nullptr == pNodeParams)
        return UPTKErrorInvalidValue;
    cudaMemcpy3DParms cuda_pNodeParams;
    UPTKMemcpy3DParmsTocudaMemcpy3DParms(pNodeParams, &cuda_pNodeParams);
    cudaError_t cuda_res;
    cuda_res = cudaGraphMemcpyNodeSetParams((cudaGraphNode_t) node, &cuda_pNodeParams);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphMemcpyNodeSetParams1D(UPTKGraphNode_t node,void * dst,const void * src,size_t count,enum UPTKMemcpyKind kind)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaGraphMemcpyNodeSetParams1D((cudaGraphNode_t) node, dst, src, count, cuda_kind);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphMemcpyNodeSetParamsFromSymbol(UPTKGraphNode_t node,void * dst,const void * symbol,size_t count,size_t offset,enum UPTKMemcpyKind kind)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaGraphMemcpyNodeSetParamsFromSymbol((cudaGraphNode_t) node, dst, symbol, count, offset, cuda_kind);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphMemcpyNodeSetParamsToSymbol(UPTKGraphNode_t node,const void * symbol,const void * src,size_t count,size_t offset,enum UPTKMemcpyKind kind)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaGraphMemcpyNodeSetParamsToSymbol((cudaGraphNode_t) node, symbol, src, count, offset, cuda_kind);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphMemsetNodeGetParams(UPTKGraphNode_t node,struct UPTKMemsetParams * pNodeParams)
{
    if (nullptr == pNodeParams)
        return UPTKErrorInvalidValue;
    cudaMemsetParams cuda_pNodeParams;
    UPTKMemsetParamsTocudaMemsetParams(pNodeParams, &cuda_pNodeParams);
    cudaError_t cuda_res;
    cuda_res = cudaGraphMemsetNodeGetParams((cudaGraphNode_t) node, &cuda_pNodeParams);
    if (cudaSuccess == cuda_res)
        cudaMemsetParamsToUPTKMemsetParams(&cuda_pNodeParams, pNodeParams);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphMemsetNodeSetParams(UPTKGraphNode_t node,const struct UPTKMemsetParams * pNodeParams)
{
    if (nullptr == pNodeParams)
        return UPTKErrorInvalidValue;
    cudaMemsetParams cuda_pNodeParams;
    UPTKMemsetParamsTocudaMemsetParams(pNodeParams, &cuda_pNodeParams);
    cudaError_t cuda_res;
    cuda_res = cudaGraphMemsetNodeSetParams((cudaGraphNode_t) node, &cuda_pNodeParams);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphNodeFindInClone(UPTKGraphNode_t * pNode,UPTKGraphNode_t originalNode,UPTKGraph_t clonedGraph)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphNodeFindInClone((cudaGraphNode_t *) pNode, (cudaGraphNode_t) originalNode, (cudaGraph_t) clonedGraph);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphNodeGetDependencies(UPTKGraphNode_t node,UPTKGraphNode_t * pDependencies,size_t * pNumDependencies)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphNodeGetDependencies((cudaGraphNode_t) node, (cudaGraphNode_t *) pDependencies, pNumDependencies);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphNodeGetDependentNodes(UPTKGraphNode_t node,UPTKGraphNode_t * pDependentNodes,size_t * pNumDependentNodes)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphNodeGetDependentNodes((cudaGraphNode_t) node, (cudaGraphNode_t *) pDependentNodes, pNumDependentNodes);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphNodeGetType(UPTKGraphNode_t node,enum UPTKGraphNodeType * pType)
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

__host__ UPTKError UPTKGraphRemoveDependencies(UPTKGraph_t graph,const UPTKGraphNode_t * from,const UPTKGraphNode_t * to,size_t numDependencies)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphRemoveDependencies((cudaGraph_t) graph, (const cudaGraphNode_t *) from, (const cudaGraphNode_t *) to, numDependencies);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphUpload(UPTKGraphExec_t graphExec,UPTKStream_t stream)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphUpload((cudaGraphExec_t) graphExec, (cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

#if UPTK_ENABLE_GL==0
__host__ UPTKError UPTKGraphicsGLRegisterBuffer(struct UPTKGraphicsResource ** resource,GLuint buffer,unsigned int flags)
{
    Debug();
    return UPTKErrorNotSupported;
}

__host__ UPTKError UPTKGraphicsGLRegisterImage(struct UPTKGraphicsResource ** resource,GLuint image,GLenum target,unsigned int flags)
{
    Debug();
    return UPTKErrorNotSupported;
}

__host__ UPTKError UPTKGraphicsMapResources(int count, UPTKGraphicsResource_t *resources, UPTKStream_t stream)
{
    Debug();
    return UPTKErrorNotSupported;
}

__host__ UPTKError UPTKGraphicsResourceGetMappedPointer(void ** devPtr,size_t * size,UPTKGraphicsResource_t resource)
{
    Debug();
    return UPTKErrorNotSupported;
}

__host__ UPTKError UPTKGraphicsSubResourceGetMappedArray(UPTKArray_t * array,UPTKGraphicsResource_t resource,unsigned int arrayIndex,unsigned int mipLevel)
{
    Debug();
    return UPTKErrorNotSupported;
}

__host__ UPTKError UPTKGraphicsUnregisterResource(UPTKGraphicsResource_t resource)
{
    Debug();
    return UPTKErrorNotSupported;
}
#else
__host__ UPTKError UPTKGraphicsGLRegisterBuffer(struct UPTKGraphicsResource ** resource,GLuint buffer,unsigned int flags)
{
    cudaGraphicsResource *cuda_resource;
    cudaError_t cuda_res;
    cuda_res = cudaGraphicsGLRegisterBuffer(&cuda_resource, buffer, flags);
    if (cuda_res == cudaSuccess){
        *resource = (struct UPTKGraphicsResource *) cuda_resource;
    }
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphicsGLRegisterImage(struct UPTKGraphicsResource ** resource,GLuint image,GLenum target,unsigned int flags)
{
    cudaGraphicsResource *cuda_resource;
    cudaError_t cuda_res;
    cuda_res = cudaGraphicsGLRegisterImage(&cuda_resource, image, target, flags);
    if (cuda_res == cudaSuccess){
        *resource = (struct UPTKGraphicsResource *) cuda_resource;
    }
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphicsMapResources(int count, UPTKGraphicsResource_t *resources, UPTKStream_t stream)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphicsMapResources(count, (cudaGraphicsResource_t *) resources, (cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphicsResourceGetMappedPointer(void ** devPtr,size_t * size,UPTKGraphicsResource_t resource)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphicsResourceGetMappedPointer(devPtr, size, (cudaGraphicsResource_t) resource);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphicsSubResourceGetMappedArray(UPTKArray_t * array,UPTKGraphicsResource_t resource,unsigned int arrayIndex,unsigned int mipLevel)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphicsSubResourceGetMappedArray((cudaArray_t *) array, (cudaGraphicsResource_t) resource, arrayIndex, mipLevel);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKGraphicsUnregisterResource(UPTKGraphicsResource_t resource)
{
    cudaError_t cuda_res;
    cuda_res = cudaGraphicsUnregisterResource((cudaGraphicsResource_t) resource);
    return cudaErrorToUPTKError(cuda_res);
}
#endif
// UNSUPPORTED
__host__ UPTKError UPTKGraphicsResourceGetMappedMipmappedArray(UPTKMipmappedArray_t * mipmappedArray,UPTKGraphicsResource_t resource)
{
    Debug();
    return UPTKErrorNotSupported;
}

// UNSUPPORTED
__host__ UPTKError UPTKGraphicsResourceSetMapFlags(UPTKGraphicsResource_t resource,unsigned int flags)
{
    Debug();
    return UPTKErrorNotSupported;
}

// UNSUPPORTED
__host__ UPTKError UPTKGraphicsUnmapResources(int count, UPTKGraphicsResource_t *resources, UPTKStream_t stream)
{
    Debug();
    return UPTKErrorNotSupported;
}

__host__ UPTKError UPTKHostAlloc(void ** pHost,size_t size,unsigned int flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaHostAlloc(pHost, size, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKHostGetDevicePointer(void ** pDevice,void * pHost,unsigned int flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaHostGetDevicePointer(pDevice, pHost, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKHostGetFlags(unsigned int * pFlags,void * pHost)
{
    cudaError_t cuda_res;
    cuda_res = cudaHostGetFlags(pFlags, pHost);
    // UPTK default to add flag UPTKHostAllocMapped
    if (cudaSuccess == cuda_res)
        *pFlags = *pFlags | UPTKHostAllocMapped;
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKHostRegister(void * ptr,size_t size,unsigned int flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaHostRegister(ptr, size, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKHostUnregister(void * ptr)
{
    cudaError_t cuda_res;
    cuda_res = cudaHostUnregister(ptr);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKImportExternalMemory(UPTKExternalMemory_t * extMem_out,const struct UPTKExternalMemoryHandleDesc * memHandleDesc)
{
    cudaExternalMemoryHandleDesc cuda_memHandleDesc;
    UPTKExternalMemoryHandleDescTocudaExternalMemoryHandleDesc(memHandleDesc, &cuda_memHandleDesc);
    cudaError_t cuda_res;
    cuda_res = cudaImportExternalMemory((cudaExternalMemory_t *) extMem_out, &cuda_memHandleDesc);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKImportExternalSemaphore(UPTKExternalSemaphore_t * extSem_out,const struct UPTKExternalSemaphoreHandleDesc * semHandleDesc)
{
    cudaExternalSemaphoreHandleDesc cuda_semHandleDesc;
    UPTKExternalSemaphoreHandleDescTocudaExternalSemaphoreHandleDesc(semHandleDesc, &cuda_semHandleDesc);
    cudaError_t cuda_res;
    cuda_res = cudaImportExternalSemaphore((cudaExternalSemaphore_t *) extSem_out, &cuda_semHandleDesc);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKIpcCloseMemHandle(void * devPtr)
{
    cudaError_t cuda_res;
    cuda_res = cudaIpcCloseMemHandle(devPtr);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKIpcGetEventHandle(UPTKIpcEventHandle_t * handle,UPTKEvent_t event)
{
    cudaError_t cuda_res;
    cuda_res = cudaIpcGetEventHandle((cudaIpcEventHandle_t *) handle, (cudaEvent_t) event);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKIpcGetMemHandle(UPTKIpcMemHandle_t * handle,void * devPtr)
{
    cudaError_t cuda_res;
    cuda_res = cudaIpcGetMemHandle((cudaIpcMemHandle_t *) handle, devPtr);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKIpcOpenEventHandle(UPTKEvent_t * event,UPTKIpcEventHandle_t handle)
{    
    cudaIpcEventHandle_t cuda_handle;
    UPTKIpcEventHandleTocudaIpcEventHandle(&handle, &cuda_handle);
    cudaError_t cuda_res;
    cuda_res = cudaIpcOpenEventHandle((cudaEvent_t *)event, cuda_handle);
    cudaIpcEventHandleToUPTKIpcEventHandle(&cuda_handle, &handle);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKIpcOpenMemHandle(void ** devPtr,UPTKIpcMemHandle_t handle,unsigned int flags)
{
    cudaIpcMemHandle_t cuda_handle;
    UPTKIpcMemHandleTocudaIpcMemHandle(&handle, &cuda_handle);
    if (UPTKIpcMemLazyEnablePeerAccess == flags)
        flags = cudaIpcMemLazyEnablePeerAccess;
    cudaError_t cuda_res;
    cuda_res = cudaIpcOpenMemHandle(devPtr, cuda_handle, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKLaunchCooperativeKernel(const void * func,dim3 gridDim,dim3 blockDim,void ** args,size_t sharedMem,UPTKStream_t stream)
{
    cudaError_t cuda_res;
    cuda_res = cudaLaunchCooperativeKernel(func, gridDim, blockDim, args, sharedMem, (cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

// UNSUPPORTED
 __host__ UPTKError UPTKLaunchCooperativeKernelMultiDevice(struct UPTKLaunchParams *launchParamsList, unsigned int numDevices, unsigned int flags)
 {
    //Debug();
    cudaError_t cuda_res;
    cuda_res = cudaLaunchCooperativeKernelMultiDevice((cudaLaunchParams *)launchParamsList, numDevices, flags);
    return cudaErrorToUPTKError(cuda_res);
}

// UNSUPPORTED
__host__ UPTKError UPTKLaunchHostFunc(UPTKStream_t stream,UPTKHostFn_t fn,void * userData)
{
    cudaError_t cuda_res;
    cuda_res = cudaLaunchHostFunc((cudaStream_t) stream, (cudaHostFn_t) fn, userData);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMalloc(void ** devPtr,size_t size)
{
    cudaError_t cuda_res;
    cuda_res = cudaMalloc(devPtr, size);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMalloc3D(struct UPTKPitchedPtr * pitchedDevPtr,struct UPTKExtent extent)
{
    cudaExtent cuda_extent;
    UPTKExtentTocudaExtent(&extent, &cuda_extent);
    cudaError_t cuda_res;
    cuda_res = cudaMalloc3D((cudaPitchedPtr*) pitchedDevPtr, cuda_extent);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMalloc3DArray(UPTKArray_t *array, const struct UPTKChannelFormatDesc* desc, struct UPTKExtent extent, unsigned int flags)
{
    if (nullptr == desc)
        return UPTKErrorInvalidValue;
    cudaExtent cuda_extent;
    UPTKExtentTocudaExtent(&extent, &cuda_extent);
    cudaChannelFormatDesc cuda_desc;
    UPTKChannelFormatDescTocudaChannelFormatDesc(desc, &cuda_desc);
    cudaError_t cuda_res;
    cuda_res = cudaMalloc3DArray((cudaArray_t*) array, &cuda_desc, cuda_extent, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMallocArray(UPTKArray_t *array, const struct UPTKChannelFormatDesc *desc, size_t width, size_t height, unsigned int flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaMallocArray((cudaArray_t*) array, (cudaChannelFormatDesc *)desc, width, height, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMallocHost(void ** ptr,size_t size)
{
    cudaError_t cuda_res;
    cuda_res = cudaMallocHost(ptr, size);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMallocManaged(void ** devPtr,size_t size,unsigned int flags)
{
    // If the size of the NV environment is 0, the UPTKErrorInvalidValue should be returned theoretically according to the UPTK official documentation.
    // However, in the actual NV environment verification, UPTKSuccess is returned when size is 0, and the returned device pointer is nullptr, if the actual operation of this memory, it will crash.
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

__host__ UPTKError UPTKMallocMipmappedArray(UPTKMipmappedArray_t *mipmappedArray, const struct UPTKChannelFormatDesc* desc, struct UPTKExtent extent, unsigned int numLevels, unsigned int flags)
{
    cudaError_t cuda_res;
    cudaChannelFormatDesc cuda_desc;
    cudaExtent cuda_extent;
    if (desc == nullptr){
        return UPTKErrorInvalidValue;
    }
    UPTKExtentTocudaExtent(&extent, &cuda_extent);
    UPTKChannelFormatDescTocudaChannelFormatDesc(desc, &cuda_desc);
    cuda_res = cudaMallocMipmappedArray((cudaMipmappedArray_t *)mipmappedArray, &cuda_desc, cuda_extent, numLevels, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMallocPitch(void ** devPtr,size_t * pitch,size_t width,size_t height)
{
    cudaError_t cuda_res;
    cuda_res = cudaMallocPitch(devPtr, pitch, width, height);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemAdvise(const void * devPtr,size_t count,enum UPTKMemoryAdvise advice,int device)
{
    cudaMemoryAdvise cuda_advice = UPTKMemoryAdviseTocudaMemoryAdvise(advice);
    cudaError_t cuda_res;
    cuda_res = cudaMemAdvise(devPtr, count, cuda_advice, device);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemGetInfo(size_t * free,size_t * total)
{
    cudaError_t cuda_res;
    cuda_res = cudaMemGetInfo(free, total);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, UPTKStream_t stream)
{
    cudaError_t cuda_res;
    cuda_res = cudaMemPrefetchAsync(devPtr, count, dstDevice, (cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemRangeGetAttribute(void * data,size_t dataSize,enum UPTKMemRangeAttribute attribute,const void * devPtr,size_t count)
{
    cudaMemRangeAttribute cuda_attribute = UPTKMemRangeAttributeTocudaMemRangeAttribute(attribute);
    cudaError_t cuda_res;
    cuda_res = cudaMemRangeGetAttribute(data, dataSize, cuda_attribute, devPtr, count);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemRangeGetAttributes(void ** data,size_t * dataSizes,enum UPTKMemRangeAttribute * attributes,size_t numAttributes,const void * devPtr,size_t count)
{
    cudaMemRangeAttribute cuda_attributes;
    cudaError_t cuda_res;
    cuda_res = cudaMemRangeGetAttributes(data, dataSizes, &cuda_attributes, numAttributes, devPtr, count);
    if ( cuda_res == cudaSuccess )
    {
        //TODO
        *attributes = cudaMemRangeAttributeToUPTKMemRangeAttribute(cuda_attributes);
    }
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpy(void * dst,const void * src,size_t count,enum UPTKMemcpyKind kind)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpy(dst, src, count, cuda_kind);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpy2D(void * dst,size_t dpitch,const void * src,size_t spitch,size_t width,size_t height,enum UPTKMemcpyKind kind)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cuda_kind);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpy2DArrayToArray(UPTKArray_t dst,size_t wOffsetDst,size_t hOffsetDst,UPTKArray_const_t src,size_t wOffsetSrc,size_t hOffsetSrc,size_t width,size_t height,enum UPTKMemcpyKind kind)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpy2DArrayToArray((cudaArray_t) dst, wOffsetDst, hOffsetDst, (cudaArray_t) src, wOffsetSrc, hOffsetSrc, width, height, cuda_kind);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum UPTKMemcpyKind kind, UPTKStream_t stream)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cuda_kind, (cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpy2DAsync_ptsz(void * dst,size_t dpitch,const void * src,size_t spitch,size_t width,size_t height,enum UPTKMemcpyKind kind,UPTKStream_t stream)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cuda_kind, (cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpy2DFromArray(void * dst,size_t dpitch,UPTKArray_const_t src,size_t wOffset,size_t hOffset,size_t width,size_t height,enum UPTKMemcpyKind kind)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpy2DFromArray(dst, dpitch, (cudaArray_t) src, wOffset, hOffset, width, height, cuda_kind);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpy2DFromArrayAsync(void *dst, size_t dpitch, UPTKArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum UPTKMemcpyKind kind, UPTKStream_t stream)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpy2DFromArrayAsync(dst, dpitch, (cudaArray_t) src, wOffset, hOffset, width, height, cuda_kind, (cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpy2DToArray(UPTKArray_t dst,size_t wOffset,size_t hOffset,const void * src,size_t spitch,size_t width,size_t height,enum UPTKMemcpyKind kind)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpy2DToArray((cudaArray_t) dst, wOffset, hOffset, src, spitch, width, height, cuda_kind);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpy2DToArrayAsync(UPTKArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum UPTKMemcpyKind kind, UPTKStream_t stream)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpy2DToArrayAsync((cudaArray_t) dst, wOffset, hOffset, src, spitch, width, height, cuda_kind,(cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpy3D(const struct UPTKMemcpy3DParms * p)
{
    if (nullptr == p)
        return UPTKErrorInvalidValue;
    struct cudaMemcpy3DParms cuda_p;
    UPTKMemcpy3DParmsTocudaMemcpy3DParms(p, &cuda_p);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpy3D(&cuda_p);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpy3DAsync(const struct UPTKMemcpy3DParms *p, UPTKStream_t stream)
{
    if (nullptr == p)
        return UPTKErrorInvalidValue;
    struct cudaMemcpy3DParms cuda_p;
    UPTKMemcpy3DParmsTocudaMemcpy3DParms(p, &cuda_p);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpy3DAsync(&cuda_p, (cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpy3DPeerAsync(const struct UPTKMemcpy3DPeerParms *p, UPTKStream_t stream)
{
    if (p == nullptr)
    {
        return UPTKErrorInvalidValue;
    }
    cudaError_t cuda_res;
    cudaMemcpy3DPeerParms h_p;
    UPTKMemcpy3DPeerParmsTocudaMemcpy3DPeerParms(p, &h_p);
    cuda_res = cudaMemcpy3DPeerAsync(&h_p, (cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset, enum UPTKMemcpyKind kind)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpyFromSymbol(dst, symbol, count, offset, cuda_kind);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, enum UPTKMemcpyKind kind)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpyToSymbol(symbol, src, count, offset, cuda_kind);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset, enum UPTKMemcpyKind kind, UPTKStream_t stream)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpyToSymbolAsync(symbol, src, count, offset, cuda_kind, (cudaStream_t)stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpy3DPeer(const struct UPTKMemcpy3DPeerParms * p)
{
    if (p == nullptr)
    {
        return UPTKErrorInvalidValue;
    }
    cudaError_t cuda_res;
    cudaMemcpy3DPeerParms h_p;
    UPTKMemcpy3DPeerParmsTocudaMemcpy3DPeerParms(p, &h_p);
    cuda_res = cudaMemcpy3DPeer(&h_p);
    return cudaErrorToUPTKError(cuda_res);
}


__host__ UPTKError UPTKMemcpyArrayToArray(UPTKArray_t dst,size_t wOffsetDst,size_t hOffsetDst,UPTKArray_const_t src,size_t wOffsetSrc,size_t hOffsetSrc,size_t count,enum UPTKMemcpyKind kind)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpyArrayToArray((cudaArray_t) dst, wOffsetDst, hOffsetDst, (cudaArray_t) src, wOffsetSrc, hOffsetSrc, count, cuda_kind);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpyAsync(void *dst, const void *src, size_t count, enum UPTKMemcpyKind kind, UPTKStream_t stream)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpyAsync(dst, src, count, cuda_kind, (cudaStream_t)stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpyFromArray(void * dst,UPTKArray_const_t src,size_t wOffset,size_t hOffset,size_t count,enum UPTKMemcpyKind kind)
{
    if (src == nullptr)
    {
        return UPTKErrorInvalidResourceHandle;
    }
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpyFromArray(dst, (cudaArray_t) src, wOffset, hOffset, count, cuda_kind);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpyFromArrayAsync(void *dst, UPTKArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum UPTKMemcpyKind kind, UPTKStream_t stream)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpyFromArrayAsync(dst, (cudaArray_t) src, wOffset, hOffset, count, cuda_kind, (cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset, enum UPTKMemcpyKind kind, UPTKStream_t stream)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpyFromSymbolAsync(dst, symbol, count, offset, cuda_kind, (cudaStream_t)stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpyPeer(void * dst,int dstDevice,const void * src,int srcDevice,size_t count)
{
    cudaError_t cuda_res;
    cuda_res = cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, UPTKStream_t stream)
{
    cudaError_t cuda_res;
    cuda_res = cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, (cudaStream_t)stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpyToArray(UPTKArray_t dst,size_t wOffset,size_t hOffset,const void * src,size_t count,enum UPTKMemcpyKind kind)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpyToArray((cudaArray *) dst, wOffset, hOffset, src, count, cuda_kind);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemcpyToArrayAsync(UPTKArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum UPTKMemcpyKind kind, UPTKStream_t stream)
{
    cudaMemcpyKind cuda_kind = UPTKMemcpyKindTocudaMemcpyKind(kind);
    cudaError_t cuda_res;
    cuda_res = cudaMemcpyToArrayAsync((cudaArray_t) dst, wOffset, hOffset, src, count, cuda_kind, (cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemset(void * devPtr,int value,size_t count)
{
    cudaError_t cuda_res;
    cuda_res = cudaMemset(devPtr, value, count);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemset2D(void * devPtr,size_t pitch,int value,size_t width,size_t height)
{
    cudaError_t cuda_res;
    cuda_res = cudaMemset2D(devPtr, pitch, value, width, height);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemset2DAsync(void * devPtr,size_t pitch,int value,size_t width,size_t height,UPTKStream_t stream)
{
    cudaError_t cuda_res;
    cuda_res = cudaMemset2DAsync(devPtr, pitch, value, width, height, (cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemset2DAsync_ptsz(void * devPtr,size_t pitch,int value,size_t width,size_t height,UPTKStream_t stream)
{
    cudaError_t cuda_res;
    cuda_res = cudaMemset2DAsync(devPtr, pitch, value, width, height, (cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemset3D(struct UPTKPitchedPtr pitchedDevPtr,int value,struct UPTKExtent extent)
{
    cudaPitchedPtr cuda_pitchedDevPtr;
    UPTKPitchedPtrTocudaPitchedPtr(&pitchedDevPtr, &cuda_pitchedDevPtr);
    cudaExtent cuda_extent;
    UPTKExtentTocudaExtent(&extent, &cuda_extent);
    cudaError_t cuda_res;
    cuda_res = cudaMemset3D(cuda_pitchedDevPtr, value, cuda_extent);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemset3DAsync(struct UPTKPitchedPtr pitchedDevPtr, int value, struct UPTKExtent extent, UPTKStream_t stream)
{
    cudaPitchedPtr cuda_pitchedDevPtr;
    UPTKPitchedPtrTocudaPitchedPtr(&pitchedDevPtr, &cuda_pitchedDevPtr);
    cudaExtent cuda_extent;
    UPTKExtentTocudaExtent(&extent, &cuda_extent);
    cudaError_t cuda_res;
    cuda_res = cudaMemset3DAsync(cuda_pitchedDevPtr, value, cuda_extent , (cudaStream_t)stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKMemsetAsync(void *devPtr, int value, size_t count, UPTKStream_t stream)
{
    cudaError_t cuda_res;
    cuda_res = cudaMemsetAsync(devPtr, value, count, (cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks,const void * func,int blockSize,size_t dynamicSMemSize)
{
    cudaError_t cuda_res;
    cuda_res = cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks,const void * func,int blockSize,size_t dynamicSMemSize,unsigned int flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKPeekAtLastError(void)
{
    cudaError_t cuda_res;
    cuda_res = cudaPeekAtLastError();
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKPointerGetAttributes(struct UPTKPointerAttributes * attributes,const void * ptr)
{
    if(nullptr == attributes)
    {
        return UPTKErrorInvalidValue;
    }

    memset(attributes, 0, sizeof(UPTKPointerAttributes));
    
    cudaPointerAttributes cuda_attributes;
    memset(&cuda_attributes, 0, sizeof(cudaPointerAttributes));

    cudaError_t cuda_res;
    cuda_res = cudaPointerGetAttributes(&cuda_attributes, ptr);
    if ( cuda_res == cudaSuccess )
    {
        cudaPointerAttributesToUPTKPointerAttributes(&cuda_attributes, attributes);
    }
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKProfilerStart(void)
{
    return UPTKSuccess;
}

__host__ UPTKError UPTKProfilerStop(void)
{
    return UPTKSuccess;
}

__host__ UPTKError UPTKRuntimeGetVersion(int * runtimeVersion)
{
    cudaError_t cuda_res;
    cuda_res = cudaRuntimeGetVersion(runtimeVersion);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKSetDevice(int device)
{
    cudaError_t cuda_res;
    cuda_res = cudaSetDevice(device);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKSetDeviceFlags(unsigned int flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaSetDeviceFlags(flags);
    return cudaErrorToUPTKError(cuda_res);
}

// UNSUPPORTED
__host__ UPTKError UPTKSetDoubleForDevice(double * d)
{
    Debug();
    return UPTKErrorNotSupported;
}

// UNSUPPORTED
__host__ UPTKError UPTKSetDoubleForHost(double * d)
{
    Debug();
    return UPTKErrorNotSupported;
}

__host__ UPTKError UPTKSetValidDevices(int * device_arr,int len)
{
    cudaError_t cuda_res;
    cuda_res = cudaSetValidDevices (device_arr, len);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKSignalExternalSemaphoresAsync(const UPTKExternalSemaphore_t *extSemArray, const struct UPTKExternalSemaphoreSignalParams *paramsArray, unsigned int numExtSems, UPTKStream_t stream)
{
    cudaError_t cuda_res;
    if (paramsArray == nullptr){
        return UPTKErrorInvalidValue;
    }
    cudaExternalSemaphoreSignalParams *cuda_paramsArray = (cudaExternalSemaphoreSignalParams *)malloc(sizeof(cudaExternalSemaphoreSignalParams) * numExtSems);
    for (int i = 0; i < numExtSems; i++){
        cuda_paramsArray[i] = UPTKExternalSemaphoreSignalParamsTocudaExternalSemaphoreSignalParams(paramsArray[i]);
    }
    cuda_res = cudaSignalExternalSemaphoresAsync((cudaExternalSemaphore_t *)extSemArray, cuda_paramsArray, numExtSems, (cudaStream_t) stream);
    free(cuda_paramsArray);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKStreamAddCallback(UPTKStream_t stream,UPTKStreamCallback_t callback,void * userData,unsigned int flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaStreamAddCallback((cudaStream_t) stream, (cudaStreamCallback_t) callback, userData, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKStreamAttachMemAsync(UPTKStream_t stream, void *devPtr, size_t length, unsigned int flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaStreamAttachMemAsync((cudaStream_t) stream, devPtr, length, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKStreamBeginCapture(UPTKStream_t stream,enum UPTKStreamCaptureMode mode)
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

__host__ UPTKError UPTKStreamCreate(UPTKStream_t * pStream)
{
    cudaError_t cuda_res;
    cuda_res = cudaStreamCreate((cudaStream_t *) pStream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKStreamCreateWithFlags(UPTKStream_t * pStream,unsigned int flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaStreamCreateWithFlags((cudaStream_t *) pStream, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKStreamCreateWithPriority(UPTKStream_t * pStream,unsigned int flags,int priority)
{
    cudaError_t cuda_res;
    cuda_res = cudaStreamCreateWithPriority((cudaStream_t *) pStream, flags, priority);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKStreamDestroy(UPTKStream_t stream)
{
    cudaError_t cuda_res;
    cuda_res = cudaStreamDestroy((cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKStreamEndCapture(UPTKStream_t stream,UPTKGraph_t * pGraph)
{
    cudaError_t cuda_res;
    cuda_res = cudaStreamEndCapture((cudaStream_t) stream, (cudaGraph_t *) pGraph);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKStreamGetFlags(UPTKStream_t hStream,unsigned int * flags)
{
    // UPTK is ok for null stream
    if (nullptr == hStream) {
        *flags = 0;
        return UPTKSuccess;
    }
    cudaError_t cuda_res;
    cuda_res = cudaStreamGetFlags((cudaStream_t) hStream, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKStreamGetPriority(UPTKStream_t hStream,int * priority)
{
    cudaError_t cuda_res;
    cuda_res = cudaStreamGetPriority((cudaStream_t) hStream, priority);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKStreamIsCapturing(UPTKStream_t stream,enum UPTKStreamCaptureStatus * pCaptureStatus)
{
    if (nullptr == pCaptureStatus)
        return UPTKErrorInvalidValue;
    cudaError_t cuda_res;
    cuda_res = cudaStreamIsCapturing((cudaStream_t) stream, (cudaStreamCaptureStatus *)pCaptureStatus);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKStreamQuery(UPTKStream_t stream)
{
    cudaError_t cuda_res;
    cuda_res = cudaStreamQuery((cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKStreamSynchronize(UPTKStream_t stream)
{
    cudaError_t cuda_res;
    cuda_res = cudaStreamSynchronize((cudaStream_t) stream);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKStreamWaitEvent(UPTKStream_t stream,UPTKEvent_t event,unsigned int flags)
{
    cudaError_t cuda_res;
    cuda_res = cudaStreamWaitEvent((cudaStream_t) stream, (cudaEvent_t) event, flags);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKThreadExchangeStreamCaptureMode(enum UPTKStreamCaptureMode * mode)
{
    cudaStreamCaptureMode cuda_mode;
    if (mode != nullptr) {
        cuda_mode = UPTKStreamCaptureModeTocudaStreamCaptureMode(*mode);
    }else{
        return UPTKErrorInvalidValue;
    }
    cudaError_t cuda_res;
    cuda_res = cudaThreadExchangeStreamCaptureMode(&cuda_mode);
    if ( cuda_res == cudaSuccess )
    {
        *mode = cudaStreamCaptureModeToUPTKStreamCaptureMode(cuda_mode);
    }
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKThreadExit(void)
{
    cudaError_t cuda_res;
    cuda_res = cudaDeviceReset();
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKThreadGetCacheConfig(enum UPTKFuncCache * pCacheConfig)
{
    if (nullptr == pCacheConfig)
        return UPTKErrorInvalidValue;
    cudaFuncCache cuda_cacheConfig;
    cudaError_t cuda_res;
    cuda_res = cudaDeviceGetCacheConfig(&cuda_cacheConfig);
    if ( cuda_res == cudaSuccess )
    {
        *pCacheConfig = cudaFuncCacheToUPTKFuncCache(cuda_cacheConfig);
    }
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKThreadGetLimit(size_t * pValue,enum UPTKLimit limit)
{
    enum cudaLimit cuda_limit = UPTKLimitTocudaLimit(limit);
    cudaError_t cuda_res;
    cuda_res = cudaThreadGetLimit(pValue, cuda_limit);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKThreadSetCacheConfig(enum UPTKFuncCache cacheConfig)
{
    cudaFuncCache cuda_cacheConfig = UPTKFuncCacheTocudaFuncCache(cacheConfig);
    cudaError_t cuda_res;
    cuda_res = cudaDeviceSetCacheConfig(cuda_cacheConfig);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKThreadSetLimit(enum UPTKLimit limit,size_t value)
{
    enum cudaLimit cuda_limit = UPTKLimitTocudaLimit(limit);
    cudaError_t cuda_res;
    cuda_res = cudaThreadSetLimit(cuda_limit, value);
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKThreadSynchronize(void)
{
    cudaError_t cuda_res;
    cuda_res = cudaDeviceSynchronize();
    return cudaErrorToUPTKError(cuda_res);
}

__host__ UPTKError UPTKWaitExternalSemaphoresAsync(const UPTKExternalSemaphore_t *extSemArray, const struct UPTKExternalSemaphoreWaitParams *paramsArray, unsigned int numExtSems, UPTKStream_t stream)
{
    cudaError_t cuda_res;
    if (paramsArray == nullptr){
        return UPTKErrorInvalidValue;
    }
    cudaExternalSemaphoreWaitParams *cuda_paramsArray = (cudaExternalSemaphoreWaitParams *)malloc(sizeof(cudaExternalSemaphoreWaitParams) * numExtSems);
    for (int i = 0; i < numExtSems; i++){
        cuda_paramsArray[i] = UPTKExternalSemaphoreWaitParamsTocudaExternalSemaphoreWaitParams(paramsArray[i]);
    }
    cuda_res = cudaWaitExternalSemaphoresAsync((cudaExternalSemaphore_t *) extSemArray, cuda_paramsArray, numExtSems, (cudaStream_t) stream);
    free(cuda_paramsArray);
    return cudaErrorToUPTKError(cuda_res);
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */

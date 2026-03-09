#include "driver.hpp"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

UPTKError  UPInit(unsigned int Flags)
{
    CUresult cuda_res;
    cuda_res = cuInit(Flags);
    return CUresultToUPTKError(cuda_res);
}

UPTKError  UPMemGetInfo_v2(size_t * free, size_t * total)
{
    CUresult cu_res;
    cu_res = cuMemGetInfo_v2(free, total);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags)
{
    CUresult cu_res;
    cu_res = cuMemHostAlloc(pp, bytesize, Flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPCtxSetCurrent(UPTKcontext ctx)
{
    CUresult cu_res;
    cu_res = cuCtxSetCurrent((CUcontext) ctx);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPStreamSynchronize(UPTKStream_t * hStream)
{
    CUresult cu_res;
    cu_res = cuStreamSynchronize((CUstream) hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPMemcpyDtoHAsync_v2(void * dstHost, UPTKdeviceptr srcDevice,size_t ByteCount,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemcpyDtoHAsync_v2(dstHost, (CUdeviceptr)srcDevice, ByteCount, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPDeviceGet(UPTKdevice * device,int ordinal)
{
    CUresult cu_res;
    cu_res = cuDeviceGet((CUdevice *)device, ordinal);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPModuleLoad(UPTKmodule * module,const char * fname)
{
    CUresult cuda_res;
    cuda_res = cuModuleLoad((CUmodule *)module, fname);
    return CUresultToUPTKError(cuda_res);
}

UPTKError  UPModuleLoadData(UPTKmodule * module,const void * image)
{
    CUresult cuda_res;
    cuda_res = cuModuleLoadData((CUmodule *)module, image);
    return CUresultToUPTKError(cuda_res);
}

UPTKError  UPModuleUnload(UPTKmodule hmod)
{
    CUresult cuda_res;
    cuda_res = cuModuleUnload((CUmodule) hmod);
    return CUresultToUPTKError(cuda_res);
}

UPTKError  UPModuleGetFunction(UPTKfunction * hfunc,UPTKmodule hmod,const char * name)
{
    CUresult cuda_res;
    cuda_res = cuModuleGetFunction((CUfunction *)hfunc, (CUmodule) hmod, name);
    return CUresultToUPTKError(cuda_res);
}

UPTKError  UPModuleGetGlobal(UPTKdeviceptr * dptr, size_t * bytes,UPTKmodule hmod,const char * name)
{
    CUresult cuda_res;
    cuda_res = cuModuleGetGlobal((CUdeviceptr *)dptr, bytes, (CUmodule) hmod, name);
    return CUresultToUPTKError(cuda_res);
}

UPTKError UPLinkCreate(unsigned int numOptions,UPTKjit_option * options,void ** optionValues,UPTKlinkState * stateOut)
{
    CUresult nvrtc_res;
    CUjit_option* nvrtc_options = (CUjit_option*)malloc(sizeof(CUjit_option) * numOptions);
    /*if (nullptr != options) {
        if (nullptr == nvrtc_options) {
          return UPTK_ERROR_OUT_OF_MEMORY;
        }
        for (int i = 0; i < numOptions; i++) {
            nvrtc_options[i] = UPTKjit_optionTonvrtcJIT_option(options[i]);
        }
    }*/
    //nvrtc_res = cuLinkCreate(numOptions, nvrtc_options, optionValues, (CUlinkState* )stateOut);
    nvrtc_res = cuLinkCreate(numOptions, nvrtc_options, optionValues, (CUlinkState* )stateOut);
    //free(nvrtc_options);
    return CUresultToUPTKError(nvrtc_res);
}

UPTKError  UPLinkDestroy(UPTKlinkState state)
{
    CUresult nvrtc_res;
    nvrtc_res = cuLinkDestroy((CUlinkState) state);
    return CUresultToUPTKError(nvrtc_res);
}

UPTKError  UPMemsetD8Async(UPTKdeviceptr dstDevice,unsigned char uc,size_t N, UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemsetD8Async((CUdeviceptr)dstDevice, uc, N, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPMemsetD32Async(UPTKdeviceptr dstDevice,unsigned int ui,size_t N, UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemsetD32Async((CUdeviceptr)dstDevice, (int) ui, N, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPMemFree(UPTKdeviceptr dstDevice)
{
    CUresult cu_res;
    cu_res = cuMemFree((CUdeviceptr)dstDevice);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPDriverGetVersion(int * driverVersion)
{
    CUresult cu_res;
    cu_res = cuDriverGetVersion(driverVersion);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPModuleGetGlobal_v2(UPTKdeviceptr * dptr, size_t * bytes, UPTKmodule hmod,const char * name)
{
    CUresult cu_res;
    cu_res = cuModuleGetGlobal_v2((CUdeviceptr *)dptr, bytes, (CUmodule)hmod, name);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPMemFreeHost(void *p)
{
    CUresult cu_res;
    cu_res = cuMemFreeHost(p);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPMemcpyHtoDAsync_v2(UPTKdeviceptr dstDevice,const void * srcHost,size_t ByteCount,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemcpyHtoDAsync_v2((CUdeviceptr)dstDevice, srcHost, ByteCount, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPEventDestroy(UPTKEvent_t hEvent)
{
    CUresult cu_res;
    cu_res = cuEventDestroy((CUevent) hEvent);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPDevicePrimaryCtxRelease(UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuDevicePrimaryCtxRelease((CUdevice) dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPMemAllocHost(void **pp, size_t bytesize)
{
    CUresult cu_res;
    cu_res = cuMemAllocHost(pp, bytesize);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPEventSynchronize(UPTKEvent_t hEvent)
{
    CUresult cu_res;
    cu_res = cuEventSynchronize((cudaEvent_t) hEvent);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPEventRecord(UPTKEvent_t hEvent,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuEventRecord((cudaEvent_t) hEvent, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPMemsetD16Async(UPTKdeviceptr dstDevice,unsigned short us,size_t N,UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuMemsetD16Async((CUdeviceptr)dstDevice, us, N, (CUstream)hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPStreamCreate(UPTKStream_t * phStream,unsigned int Flags)
{
    CUresult cu_res;
    cu_res = cuStreamCreate((CUstream *)phStream, Flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPCtxGetCurrent(UPTKcontext * pctx)
{
    CUresult cu_res;
    cu_res = cuCtxGetCurrent((CUcontext *) pctx);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPStreamDestroy(UPTKStream_t hStream)
{
    CUresult cu_res;
    cu_res = cuStreamDestroy((CUstream) hStream);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPMemcpyHtoD_v2(UPTKdeviceptr dstDevice,const void * srcHost,size_t ByteCount)
{
    CUresult cu_res;
    cu_res = cuMemcpyHtoD_v2((CUdeviceptr) dstDevice, const_cast<void *>(srcHost), ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPModuleLoadDataEx(UPTKmodule * module,const void * image,unsigned int numOptions, UPTKjit_option * options,void ** optionValues)
{
    CUresult cu_res;
    cu_res = cuModuleLoadDataEx((CUmodule *) module, image, numOptions, (CUjit_option *)options, optionValues);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPMemcpyDtoH_v2(void * dstHost, UPTKdeviceptr srcDevice, size_t ByteCount)
{
    CUresult cu_res;
    cu_res = cuMemcpyDtoH_v2(dstHost, (CUdeviceptr)srcDevice, ByteCount);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPEventCreate(UPTKEvent_t * phEvent,unsigned int Flags)
{
    CUresult cu_res;
    cu_res = cuEventCreate((CUevent *) phEvent, Flags);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPDevicePrimaryCtxRetain(UPTKcontext * pctx,UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuDevicePrimaryCtxRetain((CUcontext *)pctx, (CUdevice) dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPMemAlloc_v2(UPTKdeviceptr * dptr,size_t bytesize)
{
    CUresult cu_res;
    cu_res = cuMemAlloc_v2((CUdeviceptr *)dptr, bytesize);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPDeviceTotalMem_v2(size_t * bytes, UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuDeviceTotalMem_v2(bytes, (CUdevice) dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError UPDeviceGetName(char * name,int len,UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuDeviceGetName(name, len, (CUdevice) dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPDeviceGetCount(int * count)
{
    CUresult cu_res;
    cu_res = cuDeviceGetCount(count);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPEventElapsedTime(float * pMilliseconds, UPTKEvent_t hStart, UPTKEvent_t hEnd)
{
    CUresult cu_res;
    cu_res = cuEventElapsedTime(pMilliseconds, (CUevent) hStart, (CUevent) hEnd);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPDeviceGetAttribute(int * pi, UPTKDeviceAttr attrib,UPTKdevice dev)
{
    CUresult cu_res;
    cu_res = cuDeviceGetAttribute(pi, (CUdevice_attribute)attrib, (CUdevice) dev);
    return CUresultToUPTKError(cu_res);
}

UPTKError  UPLaunchKernel(UPTKfunction f,unsigned int gridDimX,unsigned int gridDimY,unsigned int gridDimZ,unsigned int blockDimX,unsigned int blockDimY,unsigned int blockDimZ,unsigned int sharedMemBytes,UPTKStream_t hStream,void ** kernelParams,void ** extra)
{
    CUresult cu_res;
    cu_res = cuLaunchKernel((CUfunction)f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, (CUstream)hStream, kernelParams, extra);
    return CUresultToUPTKError(cu_res);
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */

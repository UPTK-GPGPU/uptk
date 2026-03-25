#include "fft.hpp"

UPTKfftResult UPTKFFTAPI UPTKfftXtClearCallback(UPTKfftHandle plan, UPTKfftXtCallbackType cbType)
{
    cufftXtCallbackType cuda_cbtype = UPTKfftXtCallbackTypeTocufftXtCallbackType(cbType);
    cufftResult cuda_res;
    cuda_res = cufftXtClearCallback((cufftHandle)plan, cuda_cbtype);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftXtSetCallback(UPTKfftHandle plan, void **callback_routine, UPTKfftXtCallbackType cbType, void **caller_info)
{
    cufftXtCallbackType cuda_cbtype = UPTKfftXtCallbackTypeTocufftXtCallbackType(cbType);
    cufftResult cuda_res;
    cuda_res = cufftXtSetCallback((cufftHandle)plan, callback_routine, cuda_cbtype, caller_info);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftXtSetCallbackSharedSize(UPTKfftHandle plan, UPTKfftXtCallbackType cbType, size_t sharedSize)
{
    cufftXtCallbackType cuda_cbtype = UPTKfftXtCallbackTypeTocufftXtCallbackType(cbType);
    cufftResult cuda_res;
    cuda_res = cufftXtSetCallbackSharedSize((cufftHandle)plan, cuda_cbtype, sharedSize);
    return cufftResultToUPTKfftResult(cuda_res);
}

// UNSUPPORTED
UPTKfftResult UPTKFFTAPI UPTKfftXtExec(UPTKfftHandle plan, void *input, void *output, int direction)
{
    return cufftResultToUPTKfftResult(cufftXtExec((cufftHandle)plan, input, output, direction));
}

UPTKfftResult UPTKFFTAPI UPTKfftXtExecDescriptor(UPTKfftHandle plan, UPTKLibXtDesc *input, UPTKLibXtDesc *output, int direction)
{
    return cufftResultToUPTKfftResult(cufftXtExecDescriptor((cufftHandle)plan, (cudaLibXtDesc_t *)input, (cudaLibXtDesc_t *)output, direction));
}

UPTKfftResult UPTKFFTAPI UPTKfftXtExecDescriptorC2C(UPTKfftHandle plan, UPTKLibXtDesc *input, UPTKLibXtDesc *output, int direction)
{
    return cufftResultToUPTKfftResult(cufftXtExecDescriptorC2C((cufftHandle)plan, (cudaLibXtDesc_t *)input, (cudaLibXtDesc_t *)output, direction));
}

UPTKfftResult UPTKFFTAPI UPTKfftXtExecDescriptorC2R(UPTKfftHandle plan, UPTKLibXtDesc *input, UPTKLibXtDesc *output)
{
    return cufftResultToUPTKfftResult(cufftXtExecDescriptorC2R((cufftHandle)plan, (cudaLibXtDesc_t *)input, (cudaLibXtDesc_t *)output));
}

UPTKfftResult UPTKFFTAPI UPTKfftXtExecDescriptorD2Z(UPTKfftHandle plan, UPTKLibXtDesc *input, UPTKLibXtDesc *output)
{
    return cufftResultToUPTKfftResult(cufftXtExecDescriptorD2Z((cufftHandle)plan, (cudaLibXtDesc_t *)input, (cudaLibXtDesc_t *)output));
}

UPTKfftResult UPTKFFTAPI UPTKfftXtExecDescriptorR2C(UPTKfftHandle plan, UPTKLibXtDesc *input, UPTKLibXtDesc *output)
{
    return cufftResultToUPTKfftResult(cufftXtExecDescriptorR2C((cufftHandle)plan, (cudaLibXtDesc_t *)input, (cudaLibXtDesc_t *)output));
}

UPTKfftResult UPTKFFTAPI UPTKfftXtExecDescriptorZ2D(UPTKfftHandle plan, UPTKLibXtDesc *input, UPTKLibXtDesc *output)
{
    return cufftResultToUPTKfftResult(cufftXtExecDescriptorZ2D((cufftHandle)plan, (cudaLibXtDesc_t *)input, (cudaLibXtDesc_t *)output));
}

UPTKfftResult UPTKFFTAPI UPTKfftXtExecDescriptorZ2Z(UPTKfftHandle plan, UPTKLibXtDesc *input, UPTKLibXtDesc *output, int direction)
{
    return cufftResultToUPTKfftResult(cufftXtExecDescriptorZ2Z((cufftHandle)plan, (cudaLibXtDesc_t *)input, (cudaLibXtDesc_t *)output, direction));
}

UPTKfftResult UPTKFFTAPI UPTKfftXtFree(UPTKLibXtDesc *descriptor)
{
    return cufftResultToUPTKfftResult(cufftXtFree((cudaLibXtDesc_t *)descriptor));
}

UPTKfftResult UPTKFFTAPI UPTKfftXtGetSizeMany(UPTKfftHandle plan, int rank, long long int *n, long long int *inembed, long long int istride, long long int idist, UPTKDataType inputtype, long long int *onembed, long long int ostride, long long int odist, UPTKDataType outputtype, long long int batch, size_t *workSize, UPTKDataType executiontype)
{
    return cufftResultToUPTKfftResult(cufftXtGetSizeMany(
        (cufftHandle)plan, rank,n, inembed,
        istride, idist, 
        UPTKDataTypeTocudaDataType(inputtype), onembed,  ostride, 
        odist, UPTKDataTypeTocudaDataType(outputtype), 
        batch, workSize,
        UPTKDataTypeTocudaDataType(executiontype)));

}

UPTKfftResult UPTKFFTAPI UPTKfftXtMakePlanMany(UPTKfftHandle plan, int rank, long long int *n, long long int *inembed, long long int istride, long long int idist, UPTKDataType inputtype, long long int *onembed, long long int ostride, long long int odist, UPTKDataType outputtype, long long int batch, size_t *workSize, UPTKDataType executiontype)
{
    return cufftResultToUPTKfftResult(cufftXtMakePlanMany(
        (cufftHandle)plan, rank,n, inembed,
        istride, idist, 
        UPTKDataTypeTocudaDataType(inputtype), onembed,  ostride, 
        odist, UPTKDataTypeTocudaDataType(outputtype), 
        batch, workSize,
        UPTKDataTypeTocudaDataType(executiontype)));
}

UPTKfftResult UPTKFFTAPI UPTKfftXtMalloc(UPTKfftHandle plan, UPTKLibXtDesc **descriptor, UPTKfftXtSubFormat format)
{
    return cufftResultToUPTKfftResult(cufftXtMalloc((cufftHandle)plan, (cudaLibXtDesc_t **)descriptor, UPTKfftXtSubFormatTocuftXtSubFormat(format)));
}

UPTKfftResult UPTKFFTAPI UPTKfftXtMemcpy(UPTKfftHandle plan, void *dstPointer, void *srcPointer, UPTKfftXtCopyType type)
{
    return cufftResultToUPTKfftResult(cufftXtMemcpy((cufftHandle)plan, dstPointer, srcPointer, UPTKfftXtCopyTypeTocufftXtCopyType(type)));
}

UPTKfftResult UPTKFFTAPI UPTKfftXtSetGPUs(UPTKfftHandle handle, int nGPUs, int *whichGPUs)
{
    return cufftResultToUPTKfftResult(cufftXtSetGPUs((cufftHandle)handle, nGPUs, whichGPUs));
}

//unsupport
UPTKfftResult UPTKFFTAPI UPTKfftXtQueryPlan(UPTKfftHandle plan, void *queryStruct, UPTKfftXtQueryType queryType)
{
    return cufftResultToUPTKfftResult(cufftXtQueryPlan((cufftHandle)plan, queryStruct, UPTKfftXtQueryTypeTohcufftXtQueryType(queryType))); 
}

//unsupport
UPTKfftResult UPTKFFTAPI UPTKfftXtSetWorkArea(UPTKfftHandle plan, void **workArea)
{
    return cufftResultToUPTKfftResult(cufftXtSetWorkArea((cufftHandle)plan,workArea)); 
}

//unsupport 
UPTKfftResult UPTKFFTAPI UPTKfftXtSetWorkAreaPolicy(UPTKfftHandle plan, UPTKfftXtWorkAreaPolicy policy, size_t *workSize)
{
    return cufftResultToUPTKfftResult(cufftXtSetWorkAreaPolicy((cufftHandle)plan, UPTKfftXtWorkAreaPolicyTocufftXtWorkAreaPolicy(policy), workSize)); 
}

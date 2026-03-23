#include "fft.hpp"

#if defined(__cplusplus)
extern "C"
{
#endif /* __cplusplus */

UPTKfftResult UPTKFFTAPI UPTKfftCreate(UPTKfftHandle *handle)
{
    cufftResult cuda_res;
    cuda_res = cufftCreate((cufftHandle *)handle);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftDestroy(UPTKfftHandle plan)
{
    cufftResult cuda_res;
    cuda_res = cufftDestroy((cufftHandle)plan);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftEstimate1d(int nx, UPTKfftType type, int batch, size_t *workSize)
{
    cufftType cuda_type = UPTKfftTypeTocufftType(type);
    cufftResult cuda_res;
    cuda_res = cufftEstimate1d(nx, cuda_type, batch, workSize);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftEstimate2d(int nx, int ny, UPTKfftType type, size_t *workSize)
{
    cufftType cuda_type = UPTKfftTypeTocufftType(type);
    cufftResult cuda_res;
    cuda_res = cufftEstimate2d(nx, ny, cuda_type, workSize);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftEstimate3d(int nx, int ny, int nz, UPTKfftType type, size_t *workSize)
{
    cufftType cuda_type = UPTKfftTypeTocufftType(type);
    cufftResult cuda_res;
    cuda_res = cufftEstimate3d(nx, ny, nz, cuda_type, workSize);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftEstimateMany(int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, UPTKfftType type, int batch, size_t *workSize)
{
    cufftType cuda_type = UPTKfftTypeTocufftType(type);
    cufftResult cuda_res;
    cuda_res = cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, cuda_type, batch, workSize);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftExecC2C(UPTKfftHandle plan, UPTKfftComplex *idata, UPTKfftComplex *odata, int direction)
{
    cufftResult cuda_res;
    cuda_res = cufftExecC2C((cufftHandle)plan, (cufftComplex *)idata, (cufftComplex *)odata, direction);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftExecC2R(UPTKfftHandle plan, UPTKfftComplex *idata, UPTKfftReal *odata)
{
    cufftResult cuda_res;
    cuda_res = cufftExecC2R((cufftHandle)plan, (cufftComplex *)idata, (cufftReal *)odata);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftExecD2Z(UPTKfftHandle plan, UPTKfftDoubleReal *idata, UPTKfftDoubleComplex *odata)
{
    cufftResult cuda_res;
    cuda_res = cufftExecD2Z((cufftHandle)plan, (cufftDoubleReal *)idata, (cufftDoubleComplex *)odata);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftExecR2C(UPTKfftHandle plan, UPTKfftReal *idata, UPTKfftComplex *odata)
{
    cufftResult cuda_res;
    cuda_res = cufftExecR2C((cufftHandle)plan, (cufftReal *)idata, (cufftComplex *)odata);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftExecZ2D(UPTKfftHandle plan, UPTKfftDoubleComplex *idata, UPTKfftDoubleReal *odata)
{
    cufftResult cuda_res;
    cuda_res = cufftExecZ2D((cufftHandle)plan, (cufftDoubleComplex *)idata, (cufftDoubleReal *)odata);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftExecZ2Z(UPTKfftHandle plan, UPTKfftDoubleComplex *idata, UPTKfftDoubleComplex *odata, int direction)
{
    cufftResult cuda_res;
    cuda_res = cufftExecZ2Z((cufftHandle)plan, (cufftDoubleComplex *)idata, (cufftDoubleComplex *)odata, direction);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftGetProperty(UPTKlibraryPropertyType type, int *value)
{
    if (nullptr == value)
    {
        return UPTKFFT_INVALID_VALUE;
    }

    switch (type)
    {
    case UPTK_MAJOR_VERSION:
        *value = UPTKFFT_VER_MAJOR;
        break;
    case UPTK_MINOR_VERSION:
        *value = UPTKFFT_VER_MINOR;
        break;
    case UPTK_PATCH_LEVEL:
        *value = UPTKFFT_VER_PATCH;
        break;
    default:
        return UPTKFFT_INVALID_TYPE;
    }

    return UPTKFFT_SUCCESS;
}

UPTKfftResult UPTKFFTAPI UPTKfftGetSize(UPTKfftHandle handle, size_t *workSize)
{
    cufftResult cuda_res;
    cuda_res = cufftGetSize((cufftHandle)handle, workSize);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftGetSize1d(UPTKfftHandle handle, int nx, UPTKfftType type, int batch, size_t *workSize)
{
    cufftType cuda_type = UPTKfftTypeTocufftType(type);
    cufftResult cuda_res;
    cuda_res = cufftGetSize1d((cufftHandle)handle, nx, cuda_type, batch, workSize);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftGetSize2d(UPTKfftHandle handle, int nx, int ny, UPTKfftType type, size_t *workSize)
{
    cufftType cuda_type = UPTKfftTypeTocufftType(type);
    cufftResult cuda_res;
    cuda_res = cufftGetSize2d((cufftHandle)handle, nx, ny, cuda_type, workSize);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftGetSize3d(UPTKfftHandle handle, int nx, int ny, int nz, UPTKfftType type, size_t *workSize)
{
    cufftType cuda_type = UPTKfftTypeTocufftType(type);
    cufftResult cuda_res;
    cuda_res = cufftGetSize3d((cufftHandle)handle, nx, ny, nz, cuda_type, workSize);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftGetSizeMany(UPTKfftHandle handle, int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, UPTKfftType type, int batch, size_t *workArea)
{
    cufftType cuda_type = UPTKfftTypeTocufftType(type);
    cufftResult cuda_res;
    cuda_res = cufftGetSizeMany((cufftHandle)handle, rank, n, inembed, istride, idist, onembed, ostride, odist, cuda_type, batch, workArea);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftGetSizeMany64(UPTKfftHandle plan, int rank, long long int *n, long long int *inembed, long long int istride, long long int idist, long long int *onembed, long long int ostride, long long int odist, UPTKfftType type, long long int batch, size_t *workSize)
{
    cufftType cuda_type = UPTKfftTypeTocufftType(type);
    cufftResult cuda_res;
    cuda_res = cufftGetSizeMany64((cufftHandle)plan, rank, n, inembed, istride, idist, onembed, ostride, odist, cuda_type, batch, workSize);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftGetVersion(int *version)
{
    if (nullptr == version)
    {
        return UPTKFFT_INVALID_VALUE;
    }

    *version = UPTKFFT_VERSION;

    return UPTKFFT_SUCCESS;
}

UPTKfftResult UPTKFFTAPI UPTKfftMakePlan1d(UPTKfftHandle plan, int nx, UPTKfftType type, int batch, size_t *workSize)
{
    cufftType cuda_type = UPTKfftTypeTocufftType(type);
    cufftResult cuda_res;
    cuda_res = cufftMakePlan1d((cufftHandle)plan, nx, cuda_type, batch, workSize);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftMakePlan2d(UPTKfftHandle plan, int nx, int ny, UPTKfftType type, size_t *workSize)
{
    cufftType cuda_type = UPTKfftTypeTocufftType(type);
    cufftResult cuda_res;
    cuda_res = cufftMakePlan2d((cufftHandle)plan, nx, ny, cuda_type, workSize);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftMakePlan3d(UPTKfftHandle plan, int nx, int ny, int nz, UPTKfftType type, size_t *workSize)
{
    cufftType cuda_type = UPTKfftTypeTocufftType(type);
    cufftResult cuda_res;
    cuda_res = cufftMakePlan3d((cufftHandle)plan, nx, ny, nz, cuda_type, workSize);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftMakePlanMany(UPTKfftHandle plan, int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, UPTKfftType type, int batch, size_t *workSize)
{
    cufftType cuda_type = UPTKfftTypeTocufftType(type);
    cufftResult cuda_res;
    cuda_res = cufftMakePlanMany((cufftHandle)plan, rank, n, inembed, istride, idist, onembed, ostride, odist, cuda_type, batch, workSize);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftMakePlanMany64(UPTKfftHandle plan, int rank, long long int *n, long long int *inembed, long long int istride, long long int idist, long long int *onembed, long long int ostride, long long int odist, UPTKfftType type, long long int batch, size_t *workSize)
{
    cufftType cuda_type = UPTKfftTypeTocufftType(type);
    cufftResult cuda_res;
    cuda_res = cufftMakePlanMany64((cufftHandle)plan, rank, n, inembed, istride, idist, onembed, ostride, odist, cuda_type, batch, workSize);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftPlan1d(UPTKfftHandle *plan, int nx, UPTKfftType type, int batch)
{
    cufftType cuda_type = UPTKfftTypeTocufftType(type);
    cufftResult cuda_res;
    cuda_res = cufftPlan1d((cufftHandle *)plan, nx, cuda_type, batch);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftPlan2d(UPTKfftHandle *plan, int nx, int ny, UPTKfftType type)
{
    cufftType cuda_type = UPTKfftTypeTocufftType(type);
    cufftResult cuda_res;
    cuda_res = cufftPlan2d((cufftHandle *)plan, nx, ny, cuda_type);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftPlan3d(UPTKfftHandle *plan, int nx, int ny, int nz, UPTKfftType type)
{
    cufftType cuda_type = UPTKfftTypeTocufftType(type);
    cufftResult cuda_res;
    cuda_res = cufftPlan3d((cufftHandle *)plan, nx, ny, nz, cuda_type);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftPlanMany(UPTKfftHandle *plan, int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, UPTKfftType type, int batch)
{
    cufftType cuda_type = UPTKfftTypeTocufftType(type);
    cufftResult cuda_res;
    cuda_res = cufftPlanMany((cufftHandle *)plan, rank, n, inembed, istride, idist, onembed, ostride, odist, cuda_type, batch);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftSetAutoAllocation(UPTKfftHandle plan, int autoAllocate)
{
    cufftResult cuda_res;
    cuda_res = cufftSetAutoAllocation((cufftHandle)plan, autoAllocate);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftSetStream(UPTKfftHandle plan, UPTKStream_t stream)
{
    cufftResult cuda_res;
    cuda_res = cufftSetStream((cufftHandle)plan, (cudaStream_t)stream);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftSetWorkArea(UPTKfftHandle plan, void *workArea)
{
    cufftResult cuda_res;
    cuda_res = cufftSetWorkArea((cufftHandle)plan, workArea);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftSetPlanPropertyInt64(UPTKfftHandle plan,
                                                        UPTKfftProperty property,
                                                        const long long int inputValueInt)
{
    cufftResult cuda_res;
    cuda_res = cufftSetPlanPropertyInt64((cufftHandle)plan, UPTKfftPropertyTocufftProperty(property), inputValueInt);
    return cufftResultToUPTKfftResult(cuda_res);
}

UPTKfftResult UPTKFFTAPI UPTKfftGetPlanPropertyInt64(UPTKfftHandle plan,
                                                        UPTKfftProperty property,
                                                        long long int *returnPtrValue)
{
    cufftResult cuda_res;
    cuda_res = cufftGetPlanPropertyInt64((cufftHandle)plan, UPTKfftPropertyTocufftProperty(property), returnPtrValue);
    return cufftResultToUPTKfftResult(cuda_res);
}
UPTKfftResult UPTKFFTAPI UPTKfftResetPlanProperty(UPTKfftHandle plan, UPTKfftProperty property)
{
    cufftResult cuda_res;
    cuda_res = cufftResetPlanProperty((cufftHandle)plan, UPTKfftPropertyTocufftProperty(property));
    return cufftResultToUPTKfftResult(cuda_res);
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */
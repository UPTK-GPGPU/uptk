#include "blas.hpp"
#include <iostream>
#include <cuda_runtime.h>

#define BLASXT_HANDLE_PARSE                                                   \
    UPTKblasXtHandle_t blasHandle = ((GPUfusionXtContext *)handle)->XtHandle; \
    if (nullptr == blasHandle)                                                \
    {                                                                         \
        return UPTKBLAS_STATUS_INTERNAL_ERROR;                                \
    }

typedef struct UPTKblasXtContext_Xt
{
    UPTKblasXtHandle_t XtHandle = nullptr;
    int blockDim = 1024;
    UPTKblasXtPinnedMemMode_t mode = UPTKBLASXT_PINNING_DISABLED;
} GPUfusionXtContext;

#define HIP_CHECK(cmd, fun)                                                         \
    {                                                                               \
        cudaError_t status = cmd;                                                   \
        if (status != cudaSuccess)                                                  \
        {                                                                           \
            std::cout << "error: #" << status << " (" << cudaGetErrorString(status) \
                      << ") at function:" << fun << ": " << #cmd << std::endl;      \
            abort();                                                                \
        }                                                                           \
    }

template <typename Ptr>
class MemoryTools
{
    enum class Flag
    {
        none,
        reg,
        copy,
    };

public:
    template <typename P>
    MemoryTools(P ptr, size_t length, const char *fun = nullptr) : device_ptr(nullptr),
                                                                   host_ptr(nullptr),
                                                                   length(length),
                                                                   fun_ptr(fun)
    {
        cudaPointerAttributes attribs;
        cudaError_t status = cudaPointerGetAttributes(&attribs, ptr);
        if (status == cudaErrorInvalidValue)
        {
            flag = Flag::reg;
            host_ptr = const_cast<Ptr>(ptr);
            HIP_CHECK(cudaHostRegister((void *)ptr, length, cudaHostRegisterMapped), fun_ptr);
            HIP_CHECK(cudaHostGetDevicePointer((void **)&device_ptr, (void *)ptr, cudaHostRegisterDefault), fun_ptr);
        }
        else if (attribs.devicePointer == nullptr)
        {
            flag = Flag::copy;
            host_ptr = const_cast<Ptr>(ptr);
            HIP_CHECK(cudaMalloc((void **)&device_ptr, length), fun_ptr);
            HIP_CHECK(cudaMemcpy((void *)device_ptr, (void *)host_ptr, length, cudaMemcpyHostToDevice), fun_ptr);
        }
        else if (attribs.devicePointer != nullptr)
        {
            flag = Flag::none;
            host_ptr = nullptr;
            device_ptr = reinterpret_cast<Ptr>(attribs.devicePointer);
        }
    }
    ~MemoryTools()
    {
        if (flag == Flag::reg)
        {
            HIP_CHECK(cudaHostUnregister((void *)host_ptr), fun_ptr);
        }
        if (flag == Flag::copy)
        {
            if (!std::is_const_v<std::remove_pointer_t<Ptr>>)
            {
                HIP_CHECK(cudaMemcpy((void *)host_ptr, (void *)device_ptr, length, cudaMemcpyDeviceToHost), fun_ptr);
            }
            HIP_CHECK(cudaFree((void *)device_ptr), fun_ptr);
        }
    }
    Ptr device()
    {
        return device_ptr;
    }
    Ptr host()
    {
        return host_ptr;
    }
    size_t size()
    {
        return length;
    }

private:
    Flag flag;
    Ptr host_ptr;
    Ptr device_ptr;
    size_t length;
    const char *fun_ptr;
};

template <typename P>
MemoryTools(P, size_t, const char *) -> MemoryTools<P>;

#define CREATE_MEMORY_TOOLS(ptr, length) MemoryTools(ptr, length, __FUNCTION__)

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCreate(UPTKblasXtHandle_t *handle)
{
    *handle = (UPTKblasXtHandle_t)malloc(sizeof(GPUfusionXtContext));
    memset(*handle, 0, sizeof(GPUfusionXtContext));
    // UPTKblasXtHandle_t *blasHandle = &(((GPUfusionXtContext *)(*handle))->XtHandle);
    // cublasStatus_t hip_res;
    // hip_res = cublasCreate_v2((cublasHandle_t *)blasHandle);
    // return cublasStatusToUPTKblasStatus(hip_res);
    return UPTKBLAS_STATUS_SUCCESS;
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtDestroy(UPTKblasXtHandle_t handle)
{
    // UPTKblasXtHandle_t blasHandle = ((GPUfusionXtContext *)handle)->XtHandle;
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasDestroy_v2((cublasHandle_t)blasHandle);
    free(handle);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtGetNumBoards(int nbDevices, int deviceId[], int *nbBoards)
{
    if (nullptr == nbBoards)
    {
        return UPTKBLAS_STATUS_INVALID_VALUE;
    }

    *nbBoards = 64;

    return UPTKBLAS_STATUS_SUCCESS;
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtMaxBoards(int *nbGpuBoards)
{
    if (nullptr == nbGpuBoards)
    {
        return UPTKBLAS_STATUS_INVALID_VALUE;
    }

    *nbGpuBoards = 1024;

    return UPTKBLAS_STATUS_SUCCESS;
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtDeviceSelect(UPTKblasXtHandle_t handle, int nbDevices, int deviceId[])
{
    if (nbDevices <= 0)
    {
        return UPTKBLAS_STATUS_INVALID_VALUE;
    }

    cudaError_t hip_res;
    hip_res = cudaSetDevice(deviceId[0]);

    if (cudaSuccess == hip_res)
    {
        UPTKblasXtHandle_t blasHandle = nullptr;
        cublasStatus_t status = cublasCreate_v2((cublasHandle_t *)&blasHandle);

        if (CUBLAS_STATUS_SUCCESS == status)
        {
            ((GPUfusionXtContext *)handle)->XtHandle = blasHandle;
        }

        return cublasStatusToUPTKblasStatus(status);
    }

    return UPTKBLAS_STATUS_INTERNAL_ERROR;
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtSetBlockDim(UPTKblasXtHandle_t handle, int blockDim)
{
    GPUfusionXtContext *blasHandle = (GPUfusionXtContext *)handle;

    if (blockDim <= std::numeric_limits<int>::max() && blockDim > 0)
    {
        blasHandle->blockDim = blockDim;
    }
    else
    {
        return UPTKBLAS_STATUS_INVALID_VALUE;
    }
    return UPTKBLAS_STATUS_SUCCESS;
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtGetBlockDim(UPTKblasXtHandle_t handle, int *blockDim)
{
    if (nullptr == blockDim)
    {
        return UPTKBLAS_STATUS_INVALID_VALUE;
    }

    GPUfusionXtContext *blasHandle = (GPUfusionXtContext *)handle;
    *blockDim = blasHandle->blockDim;

    return UPTKBLAS_STATUS_SUCCESS;
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtGetPinningMemMode(UPTKblasXtHandle_t handle, UPTKblasXtPinnedMemMode_t *mode)
{
    if (nullptr == mode)
    {
        return UPTKBLAS_STATUS_INVALID_VALUE;
    }

    GPUfusionXtContext *blasHandle = (GPUfusionXtContext *)handle;
    *mode = blasHandle->mode;
    return UPTKBLAS_STATUS_SUCCESS;
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtSetPinningMemMode(UPTKblasXtHandle_t handle, UPTKblasXtPinnedMemMode_t mode)
{
    GPUfusionXtContext *blasHandle = (GPUfusionXtContext *)handle;
    blasHandle->mode = mode;
    return UPTKBLAS_STATUS_SUCCESS;
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtSetCpuRoutine(UPTKblasXtHandle_t handle, UPTKblasXtBlasOp_t blasOp, UPTKblasXtOpType_t type, void *blasFunctor)
{
    return UPTKBLAS_STATUS_SUCCESS;
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtSetCpuRatio(UPTKblasXtHandle_t handle, UPTKblasXtBlasOp_t blasOp, UPTKblasXtOpType_t type, float ratio)
{
    return UPTKBLAS_STATUS_SUCCESS;
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtSgemm(UPTKblasXtHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, size_t m, size_t n, size_t k, const float *alpha, const float *A, size_t lda, const float *B, size_t ldb, const float *beta, float *C, size_t ldc)
{
    size_t sizeA = transa == UPTKBLAS_OP_N ? lda * k : lda * m;
    size_t sizeB = transb == UPTKBLAS_OP_N ? ldb * n : ldb * k;
    size_t sizeC = ldc * n;

    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(float) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(float) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(float) * sizeC);

    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t hip_transB = UPTKblasOperationTocublasOperation(transb);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasSgemm_v2((cublasHandle_t)blasHandle, hip_transA, hip_transB, m, n, k, alpha, dA.device(), lda, dB.device(), ldb, beta, dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtDgemm(UPTKblasXtHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, size_t m, size_t n, size_t k, const double *alpha, const double *A, size_t lda, const double *B, size_t ldb, const double *beta, double *C, size_t ldc)
{
    size_t sizeA = transa == UPTKBLAS_OP_N ? lda * k : lda * m;
    size_t sizeB = transb == UPTKBLAS_OP_N ? ldb * n : ldb * k;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(double) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(double) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(double) * sizeC);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t hip_transB = UPTKblasOperationTocublasOperation(transb);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasDgemm_v2((cublasHandle_t)blasHandle, hip_transA, hip_transB, m, n, k, alpha, dA.device(), lda, dB.device(), ldb, beta, dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCgemm(UPTKblasXtHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, size_t m, size_t n, size_t k, const cuComplex *alpha, const cuComplex *A, size_t lda, const cuComplex *B, size_t ldb, const cuComplex *beta, cuComplex *C, size_t ldc)
{
    size_t sizeA = transa == UPTKBLAS_OP_N ? lda * k : lda * m;
    size_t sizeB = transb == UPTKBLAS_OP_N ? ldb * n : ldb * k;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuComplex) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(cuComplex) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(cuComplex) * sizeC);

    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t hip_transB = UPTKblasOperationTocublasOperation(transb);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasCgemm_v2((cublasHandle_t)blasHandle, hip_transA, hip_transB, m, n, k, (const cuComplex *)alpha, (const cuComplex *)dA.device(), lda, (const cuComplex *)dB.device(), ldb, (const cuComplex *)beta, (cuComplex *)dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZgemm(UPTKblasXtHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, size_t m, size_t n, size_t k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, size_t lda, const cuDoubleComplex *B, size_t ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, size_t ldc)
{
    size_t sizeA = transa == UPTKBLAS_OP_N ? lda * k : lda * m;
    size_t sizeB = transb == UPTKBLAS_OP_N ? ldb * n : ldb * k;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuDoubleComplex) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(cuDoubleComplex) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(cuDoubleComplex) * sizeC);

    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t hip_transB = UPTKblasOperationTocublasOperation(transb);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasZgemm_v2((cublasHandle_t)blasHandle, hip_transA, hip_transB, m, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)dA.device(), lda, (const cuDoubleComplex *)dB.device(), ldb, (const cuDoubleComplex *)beta, (cuDoubleComplex *)dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtSsyrk(UPTKblasXtHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, size_t n, size_t k, const float *alpha, const float *A, size_t lda, const float *beta, float *C, size_t ldc)
{
    size_t sizeA = trans == UPTKBLAS_OP_N ? lda * k : lda * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(float) * sizeA);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(float) * sizeC);

    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasSsyrk_v2((cublasHandle_t)blasHandle, hip_uplo, hip_transA, n, k, alpha, dA.device(), lda, beta, dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtDsyrk(UPTKblasXtHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, size_t n, size_t k, const double *alpha, const double *A, size_t lda, const double *beta, double *C, size_t ldc)
{
    size_t sizeA = trans == UPTKBLAS_OP_N ? lda * k : lda * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(double) * sizeA);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(double) * sizeC);

    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasDsyrk_v2((cublasHandle_t)blasHandle, hip_uplo, hip_transA, n, k, alpha, dA.device(), lda, beta, dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCsyrk(UPTKblasXtHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, size_t n, size_t k, const cuComplex *alpha, const cuComplex *A, size_t lda, const cuComplex *beta, cuComplex *C, size_t ldc)
{
    size_t sizeA = trans == UPTKBLAS_OP_N ? lda * k : lda * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuComplex) * sizeA);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(cuComplex) * sizeC);

    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasCsyrk_v2((cublasHandle_t)blasHandle, hip_uplo, hip_transA, n, k, (const cuComplex *)alpha, (const cuComplex *)dA.device(), lda, (const cuComplex *)beta, (cuComplex *)dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZsyrk(UPTKblasXtHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, size_t n, size_t k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, size_t lda, const cuDoubleComplex *beta, cuDoubleComplex *C, size_t ldc)
{
    size_t sizeA = trans == UPTKBLAS_OP_N ? lda * k : lda * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuDoubleComplex) * sizeA);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(cuDoubleComplex) * sizeC);

    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasZsyrk_v2((cublasHandle_t)blasHandle, hip_uplo, hip_transA, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)dA.device(), lda, (const cuDoubleComplex *)beta, (cuDoubleComplex *)dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCherk(UPTKblasXtHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, size_t n, size_t k, const float *alpha, const cuComplex *A, size_t lda, const float *beta, cuComplex *C, size_t ldc)
{
    size_t sizeA = trans == UPTKBLAS_OP_N ? lda * k : lda * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuComplex) * sizeA);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(cuComplex) * sizeC);

    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasCherk_v2((cublasHandle_t)blasHandle, hip_uplo, hip_transA, n, k, alpha, (const cuComplex *)dA.device(), lda, beta, (cuComplex *)dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZherk(UPTKblasXtHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, size_t n, size_t k, const double *alpha, const cuDoubleComplex *A, size_t lda, const double *beta, cuDoubleComplex *C, size_t ldc)
{
    size_t sizeA = trans == UPTKBLAS_OP_N ? lda * k : lda * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuDoubleComplex) * sizeA);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(cuDoubleComplex) * sizeC);

    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasZherk_v2((cublasHandle_t)blasHandle, hip_uplo, hip_transA, n, k, alpha, (const cuDoubleComplex *)dA.device(), lda, beta, (cuDoubleComplex *)dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtSsyr2k(UPTKblasXtHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, size_t n, size_t k, const float *alpha, const float *A, size_t lda, const float *B, size_t ldb, const float *beta, float *C, size_t ldc)
{
    size_t sizeA = trans == UPTKBLAS_OP_N ? lda * k : lda * n;
    size_t sizeB = trans == UPTKBLAS_OP_N ? ldb * k : ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(float) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(float) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(float) * sizeC);

    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasSsyr2k_v2((cublasHandle_t)blasHandle, hip_uplo, hip_transA, n, k, alpha, dA.device(), lda, dB.device(), ldb, beta, dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtDsyr2k(UPTKblasXtHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, size_t n, size_t k, const double *alpha, const double *A, size_t lda, const double *B, size_t ldb, const double *beta, double *C, size_t ldc)
{
    size_t sizeA = trans == UPTKBLAS_OP_N ? lda * k : lda * n;
    size_t sizeB = trans == UPTKBLAS_OP_N ? ldb * k : ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(double) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(double) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(double) * sizeC);

    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasDsyr2k_v2((cublasHandle_t)blasHandle, hip_uplo, hip_transA, n, k, alpha, dA.device(), lda, dB.device(), ldb, beta, dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCsyr2k(UPTKblasXtHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, size_t n, size_t k, const cuComplex *alpha, const cuComplex *A, size_t lda, const cuComplex *B, size_t ldb, const cuComplex *beta, cuComplex *C, size_t ldc)
{
    size_t sizeA = trans == UPTKBLAS_OP_N ? lda * k : lda * n;
    size_t sizeB = trans == UPTKBLAS_OP_N ? ldb * k : ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuComplex) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(cuComplex) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(cuComplex) * sizeC);

    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasCsyr2k_v2((cublasHandle_t)blasHandle, hip_uplo, hip_transA, n, k, (const cuComplex *)alpha, (const cuComplex *)dA.device(), lda, (const cuComplex *)dB.device(), ldb, (const cuComplex *)beta, (cuComplex *)dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZsyr2k(UPTKblasXtHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, size_t n, size_t k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, size_t lda, const cuDoubleComplex *B, size_t ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, size_t ldc)
{
    size_t sizeA = trans == UPTKBLAS_OP_N ? lda * k : lda * n;
    size_t sizeB = trans == UPTKBLAS_OP_N ? ldb * k : ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuDoubleComplex) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(cuDoubleComplex) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(cuDoubleComplex) * sizeC);

    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasZsyr2k_v2((cublasHandle_t)blasHandle, hip_uplo, hip_transA, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)dA.device(), lda, (const cuDoubleComplex *)dB.device(), ldb, (const cuDoubleComplex *)beta, (cuDoubleComplex *)dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCherkx(UPTKblasXtHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, size_t n, size_t k, const cuComplex *alpha, const cuComplex *A, size_t lda, const cuComplex *B, size_t ldb, const float *beta, cuComplex *C, size_t ldc)
{
    size_t sizeA = trans == UPTKBLAS_OP_N ? lda * k : lda * n;
    size_t sizeB = trans == UPTKBLAS_OP_N ? ldb * k : ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuComplex) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(cuComplex) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(cuComplex) * sizeC);

    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasCherkx((cublasHandle_t)blasHandle, hip_uplo, hip_transA, n, k, (const cuComplex *)alpha, (const cuComplex *)dA.device(), lda, (const cuComplex *)dB.device(), ldb, beta, (cuComplex *)dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZherkx(UPTKblasXtHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, size_t n, size_t k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, size_t lda, const cuDoubleComplex *B, size_t ldb, const double *beta, cuDoubleComplex *C, size_t ldc)
{
    size_t sizeA = trans == UPTKBLAS_OP_N ? lda * k : lda * n;
    size_t sizeB = trans == UPTKBLAS_OP_N ? ldb * k : ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuDoubleComplex) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(cuDoubleComplex) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(cuDoubleComplex) * sizeC);

    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasZherkx((cublasHandle_t)blasHandle, hip_uplo, hip_transA, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)dA.device(), lda, (const cuDoubleComplex *)dB.device(), ldb, beta, (cuDoubleComplex *)dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtStrsm(UPTKblasXtHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, size_t m, size_t n, const float *alpha, const float *A, size_t lda, float *B, size_t ldb)
{
    size_t sizeA = side == UPTKBLAS_SIDE_LEFT ? lda * m : lda * n;
    size_t sizeB = ldb * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(float) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(float) * sizeB);

    cublasSideMode_t hip_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t hip_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasStrsm_v2((cublasHandle_t)blasHandle, hip_side, hip_uplo, hip_transA, hip_diag, m, n, alpha, const_cast<float *>(dA.device()), lda, dB.device(), ldb);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtDtrsm(UPTKblasXtHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, size_t m, size_t n, const double *alpha, const double *A, size_t lda, double *B, size_t ldb)
{
    size_t sizeA = side == UPTKBLAS_SIDE_LEFT ? lda * m : lda * n;
    size_t sizeB = ldb * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(double) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(double) * sizeB);

    cublasSideMode_t hip_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t hip_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasDtrsm_v2((cublasHandle_t)blasHandle, hip_side, hip_uplo, hip_transA, hip_diag, m, n, alpha, const_cast<double *>(dA.device()), lda, dB.device(), ldb);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCtrsm(UPTKblasXtHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, size_t m, size_t n, const cuComplex *alpha, const cuComplex *A, size_t lda, cuComplex *B, size_t ldb)
{
    size_t sizeA = side == UPTKBLAS_SIDE_LEFT ? lda * m : lda * n;
    size_t sizeB = ldb * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuComplex) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(cuComplex) * sizeB);

    cublasSideMode_t hip_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t hip_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasCtrsm_v2((cublasHandle_t)blasHandle, hip_side, hip_uplo, hip_transA, hip_diag, m, n, (const cuComplex *)alpha, (cuComplex *)dA.device(), lda, (cuComplex *)dB.device(), ldb);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZtrsm(UPTKblasXtHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, size_t m, size_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, size_t lda, cuDoubleComplex *B, size_t ldb)
{
    size_t sizeA = side == UPTKBLAS_SIDE_LEFT ? lda * m : lda * n;
    size_t sizeB = ldb * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuDoubleComplex) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(cuDoubleComplex) * sizeB);

    cublasSideMode_t hip_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t hip_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasZtrsm_v2((cublasHandle_t)blasHandle, hip_side, hip_uplo, hip_transA, hip_diag, m, n, (const cuDoubleComplex *)alpha, (cuDoubleComplex *)dA.device(), lda, (cuDoubleComplex *)dB.device(), ldb);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtSsymm(UPTKblasXtHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, size_t m, size_t n, const float *alpha, const float *A, size_t lda, const float *B, size_t ldb, const float *beta, float *C, size_t ldc)
{
    size_t sizeA = side == UPTKBLAS_SIDE_LEFT ? lda * m : lda * n;
    size_t sizeB = ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(float) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(float) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(float) * sizeC);

    cublasSideMode_t hip_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasSsymm_v2((cublasHandle_t)blasHandle, hip_side, hip_uplo, m, n, alpha, dA.device(), lda, dB.device(), ldb, beta, dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtDsymm(UPTKblasXtHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, size_t m, size_t n, const double *alpha, const double *A, size_t lda, const double *B, size_t ldb, const double *beta, double *C, size_t ldc)
{
    size_t sizeA = side == UPTKBLAS_SIDE_LEFT ? lda * m : lda * n;
    size_t sizeB = ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(double) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(double) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(double) * sizeC);

    cublasSideMode_t hip_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasDsymm_v2((cublasHandle_t)blasHandle, hip_side, hip_uplo, m, n, alpha, dA.device(), lda, dB.device(), ldb, beta, dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCsymm(UPTKblasXtHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, size_t m, size_t n, const cuComplex *alpha, const cuComplex *A, size_t lda, const cuComplex *B, size_t ldb, const cuComplex *beta, cuComplex *C, size_t ldc)
{
    size_t sizeA = side == UPTKBLAS_SIDE_LEFT ? lda * m : lda * n;
    size_t sizeB = ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuComplex) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(cuComplex) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(cuComplex) * sizeC);

    cublasSideMode_t hip_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasCsymm_v2((cublasHandle_t)blasHandle, hip_side, hip_uplo, m, n, (const cuComplex *)alpha, (const cuComplex *)dA.device(), lda, (const cuComplex *)dB.device(), ldb, (const cuComplex *)beta, (cuComplex *)dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZsymm(UPTKblasXtHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, size_t m, size_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, size_t lda, const cuDoubleComplex *B, size_t ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, size_t ldc)
{
    size_t sizeA = side == UPTKBLAS_SIDE_LEFT ? lda * m : lda * n;
    size_t sizeB = ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuDoubleComplex) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(cuDoubleComplex) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(cuDoubleComplex) * sizeC);

    cublasSideMode_t hip_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasZsymm_v2((cublasHandle_t)blasHandle, hip_side, hip_uplo, m, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)dA.device(), lda, (const cuDoubleComplex *)dB.device(), ldb, (const cuDoubleComplex *)beta, (cuDoubleComplex *)dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtChemm(UPTKblasXtHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, size_t m, size_t n, const cuComplex *alpha, const cuComplex *A, size_t lda, const cuComplex *B, size_t ldb, const cuComplex *beta, cuComplex *C, size_t ldc)
{
    size_t sizeA = side == UPTKBLAS_SIDE_LEFT ? lda * m : lda * n;
    size_t sizeB = ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuComplex) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(cuComplex) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(cuComplex) * sizeC);

    cublasSideMode_t hip_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasChemm_v2((cublasHandle_t)blasHandle, hip_side, hip_uplo, m, n, (const cuComplex *)alpha, (const cuComplex *)dA.device(), lda, (const cuComplex *)dB.device(), ldb, (const cuComplex *)beta, (cuComplex *)dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZhemm(UPTKblasXtHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, size_t m, size_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, size_t lda, const cuDoubleComplex *B, size_t ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, size_t ldc)
{
    size_t sizeA = side == UPTKBLAS_SIDE_LEFT ? lda * m : lda * n;
    size_t sizeB = ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuDoubleComplex) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(cuDoubleComplex) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(cuDoubleComplex) * sizeC);

    cublasSideMode_t hip_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasZhemm_v2((cublasHandle_t)blasHandle, hip_side, hip_uplo, m, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)dA.device(), lda, (const cuDoubleComplex *)dB.device(), ldb, (const cuDoubleComplex *)beta, (cuDoubleComplex *)dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtSsyrkx(UPTKblasXtHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, size_t n, size_t k, const float *alpha, const float *A, size_t lda, const float *B, size_t ldb, const float *beta, float *C, size_t ldc)
{
    size_t sizeA = trans == UPTKBLAS_OP_N ? lda * k : lda * n;
    size_t sizeB = trans == UPTKBLAS_OP_N ? ldb * k : ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(float) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(float) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(float) * sizeC);

    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasSsyrkx((cublasHandle_t)blasHandle, hip_uplo, hip_transA, n, k, alpha, dA.device(), lda, dB.device(), ldb, beta, dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtDsyrkx(UPTKblasXtHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, size_t n, size_t k, const double *alpha, const double *A, size_t lda, const double *B, size_t ldb, const double *beta, double *C, size_t ldc)
{
    size_t sizeA = trans == UPTKBLAS_OP_N ? lda * k : lda * n;
    size_t sizeB = trans == UPTKBLAS_OP_N ? ldb * k : ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(double) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(double) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(double) * sizeC);

    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasDsyrkx((cublasHandle_t)blasHandle, hip_uplo, hip_transA, n, k, alpha, dA.device(), lda, dB.device(), ldb, beta, dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCsyrkx(UPTKblasXtHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, size_t n, size_t k, const cuComplex *alpha, const cuComplex *A, size_t lda, const cuComplex *B, size_t ldb, const cuComplex *beta, cuComplex *C, size_t ldc)
{
    size_t sizeA = trans == UPTKBLAS_OP_N ? lda * k : lda * n;
    size_t sizeB = trans == UPTKBLAS_OP_N ? ldb * k : ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuComplex) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(cuComplex) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(cuComplex) * sizeC);

    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasCsyrkx((cublasHandle_t)blasHandle, hip_uplo, hip_transA, n, k, (const cuComplex *)alpha, (const cuComplex *)dA.device(), lda, (const cuComplex *)dB.device(), ldb, (const cuComplex *)beta, (cuComplex *)dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZsyrkx(UPTKblasXtHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, size_t n, size_t k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, size_t lda, const cuDoubleComplex *B, size_t ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, size_t ldc)
{
    size_t sizeA = trans == UPTKBLAS_OP_N ? lda * k : lda * n;
    size_t sizeB = trans == UPTKBLAS_OP_N ? ldb * k : ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuDoubleComplex) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(cuDoubleComplex) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(cuDoubleComplex) * sizeC);

    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasZsyrkx((cublasHandle_t)blasHandle, hip_uplo, hip_transA, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)dA.device(), lda, (const cuDoubleComplex *)dB.device(), ldb, (const cuDoubleComplex *)beta, (cuDoubleComplex *)dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCher2k(UPTKblasXtHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, size_t n, size_t k, const cuComplex *alpha, const cuComplex *A, size_t lda, const cuComplex *B, size_t ldb, const float *beta, cuComplex *C, size_t ldc)
{
    size_t sizeA = trans == UPTKBLAS_OP_N ? lda * k : lda * n;
    size_t sizeB = trans == UPTKBLAS_OP_N ? ldb * k : ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuComplex) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(cuComplex) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(cuComplex) * sizeC);

    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasCher2k_v2((cublasHandle_t)blasHandle, hip_uplo, hip_transA, n, k, (const cuComplex *)alpha, (const cuComplex *)dA.device(), lda, (const cuComplex *)dB.device(), ldb, beta, (cuComplex *)dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZher2k(UPTKblasXtHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, size_t n, size_t k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, size_t lda, const cuDoubleComplex *B, size_t ldb, const double *beta, cuDoubleComplex *C, size_t ldc)
{
    size_t sizeA = trans == UPTKBLAS_OP_N ? lda * k : lda * n;
    size_t sizeB = trans == UPTKBLAS_OP_N ? ldb * k : ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuDoubleComplex) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(cuDoubleComplex) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(cuDoubleComplex) * sizeC);

    cublasFillMode_t hip_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t hip_transA = UPTKblasOperationTocublasOperation(trans);
    BLASXT_HANDLE_PARSE
    cublasStatus_t hip_res;
    hip_res = cublasZher2k_v2((cublasHandle_t)blasHandle, hip_uplo, hip_transA, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)dA.device(), lda, (const cuDoubleComplex *)dB.device(), ldb, beta, (cuDoubleComplex *)dC.device(), ldc);
    return cublasStatusToUPTKblasStatus(hip_res);
}

template <typename T>
__device__ T zero();

template <>
__device__ float zero<float>()
{
    return 0.0f;
}

template <>
__device__ double zero<double>()
{
    return 0.0;
}

template <>
__device__ cuComplex zero<cuComplex>()
{
    return make_cuComplex(0.0f, 0.0f);
}

template <>
__device__ cuDoubleComplex zero<cuDoubleComplex>()
{
    return make_cuDoubleComplex(0.0, 0.0);
}

template <typename T>
__device__ T multiply(T a, T b);

template <>
__device__ float multiply<float>(float a, float b)
{
    return a * b;
}

template <>
__device__ double multiply<double>(double a, double b)
{
    return a * b;
}

template <>
__device__ cuComplex multiply<cuComplex>(cuComplex a, cuComplex b)
{
    return cuCmulf(a, b);
}

template <>
__device__ cuDoubleComplex multiply<cuDoubleComplex>(cuDoubleComplex a, cuDoubleComplex b)
{
    return cuCmul(a, b);
}

template <typename T>
__device__ T add(T a, T b);

template <>
__device__ float add<float>(float a, float b)
{
    return a + b;
}

template <>
__device__ double add<double>(double a, double b)
{
    return a + b;
}

template <>
__device__ cuComplex add<cuComplex>(cuComplex a, cuComplex b)
{
    return cuCaddf(a, b);
}

template <>
__device__ cuDoubleComplex add<cuDoubleComplex>(cuDoubleComplex a, cuDoubleComplex b)
{
    return cuCadd(a, b);
}

template <typename T>
__global__ void csr_spmm_kernel(const T *csrVal, const int *csrRowPtr, const int *csrColIdx, const T *B, T *C, int m, int n, int ldb, int ldc, T alpha, T beta)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m)
    {
        for (int col = 0; col < n; col++)
        {
            T sum = zero<T>();
            for (int i = csrRowPtr[row]; i < csrRowPtr[row + 1]; i++)
            {
                int idx = csrColIdx[i];
                sum = add(sum, multiply(csrVal[i], B[idx * ldb + col]));
            }
            C[row * ldc + col] = add(multiply(alpha, sum), multiply(beta, C[row * ldc + col]));
        }
    }
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtSspmm(UPTKblasXtHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, size_t m, size_t n, const float *alpha, const float *AP, const float *B, size_t ldb, const float *beta, float *C, size_t ldc)
{

    // Check if the input is empty
    if (alpha == nullptr || AP == nullptr || B == nullptr || beta == nullptr || C == nullptr)
    {
        return UPTKBLAS_STATUS_INVALID_VALUE;
    }

    int rowData[m + 1];
    for (int i = 0; i < m + 1; ++i)
    {
        float temp = AP[i];
        rowData[i] = static_cast<int>(temp);
    }

    const int nnz = rowData[m];

    int colData[nnz];
    float valData[nnz];
    for (int i = 0; i < nnz; ++i)
    {
        float temp = AP[m + 1 + i];
        colData[i] = static_cast<int>(temp);
    }

    for (int i = 0; i < nnz; ++i)
    {
        float temp = AP[m + 1 + nnz + i];
        valData[i] = temp;
    }

    // const float *APReceive = AP;
    const int *csrRowPtr = rowData;
    const int *csrColIdx = colData;
    const float *csrVal = valData; // CSR values

    size_t sizecsrRowPtr = (m + 1) * sizeof(int);
    size_t sizecsrColIdx = nnz * sizeof(int);
    size_t sizecsrVal = nnz * sizeof(float);
    size_t sizeB = ldb * n * sizeof(float);
    size_t sizeC = ldc * n * sizeof(float);

    MemoryTools dRowPtr = CREATE_MEMORY_TOOLS(csrRowPtr, sizecsrRowPtr);
    MemoryTools dColIdx = CREATE_MEMORY_TOOLS(csrColIdx, sizecsrColIdx);
    MemoryTools dcsrVal = CREATE_MEMORY_TOOLS(csrVal, sizecsrVal);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeC);

    // const int *csrColIdx = reinterpret_cast<const int *>(dAP.device() + (m + 1) * sizeof(int));
    // const float *csrVal = reinterpret_cast<const float *>(dAP.device() + (m + 1) * sizeof(int) + nnz * sizeof(int));

    // Calculate grid and thread block size
    int blockSize = 256;
    int numBlocks = (m + blockSize - 1) / blockSize;

    // Call CUDA kernel function for calculation
    csr_spmm_kernel<float><<<numBlocks, blockSize>>>(dcsrVal.device(), dRowPtr.device(), dColIdx.device(), dB.device(), dC.device(), m, n, ldb, ldc, *alpha, *beta);

    // Check CUDA error
    UPTKError_t cudaStatus = UPTKGetLastError();
    if (cudaStatus != UPTKSuccess)
    {
        return UPTKBLAS_STATUS_EXECUTION_FAILED;
    }

    // Synchronize devices
    cudaStatus = UPTKDeviceSynchronize();
    if (cudaStatus != UPTKSuccess)
    {
        return UPTKBLAS_STATUS_EXECUTION_FAILED;
    }

    UPTKMemcpy(C, dC.device(), sizeC, UPTKMemcpyDeviceToHost);

    return UPTKBLAS_STATUS_SUCCESS;
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtDspmm(UPTKblasXtHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, size_t m, size_t n, const double *alpha, const double *AP, const double *B, size_t ldb, const double *beta, double *C, size_t ldc)
{
    if (alpha == nullptr || AP == nullptr || B == nullptr || beta == nullptr || C == nullptr)
    {
        return UPTKBLAS_STATUS_INVALID_VALUE;
    }

    int rowData[m + 1];
    for (int i = 0; i < m + 1; ++i)
    {
        double temp = AP[i];
        rowData[i] = static_cast<int>(temp);
    }

    const int nnz = rowData[m];

    int colData[nnz];
    double valData[nnz];
    for (int i = 0; i < nnz; ++i)
    {
        double temp = AP[m + 1 + i];
        colData[i] = static_cast<int>(temp);
    }

    for (int i = 0; i < nnz; ++i)
    {
        double temp = AP[m + 1 + nnz + i];
        valData[i] = temp;
    }

    const int *csrRowPtr = rowData;
    const int *csrColIdx = colData;
    const double *csrVal = valData;

    size_t sizecsrRowPtr = (m + 1) * sizeof(int);
    size_t sizecsrColIdx = nnz * sizeof(int);
    size_t sizecsrVal = nnz * sizeof(double);
    size_t sizeB = ldb * n * sizeof(double);
    size_t sizeC = ldc * n * sizeof(double);

    MemoryTools dRowPtr = CREATE_MEMORY_TOOLS(csrRowPtr, sizecsrRowPtr);
    MemoryTools dColIdx = CREATE_MEMORY_TOOLS(csrColIdx, sizecsrColIdx);
    MemoryTools dcsrVal = CREATE_MEMORY_TOOLS(csrVal, sizecsrVal);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeC);

    int blockSize = 256;
    int numBlocks = (m + blockSize - 1) / blockSize;

    csr_spmm_kernel<double><<<numBlocks, blockSize>>>(dcsrVal.device(), dRowPtr.device(), dColIdx.device(), dB.device(), dC.device(), m, n, ldb, ldc, *alpha, *beta);

    UPTKError_t cudaStatus = UPTKGetLastError();
    if (cudaStatus != UPTKSuccess)
    {
        return UPTKBLAS_STATUS_EXECUTION_FAILED;
    }

    cudaStatus = UPTKDeviceSynchronize();
    if (cudaStatus != UPTKSuccess)
    {
        return UPTKBLAS_STATUS_EXECUTION_FAILED;
    }

    UPTKMemcpy(C, dC.device(), sizeC, UPTKMemcpyDeviceToHost);

    return UPTKBLAS_STATUS_SUCCESS;
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCspmm(UPTKblasXtHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, size_t m, size_t n, const cuComplex *alpha, const cuComplex *AP, const cuComplex *B, size_t ldb, const cuComplex *beta, cuComplex *C, size_t ldc)
{
    if (alpha == nullptr || AP == nullptr || B == nullptr || beta == nullptr || C == nullptr)
    {
        return UPTKBLAS_STATUS_INVALID_VALUE;
    }

    int rowData[m + 1];
    for (int i = 0; i < m + 1; ++i)
    {
        float temp = AP[i].x; // Access real part
        rowData[i] = static_cast<int>(temp);
    }

    const int nnz = rowData[m];

    int colData[nnz];
    cuComplex valData[nnz];
    for (int i = 0; i < nnz; ++i)
    {
        float temp = AP[m + 1 + i].x; // Access real part
        colData[i] = static_cast<int>(temp);
    }

    for (int i = 0; i < nnz; ++i)
    {
        valData[i] = AP[m + 1 + nnz + i];
    }

    const int *csrRowPtr = rowData;
    const int *csrColIdx = colData;
    const cuComplex *csrVal = valData;

    size_t sizecsrRowPtr = (m + 1) * sizeof(int);
    size_t sizecsrColIdx = nnz * sizeof(int);
    size_t sizecsrVal = nnz * sizeof(cuComplex);
    size_t sizeB = ldb * n * sizeof(cuComplex);
    size_t sizeC = ldc * n * sizeof(cuComplex);

    MemoryTools dRowPtr = CREATE_MEMORY_TOOLS(csrRowPtr, sizecsrRowPtr);
    MemoryTools dColIdx = CREATE_MEMORY_TOOLS(csrColIdx, sizecsrColIdx);
    MemoryTools dcsrVal = CREATE_MEMORY_TOOLS(csrVal, sizecsrVal);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeC);

    int blockSize = 256;
    int numBlocks = (m + blockSize - 1) / blockSize;

    csr_spmm_kernel<cuComplex><<<numBlocks, blockSize>>>(dcsrVal.device(), dRowPtr.device(), dColIdx.device(), dB.device(), dC.device(), m, n, ldb, ldc, *alpha, *beta);

    UPTKError_t cudaStatus = UPTKGetLastError();
    if (cudaStatus != UPTKSuccess)
    {
        return UPTKBLAS_STATUS_EXECUTION_FAILED;
    }

    cudaStatus = UPTKDeviceSynchronize();
    if (cudaStatus != UPTKSuccess)
    {
        return UPTKBLAS_STATUS_EXECUTION_FAILED;
    }

    UPTKMemcpy(C, dC.device(), sizeC, UPTKMemcpyDeviceToHost);

    return UPTKBLAS_STATUS_SUCCESS;
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZspmm(UPTKblasXtHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, size_t m, size_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *AP, const cuDoubleComplex *B, size_t ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, size_t ldc)
{
    if (alpha == nullptr || AP == nullptr || B == nullptr || beta == nullptr || C == nullptr)
    {
        return UPTKBLAS_STATUS_INVALID_VALUE;
    }

    int rowData[m + 1];
    for (int i = 0; i < m + 1; ++i)
    {
        double temp = AP[i].x; // Access real part
        rowData[i] = static_cast<int>(temp);
    }

    const int nnz = rowData[m];

    int colData[nnz];
    cuDoubleComplex valData[nnz];
    for (int i = 0; i < nnz; ++i)
    {
        double temp = AP[m + 1 + i].x; // Access real part
        colData[i] = static_cast<int>(temp);
    }

    for (int i = 0; i < nnz; ++i)
    {
        valData[i] = AP[m + 1 + nnz + i];
    }

    const int *csrRowPtr = rowData;
    const int *csrColIdx = colData;
    const cuDoubleComplex *csrVal = valData;

    size_t sizecsrRowPtr = (m + 1) * sizeof(int);
    size_t sizecsrColIdx = nnz * sizeof(int);
    size_t sizecsrVal = nnz * sizeof(cuDoubleComplex);
    size_t sizeB = ldb * n * sizeof(cuDoubleComplex);
    size_t sizeC = ldc * n * sizeof(cuDoubleComplex);

    MemoryTools dRowPtr = CREATE_MEMORY_TOOLS(csrRowPtr, sizecsrRowPtr);
    MemoryTools dColIdx = CREATE_MEMORY_TOOLS(csrColIdx, sizecsrColIdx);
    MemoryTools dcsrVal = CREATE_MEMORY_TOOLS(csrVal, sizecsrVal);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeC);

    int blockSize = 256;
    int numBlocks = (m + blockSize - 1) / blockSize;

    csr_spmm_kernel<cuDoubleComplex><<<numBlocks, blockSize>>>(dcsrVal.device(), dRowPtr.device(), dColIdx.device(), dB.device(), dC.device(), m, n, ldb, ldc, *alpha, *beta);

    UPTKError_t cudaStatus = UPTKGetLastError();
    if (cudaStatus != UPTKSuccess)
    {
        return UPTKBLAS_STATUS_EXECUTION_FAILED;
    }

    cudaStatus = UPTKDeviceSynchronize();
    if (cudaStatus != UPTKSuccess)
    {
        return UPTKBLAS_STATUS_EXECUTION_FAILED;
    }

    UPTKMemcpy(C, dC.device(), sizeC, UPTKMemcpyDeviceToHost);

    return UPTKBLAS_STATUS_SUCCESS;
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtStrmm(UPTKblasXtHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, size_t m, size_t n, const float *alpha, const float *A, size_t lda, const float *B, size_t ldb, float *C, size_t ldc)
{
    size_t sizeA = side == UPTKBLAS_SIDE_LEFT ? lda * m : lda * n;
    size_t sizeB = ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(float) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(float) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(float) * sizeC);

    BLASXT_HANDLE_PARSE
    return cublasStatusToUPTKblasStatus(cublasStrmm_v2(
        (cublasHandle_t)blasHandle,
        UPTKblasSideModeTocublasSideMode(side),
        UPTKblasFillModeTocublasFillMode(uplo),
        UPTKblasOperationTocublasOperation(trans),
        UPTKblasDiagTypeTocublasDiagType(diag),
        m,
        n,
        alpha,
        dA.device(),
        lda,
        dB.device(),
        ldb,
        dC.device(),
        ldc));
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtDtrmm(UPTKblasXtHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, size_t m, size_t n, const double *alpha, const double *A, size_t lda, const double *B, size_t ldb, double *C, size_t ldc)
{
    size_t sizeA = side == UPTKBLAS_SIDE_LEFT ? lda * m : lda * n;
    size_t sizeB = ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(double) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(double) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(double) * sizeC);

    BLASXT_HANDLE_PARSE
    return cublasStatusToUPTKblasStatus(cublasDtrmm_v2(
        (cublasHandle_t)blasHandle,
        UPTKblasSideModeTocublasSideMode(side),
        UPTKblasFillModeTocublasFillMode(uplo),
        UPTKblasOperationTocublasOperation(trans),
        UPTKblasDiagTypeTocublasDiagType(diag),
        m,
        n,
        alpha,
        dA.device(),
        lda,
        dB.device(),
        ldb,
        dC.device(),
        ldc));
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtCtrmm(UPTKblasXtHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, size_t m, size_t n, const cuComplex *alpha, const cuComplex *A, size_t lda, const cuComplex *B, size_t ldb, cuComplex *C, size_t ldc)
{
    size_t sizeA = side == UPTKBLAS_SIDE_LEFT ? lda * m : lda * n;
    size_t sizeB = ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuComplex) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(cuComplex) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(cuComplex) * sizeC);

    BLASXT_HANDLE_PARSE
    return cublasStatusToUPTKblasStatus(cublasCtrmm_v2(
        (cublasHandle_t)blasHandle,
        UPTKblasSideModeTocublasSideMode(side),
        UPTKblasFillModeTocublasFillMode(uplo),
        UPTKblasOperationTocublasOperation(trans),
        UPTKblasDiagTypeTocublasDiagType(diag),
        m,
        n,
        alpha,
        dA.device(),
        lda,
        dB.device(),
        ldb,
        dC.device(),
        ldc));
}

// UNSUPPORTED
UPTKblasStatus_t UPTKBLASAPI UPTKblasXtZtrmm(UPTKblasXtHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, size_t m, size_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, size_t lda, const cuDoubleComplex *B, size_t ldb, cuDoubleComplex *C, size_t ldc)
{
    size_t sizeA = side == UPTKBLAS_SIDE_LEFT ? lda * m : lda * n;
    size_t sizeB = ldb * n;
    size_t sizeC = ldc * n;
    MemoryTools dA = CREATE_MEMORY_TOOLS(A, sizeof(cuDoubleComplex) * sizeA);
    MemoryTools dB = CREATE_MEMORY_TOOLS(B, sizeof(cuDoubleComplex) * sizeB);
    MemoryTools dC = CREATE_MEMORY_TOOLS(C, sizeof(cuDoubleComplex) * sizeC);

    BLASXT_HANDLE_PARSE
    return cublasStatusToUPTKblasStatus(cublasZtrmm_v2(
        (cublasHandle_t)blasHandle,
        UPTKblasSideModeTocublasSideMode(side),
        UPTKblasFillModeTocublasFillMode(uplo),
        UPTKblasOperationTocublasOperation(trans),
        UPTKblasDiagTypeTocublasDiagType(diag),
        m,
        n,
        alpha,
        dA.device(),
        lda,
        dB.device(),
        ldb,
        dC.device(),
        ldc));
}

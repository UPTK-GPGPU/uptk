#include "test_common.h"
#include "blas/blas.hpp"

int main()
{
    printf("=============================================\n");
    printf("  UPTK BLAS Test Suite\n");
    printf("=============================================\n");

    /* ============================================================
     *  Section 1: Type Converter Tests (no GPU required)
     * ============================================================ */
    TEST_SECTION("BLAS Type Converters");

    /* --- UPTKblasStatus_t <-> cublasStatus_t --- */
    TEST_ENUM_ROUNDTRIP("Status SUCCESS",
        UPTKblasStatusTocublasStatus, cublasStatusToUPTKblasStatus,
        UPTKBLAS_STATUS_SUCCESS, CUBLAS_STATUS_SUCCESS);
    TEST_ENUM_ROUNDTRIP("Status NOT_INITIALIZED",
        UPTKblasStatusTocublasStatus, cublasStatusToUPTKblasStatus,
        UPTKBLAS_STATUS_NOT_INITIALIZED, CUBLAS_STATUS_NOT_INITIALIZED);
    TEST_ENUM_ROUNDTRIP("Status ALLOC_FAILED",
        UPTKblasStatusTocublasStatus, cublasStatusToUPTKblasStatus,
        UPTKBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_ALLOC_FAILED);
    TEST_ENUM_ROUNDTRIP("Status INVALID_VALUE",
        UPTKblasStatusTocublasStatus, cublasStatusToUPTKblasStatus,
        UPTKBLAS_STATUS_INVALID_VALUE, CUBLAS_STATUS_INVALID_VALUE);
    TEST_ENUM_ROUNDTRIP("Status ARCH_MISMATCH",
        UPTKblasStatusTocublasStatus, cublasStatusToUPTKblasStatus,
        UPTKBLAS_STATUS_ARCH_MISMATCH, CUBLAS_STATUS_ARCH_MISMATCH);
    TEST_ENUM_ROUNDTRIP("Status MAPPING_ERROR",
        UPTKblasStatusTocublasStatus, cublasStatusToUPTKblasStatus,
        UPTKBLAS_STATUS_MAPPING_ERROR, CUBLAS_STATUS_MAPPING_ERROR);
    TEST_ENUM_ROUNDTRIP("Status EXECUTION_FAILED",
        UPTKblasStatusTocublasStatus, cublasStatusToUPTKblasStatus,
        UPTKBLAS_STATUS_EXECUTION_FAILED, CUBLAS_STATUS_EXECUTION_FAILED);
    TEST_ENUM_ROUNDTRIP("Status INTERNAL_ERROR",
        UPTKblasStatusTocublasStatus, cublasStatusToUPTKblasStatus,
        UPTKBLAS_STATUS_INTERNAL_ERROR, CUBLAS_STATUS_INTERNAL_ERROR);
    TEST_ENUM_ROUNDTRIP("Status NOT_SUPPORTED",
        UPTKblasStatusTocublasStatus, cublasStatusToUPTKblasStatus,
        UPTKBLAS_STATUS_NOT_SUPPORTED, CUBLAS_STATUS_NOT_SUPPORTED);
    TEST_ENUM_ROUNDTRIP("Status LICENSE_ERROR",
        UPTKblasStatusTocublasStatus, cublasStatusToUPTKblasStatus,
        UPTKBLAS_STATUS_LICENSE_ERROR, CUBLAS_STATUS_LICENSE_ERROR);

    /* --- UPTKblasAtomicsMode_t <-> cublasAtomicsMode_t --- */
    TEST_ENUM_ROUNDTRIP("AtomicsMode NOT_ALLOWED",
        UPTKblasAtomicsModeTocublasAtomicsMode,
        cublasAtomicsModeToUPTKblasAtomicsMode,
        UPTKBLAS_ATOMICS_NOT_ALLOWED, CUBLAS_ATOMICS_NOT_ALLOWED);
    TEST_ENUM_ROUNDTRIP("AtomicsMode ALLOWED",
        UPTKblasAtomicsModeTocublasAtomicsMode,
        cublasAtomicsModeToUPTKblasAtomicsMode,
        UPTKBLAS_ATOMICS_ALLOWED, CUBLAS_ATOMICS_ALLOWED);

    /* --- UPTKblasMath_t <-> cublasMath_t --- */
    TEST_ENUM_ROUNDTRIP("Math DEFAULT",
        UPTKblasMathTocublasMath, cublasMathToUPTKblasMath,
        UPTKBLAS_DEFAULT_MATH, CUBLAS_DEFAULT_MATH);
    TEST_ENUM_ROUNDTRIP("Math TENSOR_OP",
        UPTKblasMathTocublasMath, cublasMathToUPTKblasMath,
        UPTKBLAS_TENSOR_OP_MATH, CUBLAS_TENSOR_OP_MATH);
    TEST_ENUM_ROUNDTRIP("Math PEDANTIC",
        UPTKblasMathTocublasMath, cublasMathToUPTKblasMath,
        UPTKBLAS_PEDANTIC_MATH, CUBLAS_PEDANTIC_MATH);
    TEST_ENUM_ROUNDTRIP("Math TF32_TENSOR_OP",
        UPTKblasMathTocublasMath, cublasMathToUPTKblasMath,
        UPTKBLAS_TF32_TENSOR_OP_MATH, CUBLAS_TF32_TENSOR_OP_MATH);
    TEST_ENUM_ROUNDTRIP("Math DISALLOW_REDUCED",
        UPTKblasMathTocublasMath, cublasMathToUPTKblasMath,
        UPTKBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
        CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);

    /* --- UPTKblasDiagType_t -> cublasDiagType_t --- */
    TEST_ENUM_CONVERT("DiagType NON_UNIT",
        UPTKblasDiagTypeTocublasDiagType,
        UPTKBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_NON_UNIT);
    TEST_ENUM_CONVERT("DiagType UNIT",
        UPTKblasDiagTypeTocublasDiagType,
        UPTKBLAS_DIAG_UNIT, CUBLAS_DIAG_UNIT);

    /* --- UPTKblasFillMode_t -> cublasFillMode_t --- */
    TEST_ENUM_CONVERT("FillMode LOWER",
        UPTKblasFillModeTocublasFillMode,
        UPTKBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_LOWER);
    TEST_ENUM_CONVERT("FillMode UPPER",
        UPTKblasFillModeTocublasFillMode,
        UPTKBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_UPPER);
    TEST_ENUM_CONVERT("FillMode FULL",
        UPTKblasFillModeTocublasFillMode,
        UPTKBLAS_FILL_MODE_FULL, CUBLAS_FILL_MODE_FULL);

    /* --- UPTKblasOperation_t <-> cublasOperation_t --- */
    TEST_ENUM_ROUNDTRIP("Operation N",
        UPTKblasOperationTocublasOperation,
        hipblasOperationTohcublasOperation,
        UPTKBLAS_OP_N, CUBLAS_OP_N);
    TEST_ENUM_ROUNDTRIP("Operation T",
        UPTKblasOperationTocublasOperation,
        hipblasOperationTohcublasOperation,
        UPTKBLAS_OP_T, CUBLAS_OP_T);
    TEST_ENUM_ROUNDTRIP("Operation C",
        UPTKblasOperationTocublasOperation,
        hipblasOperationTohcublasOperation,
        UPTKBLAS_OP_C, CUBLAS_OP_C);
    TEST_ENUM_ROUNDTRIP("Operation CONJG",
        UPTKblasOperationTocublasOperation,
        hipblasOperationTohcublasOperation,
        UPTKBLAS_OP_CONJG, CUBLAS_OP_CONJG);

    /* --- UPTKblasPointerMode_t <-> cublasPointerMode_t --- */
    TEST_ENUM_ROUNDTRIP("PointerMode HOST",
        UPTKblasPointerModeTocublasPointerMode,
        cublasPointerModeToUPTKblasPointerMode,
        UPTKBLAS_POINTER_MODE_HOST, CUBLAS_POINTER_MODE_HOST);
    TEST_ENUM_ROUNDTRIP("PointerMode DEVICE",
        UPTKblasPointerModeTocublasPointerMode,
        cublasPointerModeToUPTKblasPointerMode,
        UPTKBLAS_POINTER_MODE_DEVICE, CUBLAS_POINTER_MODE_DEVICE);

    /* --- UPTKblasSideMode_t -> cublasSideMode_t --- */
    TEST_ENUM_CONVERT("SideMode LEFT",
        UPTKblasSideModeTocublasSideMode,
        UPTKBLAS_SIDE_LEFT, CUBLAS_SIDE_LEFT);
    TEST_ENUM_CONVERT("SideMode RIGHT",
        UPTKblasSideModeTocublasSideMode,
        UPTKBLAS_SIDE_RIGHT, CUBLAS_SIDE_RIGHT);

    /* --- UPTKblasGemmAlgo_t -> cublasGemmAlgo_t (representative) --- */
    TEST_ENUM_CONVERT("GemmAlgo DEFAULT",
        UPTKblasGemmAlgoTocublasGemmAlgo,
        UPTKBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT);
    TEST_ENUM_CONVERT("GemmAlgo ALGO0",
        UPTKblasGemmAlgoTocublasGemmAlgo,
        UPTKBLAS_GEMM_ALGO0, CUBLAS_GEMM_ALGO0);
    TEST_ENUM_CONVERT("GemmAlgo ALGO15",
        UPTKblasGemmAlgoTocublasGemmAlgo,
        UPTKBLAS_GEMM_ALGO15, CUBLAS_GEMM_ALGO15);
    TEST_ENUM_CONVERT("GemmAlgo DEFAULT_TENSOR_OP",
        UPTKblasGemmAlgoTocublasGemmAlgo,
        UPTKBLAS_GEMM_DEFAULT_TENSOR_OP, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    /* --- UPTKDataType <-> cudaDataType (representative) --- */
    TEST_ENUM_ROUNDTRIP("DataType R_16F",
        UPTKDataTypeTocudaDataType, hipDataTypeTocudaDataType,
        UPTK_R_16F, CUDA_R_16F);
    TEST_ENUM_ROUNDTRIP("DataType R_32F",
        UPTKDataTypeTocudaDataType, hipDataTypeTocudaDataType,
        UPTK_R_32F, CUDA_R_32F);
    TEST_ENUM_ROUNDTRIP("DataType R_64F",
        UPTKDataTypeTocudaDataType, hipDataTypeTocudaDataType,
        UPTK_R_64F, CUDA_R_64F);
    TEST_ENUM_ROUNDTRIP("DataType C_32F",
        UPTKDataTypeTocudaDataType, hipDataTypeTocudaDataType,
        UPTK_C_32F, CUDA_C_32F);
    TEST_ENUM_ROUNDTRIP("DataType R_8I",
        UPTKDataTypeTocudaDataType, hipDataTypeTocudaDataType,
        UPTK_R_8I, CUDA_R_8I);
    TEST_ENUM_ROUNDTRIP("DataType R_32I",
        UPTKDataTypeTocudaDataType, hipDataTypeTocudaDataType,
        UPTK_R_32I, CUDA_R_32I);
    TEST_ENUM_ROUNDTRIP("DataType R_16BF",
        UPTKDataTypeTocudaDataType, hipDataTypeTocudaDataType,
        UPTK_R_16BF, CUDA_R_16BF);
    TEST_ENUM_ROUNDTRIP("DataType R_8F_E4M3",
        UPTKDataTypeTocudaDataType, hipDataTypeTocudaDataType,
        UPTK_R_8F_E4M3, CUDA_R_8F_E4M3);

    /* --- UPTKblasComputeType_t <-> cublasComputeType_t --- */
    TEST_ENUM_ROUNDTRIP("ComputeType 16F",
        UPTKblasComputeTypeTocublasComputeType,
        hipblasComputeTypeModeTocublasComputeType,
        UPTKBLAS_COMPUTE_16F, CUBLAS_COMPUTE_16F);
    TEST_ENUM_ROUNDTRIP("ComputeType 32F",
        UPTKblasComputeTypeTocublasComputeType,
        hipblasComputeTypeModeTocublasComputeType,
        UPTKBLAS_COMPUTE_32F, CUBLAS_COMPUTE_32F);
    TEST_ENUM_ROUNDTRIP("ComputeType 64F",
        UPTKblasComputeTypeTocublasComputeType,
        hipblasComputeTypeModeTocublasComputeType,
        UPTKBLAS_COMPUTE_64F, CUBLAS_COMPUTE_64F);
    TEST_ENUM_ROUNDTRIP("ComputeType 32I",
        UPTKblasComputeTypeTocublasComputeType,
        hipblasComputeTypeModeTocublasComputeType,
        UPTKBLAS_COMPUTE_32I, CUBLAS_COMPUTE_32I);
    TEST_ENUM_ROUNDTRIP("ComputeType 32F_FAST_TF32",
        UPTKblasComputeTypeTocublasComputeType,
        hipblasComputeTypeModeTocublasComputeType,
        UPTKBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_COMPUTE_32F_FAST_TF32);

    /* ============================================================
     *  Section 2: API Function Tests (GPU required)
     * ============================================================ */
    TEST_SECTION("BLAS API Functions");

    /* --- UPTKblasCreate / UPTKblasDestroy --- */
    {
        UPTKblasHandle_t handle = nullptr;
        UPTKblasStatus_t st = UPTKblasCreate(&handle);
        TEST_API_STATUS("UPTKblasCreate",
            "UPTKblasCreate(&handle)", st, UPTKBLAS_STATUS_SUCCESS);

        if (st == UPTKBLAS_STATUS_SUCCESS) {
            /* --- UPTKblasGetVersion --- */
            {
                int version = 0;
                UPTKblasStatus_t vs = UPTKblasGetVersion(handle, &version);
                TEST_API_STATUS("UPTKblasGetVersion",
                    "UPTKblasGetVersion(handle, &version)",
                    vs, UPTKBLAS_STATUS_SUCCESS);

                char exp[64], act[64];
                snprintf(exp, sizeof(exp), "%d", UPTKBLAS_VERSION);
                snprintf(act, sizeof(act), "%d", version);
                TEST_CHECK("UPTKblasGetVersion value",
                    "version", exp, act, version == UPTKBLAS_VERSION);
            }

            /* --- UPTKblasGetVersion NULL checks --- */
            {
                UPTKblasStatus_t vs = UPTKblasGetVersion(nullptr, nullptr);
                TEST_API_STATUS("UPTKblasGetVersion(NULL,NULL)",
                    "handle=NULL, version=NULL",
                    vs, UPTKBLAS_STATUS_NOT_INITIALIZED);
            }
            {
                UPTKblasStatus_t vs = UPTKblasGetVersion(handle, nullptr);
                TEST_API_STATUS("UPTKblasGetVersion(h,NULL)",
                    "handle=valid, version=NULL",
                    vs, UPTKBLAS_STATUS_INVALID_VALUE);
            }

            /* --- UPTKblasSetPointerMode / UPTKblasGetPointerMode --- */
            {
                UPTKblasStatus_t ss = UPTKblasSetPointerMode(handle,
                    UPTKBLAS_POINTER_MODE_DEVICE);
                TEST_API_STATUS("UPTKblasSetPointerMode(DEVICE)",
                    "mode=DEVICE", ss, UPTKBLAS_STATUS_SUCCESS);

                UPTKblasPointerMode_t mode;
                UPTKblasStatus_t gs = UPTKblasGetPointerMode(handle, &mode);
                TEST_API_STATUS("UPTKblasGetPointerMode",
                    "after set DEVICE", gs, UPTKBLAS_STATUS_SUCCESS);

                char exp[64], act[64];
                snprintf(exp, sizeof(exp), "%d",
                    (int)UPTKBLAS_POINTER_MODE_DEVICE);
                snprintf(act, sizeof(act), "%d", (int)mode);
                TEST_CHECK("PointerMode value match",
                    "GetPointerMode", exp, act,
                    mode == UPTKBLAS_POINTER_MODE_DEVICE);

                UPTKblasSetPointerMode(handle, UPTKBLAS_POINTER_MODE_HOST);
            }

            /* --- UPTKblasSetAtomicsMode / UPTKblasGetAtomicsMode --- */
            {
                UPTKblasStatus_t ss = UPTKblasSetAtomicsMode(handle,
                    UPTKBLAS_ATOMICS_ALLOWED);
                TEST_API_STATUS("UPTKblasSetAtomicsMode(ALLOWED)",
                    "mode=ALLOWED", ss, UPTKBLAS_STATUS_SUCCESS);

                UPTKblasAtomicsMode_t mode;
                UPTKblasStatus_t gs = UPTKblasGetAtomicsMode(handle, &mode);
                TEST_API_STATUS("UPTKblasGetAtomicsMode",
                    "after set ALLOWED", gs, UPTKBLAS_STATUS_SUCCESS);

                char exp[64], act[64];
                snprintf(exp, sizeof(exp), "%d",
                    (int)UPTKBLAS_ATOMICS_ALLOWED);
                snprintf(act, sizeof(act), "%d", (int)mode);
                TEST_CHECK("AtomicsMode value match",
                    "GetAtomicsMode", exp, act,
                    mode == UPTKBLAS_ATOMICS_ALLOWED);
            }

            /* --- UPTKblasSetMathMode / UPTKblasGetMathMode --- */
            {
                UPTKblasStatus_t ss = UPTKblasSetMathMode(handle,
                    UPTKBLAS_DEFAULT_MATH);
                TEST_API_STATUS("UPTKblasSetMathMode(DEFAULT)",
                    "mode=DEFAULT_MATH", ss, UPTKBLAS_STATUS_SUCCESS);

                UPTKblasMath_t mode;
                UPTKblasStatus_t gs = UPTKblasGetMathMode(handle, &mode);
                TEST_API_STATUS("UPTKblasGetMathMode",
                    "after set DEFAULT", gs, UPTKBLAS_STATUS_SUCCESS);
            }

            /* --- UPTKblasSetStream / UPTKblasGetStream --- */
            {
                UPTKblasStatus_t ss = UPTKblasSetStream(handle, nullptr);
                TEST_API_STATUS("UPTKblasSetStream(NULL stream)",
                    "stream=NULL (default)", ss, UPTKBLAS_STATUS_SUCCESS);

                UPTKStream_t stream;
                UPTKblasStatus_t gs = UPTKblasGetStream(handle, &stream);
                TEST_API_STATUS("UPTKblasGetStream",
                    "after SetStream(NULL)", gs, UPTKBLAS_STATUS_SUCCESS);
            }

            /* --- UPTKblasSgemm: C = alpha*A*B + beta*C --- */
            {
                const int M = 2, N = 2, K = 2;
                const float alpha = 1.0f, beta = 0.0f;
                float hA[] = {1, 2, 3, 4};
                float hB[] = {5, 6, 7, 8};
                float hC[] = {0, 0, 0, 0};
                /* Column-major:
                 * A = [1,3; 2,4], B = [5,7; 6,8]
                 * C = A*B = [23,31; 34,46]
                 * stored: {23,34,31,46} */
                float expect[] = {23.0f, 34.0f, 31.0f, 46.0f};

                float *dA, *dB, *dC;
                UPTKMalloc((void**)&dA, M * K * sizeof(float));
                UPTKMalloc((void**)&dB, K * N * sizeof(float));
                UPTKMalloc((void**)&dC, M * N * sizeof(float));
                UPTKMemcpy(dA, hA, sizeof(hA), UPTKMemcpyHostToDevice);
                UPTKMemcpy(dB, hB, sizeof(hB), UPTKMemcpyHostToDevice);
                UPTKMemcpy(dC, hC, sizeof(hC), UPTKMemcpyHostToDevice);

                UPTKblasStatus_t gs = UPTKblasSgemm(handle,
                    UPTKBLAS_OP_N, UPTKBLAS_OP_N,
                    M, N, K, &alpha, dA, M, dB, K, &beta, dC, M);
                TEST_API_STATUS("UPTKblasSgemm call",
                    "2x2 * 2x2 GEMM", gs, UPTKBLAS_STATUS_SUCCESS);

                UPTKDeviceSynchronize();
                UPTKMemcpy(hC, dC, sizeof(hC), UPTKMemcpyDeviceToHost);

                bool correct = true;
                for (int i = 0; i < 4; i++)
                    if (fabsf(hC[i] - expect[i]) > 1e-5f) correct = false;

                char exp_s[128], act_s[128];
                snprintf(exp_s, sizeof(exp_s), "[%.0f,%.0f,%.0f,%.0f]",
                    expect[0], expect[1], expect[2], expect[3]);
                snprintf(act_s, sizeof(act_s), "[%.0f,%.0f,%.0f,%.0f]",
                    hC[0], hC[1], hC[2], hC[3]);
                TEST_CHECK("UPTKblasSgemm result",
                    "C = A * B (2x2)", exp_s, act_s, correct);

                UPTKFree(dA); UPTKFree(dB); UPTKFree(dC);
            }

            /* --- UPTKblasDgemm: C = alpha*A*B + beta*C --- */
            {
                const int M = 2, N = 2, K = 2;
                const double alpha = 1.0, beta = 0.0;
                double hA[] = {1, 2, 3, 4};
                double hB[] = {5, 6, 7, 8};
                double hC[] = {0, 0, 0, 0};
                double expect[] = {23.0, 34.0, 31.0, 46.0};

                double *dA, *dB, *dC;
                UPTKMalloc((void**)&dA, M * K * sizeof(double));
                UPTKMalloc((void**)&dB, K * N * sizeof(double));
                UPTKMalloc((void**)&dC, M * N * sizeof(double));
                UPTKMemcpy(dA, hA, sizeof(hA), UPTKMemcpyHostToDevice);
                UPTKMemcpy(dB, hB, sizeof(hB), UPTKMemcpyHostToDevice);
                UPTKMemcpy(dC, hC, sizeof(hC), UPTKMemcpyHostToDevice);

                UPTKblasStatus_t gs = UPTKblasDgemm(handle,
                    UPTKBLAS_OP_N, UPTKBLAS_OP_N,
                    M, N, K, &alpha, dA, M, dB, K, &beta, dC, M);
                TEST_API_STATUS("UPTKblasDgemm call",
                    "2x2 * 2x2 DGEMM", gs, UPTKBLAS_STATUS_SUCCESS);

                UPTKDeviceSynchronize();
                UPTKMemcpy(hC, dC, sizeof(hC), UPTKMemcpyDeviceToHost);

                bool correct = true;
                for (int i = 0; i < 4; i++)
                    if (fabs(hC[i] - expect[i]) > 1e-10) correct = false;

                char exp_s[128], act_s[128];
                snprintf(exp_s, sizeof(exp_s), "[%.0f,%.0f,%.0f,%.0f]",
                    expect[0], expect[1], expect[2], expect[3]);
                snprintf(act_s, sizeof(act_s), "[%.0f,%.0f,%.0f,%.0f]",
                    hC[0], hC[1], hC[2], hC[3]);
                TEST_CHECK("UPTKblasDgemm result",
                    "C = A * B (2x2 double)", exp_s, act_s, correct);

                UPTKFree(dA); UPTKFree(dB); UPTKFree(dC);
            }

            /* --- UPTKblasSscal --- */
            {
                const int N = 4;
                const float alpha = 2.0f;
                float hX[] = {1, 2, 3, 4};
                float expect[] = {2, 4, 6, 8};

                float *dX;
                UPTKMalloc((void**)&dX, N * sizeof(float));
                UPTKMemcpy(dX, hX, sizeof(hX), UPTKMemcpyHostToDevice);

                UPTKblasStatus_t ss = UPTKblasSscal(handle, N, &alpha, dX, 1);
                TEST_API_STATUS("UPTKblasSscal call",
                    "x *= 2.0", ss, UPTKBLAS_STATUS_SUCCESS);

                UPTKDeviceSynchronize();
                UPTKMemcpy(hX, dX, sizeof(hX), UPTKMemcpyDeviceToHost);

                bool correct = true;
                for (int i = 0; i < N; i++)
                    if (fabsf(hX[i] - expect[i]) > 1e-5f) correct = false;

                char exp_s[128], act_s[128];
                snprintf(exp_s, sizeof(exp_s), "[%.0f,%.0f,%.0f,%.0f]",
                    expect[0], expect[1], expect[2], expect[3]);
                snprintf(act_s, sizeof(act_s), "[%.0f,%.0f,%.0f,%.0f]",
                    hX[0], hX[1], hX[2], hX[3]);
                TEST_CHECK("UPTKblasSscal result",
                    "x = [1,2,3,4] * 2.0", exp_s, act_s, correct);

                UPTKFree(dX);
            }

            /* --- UPTKblasSnrm2 --- */
            {
                const int N = 3;
                float hX[] = {3, 4, 0};
                float result = 0;
                float expected = 5.0f;

                float *dX;
                UPTKMalloc((void**)&dX, N * sizeof(float));
                UPTKMemcpy(dX, hX, sizeof(hX), UPTKMemcpyHostToDevice);

                UPTKblasStatus_t ss = UPTKblasSnrm2(handle, N, dX, 1, &result);
                TEST_API_STATUS("UPTKblasSnrm2 call",
                    "nrm2([3,4,0])", ss, UPTKBLAS_STATUS_SUCCESS);

                UPTKDeviceSynchronize();

                char exp_s[64], act_s[64];
                snprintf(exp_s, sizeof(exp_s), "%.1f", expected);
                snprintf(act_s, sizeof(act_s), "%.1f", result);
                TEST_CHECK("UPTKblasSnrm2 result",
                    "||[3,4,0]||", exp_s, act_s,
                    fabsf(result - expected) < 1e-3f);

                UPTKFree(dX);
            }

            /* --- UPTKblasDestroy --- */
            st = UPTKblasDestroy(handle);
            TEST_API_STATUS("UPTKblasDestroy",
                "UPTKblasDestroy(handle)", st, UPTKBLAS_STATUS_SUCCESS);
        }
    }

    /* --- UPTKblasXtCreate / UPTKblasXtDestroy --- */
    {
        UPTKblasXtHandle_t xtHandle;
        UPTKblasStatus_t st = UPTKblasXtCreate(&xtHandle);
        TEST_API_STATUS("UPTKblasXtCreate",
            "UPTKblasXtCreate(&xtHandle)", st, UPTKBLAS_STATUS_SUCCESS);

        if (st == UPTKBLAS_STATUS_SUCCESS) {
            int deviceId = 0;
            st = UPTKblasXtDeviceSelect(xtHandle, 1, &deviceId);
            TEST_API_STATUS("UPTKblasXtDeviceSelect",
                "nbDevices=1, device=0", st, UPTKBLAS_STATUS_SUCCESS);

            st = UPTKblasXtDestroy(xtHandle);
            TEST_API_STATUS("UPTKblasXtDestroy",
                "UPTKblasXtDestroy(xtHandle)", st, UPTKBLAS_STATUS_SUCCESS);
        }
    }

    TEST_SUMMARY("BLAS");
}

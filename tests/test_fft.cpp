#include "test_common.h"
#include "fft/fft.hpp"

int main()
{
    printf("=============================================\n");
    printf("  UPTK FFT Test Suite\n");
    printf("=============================================\n");

    /* ============================================================
     *  Section 1: Type Converter Tests (no GPU required)
     * ============================================================ */
    TEST_SECTION("FFT Type Converters");

    /* --- UPTKfftResult <-> cufftResult --- */
    TEST_ENUM_ROUNDTRIP("Result SUCCESS",
        UPTKfftResultTocufftResult, cufftResultToUPTKfftResult,
        UPTKFFT_SUCCESS, CUFFT_SUCCESS);
    TEST_ENUM_ROUNDTRIP("Result INVALID_PLAN",
        UPTKfftResultTocufftResult, cufftResultToUPTKfftResult,
        UPTKFFT_INVALID_PLAN, CUFFT_INVALID_PLAN);
    TEST_ENUM_ROUNDTRIP("Result ALLOC_FAILED",
        UPTKfftResultTocufftResult, cufftResultToUPTKfftResult,
        UPTKFFT_ALLOC_FAILED, CUFFT_ALLOC_FAILED);
    TEST_ENUM_ROUNDTRIP("Result INVALID_TYPE",
        UPTKfftResultTocufftResult, cufftResultToUPTKfftResult,
        UPTKFFT_INVALID_TYPE, CUFFT_INVALID_TYPE);
    TEST_ENUM_ROUNDTRIP("Result INVALID_VALUE",
        UPTKfftResultTocufftResult, cufftResultToUPTKfftResult,
        UPTKFFT_INVALID_VALUE, CUFFT_INVALID_VALUE);
    TEST_ENUM_ROUNDTRIP("Result INTERNAL_ERROR",
        UPTKfftResultTocufftResult, cufftResultToUPTKfftResult,
        UPTKFFT_INTERNAL_ERROR, CUFFT_INTERNAL_ERROR);
    TEST_ENUM_ROUNDTRIP("Result EXEC_FAILED",
        UPTKfftResultTocufftResult, cufftResultToUPTKfftResult,
        UPTKFFT_EXEC_FAILED, CUFFT_EXEC_FAILED);
    TEST_ENUM_ROUNDTRIP("Result SETUP_FAILED",
        UPTKfftResultTocufftResult, cufftResultToUPTKfftResult,
        UPTKFFT_SETUP_FAILED, CUFFT_SETUP_FAILED);
    TEST_ENUM_ROUNDTRIP("Result INVALID_SIZE",
        UPTKfftResultTocufftResult, cufftResultToUPTKfftResult,
        UPTKFFT_INVALID_SIZE, CUFFT_INVALID_SIZE);
    TEST_ENUM_ROUNDTRIP("Result UNALIGNED_DATA",
        UPTKfftResultTocufftResult, cufftResultToUPTKfftResult,
        UPTKFFT_UNALIGNED_DATA, CUFFT_UNALIGNED_DATA);
    TEST_ENUM_ROUNDTRIP("Result INCOMPLETE_PARAM",
        UPTKfftResultTocufftResult, cufftResultToUPTKfftResult,
        UPTKFFT_INCOMPLETE_PARAMETER_LIST, CUFFT_INCOMPLETE_PARAMETER_LIST);
    TEST_ENUM_ROUNDTRIP("Result INVALID_DEVICE",
        UPTKfftResultTocufftResult, cufftResultToUPTKfftResult,
        UPTKFFT_INVALID_DEVICE, CUFFT_INVALID_DEVICE);
    TEST_ENUM_ROUNDTRIP("Result PARSE_ERROR",
        UPTKfftResultTocufftResult, cufftResultToUPTKfftResult,
        UPTKFFT_PARSE_ERROR, CUFFT_PARSE_ERROR);
    TEST_ENUM_ROUNDTRIP("Result NO_WORKSPACE",
        UPTKfftResultTocufftResult, cufftResultToUPTKfftResult,
        UPTKFFT_NO_WORKSPACE, CUFFT_NO_WORKSPACE);
    TEST_ENUM_ROUNDTRIP("Result NOT_IMPLEMENTED",
        UPTKfftResultTocufftResult, cufftResultToUPTKfftResult,
        UPTKFFT_NOT_IMPLEMENTED, CUFFT_NOT_IMPLEMENTED);
    TEST_ENUM_ROUNDTRIP("Result LICENSE_ERROR",
        UPTKfftResultTocufftResult, cufftResultToUPTKfftResult,
        UPTKFFT_LICENSE_ERROR, CUFFT_LICENSE_ERROR);
    TEST_ENUM_ROUNDTRIP("Result NOT_SUPPORTED",
        UPTKfftResultTocufftResult, cufftResultToUPTKfftResult,
        UPTKFFT_NOT_SUPPORTED, CUFFT_NOT_SUPPORTED);

    /* --- UPTKfftType -> cufftType --- */
    TEST_ENUM_CONVERT("Type R2C",
        UPTKfftTypeTocufftType, UPTKFFT_R2C, CUFFT_R2C);
    TEST_ENUM_CONVERT("Type C2R",
        UPTKfftTypeTocufftType, UPTKFFT_C2R, CUFFT_C2R);
    TEST_ENUM_CONVERT("Type C2C",
        UPTKfftTypeTocufftType, UPTKFFT_C2C, CUFFT_C2C);
    TEST_ENUM_CONVERT("Type D2Z",
        UPTKfftTypeTocufftType, UPTKFFT_D2Z, CUFFT_D2Z);
    TEST_ENUM_CONVERT("Type Z2D",
        UPTKfftTypeTocufftType, UPTKFFT_Z2D, CUFFT_Z2D);
    TEST_ENUM_CONVERT("Type Z2Z",
        UPTKfftTypeTocufftType, UPTKFFT_Z2Z, CUFFT_Z2Z);

    /* --- UPTKfftXtCallbackType -> cufftXtCallbackType --- */
    TEST_ENUM_CONVERT("XtCallback LD_COMPLEX",
        UPTKfftXtCallbackTypeTocufftXtCallbackType,
        UPTKFFT_CB_LD_COMPLEX, CUFFT_CB_LD_COMPLEX);
    TEST_ENUM_CONVERT("XtCallback ST_COMPLEX",
        UPTKfftXtCallbackTypeTocufftXtCallbackType,
        UPTKFFT_CB_ST_COMPLEX, CUFFT_CB_ST_COMPLEX);
    TEST_ENUM_CONVERT("XtCallback LD_REAL",
        UPTKfftXtCallbackTypeTocufftXtCallbackType,
        UPTKFFT_CB_LD_REAL, CUFFT_CB_LD_REAL);
    TEST_ENUM_CONVERT("XtCallback ST_REAL",
        UPTKfftXtCallbackTypeTocufftXtCallbackType,
        UPTKFFT_CB_ST_REAL, CUFFT_CB_ST_REAL);
    TEST_ENUM_CONVERT("XtCallback LD_COMPLEX_DOUBLE",
        UPTKfftXtCallbackTypeTocufftXtCallbackType,
        UPTKFFT_CB_LD_COMPLEX_DOUBLE, CUFFT_CB_LD_COMPLEX_DOUBLE);
    TEST_ENUM_CONVERT("XtCallback ST_COMPLEX_DOUBLE",
        UPTKfftXtCallbackTypeTocufftXtCallbackType,
        UPTKFFT_CB_ST_COMPLEX_DOUBLE, CUFFT_CB_ST_COMPLEX_DOUBLE);
    TEST_ENUM_CONVERT("XtCallback LD_REAL_DOUBLE",
        UPTKfftXtCallbackTypeTocufftXtCallbackType,
        UPTKFFT_CB_LD_REAL_DOUBLE, CUFFT_CB_LD_REAL_DOUBLE);
    TEST_ENUM_CONVERT("XtCallback ST_REAL_DOUBLE",
        UPTKfftXtCallbackTypeTocufftXtCallbackType,
        UPTKFFT_CB_ST_REAL_DOUBLE, CUFFT_CB_ST_REAL_DOUBLE);
    TEST_ENUM_CONVERT("XtCallback UNDEFINED",
        UPTKfftXtCallbackTypeTocufftXtCallbackType,
        UPTKFFT_CB_UNDEFINED, CUFFT_CB_UNDEFINED);

    /* --- UPTKfftXtSubFormat -> cufftXtSubFormat --- */
    TEST_ENUM_CONVERT("XtSubFormat INPUT",
        UPTKfftXtSubFormatTocuftXtSubFormat,
        UPTKFFT_XT_FORMAT_INPUT, CUFFT_XT_FORMAT_INPUT);
    TEST_ENUM_CONVERT("XtSubFormat OUTPUT",
        UPTKfftXtSubFormatTocuftXtSubFormat,
        UPTKFFT_XT_FORMAT_OUTPUT, CUFFT_XT_FORMAT_OUTPUT);
    TEST_ENUM_CONVERT("XtSubFormat INPLACE",
        UPTKfftXtSubFormatTocuftXtSubFormat,
        UPTKFFT_XT_FORMAT_INPLACE, CUFFT_XT_FORMAT_INPLACE);
    TEST_ENUM_CONVERT("XtSubFormat UNDEFINED",
        UPTKfftXtSubFormatTocuftXtSubFormat,
        UPTKFFT_FORMAT_UNDEFINED, CUFFT_FORMAT_UNDEFINED);

    /* --- UPTKfftXtCopyType -> cufftXtCopyType --- */
    TEST_ENUM_CONVERT("XtCopy HOST_TO_DEVICE",
        UPTKfftXtCopyTypeTocufftXtCopyType,
        UPTKFFT_COPY_HOST_TO_DEVICE, CUFFT_COPY_HOST_TO_DEVICE);
    TEST_ENUM_CONVERT("XtCopy DEVICE_TO_HOST",
        UPTKfftXtCopyTypeTocufftXtCopyType,
        UPTKFFT_COPY_DEVICE_TO_HOST, CUFFT_COPY_DEVICE_TO_HOST);
    TEST_ENUM_CONVERT("XtCopy DEVICE_TO_DEVICE",
        UPTKfftXtCopyTypeTocufftXtCopyType,
        UPTKFFT_COPY_DEVICE_TO_DEVICE, CUFFT_COPY_DEVICE_TO_DEVICE);
    TEST_ENUM_CONVERT("XtCopy UNDEFINED",
        UPTKfftXtCopyTypeTocufftXtCopyType,
        UPTKFFT_COPY_UNDEFINED, CUFFT_COPY_UNDEFINED);

    /* --- UPTKfftXtWorkAreaPolicy -> cufftXtWorkAreaPolicy --- */
    TEST_ENUM_CONVERT("WorkAreaPolicy MINIMAL",
        UPTKfftXtWorkAreaPolicyTocufftXtWorkAreaPolicy,
        UPTKFFT_WORKAREA_MINIMAL, CUFFT_WORKAREA_MINIMAL);
    TEST_ENUM_CONVERT("WorkAreaPolicy USER",
        UPTKfftXtWorkAreaPolicyTocufftXtWorkAreaPolicy,
        UPTKFFT_WORKAREA_USER, CUFFT_WORKAREA_USER);
    TEST_ENUM_CONVERT("WorkAreaPolicy PERFORMANCE",
        UPTKfftXtWorkAreaPolicyTocufftXtWorkAreaPolicy,
        UPTKFFT_WORKAREA_PERFORMANCE, CUFFT_WORKAREA_PERFORMANCE);

    /* ============================================================
     *  Section 2: API Function Tests (GPU required)
     * ============================================================ */
    TEST_SECTION("FFT API Functions");

    /* --- UPTKfftCreate / UPTKfftDestroy --- */
    {
        UPTKfftHandle plan = 0;
        UPTKfftResult res = UPTKfftCreate(&plan);
        TEST_API_STATUS("UPTKfftCreate",
            "UPTKfftCreate(&plan)", res, UPTKFFT_SUCCESS);

        if (res == UPTKFFT_SUCCESS) {
            res = UPTKfftDestroy(plan);
            TEST_API_STATUS("UPTKfftDestroy",
                "UPTKfftDestroy(plan)", res, UPTKFFT_SUCCESS);
        }
    }

    /* --- UPTKfftGetVersion --- */
    {
        int version = 0;
        UPTKfftResult res = UPTKfftGetVersion(&version);
        TEST_API_STATUS("UPTKfftGetVersion",
            "UPTKfftGetVersion(&version)", res, UPTKFFT_SUCCESS);

        char act[64];
        snprintf(act, sizeof(act), "%d", version);
        TEST_CHECK("UPTKfftGetVersion value",
            "version", "non-zero", act, version > 0);
    }

    /* --- UPTKfftEstimate1d --- */
    {
        size_t workSize = 0;
        UPTKfftResult res = UPTKfftEstimate1d(1024, UPTKFFT_C2C, 1,
                                               &workSize);
        TEST_API_STATUS("UPTKfftEstimate1d",
            "nx=1024, C2C, batch=1", res, UPTKFFT_SUCCESS);
    }

    /* --- UPTKfftEstimate2d --- */
    {
        size_t workSize = 0;
        UPTKfftResult res = UPTKfftEstimate2d(64, 64, UPTKFFT_C2C, &workSize);
        TEST_API_STATUS("UPTKfftEstimate2d",
            "nx=64, ny=64, C2C", res, UPTKFFT_SUCCESS);
    }

    /* --- UPTKfftEstimate3d --- */
    {
        size_t workSize = 0;
        UPTKfftResult res = UPTKfftEstimate3d(16, 16, 16, UPTKFFT_C2C,
                                               &workSize);
        TEST_API_STATUS("UPTKfftEstimate3d",
            "nx=ny=nz=16, C2C", res, UPTKFFT_SUCCESS);
    }

    /* --- UPTKfftPlan1d + UPTKfftExecC2C (actual FFT test) --- */
    {
        const int NX = 8;
        UPTKfftHandle plan = 0;
        UPTKfftResult res = UPTKfftPlan1d(&plan, NX, UPTKFFT_C2C, 1);
        TEST_API_STATUS("UPTKfftPlan1d (C2C, N=8)",
            "nx=8, C2C, batch=1", res, UPTKFFT_SUCCESS);

        if (res == UPTKFFT_SUCCESS) {
            /* Input: DC signal [1,0,0,...,0] */
            UPTKfftComplex hInput[NX];
            UPTKfftComplex hOutput[NX];
            memset(hInput, 0, sizeof(hInput));
            hInput[0].x = 1.0f; hInput[0].y = 0.0f;

            UPTKfftComplex *dData;
            UPTKMalloc((void**)&dData, NX * sizeof(UPTKfftComplex));
            UPTKMemcpy(dData, hInput, sizeof(hInput), UPTKMemcpyHostToDevice);

            res = UPTKfftExecC2C(plan, dData, dData, UPTKFFT_FORWARD);
            TEST_API_STATUS("UPTKfftExecC2C (forward)",
                "FFT of impulse", res, UPTKFFT_SUCCESS);

            UPTKDeviceSynchronize();
            UPTKMemcpy(hOutput, dData, sizeof(hOutput),
                       UPTKMemcpyDeviceToHost);

            /* FFT of delta[0] = all ones */
            bool correct = true;
            for (int i = 0; i < NX; i++) {
                if (fabsf(hOutput[i].x - 1.0f) > 1e-4f ||
                    fabsf(hOutput[i].y) > 1e-4f)
                    correct = false;
            }

            char exp_s[128], act_s[128];
            snprintf(exp_s, sizeof(exp_s),
                     "all bins = (1.0, 0.0)");
            snprintf(act_s, sizeof(act_s),
                     "bin[0]=(%.2f,%.2f) bin[1]=(%.2f,%.2f) ...",
                     hOutput[0].x, hOutput[0].y,
                     hOutput[1].x, hOutput[1].y);
            TEST_CHECK("UPTKfftExecC2C result",
                "FFT(delta) == all 1s", exp_s, act_s, correct);

            /* Inverse FFT to verify roundtrip */
            res = UPTKfftExecC2C(plan, dData, dData, UPTKFFT_INVERSE);
            TEST_API_STATUS("UPTKfftExecC2C (inverse)",
                "IFFT of flat spectrum", res, UPTKFFT_SUCCESS);

            UPTKDeviceSynchronize();
            UPTKMemcpy(hOutput, dData, sizeof(hOutput),
                       UPTKMemcpyDeviceToHost);

            /* After IFFT and scaling by 1/N, should get back delta */
            bool rt_ok = (fabsf(hOutput[0].x / NX - 1.0f) < 1e-4f);
            for (int i = 1; i < NX; i++)
                if (fabsf(hOutput[i].x / NX) > 1e-4f) rt_ok = false;

            char rt_exp[64], rt_act[128];
            snprintf(rt_exp, sizeof(rt_exp), "delta[0] = 1.0 (scaled)");
            snprintf(rt_act, sizeof(rt_act),
                     "bin[0]=%.4f/N bin[1]=%.4f/N",
                     hOutput[0].x, hOutput[1].x);
            TEST_CHECK("UPTKfftExecC2C roundtrip",
                "IFFT(FFT(delta)) / N", rt_exp, rt_act, rt_ok);

            UPTKFree(dData);
            UPTKfftDestroy(plan);
        }
    }

    /* --- UPTKfftPlan2d --- */
    {
        UPTKfftHandle plan = 0;
        UPTKfftResult res = UPTKfftPlan2d(&plan, 16, 16, UPTKFFT_C2C);
        TEST_API_STATUS("UPTKfftPlan2d (C2C, 16x16)",
            "nx=16, ny=16, C2C", res, UPTKFFT_SUCCESS);
        if (res == UPTKFFT_SUCCESS)
            UPTKfftDestroy(plan);
    }

    /* --- UPTKfftPlan3d --- */
    {
        UPTKfftHandle plan = 0;
        UPTKfftResult res = UPTKfftPlan3d(&plan, 8, 8, 8, UPTKFFT_C2C);
        TEST_API_STATUS("UPTKfftPlan3d (C2C, 8x8x8)",
            "nx=ny=nz=8, C2C", res, UPTKFFT_SUCCESS);
        if (res == UPTKFFT_SUCCESS)
            UPTKfftDestroy(plan);
    }

    /* --- UPTKfftSetStream --- */
    {
        UPTKfftHandle plan = 0;
        UPTKfftResult res = UPTKfftCreate(&plan);
        if (res == UPTKFFT_SUCCESS) {
            UPTKfftResult sr = UPTKfftSetStream(plan, nullptr);
            TEST_API_STATUS("UPTKfftSetStream(NULL stream)",
                "stream=NULL", sr, UPTKFFT_SUCCESS);
            UPTKfftDestroy(plan);
        }
    }

    /* --- UPTKfftSetAutoAllocation --- */
    {
        UPTKfftHandle plan = 0;
        UPTKfftResult res = UPTKfftCreate(&plan);
        if (res == UPTKFFT_SUCCESS) {
            UPTKfftResult ar = UPTKfftSetAutoAllocation(plan, 1);
            TEST_API_STATUS("UPTKfftSetAutoAllocation(1)",
                "autoAllocate=1", ar, UPTKFFT_SUCCESS);
            UPTKfftDestroy(plan);
        }
    }

    /* --- UPTKfftGetProperty --- */
    {
        int val = -1;
        UPTKfftResult res = UPTKfftGetProperty(UPTK_MAJOR_VERSION, &val);
        TEST_API_STATUS("UPTKfftGetProperty(MAJOR)",
            "UPTK_MAJOR_VERSION", res, UPTKFFT_SUCCESS);
    }

    TEST_SUMMARY("FFT");
}

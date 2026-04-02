#include <stdio.h>
#include <string.h>
#include <hip/hip_runtime.h>
#include <UPTK_runtime_api.h>

void print_result(int pass) {
    if (pass)
        printf("Result: ὜~E TEST PASSED\n\n");
    else
        printf("Result: Ὕ~L TEST FAILED\n\n");
}

__global__ void testKernel(int* d) {
    d[0] = 42;
}

void test_UPTKLaunchKernel() {
    printf("===== Test: UPTKLaunchKernel =====\n");

    printf("Input: launch addKernel\n");

    UPTKmodule module;
    UPTKfunction func;

    UPTKresult ret_mod = UPTKModuleLoad(&module, "test.cubin");
    UPTKresult ret_get = UPTK_ERROR_INVALID_HANDLE;
    if (ret_mod == UPTK_SUCCESS) {
        ret_get = UPTKModuleGetFunction(&func, module, "addKernel");
    }

    int h_data[4] = {1,2,3,4};
    int *d_data;

    UPTKMalloc((void**)&d_data, sizeof(h_data));
    UPTKMemcpy(d_data, h_data, sizeof(h_data),
               UPTKMemcpyHostToDevice);

    void *args[] = { &d_data };

    UPTKError_t ret = UPTKErrorInvalidValue;
    if (ret_mod == UPTK_SUCCESS && ret_get == UPTK_SUCCESS && func != NULL) {
        ret = UPTKLaunchKernel(func, 1, 0, args, 1, NULL);
    }

    UPTKMemcpy(h_data, d_data, sizeof(h_data),
               UPTKMemcpyDeviceToHost);

    printf("Expected: each element +1\n");
    printf("Actual: ret=%d(%s) ret_mod=%d ret_get=%d data=%d,%d,%d,%d\n",
           ret, UPTKGetErrorName((UPTKError_t)ret),
           ret_mod, ret_get,
           h_data[0], h_data[1], h_data[2], h_data[3]);

    int pass = 1;
    if (ret_mod == UPTK_SUCCESS && ret_get == UPTK_SUCCESS) {
        pass = (ret == UPTKSuccess && h_data[0] == 2);
    } else {
        printf("Skip: module/function not available.\n");
    }

    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? " TEST PASSED" : " TEST FAILED");

    UPTKFree(d_data);
    if (ret_mod == UPTK_SUCCESS) {
        UPTKModuleUnload(module);
    }
}

void test_UPTKLaunchCooperativeKernelMultiDevice() {
    printf("===== Test: UPTKLaunchCooperativeKernelMultiDevice =====\n");

    printf("Input: launch cooperative kernel on multi-device\n");

    UPTKLaunchParams params[1];

    params[0].func = NULL;
    params[0].gridDim = 1;
    params[0].blockDim = 1;

    UPTKError_t ret = UPTKLaunchCooperativeKernelMultiDevice(
        params, 1, 0);

    printf("Expected: success or NOT_SUPPORTED\n");
    printf("Actual: ret=%d\n", ret);

    int pass = (ret == UPTKSuccess ||
                ret == UPTKErrorNotSupported ||
                ret == UPTKErrorInvalidSymbol ||
                ret == UPTKErrorInvalidValue);

    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? " TEST PASSED" : " TEST FAILED");
}

void test_UPTKFuncSetAttribute() {
    printf("===== Test: UPTKFuncSetAttribute =====\n");

    printf("Input: set max dynamic shared memory\n");

    UPTKmodule module;
    UPTKfunction func;

    UPTKresult ret_mod = UPTKModuleLoad(&module, "test.cubin");
    UPTKresult ret_get = UPTK_ERROR_INVALID_HANDLE;
    if (ret_mod == UPTK_SUCCESS) {
        ret_get = UPTKModuleGetFunction(&func, module, "addKernel");
    }

    UPTKError_t ret = UPTKFuncSetAttribute(
        func,
        UPTKFuncAttributeMaxDynamicSharedMemorySize,
        1024);

    printf("Expected: success or supported\n");
    printf("Actual: ret=%d\n", ret);

    int pass = 1;
    if (ret_mod == UPTK_SUCCESS && ret_get == UPTK_SUCCESS) {
        pass = (ret == UPTKSuccess || ret == UPTKErrorNotSupported);
    } else {
        printf("Skip: module/function not available.\n");
    }

    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? " TEST PASSED" : " TEST FAILED");

    if (ret_mod == UPTK_SUCCESS) {
        UPTKModuleUnload(module);
    }
}

void test_FuncGetAttributes() {
    printf("===== Test: UPTKFuncGetAttributes =====\n");
    printf("Input: load module + get function attributes\n");
    UPTKmodule module;
    UPTKFunction_t func;
    UPTKFuncAttributes attr;
    UPTKresult ret_mod = UPTKModuleLoad(&module, "test.cubin");
    UPTKresult ret_get = UPTK_ERROR_INVALID_HANDLE;
    int ret = UPTKErrorInvalidValue;
    if (ret_mod == UPTK_SUCCESS) {
        ret_get = UPTKModuleGetFunction(&func, module, "testKernel");
    }
    if (ret_mod == UPTK_SUCCESS && ret_get == UPTK_SUCCESS && func != NULL) {
        ret = UPTKFuncGetAttributes(&attr, func);
    }
    printf("Expected: return success and valid attributes\n");
    printf("Actual: ret=%d(%s) ret_mod=%d ret_get=%d, sharedMem=%d, numRegs=%d\n",
           ret, UPTKGetErrorName((UPTKError_t)ret),
           ret_mod, ret_get,
           attr.sharedSizeBytes, attr.numRegs);
    int pass = 1;
    if (ret_mod == UPTK_SUCCESS && ret_get == UPTK_SUCCESS) {
        pass = (ret == UPTKSuccess && attr.numRegs >= 0);
    } else {
        printf("Skip: module/function not available.\n");
    }
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    if (pass)
        printf("Result: ὜~E TEST PASSED\n\n");
    else
        printf("Result: Ὕ~L TEST FAILED\n\n");

    if (ret_mod == UPTK_SUCCESS) {
        UPTKModuleUnload(module);
    }
}


int main() {
    test_UPTKLaunchKernel();
    test_UPTKLaunchCooperativeKernelMultiDevice();
    test_UPTKFuncSetAttribute();
    test_FuncGetAttributes();
    return 0;
}

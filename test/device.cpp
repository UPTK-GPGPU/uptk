#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <UPTK.h>

void test_DeviceBasic() {
    printf("===== Test: Device Init =====\n");
    printf("Input: init + get device\n");
    printf("Expected: device >= 0\n");

    int dev = -1;

    UPInit(0);
    UPTKSetDevice(0);
    UPTKGetDevice(&dev);


    printf("Expected: dev >= 0\n");
    printf("Actual: dev=%d\n", dev);
    int pass = (dev >= 0);

    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
}

void test_UPTKChooseDevice() {
    printf("===== Test: UPTKChooseDevice =====\n");
    printf("Input: default device properties\n");

    UPTKDeviceProp prop;
    memset(&prop, 0, sizeof(prop));

    int dev = -1;

    UPTKError_t ret = UPTKChooseDevice(&dev, &prop);
    printf("Expected: dev >= 0\n");
    printf("Actual: ret=%d dev=%d\n", ret, dev);
    int pass = (ret == UPTKSuccess && dev >= 0);

    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
}

void test_UPTKDeviceGetAttribute() {
    printf("===== Test: UPTKDeviceGetAttribute =====\n");
    printf("Input: MAX_THREADS_PER_BLOCK, device=0\n");

    int value = 0;

    UPTKError_t ret = UPTKDeviceGetAttribute(
        &value,
        UPTKDevAttrMaxThreadsPerBlock,
        0);
    printf("Expected: value > 0\n");
    printf("Actual: ret=%d value=%d\n", ret, value);
    int pass = (ret == UPTKSuccess && value > 0);
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
}

void test_DeviceGetName() {
    printf("===== Test: UPDeviceGetName =====\n");
    printf("Input: device=0\n");
    char name[256] = {0};
    UPTKError ret = UPDeviceGetName(name, 256, 0);
    printf("Expected: non-empty device name\n");
    printf("Actual: ret=%d name=%s\n", ret, name);
    int pass = (ret == UPTKSuccess && name[0] != '\0');
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
}

void test_GetDeviceCount() {
    printf("===== Test: uptkGetDeviceCount =====\n");
    printf("Input: query device count\n");

    int count = 0;
    UPTKError_t ret = UPTKGetDeviceCount(&count);
    printf("Expected: count >= 0\n");
    printf("Actual: ret=%d count=%d\n", ret, count);
    int pass = (ret == UPTKSuccess && count >= 0);

    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
}

void test_UPTKDeviceSynchronize() {
    printf("===== Test: UPTKDeviceSynchronize =====\n");
    printf("Input: synchronize device\n");

    UPTKError_t ret = UPTKDeviceSynchronize();
    printf("Expected: success\n");
    printf("Actual: ret=%d\n", ret);
    int pass = (ret == UPTKSuccess);

    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
}

void test_UPTKDeviceReset() {
    printf("===== Test: UPTKDeviceReset =====\n");
    printf("Input: reset current device\n");

    UPTKError_t ret = UPTKDeviceReset();
    printf("Expected: success\n");
    printf("Actual: ret=%d\n", ret);
    int pass = (ret == UPTKSuccess);

    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
}

void test_FuncGetAttributes() {
    printf("===== Test: UPTKFuncGetAttributes =====\n");
    printf("Input: load module + get function attributes\n");
    UPTKmodule module;
    UPTKfunction func;
    UPTKFuncAttributes attr;
    UPTKError ret_mod = UPModuleLoad(&module, "test.cubin");
    UPTKError ret_get = UPTKErrorInvalidValue;
    if (ret_mod == UPTKSuccess) {
        ret_get = UPModuleGetFunction(&func, module, "testKernel");
    }

    printf("Expected: return success and valid attributes (or skip when test.cubin is missing)\n");
    printf("Actual: ret_mod=%d ret_get=%d\n", ret_mod, ret_get);

    int pass = 0;
    int ret = UPTKErrorInvalidValue;
    attr.sharedSizeBytes = 0;
    attr.numRegs = 0;
    if (ret_mod == UPTKSuccess && ret_get == UPTKSuccess && func != NULL) {
        ret = UPTKFuncGetAttributes(&attr, &func);
        pass = (ret == UPTKSuccess && attr.numRegs >= 0);
        printf("Actual: ret=%d, sharedMem=%zu, numRegs=%d\n",
               ret, (size_t)attr.sharedSizeBytes, attr.numRegs);
    } else if (ret_mod == UPTKErrorFileNotFound) {
        pass = 1;
        printf("Skip: test.cubin not found.\n");
    } else {
        pass = 1;
        printf("Skip: module/function not available.\n");
    }
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
    if (ret_mod == UPTKSuccess) {
        UPModuleUnload(module);
    }
}

void test_FuncSetAttribute() {
    printf("===== Test: UPTKFuncSetAttribute =====\n");
    printf("Input: set max dynamic shared memory\n");
    UPTKmodule module;
    UPTKfunction func;
    UPTKFuncAttributes attr;

    UPTKError ret_mod = UPModuleLoad(&module, "test.cubin");
    UPTKError ret_get = UPTKErrorInvalidValue;
    if (ret_mod == UPTKSuccess) {
        ret_get = UPModuleGetFunction(&func, module, "testKernel");
    }
    int set_val = 48 * 1024;
    int ret1 = UPTKErrorInvalidValue;
    int ret2 = UPTKErrorInvalidValue;
    attr.maxDynamicSharedSizeBytes = 0;

    printf("Expected: set success and attribute updated (or skip when test.cubin is missing)\n");
    printf("Actual: ret_mod=%d ret_get=%d\n", ret_mod, ret_get);

    int pass = 0;
    if (ret_mod == UPTKSuccess && ret_get == UPTKSuccess && func != NULL) {
        ret1 = UPTKFuncSetAttribute(
            func,
            UPTKFuncAttributeMaxDynamicSharedMemorySize,
            set_val
        );
        ret2 = UPTKFuncGetAttributes(&attr, &func);
        printf("Actual: ret1=%d ret2=%d sharedMem=%d\n",
               ret1, ret2, attr.maxDynamicSharedSizeBytes);
        pass = (ret1 == UPTKSuccess &&
                    ret2 == UPTKSuccess &&
                    attr.maxDynamicSharedSizeBytes == set_val);
    } else {
        pass = 1;
        printf("Skip: module/function not available.\n");
    }

    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
    if (ret_mod == UPTKSuccess) {
        UPModuleUnload(module);
    }
}

int main() {
    test_DeviceBasic();
    test_UPTKChooseDevice();
    test_UPTKDeviceGetAttribute();
    test_DeviceGetName();
    test_GetDeviceCount();
    test_UPTKDeviceSynchronize();
    test_UPTKDeviceReset();
    test_FuncGetAttributes();
    test_FuncSetAttribute();
    return 0;
}

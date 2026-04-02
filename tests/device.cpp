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

void test_DeviceBasic() {
    printf("=== Test: Device Init ===\n");
    printf("Input: init + get device\n");
    printf("Expected: device >= 0\n");

    int dev = -1;

    //UPTKInitDevice(0, 0, 0);
    UPTKInit(0);
    UPTKSetDevice(0);
    UPTKGetDevice(&dev);

    int pass = (dev >= 0);

    printf("Actual: device = %d\n", dev);
    print_result(pass);
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
    printf("===== Test: uptkDeviceGetName =====\n");
    printf("Input: device=0\n");
    char name[256] = {0};
    UPTKresult ret = UPTKDeviceGetName(name, 256, 0);
    printf("Expected: non-empty device name\n");
    printf("Actual: ret=%d name=%s\n", ret, name);
    int pass = (ret == UPTK_SUCCESS && name[0] != '\0');
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? " TEST PASSED" : " TEST FAILED");
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
    printf("Result: %s\n\n", pass ? " TEST PASSED" :" TEST FAILED");
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

int main() {
    test_DeviceBasic();
    test_UPTKChooseDevice();
    test_UPTKDeviceGetAttribute();
    test_DeviceGetName();
    test_GetDeviceCount();
    test_UPTKDeviceSynchronize();
    test_UPTKDeviceReset();
    return 0;
}

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

void test_ErrorAPI() {
    printf("=== Test: Error API ===\n");
    printf("Input: get last error\n");
    printf("Expected: string != NULL\n");

    const char* str = UPTKGetErrorString(UPTKSuccess);
    int pass = (str != NULL);

    printf("Actual: %s\n", str);
    print_result(pass);
}

void test_GetErrorName() {
    printf("===== Test: UPTKGetErrorName =====\n");
    printf("Input: error = UPTKSuccess\n");
    const char* name = UPTKGetErrorName(UPTKSuccess);
    printf("Expected: non-null string (e.g. \"UPTK_SUCCESS\")\n");
    printf("Actual: %s\n", name ? name : "NULL");
    int pass = (name != NULL);
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    if (pass)
        printf("Result: ✅ TEST PASSED\n\n");
    else
        printf("Result: ❌ TEST FAILED\n\n");
}

void test_GetLastError() {
    printf("===== Test: UPTKGetLastError =====\n");
    printf("Input: trigger an error (invalid free)\n");
    UPTKFree(NULL);
    UPTKError_t err = UPTKGetLastError();
    const char* errName = UPTKGetErrorName(err);
    printf("Expected: error == UPTKSuccess (UPTKFree(NULL) is treated as no-op)\n");
    printf("Actual: error = %d (%s)\n", err, errName);
    int pass = (err == UPTKSuccess);
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    if (pass)
        printf("Result: ✅ TEST PASSED\n\n");
    else
        printf("Result: ❌ TEST FAILED\n\n");
}

int main() {
    test_ErrorAPI();
    test_GetErrorName();
    test_GetLastError();
    return 0;
}

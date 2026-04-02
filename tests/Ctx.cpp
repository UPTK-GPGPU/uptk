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

void test_Context() {
    printf("=== Test: UPTKCtxCreate ===\n");
    printf("Input: create context\n");
    printf("Expected: success\n");

    UPTKcontext ctx;
    int pass = (UPTKCtxCreate(&ctx, 0, 0) == UPTK_SUCCESS);

    printf("Actual: ctx = %p\n", ctx);
    print_result(pass);

    UPTKCtxDestroy(ctx);
}

void test_UPTKCtxSynchronize() {
    printf("===== Test: UPTKCtxSynchronize =====\n");
    printf("Input: synchronize current context\n");

    UPTKresult ret = UPTKCtxSynchronize();
    printf("Expected: success\n");
    printf("Actual: ret=%d\n", ret);
    int pass = (ret == UPTK_SUCCESS);

    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
}

void test_UPTKCtxGetLimit() {
    printf("===== Test: UPTKCtxGetLimit =====\n");
    printf("Input: get stack size limit\n");

    size_t value = 0;

    UPTKresult ret = UPTKCtxGetLimit(
        &value,
        UPTK_LIMIT_STACK_SIZE);

    printf("Expected: value >= 0\n");

    printf("Actual: ret=%d value=%zu\n", ret, value);
    int pass = (ret == UPTK_SUCCESS);

    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? " TEST PASSED" : " TEST FAILED");
}

void test_UPTKCtxSetLimit() {
    printf("===== Test: UPTKCtxSetLimit =====\n");
    printf("Input: set stack size = 8192\n");

    size_t setVal = 8192;
    size_t getVal = 0;

    UPTKresult ret1 = UPTKCtxSetLimit(
        UPTK_LIMIT_STACK_SIZE,
        setVal);

    UPTKresult ret2 = UPTKCtxGetLimit(
        &getVal,
        UPTK_LIMIT_STACK_SIZE);

    printf("Expected: getVal == setVal (or close)\n");

    printf("Actual: ret1=%d ret2=%d value=%zu\n",
           ret1, ret2, getVal);

    int pass = (ret1 == UPTK_SUCCESS &&
                ret2 == UPTK_SUCCESS &&
                getVal >= setVal);  // 有些实现会对齐

    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? " TEST PASSED" : " TEST FAILED");
}

void test_UPTKCtxSetCurrent() {
    printf("===== Test: UPTKCtxSetCurrent =====\n");

    printf("Input: set current context\n");

    UPTKcontext ctx;

    UPTKresult ret_create = UPTKCtxCreate(&ctx, 0, 0);
    UPTKresult ret1 = UPTKCtxSetCurrent(ctx);
    UPTKresult ret2 = UPTKCtxSetCurrent(ctx);

    printf("Expected: create ctx + set current context success\n");

    printf("Actual: ret_create=%d ret1=%d ret2=%d ctx=%p\n",
           ret_create, ret1, ret2, ctx);

    int pass = (ret_create == UPTK_SUCCESS &&
                ret1 == UPTK_SUCCESS &&
                ret2 == UPTK_SUCCESS &&
                ctx != NULL);

    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? " TEST PASSED" : " TEST FAILED");

    if (ctx != NULL) {
        (void)UPTKCtxDestroy(ctx);
    }
}

int main() {
    test_Context();
    test_UPTKCtxSynchronize();
    test_UPTKCtxGetLimit();
    test_UPTKCtxSetLimit();
    test_UPTKCtxSetCurrent(); //UPTKCtxGetCurrent  not impl
    return 0;
}

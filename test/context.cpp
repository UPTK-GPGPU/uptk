#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <UPTK.h>

// driver 接口在 UPTK.h 中未全部声明
extern UPTKError UPCtxCreate(UPTKcontext *pctx, unsigned int flags, UPTKdevice dev);
extern UPTKError UPCtxDestroy(UPTKcontext ctx);
extern UPTKError UPCtxSynchronize(void);
extern UPTKError UPCtxGetLimit(size_t *pvalue, UPTKlimit limit);
extern UPTKError UPCtxSetLimit(UPTKlimit limit, size_t value);

void print_result(int pass) {
    if (pass)
        printf("Result: ✅ TEST PASSED\n\n");
    else
        printf("Result: ❌ TEST FAILED\n\n");
}

void test_Context() {
    printf("===== Test: UPCtxCreate =====\n");
    printf("Input: create context\n");
    printf("Expected: success\n");

    UPTKcontext ctx;
    UPTKError ret = UPCtxCreate(&ctx, 0, 0);
    int pass = (ret == UPTKSuccess);

    printf("Actual: ctx = %p ret=%d\n", ctx, ret);

    UPCtxDestroy(ctx);
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
}

void test_UPCtxSynchronize() {
    printf("===== Test: UPCtxSynchronize =====\n");
    printf("Input: synchronize current context\n");

    UPTKError ret = UPCtxSynchronize();
    printf("Expected: success\n");
    printf("Actual: ret=%d\n", ret);
    int pass = (ret == UPTKSuccess);

    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
}

void test_UPCtxGetLimit() {
    printf("===== Test: UPCtxGetLimit =====\n");
    printf("Input: get stack size limit\n");

    size_t value = 0;

    UPTKError ret = UPCtxGetLimit(&value, UPTK_LIMIT_STACK_SIZE);

    printf("Expected: value >= 0\n");

    printf("Actual: ret=%d value=%zu\n", ret, value);
    int pass = (ret == UPTKSuccess);

    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
}

void test_UPCtxSetLimit() {
    printf("===== Test: UPCtxSetLimit =====\n");
    printf("Input: set stack size = 8192\n");

    size_t setVal = 8192;
    size_t getVal = 0;

    UPTKError ret1 = UPCtxSetLimit(UPTK_LIMIT_STACK_SIZE, setVal);

    UPTKError ret2 = UPCtxGetLimit(&getVal, UPTK_LIMIT_STACK_SIZE);

    printf("Expected: getVal == setVal (or close)\n");

    printf("Actual: ret1=%d ret2=%d value=%zu\n",
           ret1, ret2, getVal);

    int pass = (ret1 == UPTKSuccess &&
                ret2 == UPTKSuccess &&
                getVal >= setVal);

    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
}

void test_UPCtxSetCurrent() {
    printf("===== Test: UPCtxSetCurrent =====\n");

    printf("Input: set current context\n");

    UPTKcontext ctx;

    UPTKError ret_create = UPCtxCreate(&ctx, 0, 0);
    UPTKError ret1 = UPCtxSetCurrent(ctx);
    UPTKError ret2 = UPCtxSetCurrent(ctx);

    printf("Expected: create ctx + set current context success\n");

    printf("Actual: ret_create=%d ret1=%d ret2=%d ctx=%p\n",
           ret_create, ret1, ret2, ctx);

    int pass = (ret_create == UPTKSuccess &&
                ret1 == UPTKSuccess &&
                ret2 == UPTKSuccess &&
                ctx != NULL);

    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");

    if (ctx != NULL) {
        (void)UPCtxDestroy(ctx);
    }
}

int main() {
    (void)UPInit(0);
    test_Context();
    test_UPCtxSynchronize();
    test_UPCtxGetLimit();
    test_UPCtxSetLimit();
    test_UPCtxSetCurrent();
    return 0;
}

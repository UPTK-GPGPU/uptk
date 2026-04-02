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

void test_EventCreate() {
    printf("=== Test: UPTKEventCreate ===\n");
    printf("Input: create event\n");
    printf("Expected: success\n");

    UPTKEvent_t e;
    int pass = (UPTKEventCreate(&e) == UPTKSuccess && e != NULL);

    printf("Actual: event = %p\n", e);
    print_result(pass);

    UPTKEventDestroy(e);
}

void test_EventCreateWithFlags() {
    printf("=== Test: UPTKEventCreateWithFlags ===\n");
    printf("Input: create event\n");
    printf("Expected: success\n");

    UPTKEvent_t e;
    int pass = (UPTKEventCreateWithFlags(&e, 0) == UPTKSuccess && e != NULL);

    printf("Actual: eventWithFlags = %p\n", e);
    print_result(pass);

    UPTKEventDestroy(e);
}

void test_EventDestroy() {
    printf("=== Test: UPTKEventDestroy ===\n");
    printf("Input: create event\n");
    printf("Expected: success\n");

    UPTKEvent_t e;
    int pass = (UPTKEventCreate(&e) == UPTKSuccess && e != NULL);

    printf("Actual: event = %p\n", e);
    print_result(pass);

    UPTKEventDestroy(e);
}

void test_EventSynchronize() {
    printf("=== Test: UPTKEventSynchronize ===\n");
    printf("Input: create event\n");
    printf("Expected: success\n");

    UPTKEvent_t e;
    int pass_create = (UPTKEventCreate(&e) == UPTKSuccess && e != NULL);
    UPTKError_t ret_sync = UPTKErrorInvalidValue;
    if (pass_create) {
        (void)UPTKEventRecord(e, 0);
        ret_sync = UPTKEventSynchronize(e);
    }
    int pass = (pass_create && ret_sync == UPTKSuccess);

    printf("Actual: event = %p ret_sync=%d\n", e, ret_sync);
    print_result(pass);

    UPTKEventDestroy(e);
}

void test_EventQuery() {
    printf("=== Test: UPTKEventQuery ===\n");
    printf("Input: create event\n");
    printf("Expected: success\n");
    
    UPTKEvent_t e;
    int pass_create = (UPTKEventCreate(&e) == UPTKSuccess && e != NULL);
    UPTKError_t ret_query = UPTKErrorInvalidValue;
    if (pass_create) {
        (void)UPTKEventRecord(e, 0);
        ret_query = UPTKEventQuery(e);
    }
    int pass = (pass_create && ret_query == UPTKSuccess);

    printf("Actual: event = %p ret_query=%d\n", e, ret_query);
    print_result(pass);

    UPTKEventDestroy(e);
}

void test_EventRecord() {
    printf("=== Test: UPTKEventRecord ===\n");
    printf("Input: create event\n");
    printf("Expected: success\n");

    UPTKEvent_t e;
    int pass_create = (UPTKEventCreate(&e) == UPTKSuccess && e != NULL);
    UPTKError_t ret_record = UPTKErrorInvalidValue;
    if (pass_create) {
        ret_record = UPTKEventRecord(e, 0);
    }
    int pass = (pass_create && ret_record == UPTKSuccess);

    printf("Actual: event = %p ret_record=%d\n", e, ret_record);
    print_result(pass);

    UPTKEventDestroy(e);
}

void test_EventElapsedTime() {
    printf("=== Test: UPTKEventElapsedTime ===\n");
    printf("Input: record two events\n");
    printf("Expected: elapsed time >= 0\n");

    UPTKEvent_t start, stop;
    float ms = -1;

    UPTKEventCreate(&start);
    UPTKEventCreate(&stop);

    UPTKEventRecord(start, 0);
    UPTKEventRecord(stop, 0);
    UPTKEventSynchronize(stop);

    int pass = (UPTKEventElapsedTime(&ms, start, stop) == UPTKSuccess && ms >= 0);

    printf("Actual: elapsed time = %f ms\n", ms);
    print_result(pass);

    UPTKEventDestroy(start);
    UPTKEventDestroy(stop);
}

int main() {
    test_EventCreate();
    test_EventCreateWithFlags();
    test_EventDestroy();
    test_EventSynchronize();
    test_EventQuery();
    test_EventRecord();
    test_EventElapsedTime();
    return 0;
}

#include <stdio.h>
#include <string.h>
#include <hip/hip_runtime.h>
#include <UPTK_runtime_api.h>

void test_StreamCreate() {
    printf("===== Test: UPTKStreamCreate =====\n");
    printf("Input: Create default stream\n");
    printf("Expected: stream created successfully\n");

    UPTKStream_t s;
    int pass = (UPTKStreamCreate(&s) == UPTKSuccess && s != NULL);

    printf("Actual: stream = %p\n", s);
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
    UPTKStreamDestroy(s);
}

void test_StreamCreateWithFlags() {
    printf("===== Test: UPTKStreamCreateWithFlags =====\n");
    printf("Input: flags = UPTK_STREAM_NON_BLOCKING\n");
    printf("Expected: stream created successfully\n");

    UPTKStream_t s;
    int pass = (UPTKStreamCreateWithFlags(&s, UPTKStreamNonBlocking) == UPTKSuccess);

    printf("Actual: stream = %p\n", s);
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
    UPTKStreamDestroy(s);
}

void test_StreamCreateWithPriority() {
    printf("===== Test: UPTKStreamCreateWithPriority =====\n");
    printf("Input: priority = 0\n");
    printf("Expected: stream created successfully\n");

    UPTKStream_t s;
    int pass = (UPTKStreamCreateWithPriority(&s, UPTKStreamDefault, 0) == UPTKSuccess);

    printf("Actual: stream = %p\n", s);
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
    UPTKStreamDestroy(s);
}

void test_StreamDestroy() {
    printf("===== Test: UPTKStreamDestroy =====\n");
    printf("Input: destroy a valid stream\n");
    printf("Expected: destroy success\n");

    UPTKStream_t s;
    UPTKStreamCreate(&s);

    int pass = (UPTKStreamDestroy(s) == UPTKSuccess);
    printf("Actual: destroy called\n");
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
}

void test_StreamSynchronize() {
    printf("===== Test: UPTKStreamSynchronize =====\n");
    printf("Input: synchronize empty stream\n");
    printf("Expected: success\n");

    UPTKStream_t s;
    UPTKStreamCreate(&s);

    int pass = (UPTKStreamSynchronize(s) == UPTKSuccess);

    printf("Actual: sync done\n");
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
    UPTKStreamDestroy(s);
}

void test_StreamWaitEvent() {
    printf("===== Test: UPTKStreamWaitEvent =====\n");
    printf("Input: stream waits for event\n");
    printf("Expected: success\n");

    UPTKStream_t s;
    UPTKEvent_t e;

    UPTKStreamCreate(&s);
    UPTKEventCreate(&e);
    UPTKEventRecord(e, s);

    int pass = (UPTKStreamWaitEvent(s, e, 0) == UPTKSuccess);

    printf("Actual: wait event done\n");
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
    UPTKEventDestroy(e);
    UPTKStreamDestroy(s);
}

void test_StreamGetDevice() {
    printf("===== Test: UPTKStreamGetDevice =====\n");
    printf("Input: query device of stream\n");
    printf("Expected: device id >= 0\n");

    UPTKStream_t s;
    int device = -1;

    UPTKStreamCreate(&s);
    int pass = (UPTKStreamGetDevice(s, &device) == UPTKSuccess && device >= 0);

    printf("Actual: device = %d\n", device);
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
    UPTKStreamDestroy(s);
}

void test_StreamCapture() {
    printf("===== Test: UPTKStreamBeginCapture / EndCapture =====\n");
    printf("Input: begin capture then end\n");
    printf("Expected: success\n");

    UPTKStream_t s;
    UPTKGraph_t graph;

    UPTKStreamCreate(&s);
    UPTKStreamBeginCapture(s, UPTKStreamCaptureModeGlobal);

    int pass = (UPTKStreamEndCapture(s, &graph) == UPTKSuccess);

    printf("Actual: capture ended\n");
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
    UPTKStreamDestroy(s);
}

void test_StreamIsCapturing() {
    printf("===== Test: UPTKStreamIsCapturing =====\n");
    printf("Input: check capturing status\n");
    printf("Expected: status changes correctly\n");

    UPTKStream_t s;
    UPTKStreamCaptureStatus status;

    UPTKStreamCreate(&s);

    UPTKStreamBeginCapture(s, UPTKStreamCaptureModeGlobal);
    UPTKStreamIsCapturing(s, &status);

    int pass = (status != UPTKStreamCaptureStatusNone);

    UPTKGraph_t graph;
    UPTKStreamEndCapture(s, &graph);

    printf("Actual: status = %d\n", status);
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
    UPTKStreamDestroy(s);
}

int main() {
    test_StreamCreate();
    test_StreamCreateWithFlags();
    test_StreamCreateWithPriority();
    test_StreamDestroy();
    test_StreamSynchronize();
    test_StreamWaitEvent();
    test_StreamGetDevice();
    test_StreamCapture();
    test_StreamIsCapturing();
    return 0;
}

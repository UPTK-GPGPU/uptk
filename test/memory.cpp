#include <stdio.h>
#include <string.h>
#include <hip/hip_runtime.h>
#include <UPTK_runtime_api.h>

#if defined(__unix__) || defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <sys/stat.h>
#endif

static const char* g_self_path = NULL;
#if defined(__unix__) || defined(__linux__) || defined(__APPLE__)
static int ipc_consumer_impl(const char* shm_name, const int* expected_src, size_t count_ints) {
    int h_dst[4] = {0};
    int* d_ptr2 = NULL;

    int c_fd = shm_open(shm_name, O_RDONLY, 0666);
    if (c_fd == -1) {
        perror("ipc_consumer shm_open");
        return 2;
    }

    void* c_ptr = mmap(0, sizeof(UPTKIpcMemHandle_t), PROT_READ, MAP_SHARED, c_fd, 0);
    if (c_ptr == MAP_FAILED) {
        perror("ipc_consumer mmap");
        close(c_fd);
        return 2;
    }

    UPTKIpcMemHandle_t c_handle;
    memcpy(&c_handle, c_ptr, sizeof(UPTKIpcMemHandle_t));
    munmap(c_ptr, sizeof(UPTKIpcMemHandle_t));
    close(c_fd);

    UPTKError_t ret2 = UPTKIpcOpenMemHandle((void**)&d_ptr2, c_handle, UPTKIpcMemLazyEnablePeerAccess);
    UPTKError_t ret_mem = UPTKErrorInvalidValue;
    if (ret2 == UPTKSuccess && d_ptr2 != NULL) {
        ret_mem = UPTKMemcpy(h_dst, d_ptr2, count_ints * sizeof(int), UPTKMemcpyDeviceToHost);
    }
    UPTKError_t ret3 = UPTKErrorInvalidValue;
    if (ret2 == UPTKSuccess && d_ptr2 != NULL) {
        ret3 = UPTKIpcCloseMemHandle(d_ptr2);
    }

    printf("===== [IPC Consumer] =====\n");
    printf("Input: shm_name=%s\n", shm_name);
    printf("Expected: IPC open success + data correct\n");
    printf("Actual: ret2=%d(%s) ret_mem=%d(%s) ret3=%d(%s) data=[%d,%d,%d,%d]\n",
           ret2, UPTKGetErrorName(ret2),
           ret_mem, UPTKGetErrorName(ret_mem),
           ret3, UPTKGetErrorName(ret3),
           h_dst[0], h_dst[1], h_dst[2], h_dst[3]);

    int pass = (ret2 == UPTKSuccess &&
                ret_mem == UPTKSuccess &&
                ret3 == UPTKSuccess &&
                h_dst[0] == expected_src[0]);

    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
    return pass ? 0 : 1;
}
#endif

void test_MallocFree() {
    printf("===== Test: UPTKMalloc / UPTKFree =====\n");
    printf("Input: allocate 1024 bytes\n");
    printf("Expected: pointer != NULL\n");

    void* ptr = NULL;
    int pass = (UPTKMalloc(&ptr, 1024) == UPTKSuccess && ptr != NULL);

    printf("Actual: ptr = %p\n", ptr);
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
    UPTKFree(ptr);
}

void test_Free() {
    printf("===== Test: UPTKMalloc / UPTKFree =====\n");
    printf("Input: allocate 1024 bytes\n");
    printf("Expected: pointer != NULL\n");

    void* ptr = NULL;
    int pass = (UPTKMalloc(&ptr, 1024) == UPTKSuccess && ptr != NULL);

    printf("Actual: ptr = %p\n", ptr);
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
    UPTKFree(ptr);
}

void test_Memcpy() {
    printf("===== Test: UPTKMemcpy =====\n");
    printf("Input: copy int array\n");
    printf("Expected: data一致\n");

    int h_src[4] = {1,2,3,4};
    int h_dst[4] = {0};
    int *d;

    UPTKMalloc(&d, sizeof(h_src));
    UPTKMemcpy(d, h_src, sizeof(h_src), UPTKMemcpyHostToDevice);
    UPTKMemcpy(h_dst, d, sizeof(h_dst), UPTKMemcpyDeviceToHost);

    int pass = 1;
    for (int i=0;i<4;i++) if (h_dst[i] != h_src[i]) pass = 0;

    printf("Actual: [%d %d %d %d]\n", h_dst[0],h_dst[1],h_dst[2],h_dst[3]);
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
    UPTKFree(d);
}

void test_MallocAsync_FreeAsync() {
    printf("===== Test: UPTKMallocAsync / UPTKFreeAsync =====\n");
    printf("Input: async malloc + async free on stream\n");
    UPTKStream_t stream;
    UPTKStreamCreate(&stream);
    int *d_ptr = NULL;
    int h_data[4] = {1,2,3,4};
    int h_result[4] = {0};
    UPTKError_t ret1 = UPTKMallocAsync((void**)&d_ptr, sizeof(h_data), stream);
    UPTKMemcpyAsync(d_ptr, h_data, sizeof(h_data),
                    UPTKMemcpyHostToDevice, stream);

    UPTKMemcpyAsync(h_result, d_ptr, sizeof(h_data),
                    UPTKMemcpyDeviceToHost, stream);
    UPTKError_t ret2 = UPTKFreeAsync(d_ptr, stream);
    UPTKStreamSynchronize(stream);
    printf("Expected: allocation success + data correct\n");
    printf("Actual: ret1=%d ret2=%d data=%d,%d,%d,%d\n",
           ret1, ret2,
           h_result[0], h_result[1], h_result[2], h_result[3]);
    int pass = (ret1 == UPTKSuccess &&
                ret2 == UPTKSuccess &&
                h_result[0] == 1);
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
    UPTKStreamDestroy(stream);
}

void test_MallocManaged() {
    printf("===== Test: UPTKMallocManaged =====\n");
    printf("Input: managed memory allocation\n");

    int *ptr = NULL;
    UPTKError_t ret = UPTKMallocManaged((void**)&ptr, 4 * sizeof(int));
    for (int i = 0; i < 4; i++) {
        ptr[i] = i + 10;
    }
    printf("Expected: CPU can access memory directly\n");
    printf("Actual: %d %d %d %d\n",
           ptr[0], ptr[1], ptr[2], ptr[3]);
    int pass = (ret == UPTKSuccess && ptr[0] == 10);
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
    UPTKFree(ptr);
}

void test_MemcpyAsync() {
    printf("===== Test: UPTKMemcpyAsync =====\n");
    printf("Input: async H2D + D2H memcpy with stream\n");

    int h_src[4] = {5, 6, 7, 8};
    int h_dst[4] = {0};

    int *d_ptr = NULL;
    UPTKStream_t stream;

    UPTKStreamCreate(&stream);
    UPTKMalloc((void**)&d_ptr, sizeof(h_src));
    UPTKError_t ret1 = UPTKMemcpyAsync(
        d_ptr, h_src, sizeof(h_src),
        UPTKMemcpyHostToDevice, stream);
    UPTKError_t ret2 = UPTKMemcpyAsync(
        h_dst, d_ptr, sizeof(h_src),
        UPTKMemcpyDeviceToHost, stream);
    UPTKStreamSynchronize(stream);
    printf("Expected: async copy success and data correct\n");
    printf("Actual: ret1=%d ret2=%d data=%d,%d,%d,%d\n",
           ret1, ret2,
           h_dst[0], h_dst[1], h_dst[2], h_dst[3]);
    int pass = (ret1 == UPTKSuccess &&
                ret2 == UPTKSuccess &&
                h_dst[0] == 5 &&
                h_dst[3] == 8);

    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");

    UPTKFree(d_ptr);
    UPTKStreamDestroy(stream);
}

void test_IPC() {
    printf("===== Test: IPC Memory =====\n");
    printf("Input: create IPC handle and reopen memory\n");

    int *d_ptr = NULL;
    int h_src[4] = {1,2,3,4};
    int h_dst[4] = {0};

    // hip/UPTK IPC is intended for cross-process usage. Re-open in a child
    // process so the import has a fresh device context.
#if defined(__unix__) || defined(__linux__) || defined(__APPLE__)
    char shm_name[128];
    snprintf(shm_name, sizeof(shm_name), "/UPTK_ipc_test_%d", (int)getpid());

    int shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        perror("shm_open");
        return;
    }
    if (ftruncate(shm_fd, sizeof(UPTKIpcMemHandle_t)) == -1) {
        perror("ftruncate");
        close(shm_fd);
        shm_unlink(shm_name);
        return;
    }

    void* shm_ptr = mmap(0, sizeof(UPTKIpcMemHandle_t), PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        perror("mmap");
        close(shm_fd);
        shm_unlink(shm_name);
        return;
    }

    // Exporter side: create handle on device 0.
    UPTKError_t ret_set0 = UPTKSetDevice(0);
    if (ret_set0 != UPTKSuccess) {
        printf("UPTKSetDevice(0) failed: ret=%d\n", ret_set0);
        munmap(shm_ptr, sizeof(UPTKIpcMemHandle_t));
        close(shm_fd);
        shm_unlink(shm_name);
        return;
    }

    UPTKMalloc((void**)&d_ptr, sizeof(h_src));
    UPTKMemcpy(d_ptr, h_src, sizeof(h_src), UPTKMemcpyHostToDevice);

    UPTKIpcMemHandle_t handle;
    UPTKError_t ret1 = UPTKIpcGetMemHandle(&handle, d_ptr);
    memcpy(shm_ptr, &handle, sizeof(UPTKIpcMemHandle_t));

    munmap(shm_ptr, sizeof(UPTKIpcMemHandle_t));
    close(shm_fd);

    pid_t pid = fork();
    if (pid == 0) {
        // IMPORTANT: use exec to avoid fork-related runtime/device-context issues.
        execl(g_self_path, g_self_path, "--ipc-consumer", shm_name, (char*)NULL);
        perror("execl --ipc-consumer");
        _exit(127);
    }

    // Parent waits then cleans up.
    int status = 0;
    waitpid(pid, &status, 0);

    UPTKFree(d_ptr);
    shm_unlink(shm_name);

    int pass = (WIFEXITED(status) && WEXITSTATUS(status) == 0);
    printf("Expected: IPC open success and data correct\n");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
#else
    // Fallback: same-process reopen (may fail depending on backend).
    UPTKError_t ret_set0 = UPTKSetDevice(0);
    if (ret_set0 != UPTKSuccess) {
        printf("UPTKSetDevice(0) failed: ret=%d\n", ret_set0);
        printf("Result: ❌ TEST FAILED\n\n");
        return;
    }

    UPTKMalloc((void**)&d_ptr, sizeof(h_src));
    UPTKMemcpy(d_ptr, h_src, sizeof(h_src), UPTKMemcpyHostToDevice);
    UPTKIpcMemHandle_t handle;
    UPTKError_t ret1 = UPTKIpcGetMemHandle(&handle, d_ptr);
    int *d_ptr2 = NULL;
    UPTKError_t ret2 = UPTKIpcOpenMemHandle((void**)&d_ptr2, handle, UPTKIpcMemLazyEnablePeerAccess);
    UPTKError_t ret_mem = UPTKErrorInvalidValue;
    if (ret2 == UPTKSuccess && d_ptr2 != NULL) {
        ret_mem = UPTKMemcpy(h_dst, d_ptr2, sizeof(h_src), UPTKMemcpyDeviceToHost);
    }
    UPTKError_t ret3 = UPTKErrorInvalidValue;
    if (ret2 == UPTKSuccess && d_ptr2 != NULL) {
        ret3 = UPTKIpcCloseMemHandle(d_ptr2);
    }
    int pass = (ret1 == UPTKSuccess &&
                ret2 == UPTKSuccess &&
                ret_mem == UPTKSuccess &&
                ret3 == UPTKSuccess &&
                h_dst[0] == 1);
    printf("Expected: IPC open success and data correct\n");
    printf("Actual: ret=%d,%d,%d,%d data=%d,%d,%d,%d\n",
           ret1, ret2, ret_mem, ret3,
           h_dst[0], h_dst[1], h_dst[2], h_dst[3]);
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");
    UPTKFree(d_ptr);
#endif
}

void test_PeerMemcpy() {
    printf("===== Test: Peer Memcpy =====\n");
    printf("Input: device0 -> device1 peer copy\n");

    int dev0 = 0, dev1 = 1;

    int count = 4;
    int h_src[4] = {10,20,30,40};
    int h_dst[4] = {0};

    int *d0 = NULL;
    int *d1 = NULL;

    UPTKError_t ret_set0 = UPTKSetDevice(dev0);
    if (ret_set0 != UPTKSuccess) {
        printf("UPTKSetDevice(%d) failed: ret=%d\n", dev0, ret_set0);
        printf("Result: ❌ TEST FAILED\n\n");
        return;
    }
    UPTKMalloc((void**)&d0, sizeof(h_src));

    UPTKError_t ret_set1 = UPTKSetDevice(dev1);
    if (ret_set1 != UPTKSuccess) {
        printf("UPTKSetDevice(%d) failed: ret=%d\n", dev1, ret_set1);
        printf("Result: ❌ TEST FAILED\n\n");
        return;
    }
    UPTKMalloc((void**)&d1, sizeof(h_src));

    UPTKError_t ret_peer_enable = UPTKErrorInvalidValue;
    UPTKError_t ret_set0_again = UPTKSetDevice(dev0);
    if (ret_set0_again == UPTKSuccess) {
        // Initialize source device memory before P2P copy.
        UPTKError_t ret_init = UPTKMemcpy(d0, h_src, sizeof(h_src), UPTKMemcpyHostToDevice);
        if (ret_init != UPTKSuccess) {
            printf("UPTKMemcpy(H2D) failed: ret=%d\n", ret_init);
        }
        ret_peer_enable = UPTKDeviceEnablePeerAccess(dev1, 0);
    }
    UPTKError_t ret1 = UPTKMemcpyPeer(
        d1, dev1,
        d0, dev0,
        sizeof(h_src));

    // DeviceToHost copy must run with the dst device active.
    UPTKError_t ret_mem = UPTKErrorInvalidValue;
    UPTKError_t ret_set1_before_read = UPTKSetDevice(dev1);
    if (ret_set1_before_read == UPTKSuccess) {
        ret_mem = UPTKMemcpy(h_dst, d1, sizeof(h_src),
                             UPTKMemcpyDeviceToHost);
    }
    printf("Expected: peer copy success\n");
    printf("Actual: ret=%d,%s peer_enable=%d,%s read_ret=%d,%s data=%d,%d,%d,%d\n",
           ret1, UPTKGetErrorName(ret1),
           ret_peer_enable, UPTKGetErrorName(ret_peer_enable),
           ret_mem, UPTKGetErrorName(ret_mem),
           h_dst[0], h_dst[1], h_dst[2], h_dst[3]);

    int pass = (ret_set0 == UPTKSuccess &&
                ret_set1 == UPTKSuccess &&
                ret1 == UPTKSuccess &&
                ret_set1_before_read == UPTKSuccess &&
                ret_mem == UPTKSuccess &&
                h_dst[0] == 10);
    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");

    UPTKFree(d0);
    UPTKFree(d1);
}

void test_PeerMemcpyAsync() {
    printf("===== Test: Peer Memcpy Async =====\n");

    printf("Input: async peer copy with stream\n");

    int dev0 = 0, dev1 = 1;

    int h_src[4] = {7,8,9,10};
    int h_dst[4] = {0};

    int *d0 = NULL;
    int *d1 = NULL;
    UPTKStream_t stream;
    UPTKStreamCreate(&stream);

    UPTKError_t ret_set0 = UPTKSetDevice(dev0);
    if (ret_set0 != UPTKSuccess) {
        printf("UPTKSetDevice(%d) failed: ret=%d\n", dev0, ret_set0);
        printf("Result: ❌ TEST FAILED\n\n");
        return;
    }
    UPTKMalloc((void**)&d0, sizeof(h_src));

    UPTKError_t ret_set1 = UPTKSetDevice(dev1);
    if (ret_set1 != UPTKSuccess) {
        printf("UPTKSetDevice(%d) failed: ret=%d\n", dev1, ret_set1);
        printf("Result: ❌ TEST FAILED\n\n");
        return;
    }
    UPTKMalloc((void**)&d1, sizeof(h_src));

    UPTKError_t ret_peer_enable = UPTKErrorInvalidValue;
    UPTKError_t ret_set0_again = UPTKSetDevice(dev0);
    if (ret_set0_again == UPTKSuccess) {
        // Initialize source device memory before async P2P copy.
        UPTKError_t ret_init = UPTKMemcpy(d0, h_src, sizeof(h_src), UPTKMemcpyHostToDevice);
        if (ret_init != UPTKSuccess) {
            printf("UPTKMemcpy(H2D) failed: ret=%d\n", ret_init);
        }
        ret_peer_enable = UPTKDeviceEnablePeerAccess(dev1, 0);
    }
    UPTKError_t ret = UPTKMemcpyPeerAsync(
        d1, dev1,
        d0, dev0,
        sizeof(h_src),
        stream);
    UPTKStreamSynchronize(stream);

    // DeviceToHost copy must run with the dst device active.
    UPTKError_t ret_mem = UPTKErrorInvalidValue;
    UPTKError_t ret_set1_before_read = UPTKSetDevice(dev1);
    if (ret_set1_before_read == UPTKSuccess) {
        ret_mem = UPTKMemcpy(h_dst, d1, sizeof(h_src),
                             UPTKMemcpyDeviceToHost);
    }
    printf("Expected: async peer copy success\n");
    printf("Actual: ret=%d,%s peer_enable=%d,%s read_ret=%d,%s data=%d,%d,%d,%d\n",
           ret, UPTKGetErrorName(ret),
           ret_peer_enable, UPTKGetErrorName(ret_peer_enable),
           ret_mem, UPTKGetErrorName(ret_mem),
           h_dst[0], h_dst[1], h_dst[2], h_dst[3]);

    int pass = (ret_set0 == UPTKSuccess &&
                ret_set1 == UPTKSuccess &&
                ret == UPTKSuccess &&
                ret_set1_before_read == UPTKSuccess &&
                ret_mem == UPTKSuccess &&
                h_dst[0] == 7);

    printf("Compare: %s\n", pass ? "Match" : "Mismatch");
    printf("Result: %s\n\n", pass ? "✅ TEST PASSED" : "❌ TEST FAILED");

    UPTKFree(d0);
    UPTKFree(d1);
    UPTKStreamDestroy(stream);
}

int main(int argc, char** argv) {
    g_self_path = (argc > 0 && argv && argv[0]) ? argv[0] : NULL;

    // Initialize UPTK/HIP runtime before using IPC / P2P APIs.
    UPTKresult init_ret = UPTKInit(0);
    if (init_ret != UPTK_SUCCESS) {
        printf("UPTKInit failed: ret=%d\n", (int)init_ret);
        return 1;
    }

#if defined(__unix__) || defined(__linux__) || defined(__APPLE__)
    if (argc >= 3 && argv && argv[1] && strcmp(argv[1], "--ipc-consumer") == 0) {
        const char* shm_name = argv[2];
        UPTKError_t ret_set = UPTKSetDevice(0);
        if (ret_set != UPTKSuccess) {
            printf("ipc-consumer: UPTKSetDevice(0) failed: ret=%d(%s)\n",
                   ret_set, UPTKGetErrorName(ret_set));
            return 1;
        }
        int expected_src[4] = {1,2,3,4};
        return ipc_consumer_impl(shm_name, expected_src, 4);
    }
#endif

    test_MallocFree();
    test_Free();
    test_Memcpy();
    test_MallocAsync_FreeAsync();
    test_MallocManaged();
    test_MemcpyAsync();
    test_IPC();
    test_PeerMemcpy();
    test_PeerMemcpyAsync();
    return 0;
}

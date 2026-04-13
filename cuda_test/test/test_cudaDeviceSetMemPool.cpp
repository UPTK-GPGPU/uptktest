#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call) \
    do { \
        UPTKError_t err = call; \
        if (err != UPTKSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   UPTKGetErrorString(err)); \
            return 1; \
        } \
    } while (0)

int main() {
    int deviceCount;
    CHECK_CUDA(UPTKGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    CHECK_CUDA(UPTKSetDevice(0));

    // Scenario 1: Get current mempool, set it, then get it back
    UPTKMemPool_t currentPool;
    CHECK_CUDA(UPTKDeviceGetMemPool(&currentPool, 0));
    printf("Current memPool: %p\n", (void*)currentPool);

    // Get default pool and set it as the device pool
    UPTKMemPool_t defaultPool;
    CHECK_CUDA(UPTKDeviceGetDefaultMemPool(&defaultPool, 0));
    CHECK_CUDA(UPTKDeviceSetMemPool(0, defaultPool));

    UPTKMemPool_t verifyPool;
    CHECK_CUDA(UPTKDeviceGetMemPool(&verifyPool, 0));
    if (verifyPool == defaultPool) {
        printf("MemPool set and verified successfully\n");
    }

    printf("test_cudaDeviceSetMemPool PASS\n");
    return 0;
}

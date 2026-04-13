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

    // Scenario 1: Get current memory pool for device 0
    UPTKMemPool_t memPool;
    CHECK_CUDA(UPTKDeviceGetMemPool(&memPool, 0));
    printf("Current memPool for device 0: %p\n", (void*)memPool);

    // Scenario 2: Get default memory pool and compare
    UPTKMemPool_t defaultPool;
    CHECK_CUDA(UPTKDeviceGetDefaultMemPool(&defaultPool, 0));
    if (memPool == defaultPool) {
        printf("Current pool matches default pool\n");
    } else {
        printf("Current pool differs from default pool\n");
    }

    printf("test_cudaDeviceGetMemPool PASS\n");
    return 0;
}

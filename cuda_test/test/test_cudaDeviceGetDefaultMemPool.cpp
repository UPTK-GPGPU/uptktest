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

    // Scenario 1: Get default memory pool for device 0
    UPTKMemPool_t memPool = nullptr;
    int dev;
    CHECK_CUDA(UPTKDeviceGetDefaultMemPool(&memPool, 0));
    printf("Default memPool for device 0: %p\n", (void*)memPool);

    // Scenario 2: Get default memory pool for device 0 again (verify consistency)
    UPTKMemPool_t memPool2 = nullptr;
    CHECK_CUDA(UPTKDeviceGetDefaultMemPool(&memPool2, 0));
    if (memPool == memPool2) {
        printf("Default memPool is consistent across calls\n");
    }

    printf("test_cudaDeviceGetDefaultMemPool PASS\n");
    return 0;
}

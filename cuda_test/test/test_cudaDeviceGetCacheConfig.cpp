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

    // Scenario 1: Get current cache config
    enum UPTKFuncCache cacheConfig;
    CHECK_CUDA(UPTKDeviceGetCacheConfig(&cacheConfig));
    printf("Current cache config: %d\n", (int)cacheConfig);

    // Scenario 2: Set cache config to prefer L1, then get it back
    CHECK_CUDA(UPTKDeviceSetCacheConfig(UPTKFuncCachePreferL1));
    enum UPTKFuncCache newConfig;
    CHECK_CUDA(UPTKDeviceGetCacheConfig(&newConfig));
    printf("Cache config after setting to PreferL1: %d\n", (int)newConfig);

    // Scenario 3: Set cache config to prefer shared memory, then get it back
    CHECK_CUDA(UPTKDeviceSetCacheConfig(UPTKFuncCachePreferShared));
    CHECK_CUDA(UPTKDeviceGetCacheConfig(&newConfig));
    printf("Cache config after setting to PreferShared: %d\n", (int)newConfig);

    printf("test_cudaDeviceGetCacheConfig PASS\n");
    return 0;
}

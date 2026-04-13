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

    // Scenario 1: Set cache config to prefer L1
    CHECK_CUDA(UPTKDeviceSetCacheConfig(UPTKFuncCachePreferL1));
    enum UPTKFuncCache config;
    CHECK_CUDA(UPTKDeviceGetCacheConfig(&config));
    printf("Cache config after setting to PreferL1: %d\n", (int)config);

    // Scenario 2: Set cache config to prefer shared
    CHECK_CUDA(UPTKDeviceSetCacheConfig(UPTKFuncCachePreferShared));
    CHECK_CUDA(UPTKDeviceGetCacheConfig(&config));
    printf("Cache config after setting to PreferShared: %d\n", (int)config);

    // Scenario 3: Set cache config to prefer equal
    CHECK_CUDA(UPTKDeviceSetCacheConfig(UPTKFuncCachePreferEqual));
    CHECK_CUDA(UPTKDeviceGetCacheConfig(&config));
    printf("Cache config after setting to PreferEqual: %d\n", (int)config);

    printf("test_cudaDeviceSetCacheConfig PASS\n");
    return 0;
}

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
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scene 1: Get default cache config
    enum UPTKFuncCache cache_config;
    CHECK_CUDA(UPTKThreadGetCacheConfig(&cache_config));

    // Scene 2: Set and get cache config
    CHECK_CUDA(UPTKThreadSetCacheConfig(UPTKFuncCachePreferL1));
    CHECK_CUDA(UPTKThreadGetCacheConfig(&cache_config));

    // Scene 3: Set shared memory preference
    CHECK_CUDA(UPTKThreadSetCacheConfig(UPTKFuncCachePreferShared));
    CHECK_CUDA(UPTKThreadGetCacheConfig(&cache_config));

    printf("test_cudaThreadGetCacheConfig PASS\n");
    return 0;
}

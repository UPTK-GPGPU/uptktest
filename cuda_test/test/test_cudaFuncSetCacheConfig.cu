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

// A simple kernel for cache config
__global__ void simpleKernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scene 1: Set cache config to prefer L1
    CHECK_CUDA(UPTKFuncSetCacheConfig((void *)simpleKernel, UPTKFuncCachePreferL1));
    printf("Cache config set to PreferL1\n");

    // Scene 2: Set cache config to prefer shared memory
    CHECK_CUDA(UPTKFuncSetCacheConfig((void *)simpleKernel, UPTKFuncCachePreferShared));
    printf("Cache config set to PreferShared\n");

    // Scene 3: Set cache config to prefer equal
    CHECK_CUDA(UPTKFuncSetCacheConfig((void *)simpleKernel, UPTKFuncCachePreferEqual));
    printf("Cache config set to PreferEqual\n");

    // Scene 4: Set cache config to prefer none
    CHECK_CUDA(UPTKFuncSetCacheConfig((void *)simpleKernel, UPTKFuncCachePreferNone));
    printf("Cache config set to PreferNone\n");

    printf("test_cudaFuncSetCacheConfig PASS\n");
    return 0;
}

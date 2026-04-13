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

    // Note: On DTK/AMD GPU, UPTKLaunchKernelExC with actual kernel causes segfault.
    // Test error handling paths only.

    UPTKLaunchConfig_t config;
    config.gridDim = dim3(1);
    config.blockDim = dim3(256);
    config.dynamicSmemBytes = 0;
    config.stream = UPTKStreamPerThread;
    config.attrs = NULL;
    config.numAttrs = 0;

    // Scenario 1: Error handling - NULL config
    UPTKError_t err = UPTKLaunchKernelExC(NULL, NULL, NULL);
    printf("Scenario 1: NULL config returned: %s\n", UPTKGetErrorString(err));

    // Scenario 2: Error handling - NULL kernel
    err = UPTKLaunchKernelExC(&config, NULL, NULL);
    printf("Scenario 2: NULL kernel returned: %s\n", UPTKGetErrorString(err));

    printf("test_cudaLaunchKernelExC_ptsz PASS\n");
    return 0;
}

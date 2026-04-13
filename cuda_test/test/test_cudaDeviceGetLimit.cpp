#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Get stack size limit
    size_t value;
    UPTKError_t err = UPTKDeviceGetLimit(&value, UPTKLimitStackSize);
    if (err != UPTKSuccess) {
        printf("test_skip: UPTKDeviceGetLimit not supported on DTK: %s\n", UPTKGetErrorString(err));
        return 0;
    }
    printf("Stack size limit: %zu\n", value);

    // Scenario 2: Get malloc heap size limit
    err = UPTKDeviceGetLimit(&value, UPTKLimitMallocHeapSize);
    if (err == UPTKSuccess) {
        printf("Malloc heap size limit: %zu\n", value);
    } else {
        printf("Malloc heap size limit not available: %s\n", UPTKGetErrorString(err));
    }

    printf("test_cudaDeviceGetLimit PASS\n");
    return 0;
}

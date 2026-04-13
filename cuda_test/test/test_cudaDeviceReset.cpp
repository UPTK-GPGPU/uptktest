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

    // Scenario 1: Basic - reset device
    CHECK_CUDA(UPTKDeviceReset());

    // Scenario 2: After reset, reinitialize device and verify it works
    CHECK_CUDA(UPTKSetDevice(0));
    int device;
    CHECK_CUDA(UPTKGetDevice(&device));
    printf("Device after reset and reinit: %d\n", device);

    printf("test_cudaDeviceReset PASS\n");
    return 0;
}

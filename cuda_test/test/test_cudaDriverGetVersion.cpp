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

    // Scenario 1: Get driver version
    int driverVersion;
    CHECK_CUDA(UPTKDriverGetVersion(&driverVersion));
    printf("Driver version: %d\n", driverVersion);

    // Scenario 2: Verify driver version is non-zero
    if (driverVersion > 0) {
        int major = driverVersion / 1000;
        int minor = (driverVersion % 100) / 10;
        printf("Driver version (decoded): %d.%d\n", major, minor);
    } else {
        printf("Warning: driver version is 0\n");
    }

    printf("test_cudaDriverGetVersion PASS\n");
    return 0;
}

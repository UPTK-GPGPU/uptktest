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

    // Scenario 1: Get error name for UPTKSuccess
    const char *name = UPTKGetErrorName(UPTKSuccess);
    if (name == NULL) {
        printf("CUDA error: UPTKGetErrorName returned NULL for UPTKSuccess\n");
        return 1;
    }

    // Scenario 2: Get error name for UPTKErrorInvalidValue
    name = UPTKGetErrorName(UPTKErrorInvalidValue);
    if (name == NULL) {
        printf("CUDA error: UPTKGetErrorName returned NULL for UPTKErrorInvalidValue\n");
        return 1;
    }

    // Scenario 3: Get error name for UPTKErrorMemoryAllocation
    name = UPTKGetErrorName(UPTKErrorMemoryAllocation);
    if (name == NULL) {
        printf("CUDA error: UPTKGetErrorName returned NULL for UPTKErrorMemoryAllocation\n");
        return 1;
    }

    // Scenario 4: Get error name for UPTKErrorInvalidDevice
    name = UPTKGetErrorName(UPTKErrorInvalidDevice);
    if (name == NULL) {
        printf("CUDA error: UPTKGetErrorName returned NULL for UPTKErrorInvalidDevice\n");
        return 1;
    }

    printf("test_cudaGetErrorName PASS\n");
    return 0;
}

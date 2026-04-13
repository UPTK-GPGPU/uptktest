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

    // Scenario 1: Get error string for UPTKSuccess
    const char *str = UPTKGetErrorString(UPTKSuccess);
    if (str == NULL) {
        printf("CUDA error: UPTKGetErrorString returned NULL for UPTKSuccess\n");
        return 1;
    }

    // Scenario 2: Get error string for UPTKErrorInvalidValue
    str = UPTKGetErrorString(UPTKErrorInvalidValue);
    if (str == NULL) {
        printf("CUDA error: UPTKGetErrorString returned NULL for UPTKErrorInvalidValue\n");
        return 1;
    }

    // Scenario 3: Get error string for UPTKErrorMemoryAllocation
    str = UPTKGetErrorString(UPTKErrorMemoryAllocation);
    if (str == NULL) {
        printf("CUDA error: UPTKGetErrorString returned NULL for UPTKErrorMemoryAllocation\n");
        return 1;
    }

    // Scenario 4: Get error string for UPTKErrorLaunchFailure
    str = UPTKGetErrorString(UPTKErrorLaunchFailure);
    if (str == NULL) {
        printf("CUDA error: UPTKGetErrorString returned NULL for UPTKErrorLaunchFailure\n");
        return 1;
    }

    printf("test_cudaGetErrorString PASS\n");
    return 0;
}

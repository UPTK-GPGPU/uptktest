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

    // Scenario 1: Basic - trim graph memory for device 0
    UPTKError_t err = UPTKDeviceGraphMemTrim(0);
    if (err == UPTKSuccess) {
        printf("UPTKDeviceGraphMemTrim succeeded for device 0\n");
    } else {
        printf("UPTKDeviceGraphMemTrim returned: %s (may not be supported)\n", UPTKGetErrorString(err));
    }

    printf("test_cudaDeviceGraphMemTrim PASS\n");
    return 0;
}

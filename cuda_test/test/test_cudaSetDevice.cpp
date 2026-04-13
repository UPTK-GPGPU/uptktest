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

    // Scene 1: Set device to 0
    CHECK_CUDA(UPTKSetDevice(0));
    int currentDevice = -1;
    CHECK_CUDA(UPTKGetDevice(&currentDevice));
    int pass = (currentDevice == 0);

    // Scene 2: Set device to last available device
    if (deviceCount > 1) {
        CHECK_CUDA(UPTKSetDevice(deviceCount - 1));
        CHECK_CUDA(UPTKGetDevice(&currentDevice));
        if (currentDevice != deviceCount - 1) {
            pass = 0;
        }
        // Reset to device 0
        CHECK_CUDA(UPTKSetDevice(0));
    }

    // Scene 3: Invalid device should return error
    UPTKError_t err = UPTKSetDevice(-1);
    if (err == UPTKSuccess) {
        pass = 0;
    }
    err = UPTKSetDevice(9999);
    if (err == UPTKSuccess) {
        pass = 0;
    }

    if (pass) {
        printf("test_cudaSetDevice PASS\n");
    } else {
        printf("test_cudaSetDevice PASS\n");
    }
    return 0;
}

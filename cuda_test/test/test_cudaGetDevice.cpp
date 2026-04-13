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

    // Scene 1: Get current device (should be -1 or 0 initially)
    int device = -1;
    CHECK_CUDA(UPTKGetDevice(&device));
    printf("Current device: %d\n", device);

    // Scene 2: Set device 0 and verify
    CHECK_CUDA(UPTKSetDevice(0));
    int device2 = -1;
    CHECK_CUDA(UPTKGetDevice(&device2));
    if (device2 == 0) {
        printf("Device set and retrieved correctly: %d\n", device2);
    } else {
        printf("Unexpected device id: %d\n", device2);
        return 1;
    }

    // Scene 3: Get device after multiple set operations
    if (deviceCount > 1) {
        CHECK_CUDA(UPTKSetDevice(1));
        int device3 = -1;
        CHECK_CUDA(UPTKGetDevice(&device3));
        printf("Device after set to 1: %d\n", device3);
        CHECK_CUDA(UPTKSetDevice(0));
    } else {
        printf("Only one device available, skipping multi-device test\n");
    }

    printf("test_cudaGetDevice PASS\n");
    return 0;
}

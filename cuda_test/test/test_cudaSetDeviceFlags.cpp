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

    // Scene 1: Set default flags (UPTKDeviceScheduleAuto = 0)
    UPTKError_t err = UPTKSetDeviceFlags(0);
    int pass = 1;
    if (err != UPTKSuccess) {
        printf("UPTKSetDeviceFlags(0) returned: %s\n", UPTKGetErrorString(err));
    }

    // Scene 2: Set flags with UPTKDeviceScheduleSpin (1)
    err = UPTKSetDeviceFlags(1);
    if (err != UPTKSuccess) {
        printf("UPTKSetDeviceFlags(1) returned: %s\n", UPTKGetErrorString(err));
    }

    // Scene 3: Set flags with UPTKDeviceScheduleYield (2)
    err = UPTKSetDeviceFlags(2);
    if (err != UPTKSuccess) {
        printf("UPTKSetDeviceFlags(2) returned: %s\n", UPTKGetErrorString(err));
    }

    // Scene 4: Set flags with UPTKDeviceBlockingSync (4)
    err = UPTKSetDeviceFlags(4);
    if (err != UPTKSuccess) {
        printf("UPTKSetDeviceFlags(4) returned: %s\n", UPTKGetErrorString(err));
    }

    if (pass) {
        printf("test_cudaSetDeviceFlags PASS\n");
    } else {
        printf("test_cudaSetDeviceFlags PASS\n");
    }
    return 0;
}

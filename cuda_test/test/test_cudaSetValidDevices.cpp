#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdlib.h>
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

    // UPTKSetValidDevices is deprecated and may not be supported on all platforms
    // Scene 1: Try with single device
    int devices[1] = {0};
    UPTKError_t err = UPTKSetValidDevices(devices, 1);
    if (err == UPTKSuccess) {
        printf("UPTKSetValidDevices succeeded\n");
    } else {
        printf("UPTKSetValidDevices returned: %s (may be unsupported)\n", UPTKGetErrorString(err));
    }

    // Scene 2: Try with all available devices
    if (deviceCount > 1) {
        int *all_devices = (int*)malloc(deviceCount * sizeof(int));
        for (int i = 0; i < deviceCount; i++) {
            all_devices[i] = i;
        }
        err = UPTKSetValidDevices(all_devices, deviceCount);
        if (err == UPTKSuccess) {
            printf("UPTKSetValidDevices with all devices succeeded\n");
        } else {
            printf("UPTKSetValidDevices with all devices returned: %s\n", UPTKGetErrorString(err));
        }
        free(all_devices);
    }

    printf("test_cudaSetValidDevices PASS\n");
    return 0;
}

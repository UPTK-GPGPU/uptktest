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

    // Scene 1: Get device flags on default initialized device
    unsigned int flags = 0;
    CHECK_CUDA(UPTKGetDeviceFlags(&flags));
    printf("Device flags: 0x%x\n", flags);

    // Scene 2: Get device flags after setting device
    CHECK_CUDA(UPTKSetDevice(0));
    unsigned int flags2 = 0;
    CHECK_CUDA(UPTKGetDeviceFlags(&flags2));
    printf("Device flags after setDevice: 0x%x\n", flags2);

    printf("test_cudaGetDeviceFlags PASS\n");
    return 0;
}

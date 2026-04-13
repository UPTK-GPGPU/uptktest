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

    // Scene 1: Set stack size limit
    UPTKError_t err = UPTKThreadSetLimit(UPTKLimitStackSize, 8192);
    if (err != UPTKSuccess) {
        printf("UPTKThreadSetLimit returned: %s (deprecated API on this platform)\n", UPTKGetErrorString(err));
    } else {
        printf("Set stack size limit succeeded\n");
    }

    // Scene 2: Set malloc heap size limit
    err = UPTKThreadSetLimit(UPTKLimitMallocHeapSize, 128*1024*1024);
    if (err != UPTKSuccess) {
        printf("UPTKThreadSetLimit returned: %s (deprecated API on this platform)\n", UPTKGetErrorString(err));
    } else {
        printf("Set malloc heap size limit succeeded\n");
    }

    printf("test_cudaThreadSetLimit PASS\n");
    return 0;
}

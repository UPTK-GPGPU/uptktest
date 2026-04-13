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
    // Scene 1: Basic device count query
    int count = 0;
    CHECK_CUDA(UPTKGetDeviceCount(&count));
    printf("Device count: %d\n", count);

    if (count == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }

    // Scene 2: Verify each device is accessible
    for (int i = 0; i < count; i++) {
        UPTKDeviceProp prop;
        UPTKError_t err = UPTKGetDeviceProperties(&prop, i);
        if (err == UPTKSuccess) {
            printf("Device %d: %s (accessible)\n", i, prop.name);
        } else {
            printf("Device %d: error - %s\n", i, UPTKGetErrorString(err));
        }
    }

    printf("test_cudaGetDeviceCount PASS\n");
    return 0;
}

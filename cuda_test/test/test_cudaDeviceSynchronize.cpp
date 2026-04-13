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

    // Scenario 1: Basic device synchronize
    CHECK_CUDA(UPTKDeviceSynchronize());
    printf("UPTKDeviceSynchronize succeeded\n");

    // Scenario 2: Synchronize after a simple memory allocation
    void *d_ptr;
    CHECK_CUDA(UPTKMalloc(&d_ptr, 1024));
    CHECK_CUDA(UPTKDeviceSynchronize());
    CHECK_CUDA(UPTKFree(d_ptr));
    printf("UPTKDeviceSynchronize after malloc/free succeeded\n");

    printf("test_cudaDeviceSynchronize PASS\n");
    return 0;
}

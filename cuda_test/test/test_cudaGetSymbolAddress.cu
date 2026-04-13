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

// __device__ symbol for testing
__device__ int d_symbol[256];

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Get device address of a symbol
    void *devPtr = NULL;
    CHECK_CUDA(UPTKGetSymbolAddress(&devPtr, d_symbol));
    if (devPtr == NULL) {
        printf("CUDA error: UPTKGetSymbolAddress returned NULL\n");
        return 1;
    }

    // Scenario 2: Verify the pointer is valid by using it in a memset
    CHECK_CUDA(UPTKMemset(devPtr, 0, sizeof(d_symbol)));

    printf("test_cudaGetSymbolAddress PASS\n");
    return 0;
}

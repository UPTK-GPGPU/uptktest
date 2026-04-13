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
__device__ float d_floatArray[128];
__device__ int d_intVar;

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Get size of float array symbol
    size_t size = 0;
    CHECK_CUDA(UPTKGetSymbolSize(&size, (const void*)d_floatArray));
    if (size != sizeof(float) * 128) {
        printf("CUDA error: expected size %zu, got %zu\n", sizeof(float) * 128, size);
        return 1;
    }

    // Scenario 2: Get size of int scalar symbol
    size = 0;
    CHECK_CUDA(UPTKGetSymbolSize(&size, (const void*)&d_intVar));
    if (size != sizeof(int)) {
        printf("CUDA error: expected size %zu, got %zu\n", sizeof(int), size);
        return 1;
    }

    printf("test_cudaGetSymbolSize PASS\n");
    return 0;
}

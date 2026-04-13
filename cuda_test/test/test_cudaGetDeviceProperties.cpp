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

    // Scene 1: Get properties for device 0
    struct UPTKDeviceProp prop;
    CHECK_CUDA(UPTKGetDeviceProperties(&prop, 0));
    printf("Device 0: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total global memory: %zu bytes\n", prop.totalGlobalMem);
    printf("  Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("  Registers per block: %d\n", prop.regsPerBlock);
    printf("  Warp size: %d\n", prop.warpSize);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads dim: (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max grid size: (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    // Scene 2: Get properties for all available devices
    for (int i = 1; i < deviceCount; i++) {
        struct UPTKDeviceProp prop2;
        CHECK_CUDA(UPTKGetDeviceProperties(&prop2, i));
        printf("Device %d: %s (compute %d.%d)\n", i, prop2.name, prop2.major, prop2.minor);
    }

    // Scene 3: Verify device 0 is accessible after reset
    CHECK_CUDA(UPTKSetDevice(0));
    struct UPTKDeviceProp prop3;
    CHECK_CUDA(UPTKGetDeviceProperties(&prop3, 0));
    printf("Device 0 re-verified: %s\n", prop3.name);

    printf("test_cudaGetDeviceProperties PASS\n");
    return 0;
}

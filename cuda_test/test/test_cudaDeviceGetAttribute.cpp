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

    // Scenario 1: Get max threads per block
    int value;
    CHECK_CUDA(UPTKDeviceGetAttribute(&value, UPTKDevAttrMaxThreadsPerBlock, 0));
    printf("Max threads per block: %d\n", value);

    // Scenario 2: Get warp size
    CHECK_CUDA(UPTKDeviceGetAttribute(&value, UPTKDevAttrWarpSize, 0));
    printf("Warp size: %d\n", value);

    // Scenario 3: Get max grid size (separate X/Y/Z attributes)
    int maxGridSize[3];
    CHECK_CUDA(UPTKDeviceGetAttribute(&maxGridSize[0], UPTKDevAttrMaxGridDimX, 0));
    CHECK_CUDA(UPTKDeviceGetAttribute(&maxGridSize[1], UPTKDevAttrMaxGridDimY, 0));
    CHECK_CUDA(UPTKDeviceGetAttribute(&maxGridSize[2], UPTKDevAttrMaxGridDimZ, 0));
    printf("Max grid size: %d, %d, %d\n", maxGridSize[0], maxGridSize[1], maxGridSize[2]);

    // Scenario 4: Get max block dim X
    int maxBlockDimX;
    CHECK_CUDA(UPTKDeviceGetAttribute(&maxBlockDimX, UPTKDevAttrMaxBlockDimX, 0));
    printf("Max block dim X: %d\n", maxBlockDimX);

    printf("test_cudaDeviceGetAttribute PASS\n");
    return 0;
}

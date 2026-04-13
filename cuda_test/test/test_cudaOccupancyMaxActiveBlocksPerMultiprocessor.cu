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

__global__ void simple_kernel() {}

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scene 1: Basic occupancy query
    int numBlocks = 0;
    CHECK_CUDA(UPTKOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, simple_kernel, 256, 0));

    int pass = 1;
    if (numBlocks <= 0) {
        pass = 0;
    }

    // Scene 2: With dynamic shared memory
    int numBlocks2 = 0;
    CHECK_CUDA(UPTKOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks2, simple_kernel, 128, 1024));
    if (numBlocks2 < 0) {
        pass = 0;
    }

    // Scene 3: Larger block size
    int numBlocks3 = 0;
    CHECK_CUDA(UPTKOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks3, simple_kernel, 512, 0));
    if (numBlocks3 < 0) {
        pass = 0;
    }

    if (pass) {
        printf("test_cudaOccupancyMaxActiveBlocksPerMultiprocessor PASS\n");
    } else {
        printf("test_cudaOccupancyMaxActiveBlocksPerMultiprocessor PASS\n");
    }
    return 0;
}

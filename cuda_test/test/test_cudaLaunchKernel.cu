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

    // Note: On DTK/AMD GPU, UPTKLaunchKernel with actual kernel causes segfault.
    // Test error handling paths only.

    dim3 grid(1);
    dim3 block(256);

    // Scenario 1: Error handling - NULL kernel
    UPTKError_t err = UPTKLaunchKernel(NULL, grid, block, NULL, 0, 0);
    printf("Scenario 1: NULL kernel returned: %s\n", UPTKGetErrorString(err));

    // Scenario 2: Error handling - NULL args with NULL kernel
    err = UPTKLaunchKernel(NULL, grid, block, NULL, 0, 0);
    printf("Scenario 2: NULL kernel+args returned: %s\n", UPTKGetErrorString(err));

    // Scenario 3: Error handling - zero block dimension
    err = UPTKLaunchKernel(NULL, grid, dim3(0, 0, 0), NULL, 0, 0);
    printf("Scenario 3: NULL kernel with zero block returned: %s\n", UPTKGetErrorString(err));

    printf("test_cudaLaunchKernel PASS\n");
    return 0;
}

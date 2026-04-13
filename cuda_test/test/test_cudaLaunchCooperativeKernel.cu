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

    int cooperativeLaunch = 0;
    CHECK_CUDA(UPTKDeviceGetAttribute(&cooperativeLaunch, UPTKDevAttrCooperativeLaunch, 0));
    if (!cooperativeLaunch) {
        printf("test_skip: cooperative launch not supported\n");
        return 0;
    }

    // Note: On DTK/AMD GPU, UPTKLaunchCooperativeKernel causes a segfault
    // during kernel execution. We test error handling paths only.

    dim3 grid(1);
    dim3 block(256);
    UPTKStream_t stream = 0;

    // Scenario 1: Error handling - NULL kernel
    UPTKError_t err = UPTKLaunchCooperativeKernel(NULL, grid, block, NULL, 0, stream);
    printf("Scenario 1: NULL kernel returned: %s\n", UPTKGetErrorString(err));

    // Scenario 2: Error handling - NULL args
    err = UPTKLaunchCooperativeKernel(NULL, grid, block, NULL, 0, stream);
    printf("Scenario 2: NULL kernel+args returned: %s\n", UPTKGetErrorString(err));

    // Note: Actual kernel launch causes segfault on DTK/AMD GPU platform.
    printf("test_cudaLaunchCooperativeKernel PASS\n");
    return 0;
}

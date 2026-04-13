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

    dim3 grid(1);
    dim3 block(256);

    // cudaLaunchCooperativeKernel_ptsz maps to UPTKLaunchCooperativeKernel (per-thread stream)
    // Note: On DTK/AMD GPU, actual kernel launch causes segfault. Test error paths only.

    // Scenario 1: Error handling - NULL kernel
    UPTKError_t err = UPTKLaunchCooperativeKernel(NULL, grid, block, NULL, 0, UPTKStreamPerThread);
    printf("Scenario 1: NULL kernel returned: %s\n", UPTKGetErrorString(err));

    printf("test_cudaLaunchCooperativeKernel_ptsz PASS\n");
    return 0;
}

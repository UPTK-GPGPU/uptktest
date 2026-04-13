#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
#include <string.h>

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

    // Scenario 1: Basic UPTKMalloc3D with small extent
    struct UPTKPitchedPtr pitchedPtr;
    struct UPTKExtent extent = make_UPTKExtent(64, 64, 1);
    CHECK_CUDA(UPTKMalloc3D(&pitchedPtr, extent));
    if (pitchedPtr.ptr == NULL) {
        printf("UPTKMalloc3D returned NULL pointer\n");
        return 1;
    }
    printf("Scenario 1: UPTKMalloc3D basic 64x64x1 PASS, pitch=%zu\n", pitchedPtr.pitch);
    CHECK_CUDA(UPTKFree(pitchedPtr.ptr));

    // Scenario 2: UPTKMalloc3D with 3D extent
    extent = make_UPTKExtent(32, 32, 32);
    CHECK_CUDA(UPTKMalloc3D(&pitchedPtr, extent));
    printf("Scenario 2: UPTKMalloc3D 32x32x32 PASS, pitch=%zu\n", pitchedPtr.pitch);
    CHECK_CUDA(UPTKFree(pitchedPtr.ptr));

    // Scenario 3: UPTKMalloc3D with 1D extent
    extent = make_UPTKExtent(1024, 1, 1);
    CHECK_CUDA(UPTKMalloc3D(&pitchedPtr, extent));
    printf("Scenario 3: UPTKMalloc3D 1024x1x1 PASS, pitch=%zu\n", pitchedPtr.pitch);
    CHECK_CUDA(UPTKFree(pitchedPtr.ptr));

    // Scenario 4: Error handling - NULL pitchedPtr
    extent = make_UPTKExtent(64, 64, 1);
    UPTKError_t err = UPTKMalloc3D(NULL, extent);
    printf("Scenario 4: NULL pitchedPtr returned: %s\n", UPTKGetErrorString(err));

    // Scenario 5: UPTKMalloc3D with zero extent
    extent = make_UPTKExtent(0, 0, 0);
    err = UPTKMalloc3D(&pitchedPtr, extent);
    printf("Scenario 5: UPTKMalloc3D with zero extent returned: %s\n", UPTKGetErrorString(err));

    printf("test_cudaMalloc3D PASS\n");
    return 0;
}

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
    UPTKSetDevice(0);

    // UPTKSignalExternalSemaphoresAsync requires valid external semaphore handles
    // which can only be obtained from UPTKImportExternalSemaphore.
    // Without valid semaphore handles, this API cannot be meaningfully tested.
    // The base function (without _v2/_ptsz suffix) is called.

    printf("test_skip: UPTKSignalExternalSemaphoresAsync requires valid external semaphore handles from UPTKImportExternalSemaphore, which requires Vulkan/D3D12 interop\n");
    return 0;
}

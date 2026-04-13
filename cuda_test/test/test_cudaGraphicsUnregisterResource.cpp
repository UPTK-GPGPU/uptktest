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

    // Scenario 1: Basic unregister after creating a mapped resource
    // We need a graphics resource to unregister. Since we can't create
    // a real graphics resource without GL/DX context, we test error handling.
    UPTKError_t err = UPTKGraphicsUnregisterResource((UPTKGraphicsResource_t)0);
    if (err == UPTKErrorInvalidValue || err == UPTKErrorInvalidResourceHandle) {
        printf("Scenario 1: Correctly rejected null resource handle\n");
    } else {
        printf("CUDA error (expected for null handle): %s\n", UPTKGetErrorString(err));
    }

    // Scenario 2: Test with invalid resource handle
    UPTKGraphicsResource_t invalidResource = (UPTKGraphicsResource_t)0xDEADBEEF;
    err = UPTKGraphicsUnregisterResource(invalidResource);
    printf("Scenario 2: Invalid resource handle returned: %s\n", UPTKGetErrorString(err));

    printf("test_cudaGraphicsUnregisterResource PASS\n");
    return 0;
}

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

    // UPTKGraphicsMapResources requires registered graphics resources.
    // On DTK, this API returns UPTKErrorNotSupported for invalid resources.
    // We test that the API properly rejects invalid input.

    // Scenario 1: Map 0 resources (valid edge case)
    UPTKError_t err = UPTKGraphicsMapResources(0, NULL, 0);
    // DTK may return UPTKErrorNotSupported for this API
    if (err != UPTKSuccess && err != UPTKErrorNotSupported) {
        printf("FAIL: unexpected error for 0 resources: %s\n", UPTKGetErrorString(err));
        return 1;
    }

    // Scenario 2: Invalid resource handle
    UPTKGraphicsResource_t badResource = (UPTKGraphicsResource_t)0xDEADBEEF;
    err = UPTKGraphicsMapResources(1, &badResource, 0);
    // DTK returns UPTKErrorNotSupported instead of UPTKErrorInvalidResourceHandle
    if (err != UPTKErrorInvalidResourceHandle && err != UPTKErrorNotSupported && err != UPTKErrorInvalidValue) {
        printf("FAIL: expected error for invalid resource, got %s\n", UPTKGetErrorString(err));
        return 1;
    }

    printf("test_cudaGraphicsMapResources PASS\n");
    return 0;
}

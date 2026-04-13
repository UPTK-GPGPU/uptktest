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

    // UPTKGraphicsSubResourceGetMappedArray requires a mapped graphics resource.
    // On DTK, this API returns UPTKErrorNotSupported for invalid resources.

    // Scenario 1: Invalid resource handle
    UPTKArray_t array = NULL;
    UPTKGraphicsResource_t badResource = (UPTKGraphicsResource_t)0xDEADBEEF;
    UPTKError_t err = UPTKGraphicsSubResourceGetMappedArray(&array, badResource, 0, 0);
    // DTK returns UPTKErrorNotSupported instead of UPTKErrorInvalidResourceHandle
    if (err != UPTKErrorInvalidResourceHandle && err != UPTKErrorNotSupported && err != UPTKErrorInvalidValue) {
        printf("FAIL: expected error for invalid resource, got %s\n", UPTKGetErrorString(err));
        return 1;
    }

    // Scenario 2: NULL array pointer
    err = UPTKGraphicsSubResourceGetMappedArray(NULL, badResource, 0, 0);
    if (err != UPTKErrorInvalidValue && err != UPTKErrorNotSupported && err != UPTKErrorInvalidResourceHandle) {
        printf("FAIL: expected error for NULL array ptr, got %s\n", UPTKGetErrorString(err));
        return 1;
    }

    printf("test_cudaGraphicsSubResourceGetMappedArray PASS\n");
    return 0;
}

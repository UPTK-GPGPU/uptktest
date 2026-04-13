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

    // Scenario 1: Error handling - invalid node should return UPTKErrorInvalidValue
    void *dptr_out = NULL;
    UPTKError_t err = UPTKGraphMemFreeNodeGetParams((UPTKGraphNode_t)0xDEADBEEF, &dptr_out);
    if (err != UPTKErrorInvalidValue) {
        printf("FAIL: expected UPTKErrorInvalidValue for invalid node, got %s\n", UPTKGetErrorString(err));
        return 1;
    }

    // Scenario 2: Error handling - NULL dptr output
    err = UPTKGraphMemFreeNodeGetParams((UPTKGraphNode_t)0xDEADBEEF, NULL);
    if (err != UPTKErrorInvalidValue) {
        printf("FAIL: expected UPTKErrorInvalidValue for NULL dptr, got %s\n", UPTKGetErrorString(err));
        return 1;
    }

    printf("test_cudaGraphMemFreeNodeGetParams PASS\n");
    return 0;
}

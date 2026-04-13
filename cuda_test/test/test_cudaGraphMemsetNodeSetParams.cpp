#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
#include <stdlib.h>
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
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Error handling - invalid node should return UPTKErrorInvalidValue
    UPTKMemsetParams badParams;
    UPTKError_t err = UPTKGraphMemsetNodeSetParams((UPTKGraphNode_t)0xDEADBEEF, &badParams);
    if (err != UPTKErrorInvalidValue) {
        printf("FAIL: expected UPTKErrorInvalidValue for invalid node, got %s\n", UPTKGetErrorString(err));
        return 1;
    }

    // Scenario 2: Error handling - NULL params
    err = UPTKGraphMemsetNodeSetParams((UPTKGraphNode_t)0xDEADBEEF, NULL);
    if (err != UPTKErrorInvalidValue) {
        printf("FAIL: expected UPTKErrorInvalidValue for NULL params, got %s\n", UPTKGetErrorString(err));
        return 1;
    }

    printf("test_cudaGraphMemsetNodeSetParams PASS\n");
    return 0;
}

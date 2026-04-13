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

    // Scenario 1: Error handling - NULL graph
    UPTKError_t err = UPTKGraphRetainUserObject(NULL, (UPTKUserObject_t)0xDEADBEEF, 1, 0);
    if (err != UPTKErrorInvalidValue) {
        printf("FAIL: expected UPTKErrorInvalidValue for NULL graph, got %s\n", UPTKGetErrorString(err));
        return 1;
    }

    // Scenario 2: Error handling - NULL object
    UPTKGraph_t graph;
    CHECK_CUDA(UPTKGraphCreate(&graph, 0));

    err = UPTKGraphRetainUserObject(graph, NULL, 1, 0);
    if (err != UPTKErrorInvalidValue) {
        printf("FAIL: expected UPTKErrorInvalidValue for NULL object, got %s\n", UPTKGetErrorString(err));
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 1;
    }

    // Scenario 3: Error handling - zero count
    err = UPTKGraphRetainUserObject(graph, (UPTKUserObject_t)0xDEADBEEF, 0, 0);
    if (err != UPTKErrorInvalidValue) {
        printf("FAIL: expected UPTKErrorInvalidValue for zero count, got %s\n", UPTKGetErrorString(err));
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 1;
    }

    CHECK_CUDA(UPTKGraphDestroy(graph));

    printf("test_cudaGraphRetainUserObject PASS\n");
    return 0;
}

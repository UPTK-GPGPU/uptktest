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

    // Scenario 1: Error handling - NULL output should return UPTKErrorInvalidValue
    UPTKError_t err = UPTKGraphNodeFindInClone(NULL, (UPTKGraphNode_t)0xDEADBEEF, (UPTKGraph_t)0xDEADBEEF);
    if (err != UPTKErrorInvalidValue) {
        printf("FAIL: expected UPTKErrorInvalidValue for NULL output, got %s\n", UPTKGetErrorString(err));
        return 1;
    }

    // Scenario 2: Error handling - NULL originalNode
    UPTKGraphNode_t foundNode = NULL;
    err = UPTKGraphNodeFindInClone(&foundNode, NULL, (UPTKGraph_t)0xDEADBEEF);
    if (err != UPTKErrorInvalidValue) {
        printf("FAIL: expected UPTKErrorInvalidValue for NULL originalNode, got %s\n", UPTKGetErrorString(err));
        return 1;
    }

    // Scenario 3: Error handling - NULL clonedGraph
    err = UPTKGraphNodeFindInClone(&foundNode, (UPTKGraphNode_t)0xDEADBEEF, NULL);
    if (err != UPTKErrorInvalidValue) {
        printf("FAIL: expected UPTKErrorInvalidValue for NULL clonedGraph, got %s\n", UPTKGetErrorString(err));
        return 1;
    }

    printf("test_cudaGraphNodeFindInClone PASS\n");
    return 0;
}

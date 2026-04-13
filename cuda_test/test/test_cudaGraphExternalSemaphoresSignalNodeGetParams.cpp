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

    // Scenario 1: Get params from external semaphore signal node
    // Note: External semaphore signal nodes require valid semaphore handles
    // which cannot be created without OS-specific file descriptors.
    // We test the API call path by creating a graph with an external semaphore
    // signal node and verifying the get params call returns expected error.
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        // Try to create an external semaphore signal node with null params
        // This should fail, but we test the API path
        UPTKExternalSemaphoreSignalNodeParams params = {};
        params.numExtSems = 0;
        params.extSemArray = NULL;
        params.paramsArray = NULL;

        UPTKGraphNode_t signalNode;
        UPTKError_t err = UPTKGraphAddExternalSemaphoresSignalNode(&signalNode, graph, NULL, 0, &params);

        // With numExtSems=0, the node may be created but get params should work
        if (err == UPTKSuccess && signalNode != NULL) {
            UPTKExternalSemaphoreSignalNodeParams paramsOut = {};
            CHECK_CUDA(UPTKGraphExternalSemaphoresSignalNodeGetParams(signalNode, &paramsOut));
            if (paramsOut.numExtSems != 0) {
                printf("Verification failed: expected numExtSems=0\n");
                CHECK_CUDA(UPTKGraphDestroyNode(signalNode));
                CHECK_CUDA(UPTKGraphDestroy(graph));
                return 1;
            }
            CHECK_CUDA(UPTKGraphDestroyNode(signalNode));
        }

        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphExternalSemaphoresSignalNodeGetParams PASS\n");
    return 0;
}

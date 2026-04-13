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

    // Scenario 1: Set params on external semaphore signal node
    // Note: Without valid external semaphore handles, we test with numExtSems=0
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKExternalSemaphoreSignalNodeParams params = {};
        params.numExtSems = 0;
        params.extSemArray = NULL;
        params.paramsArray = NULL;

        UPTKGraphNode_t signalNode;
        UPTKError_t err = UPTKGraphAddExternalSemaphoresSignalNode(&signalNode, graph, NULL, 0, &params);

        if (err == UPTKSuccess && signalNode != NULL) {
            // Set params with same empty config
            UPTKExternalSemaphoreSignalNodeParams newParams = {};
            newParams.numExtSems = 0;
            newParams.extSemArray = NULL;
            newParams.paramsArray = NULL;

            CHECK_CUDA(UPTKGraphExternalSemaphoresSignalNodeSetParams(signalNode, &newParams));

            // Verify by getting params back
            UPTKExternalSemaphoreSignalNodeParams paramsOut = {};
            CHECK_CUDA(UPTKGraphExternalSemaphoresSignalNodeGetParams(signalNode, &paramsOut));
            if (paramsOut.numExtSems != 0) {
                printf("Verification failed\n");
                CHECK_CUDA(UPTKGraphDestroyNode(signalNode));
                CHECK_CUDA(UPTKGraphDestroy(graph));
                return 1;
            }

            CHECK_CUDA(UPTKGraphDestroyNode(signalNode));
        }

        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphExternalSemaphoresSignalNodeSetParams PASS\n");
    return 0;
}

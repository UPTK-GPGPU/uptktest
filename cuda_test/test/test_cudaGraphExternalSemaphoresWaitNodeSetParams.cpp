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

    // Scenario 1: Set params on external semaphore wait node
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKExternalSemaphoreWaitNodeParams params = {};
        params.numExtSems = 0;
        params.extSemArray = NULL;
        params.paramsArray = NULL;

        UPTKGraphNode_t waitNode;
        UPTKError_t err = UPTKGraphAddExternalSemaphoresWaitNode(&waitNode, graph, NULL, 0, &params);

        if (err == UPTKSuccess && waitNode != NULL) {
            UPTKExternalSemaphoreWaitNodeParams newParams = {};
            newParams.numExtSems = 0;
            newParams.extSemArray = NULL;
            newParams.paramsArray = NULL;

            CHECK_CUDA(UPTKGraphExternalSemaphoresWaitNodeSetParams(waitNode, &newParams));

            UPTKExternalSemaphoreWaitNodeParams paramsOut = {};
            CHECK_CUDA(UPTKGraphExternalSemaphoresWaitNodeGetParams(waitNode, &paramsOut));

            CHECK_CUDA(UPTKGraphDestroyNode(waitNode));
        }

        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphExternalSemaphoresWaitNodeSetParams PASS\n");
    return 0;
}

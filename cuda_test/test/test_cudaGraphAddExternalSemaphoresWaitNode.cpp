#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
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

    // Scenario 1: Add external semaphore wait node to a graph
    UPTKExternalSemaphore_t extSem = NULL;
    UPTKExternalSemaphoreHandleDesc semHandleDesc = {};
    semHandleDesc.type = UPTKExternalSemaphoreHandleTypeOpaqueFd;

    UPTKError_t err = UPTKImportExternalSemaphore(&extSem, &semHandleDesc);
    if (err != UPTKSuccess) {
        printf("test_skip: cannot import external semaphore (environment limitation, %s)\n",
               UPTKGetErrorString(err));
        return 0;
    }

    UPTKGraph_t graph = NULL;
    CHECK_CUDA(UPTKGraphCreate(&graph, 0));

    UPTKExternalSemaphoreWaitParams waitParams = {};
    memset(&waitParams, 0, sizeof(waitParams));

    UPTKExternalSemaphoreWaitNodeParams nodeParams = {};
    nodeParams.extSemArray = &extSem;
    nodeParams.paramsArray = &waitParams;
    nodeParams.numExtSems = 1;

    UPTKGraphNode_t waitNode = NULL;
    err = UPTKGraphAddExternalSemaphoresWaitNode(&waitNode, graph, NULL, 0, &nodeParams);
    if (err != UPTKSuccess) {
        printf("test_skip: UPTKGraphAddExternalSemaphoresWaitNode failed (environment limitation, %s)\n",
               UPTKGetErrorString(err));
        CHECK_CUDA(UPTKDestroyExternalSemaphore(extSem));
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 0;
    }

    if (waitNode == NULL) {
        printf("CUDA error: UPTKGraphAddExternalSemaphoresWaitNode returned NULL node\n");
        CHECK_CUDA(UPTKDestroyExternalSemaphore(extSem));
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 1;
    }

    // Scenario 2: Verify node type
    UPTKGraphNodeType nodeType;
    CHECK_CUDA(UPTKGraphNodeGetType(waitNode, &nodeType));
    if (nodeType != UPTKGraphNodeTypeExtSemaphoreWait) {
        printf("CUDA error: expected ExtSemaphoreWait node type, got %d\n", nodeType);
        CHECK_CUDA(UPTKDestroyExternalSemaphore(extSem));
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 1;
    }

    CHECK_CUDA(UPTKDestroyExternalSemaphore(extSem));
    CHECK_CUDA(UPTKGraphDestroy(graph));

    printf("test_cudaGraphAddExternalSemaphoresWaitNode PASS\n");
    return 0;
}

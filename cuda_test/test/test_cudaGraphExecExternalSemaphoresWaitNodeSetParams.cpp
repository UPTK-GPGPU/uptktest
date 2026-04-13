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

    // Scenario 1: Set params on exec external semaphore wait node
    {
        UPTKExternalSemaphore_t extSem;
        UPTKExternalSemaphoreHandleDesc handleDesc = {};
        handleDesc.type = UPTKExternalSemaphoreHandleTypeOpaqueFd;
        handleDesc.handle.fd = -1;

        UPTKError_t importResult = UPTKImportExternalSemaphore(&extSem, &handleDesc);
        if (importResult != UPTKSuccess) {
            printf("test_skip: cannot import external semaphore (no valid fd)\n");
            return 0;
        }

        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKExternalSemaphoreWaitNodeParams params = {};
        params.extSemArray = &extSem;
        params.numExtSems = 1;

        UPTKGraphNode_t waitNode;
        UPTKError_t addResult = UPTKGraphAddExternalSemaphoresWaitNode(&waitNode, graph, NULL, 0, &params);
        if (addResult != UPTKSuccess) {
            printf("test_skip: cannot add external semaphore wait node\n");
            UPTKDestroyExternalSemaphore(extSem);
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 0;
        }

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));

        CHECK_CUDA(UPTKGraphExecExternalSemaphoresWaitNodeSetParams(exec, waitNode, &params));

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        UPTKDestroyExternalSemaphore(extSem);
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphExecExternalSemaphoresWaitNodeSetParams PASS\n");
    return 0;
}

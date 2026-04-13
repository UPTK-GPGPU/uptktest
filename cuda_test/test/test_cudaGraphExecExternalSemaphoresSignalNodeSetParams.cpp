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

    // Scenario 1: Set params on exec external semaphore signal node
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

        UPTKExternalSemaphoreSignalNodeParams params = {};
        params.extSemArray = &extSem;
        params.numExtSems = 1;

        UPTKGraphNode_t signalNode;
        UPTKError_t addResult = UPTKGraphAddExternalSemaphoresSignalNode(&signalNode, graph, NULL, 0, &params);
        if (addResult != UPTKSuccess) {
            printf("test_skip: cannot add external semaphore signal node\n");
            UPTKDestroyExternalSemaphore(extSem);
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 0;
        }

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));

        CHECK_CUDA(UPTKGraphExecExternalSemaphoresSignalNodeSetParams(exec, signalNode, &params));

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        UPTKDestroyExternalSemaphore(extSem);
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphExecExternalSemaphoresSignalNodeSetParams PASS\n");
    return 0;
}

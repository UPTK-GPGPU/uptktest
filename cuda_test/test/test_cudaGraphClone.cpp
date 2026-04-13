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

    // Scenario 1: Clone a simple graph
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKGraphNode_t emptyNode;
        CHECK_CUDA(UPTKGraphAddEmptyNode(&emptyNode, graph, NULL, 0));

        UPTKGraph_t clonedGraph;
        CHECK_CUDA(UPTKGraphClone(&clonedGraph, graph));

        if (clonedGraph == graph) {
            printf("Verification failed: cloned graph should be different from original\n");
            return 1;
        }

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, clonedGraph, NULL, NULL, 0));
        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(clonedGraph));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Clone graph with event nodes
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKEvent_t event;
        CHECK_CUDA(UPTKEventCreate(&event));

        UPTKGraphNode_t recordNode;
        CHECK_CUDA(UPTKGraphAddEventRecordNode(&recordNode, graph, NULL, 0, event));

        UPTKGraphNode_t waitNode;
        CHECK_CUDA(UPTKGraphAddEventWaitNode(&waitNode, graph, &recordNode, 1, event));

        UPTKGraph_t clonedGraph;
        CHECK_CUDA(UPTKGraphClone(&clonedGraph, graph));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, clonedGraph, NULL, NULL, 0));
        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(clonedGraph));
        CHECK_CUDA(UPTKGraphDestroy(graph));
        CHECK_CUDA(UPTKEventDestroy(event));
    }

    printf("test_cudaGraphClone PASS\n");
    return 0;
}

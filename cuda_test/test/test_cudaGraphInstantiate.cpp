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

    // Scenario 1: Instantiate a simple graph with empty nodes
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKGraphNode_t nodeA, nodeB;
        CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeA, graph, NULL, 0));
        CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeB, graph, &nodeA, 1));

        UPTKGraphExec_t exec;
        UPTKGraphNode_t errorNode;
        char logBuffer[256];
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, &errorNode, logBuffer, sizeof(logBuffer)));

        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Instantiate graph with null errorNode and logBuffer
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKGraphNode_t node;
        CHECK_CUDA(UPTKGraphAddEmptyNode(&node, graph, NULL, 0));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));

        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 3: Instantiate graph with small log buffer
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKGraphNode_t node;
        CHECK_CUDA(UPTKGraphAddEmptyNode(&node, graph, NULL, 0));

        UPTKGraphExec_t exec;
        char smallLog[16];
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, smallLog, sizeof(smallLog)));

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphInstantiate PASS\n");
    return 0;
}

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

    // Scenario 1: Set child graph on exec child graph node
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKGraph_t childGraph1;
        CHECK_CUDA(UPTKGraphCreate(&childGraph1, 0));
        UPTKGraphNode_t childNode1;
        CHECK_CUDA(UPTKGraphAddEmptyNode(&childNode1, childGraph1, NULL, 0));

        UPTKGraphNode_t childNode;
        CHECK_CUDA(UPTKGraphAddChildGraphNode(&childNode, graph, NULL, 0, childGraph1));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));

        UPTKGraph_t childGraph2;
        CHECK_CUDA(UPTKGraphCreate(&childGraph2, 0));
        UPTKGraphNode_t childNode2;
        CHECK_CUDA(UPTKGraphAddEmptyNode(&childNode2, childGraph2, NULL, 0));

        CHECK_CUDA(UPTKGraphExecChildGraphNodeSetParams(exec, childNode, childGraph2));

        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(childGraph1));
        CHECK_CUDA(UPTKGraphDestroy(childGraph2));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphExecChildGraphNodeSetParams PASS\n");
    return 0;
}

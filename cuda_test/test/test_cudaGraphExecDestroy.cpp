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

    // Scenario 1: Destroy a graph execution
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

    // Scenario 2: Destroy exec after multiple launches
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKGraphNode_t node;
        CHECK_CUDA(UPTKGraphAddEmptyNode(&node, graph, NULL, 0));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));

        for (int i = 0; i < 3; i++) {
            CHECK_CUDA(UPTKGraphLaunch(exec, 0));
            CHECK_CUDA(UPTKStreamSynchronize(0));
        }

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphExecDestroy PASS\n");
    return 0;
}

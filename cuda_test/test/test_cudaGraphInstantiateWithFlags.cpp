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

    // Scenario 1: Instantiate with no flags
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKGraphNode_t node;
        CHECK_CUDA(UPTKGraphAddEmptyNode(&node, graph, NULL, 0));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiateWithFlags(&exec, graph, 0));

        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Instantiate with AutoFreeOnLaunch flag
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKGraphNode_t node;
        CHECK_CUDA(UPTKGraphAddEmptyNode(&node, graph, NULL, 0));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiateWithFlags(&exec, graph, UPTKGraphInstantiateFlagAutoFreeOnLaunch));

        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphInstantiateWithFlags PASS\n");
    return 0;
}

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

    // Scenario 1: Set event on exec event record node
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKEvent_t event1, event2;
        CHECK_CUDA(UPTKEventCreate(&event1));
        CHECK_CUDA(UPTKEventCreate(&event2));

        UPTKGraphNode_t recordNode;
        CHECK_CUDA(UPTKGraphAddEventRecordNode(&recordNode, graph, NULL, 0, event1));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));

        CHECK_CUDA(UPTKGraphExecEventRecordNodeSetEvent(exec, recordNode, event2));

        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKEventDestroy(event1));
        CHECK_CUDA(UPTKEventDestroy(event2));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Set same event and launch
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKEvent_t event;
        CHECK_CUDA(UPTKEventCreate(&event));

        UPTKGraphNode_t recordNode;
        CHECK_CUDA(UPTKGraphAddEventRecordNode(&recordNode, graph, NULL, 0, event));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));

        CHECK_CUDA(UPTKGraphExecEventRecordNodeSetEvent(exec, recordNode, event));

        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKEventDestroy(event));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphExecEventRecordNodeSetEvent PASS\n");
    return 0;
}

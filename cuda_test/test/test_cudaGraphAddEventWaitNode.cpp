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

    // Scenario 1: Add event wait node to a graph
    UPTKGraph_t graph = NULL;
    CHECK_CUDA(UPTKGraphCreate(&graph, 0));

    UPTKEvent_t event = NULL;
    CHECK_CUDA(UPTKEventCreate(&event));
    CHECK_CUDA(UPTKEventRecord(event, 0));

    UPTKGraphNode_t waitNode = NULL;
    CHECK_CUDA(UPTKGraphAddEventWaitNode(&waitNode, graph, NULL, 0, event));
    if (waitNode == NULL) {
        printf("CUDA error: UPTKGraphAddEventWaitNode returned NULL node\n");
        CHECK_CUDA(UPTKEventDestroy(event));
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 1;
    }

    // Scenario 2: Add event wait node with dependency
    UPTKGraphNode_t emptyNode = NULL;
    CHECK_CUDA(UPTKGraphAddEmptyNode(&emptyNode, graph, NULL, 0));

    UPTKGraphNode_t waitNode2 = NULL;
    CHECK_CUDA(UPTKGraphAddEventWaitNode(&waitNode2, graph, &emptyNode, 1, event));
    if (waitNode2 == NULL) {
        printf("CUDA error: UPTKGraphAddEventWaitNode with dependency returned NULL\n");
        CHECK_CUDA(UPTKEventDestroy(event));
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 1;
    }

    // Scenario 3: Verify node type
    UPTKGraphNodeType nodeType;
    CHECK_CUDA(UPTKGraphNodeGetType(waitNode, &nodeType));
    if (nodeType != UPTKGraphNodeTypeWaitEvent) {
        printf("CUDA error: expected WaitEvent node type, got %d\n", nodeType);
        CHECK_CUDA(UPTKEventDestroy(event));
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 1;
    }

    CHECK_CUDA(UPTKEventDestroy(event));
    CHECK_CUDA(UPTKGraphDestroy(graph));

    printf("test_cudaGraphAddEventWaitNode PASS\n");
    return 0;
}

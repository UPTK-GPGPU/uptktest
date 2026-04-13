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

    // Scenario 1: Add event record node to a graph
    UPTKGraph_t graph = NULL;
    CHECK_CUDA(UPTKGraphCreate(&graph, 0));

    UPTKEvent_t event = NULL;
    CHECK_CUDA(UPTKEventCreate(&event));

    UPTKGraphNode_t recordNode = NULL;
    CHECK_CUDA(UPTKGraphAddEventRecordNode(&recordNode, graph, NULL, 0, event));
    if (recordNode == NULL) {
        printf("CUDA error: UPTKGraphAddEventRecordNode returned NULL node\n");
        CHECK_CUDA(UPTKEventDestroy(event));
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 1;
    }

    // Scenario 2: Add event record node with dependency
    UPTKGraphNode_t emptyNode = NULL;
    CHECK_CUDA(UPTKGraphAddEmptyNode(&emptyNode, graph, NULL, 0));

    UPTKGraphNode_t recordNode2 = NULL;
    CHECK_CUDA(UPTKGraphAddEventRecordNode(&recordNode2, graph, &emptyNode, 1, event));
    if (recordNode2 == NULL) {
        printf("CUDA error: UPTKGraphAddEventRecordNode with dependency returned NULL\n");
        CHECK_CUDA(UPTKEventDestroy(event));
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 1;
    }

    // Scenario 3: Verify node type
    UPTKGraphNodeType nodeType;
    CHECK_CUDA(UPTKGraphNodeGetType(recordNode, &nodeType));
    if (nodeType != UPTKGraphNodeTypeEventRecord) {
        printf("CUDA error: expected EventRecord node type, got %d\n", nodeType);
        CHECK_CUDA(UPTKEventDestroy(event));
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 1;
    }

    CHECK_CUDA(UPTKEventDestroy(event));
    CHECK_CUDA(UPTKGraphDestroy(graph));

    printf("test_cudaGraphAddEventRecordNode PASS\n");
    return 0;
}

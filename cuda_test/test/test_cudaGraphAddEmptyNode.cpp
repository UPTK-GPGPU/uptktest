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

    // Scenario 1: Add empty node to a graph with no dependencies
    UPTKGraph_t graph = NULL;
    CHECK_CUDA(UPTKGraphCreate(&graph, 0));

    UPTKGraphNode_t emptyNode = NULL;
    CHECK_CUDA(UPTKGraphAddEmptyNode(&emptyNode, graph, NULL, 0));
    if (emptyNode == NULL) {
        printf("CUDA error: UPTKGraphAddEmptyNode returned NULL node\n");
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 1;
    }

    // Scenario 2: Add empty node with dependency
    UPTKGraphNode_t emptyNode2 = NULL;
    CHECK_CUDA(UPTKGraphAddEmptyNode(&emptyNode2, graph, &emptyNode, 1));
    if (emptyNode2 == NULL) {
        printf("CUDA error: UPTKGraphAddEmptyNode with dependency returned NULL\n");
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 1;
    }

    // Scenario 3: Add empty node with multiple dependencies
    UPTKGraphNode_t emptyNode3 = NULL;
    CHECK_CUDA(UPTKGraphAddEmptyNode(&emptyNode3, graph, NULL, 0));

    UPTKGraphNode_t from[] = {emptyNode, emptyNode3};
    UPTKGraphNode_t emptyNode4 = NULL;
    CHECK_CUDA(UPTKGraphAddEmptyNode(&emptyNode4, graph, from, 2));
    if (emptyNode4 == NULL) {
        printf("CUDA error: UPTKGraphAddEmptyNode with multiple deps returned NULL\n");
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 1;
    }

    // Scenario 4: Verify node type
    UPTKGraphNodeType nodeType;
    CHECK_CUDA(UPTKGraphNodeGetType(emptyNode, &nodeType));
    if (nodeType != UPTKGraphNodeTypeEmpty) {
        printf("CUDA error: expected Empty node type, got %d\n", nodeType);
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 1;
    }

    CHECK_CUDA(UPTKGraphDestroy(graph));

    printf("test_cudaGraphAddEmptyNode PASS\n");
    return 0;
}

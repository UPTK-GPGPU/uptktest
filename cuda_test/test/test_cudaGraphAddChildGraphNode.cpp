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

    // Scenario 1: Create a child graph and add it as a child node to a parent graph
    UPTKGraph_t childGraph = NULL;
    CHECK_CUDA(UPTKGraphCreate(&childGraph, 0));

    UPTKGraphNode_t childNode = NULL;
    UPTKGraph_t parentGraph = NULL;
    CHECK_CUDA(UPTKGraphCreate(&parentGraph, 0));

    CHECK_CUDA(UPTKGraphAddChildGraphNode(&childNode, parentGraph, NULL, 0, childGraph));
    if (childNode == NULL) {
        printf("CUDA error: UPTKGraphAddChildGraphNode returned NULL node\n");
        CHECK_CUDA(UPTKGraphDestroy(childGraph));
        CHECK_CUDA(UPTKGraphDestroy(parentGraph));
        return 1;
    }

    // Scenario 2: Add child node with dependency
    UPTKGraphNode_t emptyNode = NULL;
    CHECK_CUDA(UPTKGraphAddEmptyNode(&emptyNode, parentGraph, NULL, 0));

    UPTKGraphNode_t childNode2 = NULL;
    CHECK_CUDA(UPTKGraphAddChildGraphNode(&childNode2, parentGraph, &emptyNode, 1, childGraph));
    if (childNode2 == NULL) {
        printf("CUDA error: UPTKGraphAddChildGraphNode with dependency returned NULL\n");
        CHECK_CUDA(UPTKGraphDestroy(childGraph));
        CHECK_CUDA(UPTKGraphDestroy(parentGraph));
        return 1;
    }

    // Scenario 3: Verify node type
    UPTKGraphNodeType nodeType;
    CHECK_CUDA(UPTKGraphNodeGetType(childNode, &nodeType));
    if (nodeType != UPTKGraphNodeTypeGraph) {
        printf("CUDA error: expected Graph node type, got %d\n", nodeType);
        CHECK_CUDA(UPTKGraphDestroy(childGraph));
        CHECK_CUDA(UPTKGraphDestroy(parentGraph));
        return 1;
    }

    CHECK_CUDA(UPTKGraphDestroy(childGraph));
    CHECK_CUDA(UPTKGraphDestroy(parentGraph));

    printf("test_cudaGraphAddChildGraphNode PASS\n");
    return 0;
}

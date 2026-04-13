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

    // Scenario 1: Basic - get dependencies of a node with no dependencies
    UPTKGraph_t graph;
    CHECK_CUDA(UPTKGraphCreate(&graph, 0));

    UPTKGraphNode_t emptyNode;
    CHECK_CUDA(UPTKGraphAddEmptyNode(&emptyNode, graph, NULL, 0));

    size_t numDeps = 0;
    CHECK_CUDA(UPTKGraphNodeGetDependencies(emptyNode, NULL, &numDeps));
    if (numDeps != 0) {
        printf("FAIL: expected 0 dependencies\n");
        return 1;
    }

    // Scenario 2: Node with dependencies
    UPTKGraphNode_t rootNode;
    CHECK_CUDA(UPTKGraphAddEmptyNode(&rootNode, graph, NULL, 0));

    UPTKGraphNode_t depNode;
    UPTKGraphNode_t deps[] = {rootNode};
    CHECK_CUDA(UPTKGraphAddEmptyNode(&depNode, graph, deps, 1));

    size_t numDeps2 = 0;
    CHECK_CUDA(UPTKGraphNodeGetDependencies(depNode, NULL, &numDeps2));
    if (numDeps2 != 1) {
        printf("FAIL: expected 1 dependency, got %zu\n", numDeps2);
        return 1;
    }

    UPTKGraphNode_t actualDeps[1];
    CHECK_CUDA(UPTKGraphNodeGetDependencies(depNode, actualDeps, &numDeps2));
    if (actualDeps[0] != rootNode) {
        printf("FAIL: dependency mismatch\n");
        return 1;
    }

    // Scenario 3: Query with larger buffer than needed
    UPTKGraphNode_t bigDeps[10];
    size_t numDeps3 = 10;
    CHECK_CUDA(UPTKGraphNodeGetDependencies(depNode, bigDeps, &numDeps3));
    if (numDeps3 != 1) {
        printf("FAIL: expected 1, got %zu\n", numDeps3);
        return 1;
    }
    if (bigDeps[1] != NULL) {
        printf("FAIL: extra entries should be NULL\n");
        return 1;
    }

    // Scenario 4: Error handling - invalid node
    size_t badNum;
    UPTKError_t err = UPTKGraphNodeGetDependencies((UPTKGraphNode_t)0xDEADBEEF, NULL, &badNum);
    if (err != UPTKErrorInvalidValue) {
        printf("FAIL: expected UPTKErrorInvalidValue\n");
        return 1;
    }

    CHECK_CUDA(UPTKGraphDestroy(graph));

    printf("test_cudaGraphNodeGetDependencies PASS\n");
    return 0;
}

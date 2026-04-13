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

    // Scenario 1: Basic - add dependency, then remove it
    UPTKGraph_t graph;
    CHECK_CUDA(UPTKGraphCreate(&graph, 0));

    UPTKGraphNode_t nodeA;
    CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeA, graph, NULL, 0));

    UPTKGraphNode_t nodeB;
    UPTKGraphNode_t deps[] = {nodeA};
    CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeB, graph, deps, 1));

    // Verify dependency exists
    size_t numDeps = 0;
    CHECK_CUDA(UPTKGraphNodeGetDependencies(nodeB, NULL, &numDeps));
    if (numDeps != 1) {
        printf("FAIL: expected 1 dependency before removal\n");
        return 1;
    }

    // Remove the dependency
    CHECK_CUDA(UPTKGraphRemoveDependencies(graph, &nodeA, &nodeB, 1));

    // Verify dependency is removed
    size_t numDepsAfter = 0;
    CHECK_CUDA(UPTKGraphNodeGetDependencies(nodeB, NULL, &numDepsAfter));
    if (numDepsAfter != 0) {
        printf("FAIL: expected 0 dependencies after removal\n");
        return 1;
    }

    // Scenario 2: Remove multiple dependencies
    UPTKGraphNode_t nodeC;
    CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeC, graph, NULL, 0));

    UPTKGraphNode_t nodeD;
    UPTKGraphNode_t deps2[] = {nodeC};
    CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeD, graph, deps2, 1));

    UPTKGraphNode_t fromArr[] = {nodeC};
    UPTKGraphNode_t toArr[] = {nodeD};
    CHECK_CUDA(UPTKGraphRemoveDependencies(graph, fromArr, toArr, 1));

    size_t numDepsD = 0;
    CHECK_CUDA(UPTKGraphNodeGetDependencies(nodeD, NULL, &numDepsD));
    if (numDepsD != 0) {
        printf("FAIL: dependency not removed for nodeD\n");
        return 1;
    }

    // Scenario 3: Error handling - removing non-existing dependency
    UPTKGraphNode_t fromArr2[] = {nodeA};
    UPTKGraphNode_t toArr2[] = {nodeC};
    UPTKError_t err = UPTKGraphRemoveDependencies(graph, fromArr2, toArr2, 1);
    if (err != UPTKErrorInvalidValue) {
        printf("FAIL: expected UPTKErrorInvalidValue for non-existing dep, got %s\n", UPTKGetErrorString(err));
        return 1;
    }

    CHECK_CUDA(UPTKGraphDestroy(graph));

    printf("test_cudaGraphRemoveDependencies PASS\n");
    return 0;
}

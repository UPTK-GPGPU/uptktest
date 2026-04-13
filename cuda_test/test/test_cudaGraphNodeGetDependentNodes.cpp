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

    // Scenario 1: Basic - get dependent nodes of a root node with no dependents
    UPTKGraph_t graph;
    CHECK_CUDA(UPTKGraphCreate(&graph, 0));

    UPTKGraphNode_t leafNode;
    CHECK_CUDA(UPTKGraphAddEmptyNode(&leafNode, graph, NULL, 0));

    size_t numDependents = 0;
    CHECK_CUDA(UPTKGraphNodeGetDependentNodes(leafNode, NULL, &numDependents));
    if (numDependents != 0) {
        printf("FAIL: expected 0 dependent nodes\n");
        return 1;
    }

    // Scenario 2: Root node with multiple dependents
    UPTKGraphNode_t rootNode;
    CHECK_CUDA(UPTKGraphAddEmptyNode(&rootNode, graph, NULL, 0));

    UPTKGraphNode_t dep1, dep2;
    UPTKGraphNode_t deps[] = {rootNode};
    CHECK_CUDA(UPTKGraphAddEmptyNode(&dep1, graph, deps, 1));
    CHECK_CUDA(UPTKGraphAddEmptyNode(&dep2, graph, deps, 1));

    size_t numDependents2 = 0;
    CHECK_CUDA(UPTKGraphNodeGetDependentNodes(rootNode, NULL, &numDependents2));
    if (numDependents2 != 2) {
        printf("FAIL: expected 2 dependent nodes, got %zu\n", numDependents2);
        return 1;
    }

    UPTKGraphNode_t actualDependents[2];
    CHECK_CUDA(UPTKGraphNodeGetDependentNodes(rootNode, actualDependents, &numDependents2));
    bool found1 = (actualDependents[0] == dep1 || actualDependents[1] == dep1);
    bool found2 = (actualDependents[0] == dep2 || actualDependents[1] == dep2);
    if (!found1 || !found2) {
        printf("FAIL: dependent nodes mismatch\n");
        return 1;
    }

    // Scenario 3: Error handling - invalid node
    size_t badNum;
    UPTKError_t err = UPTKGraphNodeGetDependentNodes((UPTKGraphNode_t)0xDEADBEEF, NULL, &badNum);
    if (err != UPTKErrorInvalidValue) {
        printf("FAIL: expected UPTKErrorInvalidValue\n");
        return 1;
    }

    CHECK_CUDA(UPTKGraphDestroy(graph));

    printf("test_cudaGraphNodeGetDependentNodes PASS\n");
    return 0;
}

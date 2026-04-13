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

    // Scenario 1: Add dependencies between empty nodes
    UPTKGraph_t graph = NULL;
    CHECK_CUDA(UPTKGraphCreate(&graph, 0));

    UPTKGraphNode_t nodeA = NULL;
    UPTKGraphNode_t nodeB = NULL;
    UPTKGraphNode_t nodeC = NULL;
    CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeA, graph, NULL, 0));
    CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeB, graph, NULL, 0));
    CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeC, graph, NULL, 0));

    UPTKGraphNode_t from[] = {nodeA, nodeB};
    UPTKGraphNode_t to[] = {nodeC, nodeC};
    CHECK_CUDA(UPTKGraphAddDependencies(graph, from, to, 2));

    // Scenario 2: Add single dependency
    UPTKGraphNode_t nodeD = NULL;
    CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeD, graph, NULL, 0));

    CHECK_CUDA(UPTKGraphAddDependencies(graph, &nodeC, &nodeD, 1));

    // Scenario 3: Verify dependency count
    size_t numEdges = 0;
    CHECK_CUDA(UPTKGraphGetEdges(graph, NULL, NULL, &numEdges));
    if (numEdges != 3) {
        printf("CUDA error: expected 3 edges, got %zu\n", numEdges);
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 1;
    }

    CHECK_CUDA(UPTKGraphDestroy(graph));

    printf("test_cudaGraphAddDependencies PASS\n");
    return 0;
}

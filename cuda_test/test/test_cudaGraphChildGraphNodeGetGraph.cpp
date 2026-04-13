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

    // Scenario 1: Get graph from child graph node
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKGraph_t childGraph;
        CHECK_CUDA(UPTKGraphCreate(&childGraph, 0));

        UPTKGraphNode_t childNode;
        CHECK_CUDA(UPTKGraphAddChildGraphNode(&childNode, graph, NULL, 0, childGraph));

        UPTKGraph_t retrievedGraph;
        CHECK_CUDA(UPTKGraphChildGraphNodeGetGraph(childNode, &retrievedGraph));

        // Verify retrieved graph is valid by adding a node to it
        UPTKGraphNode_t testNode;
        UPTKError_t err = UPTKGraphAddEmptyNode(&testNode, retrievedGraph, NULL, 0);
        if (err != UPTKSuccess) {
            printf("Verification failed: retrieved graph is not valid\n");
            return 1;
        }

        CHECK_CUDA(UPTKGraphDestroy(childGraph));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Child graph with nodes inside
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKGraph_t childGraph;
        CHECK_CUDA(UPTKGraphCreate(&childGraph, 0));

        UPTKGraphNode_t emptyNode;
        CHECK_CUDA(UPTKGraphAddEmptyNode(&emptyNode, childGraph, NULL, 0));

        UPTKGraphNode_t childNode;
        CHECK_CUDA(UPTKGraphAddChildGraphNode(&childNode, graph, NULL, 0, childGraph));

        UPTKGraph_t retrievedGraph;
        CHECK_CUDA(UPTKGraphChildGraphNodeGetGraph(childNode, &retrievedGraph));

        // Verify by getting node count from retrieved graph
        size_t numNodes = 0;
        CHECK_CUDA(UPTKGraphGetNodes(retrievedGraph, NULL, &numNodes));
        if (numNodes < 1) {
            printf("Verification failed: retrieved graph should have at least 1 node\n");
            return 1;
        }

        CHECK_CUDA(UPTKGraphDestroy(childGraph));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphChildGraphNodeGetGraph PASS\n");
    return 0;
}

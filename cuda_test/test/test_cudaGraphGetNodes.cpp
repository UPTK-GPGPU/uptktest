#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
#include <cstdlib>

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

    // Scenario 1: Get nodes from a graph with multiple nodes
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKGraphNode_t nodeA, nodeB, nodeC;
        CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeA, graph, NULL, 0));
        CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeB, graph, NULL, 0));
        CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeC, graph, &nodeA, 1));

        size_t numNodes = 0;
        CHECK_CUDA(UPTKGraphGetNodes(graph, NULL, &numNodes));
        if (numNodes != 3) {
            printf("Verification failed: expected 3 nodes, got %zu\n", numNodes);
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }

        UPTKGraphNode_t *nodes = (UPTKGraphNode_t *)malloc(numNodes * sizeof(UPTKGraphNode_t));
        CHECK_CUDA(UPTKGraphGetNodes(graph, nodes, &numNodes));

        // Verify all nodes are present
        bool foundA = false, foundB = false, foundC = false;
        for (size_t i = 0; i < numNodes; i++) {
            if (nodes[i] == nodeA) foundA = true;
            if (nodes[i] == nodeB) foundB = true;
            if (nodes[i] == nodeC) foundC = true;
        }
        if (!foundA || !foundB || !foundC) {
            printf("Verification failed: not all nodes found\n");
            free(nodes);
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }

        free(nodes);
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Get nodes from an empty graph
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        size_t numNodes = 0;
        CHECK_CUDA(UPTKGraphGetNodes(graph, NULL, &numNodes));
        if (numNodes != 0) {
            printf("Verification failed: expected 0 nodes, got %zu\n", numNodes);
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }

        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphGetNodes PASS\n");
    return 0;
}

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

    // Scenario 1: Get root nodes from a graph with dependencies
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKGraphNode_t nodeA, nodeB, nodeC;
        CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeA, graph, NULL, 0));
        CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeB, graph, NULL, 0));
        CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeC, graph, &nodeA, 1));

        size_t numRootNodes = 0;
        CHECK_CUDA(UPTKGraphGetRootNodes(graph, NULL, &numRootNodes));
        if (numRootNodes != 2) {
            printf("Verification failed: expected 2 root nodes, got %zu\n", numRootNodes);
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }

        UPTKGraphNode_t *rootNodes = (UPTKGraphNode_t *)malloc(numRootNodes * sizeof(UPTKGraphNode_t));
        CHECK_CUDA(UPTKGraphGetRootNodes(graph, rootNodes, &numRootNodes));

        bool foundA = false, foundB = false;
        for (size_t i = 0; i < numRootNodes; i++) {
            if (rootNodes[i] == nodeA) foundA = true;
            if (rootNodes[i] == nodeB) foundB = true;
        }
        if (!foundA || !foundB) {
            printf("Verification failed: root nodes mismatch\n");
            free(rootNodes);
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }

        free(rootNodes);
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Get root nodes from empty graph
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        size_t numRootNodes = 0;
        CHECK_CUDA(UPTKGraphGetRootNodes(graph, NULL, &numRootNodes));
        if (numRootNodes != 0) {
            printf("Verification failed: expected 0 root nodes, got %zu\n", numRootNodes);
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }

        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphGetRootNodes PASS\n");
    return 0;
}

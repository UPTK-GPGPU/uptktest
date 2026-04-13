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

    // Scenario 1: Get edges from a simple graph
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKGraphNode_t nodeA, nodeB;
        CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeA, graph, NULL, 0));
        CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeB, graph, &nodeA, 1));

        size_t numEdges = 0;
        CHECK_CUDA(UPTKGraphGetEdges(graph, NULL, NULL, &numEdges));
        if (numEdges != 1) {
            printf("Verification failed: expected 1 edge, got %zu\n", numEdges);
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }

        UPTKGraphNode_t *from = (UPTKGraphNode_t *)malloc(numEdges * sizeof(UPTKGraphNode_t));
        UPTKGraphNode_t *to = (UPTKGraphNode_t *)malloc(numEdges * sizeof(UPTKGraphNode_t));
        CHECK_CUDA(UPTKGraphGetEdges(graph, from, to, &numEdges));

        if (from[0] != nodeA || to[0] != nodeB) {
            printf("Verification failed: edge mismatch\n");
            free(from);
            free(to);
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }

        free(from);
        free(to);
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Get edges from a graph with multiple edges
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKGraphNode_t nodeA, nodeB, nodeC;
        CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeA, graph, NULL, 0));
        CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeB, graph, &nodeA, 1));
        CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeC, graph, &nodeB, 1));

        size_t numEdges = 0;
        CHECK_CUDA(UPTKGraphGetEdges(graph, NULL, NULL, &numEdges));
        if (numEdges != 2) {
            printf("Verification failed: expected 2 edges, got %zu\n", numEdges);
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }

        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 3: Get edges from a graph with no edges
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKGraphNode_t nodeA, nodeB;
        CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeA, graph, NULL, 0));
        CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeB, graph, NULL, 0));

        size_t numEdges = 0;
        CHECK_CUDA(UPTKGraphGetEdges(graph, NULL, NULL, &numEdges));
        if (numEdges != 0) {
            printf("Verification failed: expected 0 edges, got %zu\n", numEdges);
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }

        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphGetEdges PASS\n");
    return 0;
}

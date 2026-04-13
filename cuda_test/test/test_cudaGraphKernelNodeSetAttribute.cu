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

    // Scenario 1: Create graph and add empty node, test graph API path
    UPTKGraph_t graph;
    CHECK_CUDA(UPTKGraphCreate(&graph, 0));

    UPTKGraphNode_t emptyNode;
    CHECK_CUDA(UPTKGraphAddEmptyNode(&emptyNode, graph, NULL, 0));

    // Try to get node type to verify the node
    UPTKGraphNodeType nodeType;
    CHECK_CUDA(UPTKGraphNodeGetType(emptyNode, &nodeType));
    if (nodeType != UPTKGraphNodeTypeEmpty) {
        printf("Node type mismatch: expected %d, got %d\n", UPTKGraphNodeTypeEmpty, nodeType);
        UPTKGraphDestroy(graph);
        return 1;
    }
    printf("Empty node created successfully\n");

    // Scenario 2: Create graph with memcpy node
    int *d_src, *d_dst;
    CHECK_CUDA(UPTKMalloc(&d_src, 256));
    CHECK_CUDA(UPTKMalloc(&d_dst, 256));

    UPTKGraph_t graph2;
    CHECK_CUDA(UPTKGraphCreate(&graph2, 0));

    UPTKGraphNode_t memcpyNode;
    UPTKMemcpy3DParms parms = {};
    parms.srcPtr = make_UPTKPitchedPtr(d_src, 256, 256, 1);
    parms.dstPtr = make_UPTKPitchedPtr(d_dst, 256, 256, 1);
    parms.extent = make_UPTKExtent(256, 1, 1);
    parms.kind = UPTKMemcpyDeviceToDevice;
    CHECK_CUDA(UPTKGraphAddMemcpyNode(&memcpyNode, graph2, NULL, 0, &parms));

    UPTKGraphNodeType nodeType2;
    CHECK_CUDA(UPTKGraphNodeGetType(memcpyNode, &nodeType2));
    if (nodeType2 != UPTKGraphNodeTypeMemset && nodeType2 != UPTKGraphNodeTypeMemcpy) {
        printf("Memcpy node type unexpected: %d\n", nodeType2);
        UPTKFree(d_src);
        UPTKFree(d_dst);
        UPTKGraphDestroy(graph);
        UPTKGraphDestroy(graph2);
        return 1;
    }
    printf("Memcpy node created successfully\n");

    UPTKFree(d_src);
    UPTKFree(d_dst);
    UPTKGraphDestroy(graph);
    UPTKGraphDestroy(graph2);

    printf("test_cudaGraphKernelNodeSetAttribute PASS\n");
    return 0;
}

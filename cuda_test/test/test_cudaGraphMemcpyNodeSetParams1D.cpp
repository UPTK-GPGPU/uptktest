#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

    // Scenario 1: Error handling - invalid node should return UPTKErrorInvalidValue
    float h_data[256];
    float *d_data = NULL;
    UPTKError_t err = UPTKGraphMemcpyNodeSetParams1D((UPTKGraphNode_t)0xDEADBEEF, d_data, h_data, sizeof(h_data), UPTKMemcpyHostToDevice);
    if (err != UPTKErrorInvalidValue) {
        printf("FAIL: expected UPTKErrorInvalidValue for invalid node, got %s\n", UPTKGetErrorString(err));
        return 1;
    }

    // Scenario 2: Error handling - NULL dst pointer
    UPTKGraph_t graph;
    CHECK_CUDA(UPTKGraphCreate(&graph, 0));

    UPTKMemcpy3DParms initParams = {0};
    initParams.srcPtr = make_UPTKPitchedPtr(h_data, sizeof(h_data), sizeof(h_data), 1);
    initParams.srcPos = make_UPTKPos(0, 0, 0);
    initParams.dstPtr = make_UPTKPitchedPtr(h_data, sizeof(h_data), sizeof(h_data), 1);
    initParams.dstPos = make_UPTKPos(0, 0, 0);
    initParams.extent = make_UPTKExtent(sizeof(h_data), 1, 1);
    initParams.kind = UPTKMemcpyHostToHost;

    UPTKGraphNode_t memcpyNode;
    CHECK_CUDA(UPTKGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &initParams));

    err = UPTKGraphMemcpyNodeSetParams1D(memcpyNode, NULL, h_data, sizeof(h_data), UPTKMemcpyHostToHost);
    if (err != UPTKErrorInvalidValue) {
        printf("FAIL: expected UPTKErrorInvalidValue for NULL dst, got %s\n", UPTKGetErrorString(err));
        return 1;
    }

    // Scenario 3: Zero size copy (edge case)
    // Note: DTK/AMD may return UPTKErrorInvalidValue for zero size
    err = UPTKGraphMemcpyNodeSetParams1D(memcpyNode, h_data, h_data, 0, UPTKMemcpyHostToHost);
    if (err != UPTKSuccess && err != UPTKErrorInvalidValue) {
        printf("FAIL: unexpected error for zero size: %s\n", UPTKGetErrorString(err));
        return 1;
    }

    CHECK_CUDA(UPTKGraphDestroy(graph));

    printf("test_cudaGraphMemcpyNodeSetParams1D PASS\n");
    return 0;
}

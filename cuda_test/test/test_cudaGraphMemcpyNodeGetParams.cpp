#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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

    // Scenario 1: Basic - create graph with 3D memcpy node, get params
    UPTKGraph_t graph;
    CHECK_CUDA(UPTKGraphCreate(&graph, 0));

    size_t width = 64 * sizeof(float);
    size_t height = 4;
    size_t depth = 1;

    float *h_data = (float *)malloc(width * height * depth);
    memset(h_data, 0, width * height * depth);

    float *d_data = NULL;
    CHECK_CUDA(UPTKMalloc(&d_data, width * height * depth));

    UPTKMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_UPTKPitchedPtr(h_data, width, width, height);
    copyParams.srcPos = make_UPTKPos(0, 0, 0);
    copyParams.dstPtr = make_UPTKPitchedPtr(d_data, width, width, height);
    copyParams.dstPos = make_UPTKPos(0, 0, 0);
    copyParams.extent = make_UPTKExtent(width, height, depth);
    copyParams.kind = UPTKMemcpyHostToDevice;

    UPTKGraphNode_t memcpyNode;
    CHECK_CUDA(UPTKGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &copyParams));

    UPTKMemcpy3DParms retrievedParams;
    CHECK_CUDA(UPTKGraphMemcpyNodeGetParams(memcpyNode, &retrievedParams));
    if (retrievedParams.extent.width != width || retrievedParams.extent.height != height) {
        printf("FAIL: extent mismatch\n");
        return 1;
    }

    // Scenario 2: Different copy direction (D2H)
    UPTKMemcpy3DParms copyParams2 = {0};
    copyParams2.srcPtr = make_UPTKPitchedPtr(d_data, width, width, height);
    copyParams2.srcPos = make_UPTKPos(0, 0, 0);
    copyParams2.dstPtr = make_UPTKPitchedPtr(h_data, width, width, height);
    copyParams2.dstPos = make_UPTKPos(0, 0, 0);
    copyParams2.extent = make_UPTKExtent(width, height, depth);
    copyParams2.kind = UPTKMemcpyDeviceToHost;

    UPTKGraphNode_t memcpyNode2;
    CHECK_CUDA(UPTKGraphAddMemcpyNode(&memcpyNode2, graph, NULL, 0, &copyParams2));

    UPTKMemcpy3DParms retrievedParams2;
    CHECK_CUDA(UPTKGraphMemcpyNodeGetParams(memcpyNode2, &retrievedParams2));
    if (retrievedParams2.kind != UPTKMemcpyDeviceToHost) {
        printf("FAIL: kind mismatch\n");
        return 1;
    }

    // Scenario 3: Error handling - invalid node
    UPTKMemcpy3DParms badParams;
    UPTKError_t err = UPTKGraphMemcpyNodeGetParams((UPTKGraphNode_t)0xDEADBEEF, &badParams);
    if (err != UPTKErrorInvalidValue) {
        printf("FAIL: expected UPTKErrorInvalidValue for invalid node\n");
        return 1;
    }

    CHECK_CUDA(UPTKFree(d_data));
    free(h_data);
    CHECK_CUDA(UPTKGraphDestroy(graph));

    printf("test_cudaGraphMemcpyNodeGetParams PASS\n");
    return 0;
}

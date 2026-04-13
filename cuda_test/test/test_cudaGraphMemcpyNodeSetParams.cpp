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

    // Scenario 1: Basic - create memcpy node, then set new params
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

    // Set new params with different extent
    UPTKMemcpy3DParms newParams = {0};
    newParams.srcPtr = make_UPTKPitchedPtr(h_data, width, width, height);
    newParams.srcPos = make_UPTKPos(0, 0, 0);
    newParams.dstPtr = make_UPTKPitchedPtr(d_data, width, width, height);
    newParams.dstPos = make_UPTKPos(0, 0, 0);
    newParams.extent = make_UPTKExtent(width, height, depth);
    newParams.kind = UPTKMemcpyHostToDevice;
    CHECK_CUDA(UPTKGraphMemcpyNodeSetParams(memcpyNode, &newParams));

    // Verify the params were updated
    UPTKMemcpy3DParms retrievedParams;
    CHECK_CUDA(UPTKGraphMemcpyNodeGetParams(memcpyNode, &retrievedParams));
    if (retrievedParams.extent.width != width) {
        printf("FAIL: extent not updated\n");
        return 1;
    }

    // Scenario 2: Error handling - invalid node
    UPTKError_t err = UPTKGraphMemcpyNodeSetParams((UPTKGraphNode_t)0xDEADBEEF, &newParams);
    if (err != UPTKErrorInvalidValue) {
        printf("FAIL: expected UPTKErrorInvalidValue for invalid node\n");
        return 1;
    }

    CHECK_CUDA(UPTKFree(d_data));
    free(h_data);
    CHECK_CUDA(UPTKGraphDestroy(graph));

    printf("test_cudaGraphMemcpyNodeSetParams PASS\n");
    return 0;
}

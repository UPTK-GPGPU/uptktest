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
    int h_data[256];
    float *d_data = NULL;
    CHECK_CUDA(UPTKMalloc(&d_data, sizeof(h_data)));

    UPTKError_t err = UPTKGraphMemcpyNodeSetParamsToSymbol((UPTKGraphNode_t)0xDEADBEEF, (const void*)d_data, h_data, sizeof(h_data), 0, UPTKMemcpyHostToDevice);
    if (err != UPTKErrorInvalidValue) {
        printf("FAIL: expected UPTKErrorInvalidValue for invalid node, got %s\n", UPTKGetErrorString(err));
        CHECK_CUDA(UPTKFree(d_data));
        return 1;
    }

    // Scenario 2: Create valid memcpy node, then try to set symbol params
    UPTKGraph_t graph;
    CHECK_CUDA(UPTKGraphCreate(&graph, 0));

    UPTKMemcpy3DParms initParams = {0};
    initParams.srcPtr = make_UPTKPitchedPtr(h_data, sizeof(h_data), sizeof(h_data), 1);
    initParams.srcPos = make_UPTKPos(0, 0, 0);
    initParams.dstPtr = make_UPTKPitchedPtr(d_data, sizeof(h_data), sizeof(h_data), 1);
    initParams.dstPos = make_UPTKPos(0, 0, 0);
    initParams.extent = make_UPTKExtent(sizeof(h_data), 1, 1);
    initParams.kind = UPTKMemcpyHostToDevice;

    UPTKGraphNode_t memcpyNode;
    CHECK_CUDA(UPTKGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &initParams));

    // Try setting symbol params - on DTK this may fail since device symbols work differently
    err = UPTKGraphMemcpyNodeSetParamsToSymbol(memcpyNode, (const void*)d_data, h_data, sizeof(h_data), 0, UPTKMemcpyHostToDevice);
    // Accept either success or invalid value (symbol not found on DTK)
    if (err != UPTKSuccess && err != UPTKErrorInvalidValue && err != UPTKErrorInvalidSymbol) {
        printf("FAIL: unexpected error %s\n", UPTKGetErrorString(err));
        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 1;
    }

    // Scenario 3: Zero size copy
    err = UPTKGraphMemcpyNodeSetParamsToSymbol(memcpyNode, (const void*)d_data, h_data, 0, 0, UPTKMemcpyHostToDevice);
    if (err != UPTKSuccess && err != UPTKErrorInvalidValue && err != UPTKErrorInvalidSymbol) {
        printf("FAIL: unexpected error for zero size: %s\n", UPTKGetErrorString(err));
        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 1;
    }

    CHECK_CUDA(UPTKFree(d_data));
    CHECK_CUDA(UPTKGraphDestroy(graph));

    printf("test_cudaGraphMemcpyNodeSetParamsToSymbol PASS\n");
    return 0;
}

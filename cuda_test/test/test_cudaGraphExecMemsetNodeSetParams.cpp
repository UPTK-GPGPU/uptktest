#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
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

    // Scenario 1: Update memset node params in graphExec
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int *d_data;
        CHECK_CUDA(UPTKMalloc(&d_data, 64 * sizeof(int)));

        UPTKMemsetParams memsetParams = {};
        memsetParams.dst = d_data;
        memsetParams.pitch = 0;
        memsetParams.value = 0xFF;
        memsetParams.elementSize = sizeof(int);
        memsetParams.width = 64;
        memsetParams.height = 1;

        UPTKGraphNode_t memsetNode;
        CHECK_CUDA(UPTKGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));

        // Update memset params with different value and size
        UPTKMemsetParams newParams = {};
        newParams.dst = d_data;
        newParams.pitch = 0;
        newParams.value = 0xAA;
        newParams.elementSize = 1;
        newParams.width = 64 * sizeof(int);
        newParams.height = 1;

        CHECK_CUDA(UPTKGraphExecMemsetNodeSetParams(exec, memsetNode, &newParams));

        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Update memset node with different value
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int *d_data;
        CHECK_CUDA(UPTKMalloc(&d_data, 128 * sizeof(int)));

        UPTKMemsetParams memsetParams = {};
        memsetParams.dst = d_data;
        memsetParams.pitch = 0;
        memsetParams.value = 0x55;
        memsetParams.elementSize = 1;
        memsetParams.width = 128 * sizeof(int);
        memsetParams.height = 1;

        UPTKGraphNode_t memsetNode;
        CHECK_CUDA(UPTKGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));

        // Update with different value
        UPTKMemsetParams newParams = {};
        newParams.dst = d_data;
        newParams.pitch = 0;
        newParams.value = 0xCC;
        newParams.elementSize = 1;
        newParams.width = 128 * sizeof(int);
        newParams.height = 1;

        CHECK_CUDA(UPTKGraphExecMemsetNodeSetParams(exec, memsetNode, &newParams));

        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphExecMemsetNodeSetParams PASS\n");
    return 0;
}

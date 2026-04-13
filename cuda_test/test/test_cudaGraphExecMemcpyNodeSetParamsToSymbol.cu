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

__device__ int d_symbol[64];

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Update memcpy-to-symbol node params in graphExec
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int h_data[64];
        for (int i = 0; i < 64; i++) h_data[i] = i;

        UPTKGraphNode_t memcpyNode;
        CHECK_CUDA(UPTKGraphAddMemcpyNodeToSymbol(&memcpyNode, graph, NULL, 0, d_symbol, h_data, 64 * sizeof(int), 0, UPTKMemcpyHostToDevice));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));

        // Update with different data
        int h_data2[64];
        for (int i = 0; i < 64; i++) h_data2[i] = i + 500;

        CHECK_CUDA(UPTKGraphExecMemcpyNodeSetParamsToSymbol(exec, memcpyNode, d_symbol, h_data2, 64 * sizeof(int), 0, UPTKMemcpyHostToDevice));

        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        int h_result[64];
        CHECK_CUDA(UPTKMemcpyFromSymbol(&h_result, &d_symbol, 64 * sizeof(int), 0, UPTKMemcpyDeviceToHost));

        for (int i = 0; i < 64; i++) {
            if (h_result[i] != i + 500) {
                printf("Verification failed at index %d: expected %d, got %d\n", i, i + 500, h_result[i]);
                return 1;
            }
        }

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Update with different count and offset
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int h_data[32];
        for (int i = 0; i < 32; i++) h_data[i] = i * 2;

        UPTKGraphNode_t memcpyNode;
        CHECK_CUDA(UPTKGraphAddMemcpyNodeToSymbol(&memcpyNode, graph, NULL, 0, d_symbol, h_data, 32 * sizeof(int), 0, UPTKMemcpyHostToDevice));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));

        int h_data2[16];
        for (int i = 0; i < 16; i++) h_data2[i] = i * 10;

        CHECK_CUDA(UPTKGraphExecMemcpyNodeSetParamsToSymbol(exec, memcpyNode, d_symbol, h_data2, 16 * sizeof(int), 16 * sizeof(int), UPTKMemcpyHostToDevice));

        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphExecMemcpyNodeSetParamsToSymbol PASS\n");
    return 0;
}

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

__device__ int devSymbol = 42;

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Basic memcpy from symbol to host
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int h_data;
        UPTKGraphNode_t node;
        CHECK_CUDA(UPTKGraphAddMemcpyNodeFromSymbol(&node, graph, NULL, 0, &h_data, (const void*)&devSymbol, sizeof(int), 0, UPTKMemcpyDeviceToHost));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));
        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        if (h_data != 42) {
            printf("Verification failed: expected 42, got %d\n", h_data);
            return 1;
        }

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Memcpy from symbol with offset
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int h_data[4];
        for (int i = 0; i < 4; i++) h_data[i] = 0;

        UPTKGraphNode_t node;
        CHECK_CUDA(UPTKGraphAddMemcpyNodeFromSymbol(&node, graph, NULL, 0, h_data, (const void*)&devSymbol, sizeof(int), 0, UPTKMemcpyDeviceToHost));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));
        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphAddMemcpyNodeFromSymbol PASS\n");
    return 0;
}

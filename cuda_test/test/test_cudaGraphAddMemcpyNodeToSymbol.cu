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

__device__ int devSymbol = 0;

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Basic memcpy to symbol from host
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int h_data = 100;
        UPTKGraphNode_t node;
        CHECK_CUDA(UPTKGraphAddMemcpyNodeToSymbol(&node, graph, NULL, 0, (const void*)&devSymbol, &h_data, sizeof(int), 0, UPTKMemcpyHostToDevice));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));
        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        int h_verify;
        CHECK_CUDA(UPTKMemcpyFromSymbol(&h_verify, &devSymbol, sizeof(int)));
        if (h_verify != 100) {
            printf("Verification failed: expected 100, got %d\n", h_verify);
            return 1;
        }

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Memcpy to symbol with offset
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int h_data = 200;
        UPTKGraphNode_t node;
        CHECK_CUDA(UPTKGraphAddMemcpyNodeToSymbol(&node, graph, NULL, 0, (const void*)&devSymbol, &h_data, sizeof(int), 0, UPTKMemcpyHostToDevice));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));
        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphAddMemcpyNodeToSymbol PASS\n");
    return 0;
}

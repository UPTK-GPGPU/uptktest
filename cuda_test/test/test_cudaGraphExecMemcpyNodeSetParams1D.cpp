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

    // Scenario 1: Update 1D memcpy node params in graphExec (H2D)
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int h_data[64];
        for (int i = 0; i < 64; i++) h_data[i] = i;

        int *d_data;
        CHECK_CUDA(UPTKMalloc(&d_data, 64 * sizeof(int)));

        UPTKGraphNode_t memcpyNode;
        CHECK_CUDA(UPTKGraphAddMemcpyNode1D(&memcpyNode, graph, NULL, 0, d_data, h_data, 64 * sizeof(int), UPTKMemcpyHostToDevice));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));

        // Update with different source data
        int h_data2[64];
        for (int i = 0; i < 64; i++) h_data2[i] = i + 200;

        CHECK_CUDA(UPTKGraphExecMemcpyNodeSetParams1D(exec, memcpyNode, d_data, h_data2, 64 * sizeof(int), UPTKMemcpyHostToDevice));

        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        int h_result[64];
        CHECK_CUDA(UPTKMemcpy(h_result, d_data, 64 * sizeof(int), UPTKMemcpyDeviceToHost));

        for (int i = 0; i < 64; i++) {
            if (h_result[i] != i + 200) {
                printf("Verification failed at index %d: expected %d, got %d\n", i, i + 200, h_result[i]);
                return 1;
            }
        }

        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Update 1D memcpy node params with D2D copy
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int *d_src, *d_dst;
        CHECK_CUDA(UPTKMalloc(&d_src, 32 * sizeof(int)));
        CHECK_CUDA(UPTKMalloc(&d_dst, 32 * sizeof(int)));

        int h_init[32];
        for (int i = 0; i < 32; i++) h_init[i] = i * 5;
        CHECK_CUDA(UPTKMemcpy(d_src, h_init, 32 * sizeof(int), UPTKMemcpyHostToDevice));

        UPTKGraphNode_t memcpyNode;
        CHECK_CUDA(UPTKGraphAddMemcpyNode1D(&memcpyNode, graph, NULL, 0, d_dst, d_src, 32 * sizeof(int), UPTKMemcpyDeviceToDevice));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));

        // Update with different size
        CHECK_CUDA(UPTKGraphExecMemcpyNodeSetParams1D(exec, memcpyNode, d_dst, d_src, 16 * sizeof(int), UPTKMemcpyDeviceToDevice));

        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        int h_result[32];
        CHECK_CUDA(UPTKMemcpy(h_result, d_dst, 32 * sizeof(int), UPTKMemcpyDeviceToHost));

        CHECK_CUDA(UPTKFree(d_src));
        CHECK_CUDA(UPTKFree(d_dst));
        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphExecMemcpyNodeSetParams1D PASS\n");
    return 0;
}

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

    // Scenario 1: Basic H2D + D2H memcpy node
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int h_data[64];
        for (int i = 0; i < 64; i++) h_data[i] = i;

        int *d_data;
        CHECK_CUDA(UPTKMalloc(&d_data, 64 * sizeof(int)));

        UPTKMemcpy3DParms copyParams = {};
        copyParams.srcPtr = make_UPTKPitchedPtr(h_data, 64 * sizeof(int), 64, 1);
        copyParams.dstPtr = make_UPTKPitchedPtr(d_data, 64 * sizeof(int), 64, 1);
        copyParams.extent = make_UPTKExtent(64 * sizeof(int), 1, 1);
        copyParams.kind = UPTKMemcpyHostToDevice;

        UPTKGraphNode_t node;
        CHECK_CUDA(UPTKGraphAddMemcpyNode(&node, graph, NULL, 0, &copyParams));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));
        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        int h_result[64];
        CHECK_CUDA(UPTKMemcpy(h_result, d_data, 64 * sizeof(int), UPTKMemcpyDeviceToHost));

        for (int i = 0; i < 64; i++) {
            if (h_result[i] != i) {
                printf("Verification failed at index %d\n", i);
                return 1;
            }
        }

        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: D2D memcpy node
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int *d_src, *d_dst;
        CHECK_CUDA(UPTKMalloc(&d_src, 128 * sizeof(int)));
        CHECK_CUDA(UPTKMalloc(&d_dst, 128 * sizeof(int)));

        int h_init[128];
        for (int i = 0; i < 128; i++) h_init[i] = i * 2;
        CHECK_CUDA(UPTKMemcpy(d_src, h_init, 128 * sizeof(int), UPTKMemcpyHostToDevice));

        UPTKMemcpy3DParms copyParams = {};
        copyParams.srcPtr = make_UPTKPitchedPtr(d_src, 128 * sizeof(int), 128, 1);
        copyParams.dstPtr = make_UPTKPitchedPtr(d_dst, 128 * sizeof(int), 128, 1);
        copyParams.extent = make_UPTKExtent(128 * sizeof(int), 1, 1);
        copyParams.kind = UPTKMemcpyDeviceToDevice;

        UPTKGraphNode_t node;
        CHECK_CUDA(UPTKGraphAddMemcpyNode(&node, graph, NULL, 0, &copyParams));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));
        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        int h_result[128];
        CHECK_CUDA(UPTKMemcpy(h_result, d_dst, 128 * sizeof(int), UPTKMemcpyDeviceToHost));

        CHECK_CUDA(UPTKFree(d_src));
        CHECK_CUDA(UPTKFree(d_dst));
        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 3: Memcpy node with dependencies
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int h_data[32];
        for (int i = 0; i < 32; i++) h_data[i] = i + 100;

        int *d_data;
        CHECK_CUDA(UPTKMalloc(&d_data, 32 * sizeof(int)));

        UPTKMemcpy3DParms copyParams = {};
        copyParams.srcPtr = make_UPTKPitchedPtr(h_data, 32 * sizeof(int), 32, 1);
        copyParams.dstPtr = make_UPTKPitchedPtr(d_data, 32 * sizeof(int), 32, 1);
        copyParams.extent = make_UPTKExtent(32 * sizeof(int), 1, 1);
        copyParams.kind = UPTKMemcpyHostToDevice;

        UPTKGraphNode_t node1;
        CHECK_CUDA(UPTKGraphAddMemcpyNode(&node1, graph, NULL, 0, &copyParams));

        UPTKGraphNode_t node2;
        CHECK_CUDA(UPTKGraphAddMemcpyNode(&node2, graph, &node1, 1, &copyParams));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));
        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphAddMemcpyNode PASS\n");
    return 0;
}

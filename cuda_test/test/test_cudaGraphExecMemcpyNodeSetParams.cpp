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

    // Scenario 1: Update memcpy node params in graphExec (H2D)
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

        UPTKGraphNode_t memcpyNode;
        CHECK_CUDA(UPTKGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &copyParams));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));

        // Update with different data
        int h_data2[64];
        for (int i = 0; i < 64; i++) h_data2[i] = i + 100;

        UPTKMemcpy3DParms newCopyParams = {};
        newCopyParams.srcPtr = make_UPTKPitchedPtr(h_data2, 64 * sizeof(int), 64, 1);
        newCopyParams.dstPtr = make_UPTKPitchedPtr(d_data, 64 * sizeof(int), 64, 1);
        newCopyParams.extent = make_UPTKExtent(64 * sizeof(int), 1, 1);
        newCopyParams.kind = UPTKMemcpyHostToDevice;

        CHECK_CUDA(UPTKGraphExecMemcpyNodeSetParams(exec, memcpyNode, &newCopyParams));

        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        int h_result[64];
        CHECK_CUDA(UPTKMemcpy(h_result, d_data, 64 * sizeof(int), UPTKMemcpyDeviceToHost));

        for (int i = 0; i < 64; i++) {
            if (h_result[i] != i + 100) {
                printf("Verification failed at index %d: expected %d, got %d\n", i, i + 100, h_result[i]);
                return 1;
            }
        }

        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Update memcpy node params with D2D copy
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int *d_src, *d_dst;
        CHECK_CUDA(UPTKMalloc(&d_src, 32 * sizeof(int)));
        CHECK_CUDA(UPTKMalloc(&d_dst, 32 * sizeof(int)));

        int h_init[32];
        for (int i = 0; i < 32; i++) h_init[i] = i * 3;
        CHECK_CUDA(UPTKMemcpy(d_src, h_init, 32 * sizeof(int), UPTKMemcpyHostToDevice));

        UPTKMemcpy3DParms copyParams = {};
        copyParams.srcPtr = make_UPTKPitchedPtr(d_src, 32 * sizeof(int), 32, 1);
        copyParams.dstPtr = make_UPTKPitchedPtr(d_dst, 32 * sizeof(int), 32, 1);
        copyParams.extent = make_UPTKExtent(32 * sizeof(int), 1, 1);
        copyParams.kind = UPTKMemcpyDeviceToDevice;

        UPTKGraphNode_t memcpyNode;
        CHECK_CUDA(UPTKGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &copyParams));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));

        // Update with different extent
        UPTKMemcpy3DParms newCopyParams = {};
        newCopyParams.srcPtr = make_UPTKPitchedPtr(d_src, 16 * sizeof(int), 16, 1);
        newCopyParams.dstPtr = make_UPTKPitchedPtr(d_dst, 16 * sizeof(int), 16, 1);
        newCopyParams.extent = make_UPTKExtent(16 * sizeof(int), 1, 1);
        newCopyParams.kind = UPTKMemcpyDeviceToDevice;

        CHECK_CUDA(UPTKGraphExecMemcpyNodeSetParams(exec, memcpyNode, &newCopyParams));

        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        int h_result[32];
        CHECK_CUDA(UPTKMemcpy(h_result, d_dst, 32 * sizeof(int), UPTKMemcpyDeviceToHost));

        CHECK_CUDA(UPTKFree(d_src));
        CHECK_CUDA(UPTKFree(d_dst));
        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphExecMemcpyNodeSetParams PASS\n");
    return 0;
}

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

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Basic H2D 1D memcpy node
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int h_data[64];
        for (int i = 0; i < 64; i++) h_data[i] = i;

        int *d_data;
        CHECK_CUDA(UPTKMalloc(&d_data, 64 * sizeof(int)));

        UPTKGraphNode_t node;
        CHECK_CUDA(UPTKGraphAddMemcpyNode1D(&node, graph, NULL, 0, d_data, h_data, 64 * sizeof(int), UPTKMemcpyHostToDevice));

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

    // Scenario 2: D2D 1D memcpy node
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int *d_src, *d_dst;
        CHECK_CUDA(UPTKMalloc(&d_src, 128 * sizeof(int)));
        CHECK_CUDA(UPTKMalloc(&d_dst, 128 * sizeof(int)));

        int h_init[128];
        for (int i = 0; i < 128; i++) h_init[i] = i * 3;
        CHECK_CUDA(UPTKMemcpy(d_src, h_init, 128 * sizeof(int), UPTKMemcpyHostToDevice));

        UPTKGraphNode_t node;
        CHECK_CUDA(UPTKGraphAddMemcpyNode1D(&node, graph, NULL, 0, d_dst, d_src, 128 * sizeof(int), UPTKMemcpyDeviceToDevice));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));
        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        int h_result[128];
        CHECK_CUDA(UPTKMemcpy(h_result, d_dst, 128 * sizeof(int), UPTKMemcpyDeviceToHost));

        for (int i = 0; i < 128; i++) {
            if (h_result[i] != i * 3) {
                printf("Verification failed at index %d\n", i);
                return 1;
            }
        }

        CHECK_CUDA(UPTKFree(d_src));
        CHECK_CUDA(UPTKFree(d_dst));
        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 3: D2H 1D memcpy node
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int *d_data;
        CHECK_CUDA(UPTKMalloc(&d_data, 32 * sizeof(int)));

        int h_init[32];
        for (int i = 0; i < 32; i++) h_init[i] = i + 50;
        CHECK_CUDA(UPTKMemcpy(d_data, h_init, 32 * sizeof(int), UPTKMemcpyHostToDevice));

        int h_result[32];
        UPTKGraphNode_t node;
        CHECK_CUDA(UPTKGraphAddMemcpyNode1D(&node, graph, NULL, 0, h_result, d_data, 32 * sizeof(int), UPTKMemcpyDeviceToHost));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));
        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        for (int i = 0; i < 32; i++) {
            if (h_result[i] != i + 50) {
                printf("Verification failed at index %d\n", i);
                return 1;
            }
        }

        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphAddMemcpyNode1D PASS\n");
    return 0;
}

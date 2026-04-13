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

    // Scenario 1: Basic memset node with 1D region
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int *d_data;
        CHECK_CUDA(UPTKMalloc(&d_data, 64 * sizeof(int)));

        UPTKMemsetParams memsetParams = {};
        memsetParams.dst = d_data;
        memsetParams.pitch = 64 * sizeof(int);
        memsetParams.value = 0xFF;
        memsetParams.elementSize = 1;
        memsetParams.width = 64 * sizeof(int);
        memsetParams.height = 1;

        UPTKGraphNode_t node;
        CHECK_CUDA(UPTKGraphAddMemsetNode(&node, graph, NULL, 0, &memsetParams));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));
        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        unsigned char h_data[64];
        CHECK_CUDA(UPTKMemcpy(h_data, d_data, 64, UPTKMemcpyDeviceToHost));

        for (int i = 0; i < 64; i++) {
            if (h_data[i] != 0xFF) {
                printf("Verification failed at index %d: expected 0xFF, got 0x%X\n", i, h_data[i]);
                return 1;
            }
        }

        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Memset node with 2D region (height > 1)
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int *d_data;
        CHECK_CUDA(UPTKMalloc(&d_data, 16 * 16 * sizeof(int)));

        UPTKMemsetParams memsetParams = {};
        memsetParams.dst = d_data;
        memsetParams.pitch = 16 * sizeof(int);
        memsetParams.value = 0xAA;
        memsetParams.elementSize = 4;
        memsetParams.width = 16;
        memsetParams.height = 16;

        UPTKGraphNode_t node;
        CHECK_CUDA(UPTKGraphAddMemsetNode(&node, graph, NULL, 0, &memsetParams));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));
        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 3: Memset with 1-byte element size
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        unsigned char *d_data;
        CHECK_CUDA(UPTKMalloc(&d_data, 128));

        UPTKMemsetParams memsetParams = {};
        memsetParams.dst = d_data;
        memsetParams.pitch = 128;
        memsetParams.value = 0x55;
        memsetParams.elementSize = 1;
        memsetParams.width = 128;
        memsetParams.height = 1;

        UPTKGraphNode_t node;
        CHECK_CUDA(UPTKGraphAddMemsetNode(&node, graph, NULL, 0, &memsetParams));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));
        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        unsigned char h_data[128];
        CHECK_CUDA(UPTKMemcpy(h_data, d_data, 128, UPTKMemcpyDeviceToHost));

        for (int i = 0; i < 128; i++) {
            if (h_data[i] != 0x55) {
                printf("Verification failed at index %d\n", i);
                return 1;
            }
        }

        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphAddMemsetNode PASS\n");
    return 0;
}

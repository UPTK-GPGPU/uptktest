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

    // Scenario 1: Launch a graph using per-thread default stream variant
    // cudaGraphLaunch_ptsz maps to UPTKGraphLaunch with the per-thread stream
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKGraphNode_t nodeA, nodeB;
        CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeA, graph, NULL, 0));
        CHECK_CUDA(UPTKGraphAddEmptyNode(&nodeB, graph, &nodeA, 1));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));

        // Use per-thread default stream (stream 0 in ptsz context)
        UPTKStream_t ptsz = 0;
        CHECK_CUDA(UPTKGraphLaunch(exec, ptsz));
        CHECK_CUDA(UPTKStreamSynchronize(ptsz));

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Launch on per-thread stream with explicit stream creation
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKGraphNode_t node;
        CHECK_CUDA(UPTKGraphAddEmptyNode(&node, graph, NULL, 0));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));

        // Create a non-blocking stream to simulate per-thread stream behavior
        UPTKStream_t stream;
        CHECK_CUDA(UPTKStreamCreateWithFlags(&stream, UPTKStreamNonBlocking));

        CHECK_CUDA(UPTKGraphLaunch(exec, stream));
        CHECK_CUDA(UPTKStreamSynchronize(stream));

        CHECK_CUDA(UPTKStreamDestroy(stream));
        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphLaunch_ptsz PASS\n");
    return 0;
}

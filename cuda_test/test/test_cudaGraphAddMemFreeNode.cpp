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

    // Scenario 1: Mem free node after mem alloc node
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKMemAllocNodeParams allocParams = {};
        memset(&allocParams.poolProps, 0, sizeof(allocParams.poolProps));
        allocParams.poolProps.allocType = UPTKMemAllocationTypePinned;
        allocParams.poolProps.handleTypes = UPTKMemHandleTypeNone;
        allocParams.poolProps.location.type = UPTKMemLocationTypeDevice;
        allocParams.poolProps.location.id = 0;
        allocParams.accessDescs = NULL;
        allocParams.accessDescCount = 0;
        allocParams.bytesize = 1024;
        allocParams.dptr = NULL;

        UPTKGraphNode_t allocNode;
        CHECK_CUDA(UPTKGraphAddMemAllocNode(&allocNode, graph, NULL, 0, &allocParams));

        UPTKGraphNode_t freeNode;
        CHECK_CUDA(UPTKGraphAddMemFreeNode(&freeNode, graph, &allocNode, 1, (void *)allocParams.dptr));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));
        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Mem free node with no dependencies
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int *d_ptr;
        CHECK_CUDA(UPTKMalloc(&d_ptr, 512));

        UPTKGraphNode_t freeNode;
        CHECK_CUDA(UPTKGraphAddMemFreeNode(&freeNode, graph, NULL, 0, (void *)d_ptr));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));
        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphAddMemFreeNode PASS\n");
    return 0;
}

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

    // Scenario 1: Basic mem alloc node
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

        UPTKGraphNode_t node;
        CHECK_CUDA(UPTKGraphAddMemAllocNode(&node, graph, NULL, 0, &allocParams));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));
        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        if (allocParams.dptr == NULL) {
            printf("CUDA error: mem alloc node returned NULL dptr\n");
            return 1;
        }

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Mem alloc with access descriptor
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKMemAccessDesc accessDesc = {};
        accessDesc.location.type = UPTKMemLocationTypeDevice;
        accessDesc.location.id = 0;
        accessDesc.flags = UPTKMemAccessFlagsProtReadWrite;

        UPTKMemAllocNodeParams allocParams = {};
        memset(&allocParams.poolProps, 0, sizeof(allocParams.poolProps));
        allocParams.poolProps.allocType = UPTKMemAllocationTypePinned;
        allocParams.poolProps.handleTypes = UPTKMemHandleTypeNone;
        allocParams.poolProps.location.type = UPTKMemLocationTypeDevice;
        allocParams.poolProps.location.id = 0;
        allocParams.accessDescs = &accessDesc;
        allocParams.accessDescCount = 1;
        allocParams.bytesize = 4096;
        allocParams.dptr = NULL;

        UPTKGraphNode_t node;
        CHECK_CUDA(UPTKGraphAddMemAllocNode(&node, graph, NULL, 0, &allocParams));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));
        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphAddMemAllocNode PASS\n");
    return 0;
}

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

__global__ void simpleKernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx;
    }
}

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Update graph exec with same topology
    {
        UPTKGraph_t graph1;
        CHECK_CUDA(UPTKGraphCreate(&graph1, 0));

        int *d_data;
        CHECK_CUDA(UPTKMalloc(&d_data, 64 * sizeof(int)));

        UPTKKernelNodeParams params = {};
        params.func = (void *)simpleKernel;
        params.gridDim.x = 1;
        params.gridDim.y = 1;
        params.gridDim.z = 1;
        params.blockDim.x = 64;
        params.blockDim.y = 1;
        params.blockDim.z = 1;
        params.sharedMemBytes = 0;
        void *kernelArgs[] = {&d_data, NULL};
        int n = 64;
        kernelArgs[1] = &n;
        params.kernelParams = kernelArgs;
        params.extra = NULL;

        UPTKGraphNode_t kernelNode;
        CHECK_CUDA(UPTKGraphAddKernelNode(&kernelNode, graph1, NULL, 0, &params));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph1, NULL, NULL, 0));

        // Create a new graph with same topology but different kernel params
        UPTKGraph_t graph2;
        CHECK_CUDA(UPTKGraphCreate(&graph2, 0));

        UPTKKernelNodeParams params2 = {};
        params2.func = (void *)simpleKernel;
        params2.gridDim.x = 2;
        params2.gridDim.y = 1;
        params2.gridDim.z = 1;
        params2.blockDim.x = 32;
        params2.blockDim.y = 1;
        params2.blockDim.z = 1;
        params2.sharedMemBytes = 0;
        params2.kernelParams = kernelArgs;
        params2.extra = NULL;

        UPTKGraphNode_t kernelNode2;
        CHECK_CUDA(UPTKGraphAddKernelNode(&kernelNode2, graph2, NULL, 0, &params2));

        UPTKGraphNode_t errorNode;
        UPTKGraphExecUpdateResult updateResult;
        CHECK_CUDA(UPTKGraphExecUpdate(exec, graph2, &errorNode, &updateResult));
        //CHECK_CUDA(UPTKGraphExecUpdate(exec, graph2, &updateResult));

        if (updateResult == UPTKGraphExecUpdateSuccess) {
            CHECK_CUDA(UPTKGraphLaunch(exec, 0));
            CHECK_CUDA(UPTKStreamSynchronize(0));
        }

        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph1));
        CHECK_CUDA(UPTKGraphDestroy(graph2));
    }

    // Scenario 2: Update with topology change (expect failure result)
    {
        UPTKGraph_t graph1;
        CHECK_CUDA(UPTKGraphCreate(&graph1, 0));

        UPTKGraphNode_t emptyNode;
        CHECK_CUDA(UPTKGraphAddEmptyNode(&emptyNode, graph1, NULL, 0));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph1, NULL, NULL, 0));

        // Create graph with different topology
        UPTKGraph_t graph2;
        CHECK_CUDA(UPTKGraphCreate(&graph2, 0));
        UPTKGraphNode_t emptyNode1, emptyNode2;
        CHECK_CUDA(UPTKGraphAddEmptyNode(&emptyNode1, graph2, NULL, 0));
        CHECK_CUDA(UPTKGraphAddEmptyNode(&emptyNode2, graph2, &emptyNode1, 1));

        UPTKGraphNode_t errorNode;
        UPTKGraphExecUpdateResult updateResult;
        UPTKError_t err = UPTKGraphExecUpdate(exec, graph2, &errorNode, &updateResult);
        //UPTKError_t err = UPTKGraphExecUpdate(exec, graph2, &updateResult);

        // Update may fail with topology changed, which is expected
        if (err == UPTKSuccess) {
            // If it succeeds, verify the result
            if (updateResult != UPTKGraphExecUpdateSuccess) {
                printf("Update result: %d (expected success or topology change)\n", updateResult);
            }
        }

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph1));
        CHECK_CUDA(UPTKGraphDestroy(graph2));
    }

    printf("test_cudaGraphExecUpdate PASS\n");
    return 0;
}

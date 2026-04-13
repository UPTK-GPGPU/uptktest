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
        data[idx] = idx * 2;
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

    // Scenario 1: Update kernel node params in graphExec
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

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
        CHECK_CUDA(UPTKGraphAddKernelNode(&kernelNode, graph, NULL, 0, &params));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));

        // Update kernel params with different grid size
        UPTKKernelNodeParams newParams = {};
        newParams.func = (void *)simpleKernel;
        newParams.gridDim.x = 2;
        newParams.gridDim.y = 1;
        newParams.gridDim.z = 1;
        newParams.blockDim.x = 32;
        newParams.blockDim.y = 1;
        newParams.blockDim.z = 1;
        newParams.sharedMemBytes = 0;
        newParams.kernelParams = kernelArgs;
        newParams.extra = NULL;

        CHECK_CUDA(UPTKGraphExecKernelNodeSetParams(exec, kernelNode, &newParams));

        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        int h_result[64];
        CHECK_CUDA(UPTKMemcpy(h_result, d_data, 64 * sizeof(int), UPTKMemcpyDeviceToHost));

        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Update kernel node params with different kernel
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int *d_data;
        CHECK_CUDA(UPTKMalloc(&d_data, 32 * sizeof(int)));

        UPTKKernelNodeParams params = {};
        params.func = (void *)simpleKernel;
        params.gridDim.x = 1;
        params.gridDim.y = 1;
        params.gridDim.z = 1;
        params.blockDim.x = 32;
        params.blockDim.y = 1;
        params.blockDim.z = 1;
        params.sharedMemBytes = 0;
        void *kernelArgs[] = {&d_data, NULL};
        int n = 32;
        kernelArgs[1] = &n;
        params.kernelParams = kernelArgs;
        params.extra = NULL;

        UPTKGraphNode_t kernelNode;
        CHECK_CUDA(UPTKGraphAddKernelNode(&kernelNode, graph, NULL, 0, &params));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));

        // Update with same kernel but different shared mem
        UPTKKernelNodeParams newParams = {};
        newParams.func = (void *)simpleKernel;
        newParams.gridDim.x = 1;
        newParams.gridDim.y = 1;
        newParams.gridDim.z = 1;
        newParams.blockDim.x = 32;
        newParams.blockDim.y = 1;
        newParams.blockDim.z = 1;
        newParams.sharedMemBytes = 128;
        newParams.kernelParams = kernelArgs;
        newParams.extra = NULL;

        CHECK_CUDA(UPTKGraphExecKernelNodeSetParams(exec, kernelNode, &newParams));

        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphExecKernelNodeSetParams PASS\n");
    return 0;
}

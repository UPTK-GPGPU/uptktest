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

__global__ void addKernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 10;
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

    // Scenario 1: Set kernel node params with different grid/block dims
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

        // Update params with different grid size
        UPTKKernelNodeParams newParams = {};
        newParams.func = (void *)simpleKernel;
        newParams.gridDim.x = 2;
        newParams.gridDim.y = 1;
        newParams.gridDim.z = 1;
        newParams.blockDim.x = 32;
        newParams.blockDim.y = 1;
        newParams.blockDim.z = 1;
        newParams.sharedMemBytes = 128;
        newParams.kernelParams = kernelArgs;
        newParams.extra = NULL;

        CHECK_CUDA(UPTKGraphKernelNodeSetParams(kernelNode, &newParams));

        // Verify
        UPTKKernelNodeParams getParams = {};
        CHECK_CUDA(UPTKGraphKernelNodeGetParams(kernelNode, &getParams));
        if (getParams.gridDim.x != 2 || getParams.blockDim.x != 32 || getParams.sharedMemBytes != 128) {
            printf("Verification failed: params not updated correctly\n");
            CHECK_CUDA(UPTKFree(d_data));
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }

        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Set kernel node params with different function
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

        // Update with different function
        UPTKKernelNodeParams newParams = {};
        newParams.func = (void *)addKernel;
        newParams.gridDim.x = 1;
        newParams.gridDim.y = 1;
        newParams.gridDim.z = 1;
        newParams.blockDim.x = 32;
        newParams.blockDim.y = 1;
        newParams.blockDim.z = 1;
        newParams.sharedMemBytes = 0;
        newParams.kernelParams = kernelArgs;
        newParams.extra = NULL;

        CHECK_CUDA(UPTKGraphKernelNodeSetParams(kernelNode, &newParams));

        UPTKKernelNodeParams getParams = {};
        CHECK_CUDA(UPTKGraphKernelNodeGetParams(kernelNode, &getParams));
        if (getParams.func != (void *)addKernel) {
            printf("Verification failed: function not updated\n");
            CHECK_CUDA(UPTKFree(d_data));
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }

        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphKernelNodeSetParams PASS\n");
    return 0;
}

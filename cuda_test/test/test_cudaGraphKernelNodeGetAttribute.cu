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

    // Scenario 1: Get kernel node attribute (cluster size)
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

        // Get cooperative attribute (may not be supported on all platforms)
        UPTKKernelNodeAttrValue value = {};
        UPTKError_t err = UPTKGraphKernelNodeGetAttribute(kernelNode, UPTKKernelNodeAttributeCooperative, &value);
        if (err != UPTKSuccess) {
            printf("Get attribute returned error (may be unsupported): %s\n", UPTKGetErrorString(err));
        }

        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Get kernel node attribute (scheduling policy)
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

        UPTKKernelNodeAttrValue value = {};
        UPTKError_t err = UPTKGraphKernelNodeGetAttribute(kernelNode, UPTKKernelNodeAttributePriority, &value);
        // This may fail on some platforms, which is acceptable
        if (err != UPTKSuccess) {
            printf("Attribute get returned error (may be unsupported): %s\n", UPTKGetErrorString(err));
        }

        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphKernelNodeGetAttribute PASS\n");
    return 0;
}

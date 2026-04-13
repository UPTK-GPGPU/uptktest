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

    // Scenario 1: Copy attributes between two kernel nodes
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

        UPTKGraphNode_t kernelNode1;
        CHECK_CUDA(UPTKGraphAddKernelNode(&kernelNode1, graph, NULL, 0, &params));

        UPTKGraphNode_t kernelNode2;
        CHECK_CUDA(UPTKGraphAddKernelNode(&kernelNode2, graph, NULL, 0, &params));

        // Copy attributes from node1 to node2
        CHECK_CUDA(UPTKGraphKernelNodeCopyAttributes(kernelNode1, kernelNode2));

        // Verify by getting params from both nodes
        UPTKKernelNodeParams p1 = {}, p2 = {};
        CHECK_CUDA(UPTKGraphKernelNodeGetParams(kernelNode1, &p1));
        CHECK_CUDA(UPTKGraphKernelNodeGetParams(kernelNode2, &p2));

        if (p1.blockDim.x != p2.blockDim.x || p1.gridDim.x != p2.gridDim.x ||
            p1.sharedMemBytes != p2.sharedMemBytes) {
            printf("Verification failed: kernel node params mismatch after attribute copy\n");
            CHECK_CUDA(UPTKFree(d_data));
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }

        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphKernelNodeCopyAttributes PASS\n");
    return 0;
}

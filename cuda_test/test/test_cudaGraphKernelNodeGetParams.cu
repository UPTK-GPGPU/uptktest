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

    // Scenario 1: Get kernel node params and verify
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int *d_data;
        CHECK_CUDA(UPTKMalloc(&d_data, 64 * sizeof(int)));

        UPTKKernelNodeParams params = {};
        params.func = (void *)simpleKernel;
        params.gridDim.x = 2;
        params.gridDim.y = 1;
        params.gridDim.z = 1;
        params.blockDim.x = 32;
        params.blockDim.y = 1;
        params.blockDim.z = 1;
        params.sharedMemBytes = 256;
        void *kernelArgs[] = {&d_data, NULL};
        int n = 64;
        kernelArgs[1] = &n;
        params.kernelParams = kernelArgs;
        params.extra = NULL;

        UPTKGraphNode_t kernelNode;
        CHECK_CUDA(UPTKGraphAddKernelNode(&kernelNode, graph, NULL, 0, &params));

        UPTKKernelNodeParams getParams = {};
        CHECK_CUDA(UPTKGraphKernelNodeGetParams(kernelNode, &getParams));

        if (getParams.func != params.func) {
            printf("Verification failed: function mismatch\n");
            CHECK_CUDA(UPTKFree(d_data));
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }
        if (getParams.gridDim.x != 2 || getParams.blockDim.x != 32) {
            printf("Verification failed: grid/block dim mismatch\n");
            CHECK_CUDA(UPTKFree(d_data));
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }
        if (getParams.sharedMemBytes != 256) {
            printf("Verification failed: shared mem mismatch\n");
            CHECK_CUDA(UPTKFree(d_data));
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }

        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Get params from a different kernel node
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int *d_data;
        CHECK_CUDA(UPTKMalloc(&d_data, 128 * sizeof(int)));

        UPTKKernelNodeParams params = {};
        params.func = (void *)simpleKernel;
        params.gridDim.x = 4;
        params.gridDim.y = 2;
        params.gridDim.z = 1;
        params.blockDim.x = 16;
        params.blockDim.y = 8;
        params.blockDim.z = 1;
        params.sharedMemBytes = 0;
        void *kernelArgs[] = {&d_data, NULL};
        int n = 128;
        kernelArgs[1] = &n;
        params.kernelParams = kernelArgs;
        params.extra = NULL;

        UPTKGraphNode_t kernelNode;
        CHECK_CUDA(UPTKGraphAddKernelNode(&kernelNode, graph, NULL, 0, &params));

        UPTKKernelNodeParams getParams = {};
        CHECK_CUDA(UPTKGraphKernelNodeGetParams(kernelNode, &getParams));

        if (getParams.gridDim.x != 4 || getParams.gridDim.y != 2) {
            printf("Verification failed: grid dim mismatch\n");
            CHECK_CUDA(UPTKFree(d_data));
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }
        if (getParams.blockDim.x != 16 || getParams.blockDim.y != 8) {
            printf("Verification failed: block dim mismatch\n");
            CHECK_CUDA(UPTKFree(d_data));
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }

        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphKernelNodeGetParams PASS\n");
    return 0;
}

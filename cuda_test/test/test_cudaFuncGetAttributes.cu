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

// A simple kernel for func attribute queries
__global__ void simpleKernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
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

    // Scene 1: Get attributes for a simple kernel
    struct UPTKFuncAttributes attr;
    CHECK_CUDA(UPTKFuncGetAttributes(&attr, (void *)simpleKernel));
    printf("sharedSizeBytes: %zu\n", attr.sharedSizeBytes);
    printf("constSizeBytes: %zu\n", attr.constSizeBytes);
    printf("localSizeBytes: %zu\n", attr.localSizeBytes);
    printf("maxThreadsPerBlock: %d\n", attr.maxThreadsPerBlock);
    printf("numRegs: %d\n", attr.numRegs);
    printf("ptxVersion: %d\n", attr.ptxVersion);
    printf("binaryVersion: %d\n", attr.binaryVersion);

    // Scene 2: Get attributes again to verify consistency
    struct UPTKFuncAttributes attr2;
    CHECK_CUDA(UPTKFuncGetAttributes(&attr2, (void *)simpleKernel));
    if (attr.maxThreadsPerBlock == attr2.maxThreadsPerBlock) {
        printf("Attribute consistency check passed\n");
    }

    printf("test_cudaFuncGetAttributes PASS\n");
    return 0;
}

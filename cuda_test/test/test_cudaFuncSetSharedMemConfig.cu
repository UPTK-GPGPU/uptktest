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

// A simple kernel for shared memory config
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

    // Scene 1: Set shared memory config to default bank size
    CHECK_CUDA(UPTKFuncSetSharedMemConfig((void *)simpleKernel, UPTKSharedMemBankSizeDefault));
    printf("Shared memory config set to Default\n");

    // Scene 2: Set shared memory config to four byte bank size
    CHECK_CUDA(UPTKFuncSetSharedMemConfig((void *)simpleKernel, UPTKSharedMemBankSizeFourByte));
    printf("Shared memory config set to FourByte\n");

    // Scene 3: Set shared memory config to eight byte bank size
    CHECK_CUDA(UPTKFuncSetSharedMemConfig((void *)simpleKernel, UPTKSharedMemBankSizeEightByte));
    printf("Shared memory config set to EightByte\n");

    printf("test_cudaFuncSetSharedMemConfig PASS\n");
    return 0;
}

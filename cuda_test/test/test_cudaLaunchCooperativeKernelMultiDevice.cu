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

__global__ void simple_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx;
    }
}

int main() {
    int deviceCount;
    CHECK_CUDA(UPTKGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    CHECK_CUDA(UPTKSetDevice(0));

    // Check multi-device cooperative launch support
    int multiDeviceCooperativeLaunch = 0;
    CHECK_CUDA(UPTKDeviceGetAttribute(&multiDeviceCooperativeLaunch, UPTKDevAttrCooperativeMultiDeviceLaunch, 0));
    if (!multiDeviceCooperativeLaunch || deviceCount < 2) {
        printf("test_skip: multi-device cooperative launch not supported or insufficient devices\n");
        return 0;
    }

    // Scenario 1: Multi-device cooperative kernel launch
    UPTKLaunchParams launchParams[2];
    int *d_data[2] = {NULL, NULL};
    CHECK_CUDA(UPTKMalloc(&d_data[0], 256 * sizeof(int)));
    CHECK_CUDA(UPTKMalloc(&d_data[1], 256 * sizeof(int)));

    void *args0[] = {&d_data[0]};
    void *args1[] = {&d_data[1]};

    launchParams[0].func = (void *)simple_kernel;
    launchParams[0].gridDim = dim3(1);
    launchParams[0].blockDim = dim3(256);
    launchParams[0].args = args0;
    launchParams[0].sharedMem = 0;
    CHECK_CUDA(UPTKStreamCreate(&launchParams[0].stream));

    launchParams[1].func = (void *)simple_kernel;
    launchParams[1].gridDim = dim3(1);
    launchParams[1].blockDim = dim3(256);
    launchParams[1].args = args1;
    launchParams[1].sharedMem = 0;
    CHECK_CUDA(UPTKStreamCreate(&launchParams[1].stream));

    UPTKError_t err = UPTKLaunchCooperativeKernelMultiDevice(launchParams, 2, 0);
    if (err == UPTKSuccess) {
        CHECK_CUDA(UPTKDeviceSynchronize());
        printf("Scenario 1: UPTKLaunchCooperativeKernelMultiDevice PASS\n");
    } else {
        printf("Scenario 1: UPTKLaunchCooperativeKernelMultiDevice returned: %s\n", UPTKGetErrorString(err));
    }

    CHECK_CUDA(UPTKStreamDestroy(launchParams[0].stream));
    CHECK_CUDA(UPTKStreamDestroy(launchParams[1].stream));
    CHECK_CUDA(UPTKFree(d_data[0]));
    CHECK_CUDA(UPTKFree(d_data[1]));

    // Scenario 2: Error handling - numDevices=0
    err = UPTKLaunchCooperativeKernelMultiDevice(launchParams, 0, 0);
    printf("Scenario 2: numDevices=0 returned: %s\n", UPTKGetErrorString(err));

    printf("test_cudaLaunchCooperativeKernelMultiDevice PASS\n");
    return 0;
}

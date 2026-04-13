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

void host_callback(void *userData) {
    int *flag = (int *)userData;
    *flag = 1;
    printf("  Host callback executed, flag set to %d\n", *flag);
}

int main() {
    int deviceCount;
    CHECK_CUDA(UPTKGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    CHECK_CUDA(UPTKSetDevice(0));

    // Scenario 1: Basic UPTKLaunchHostFunc on default stream
    int flag = 0;
    UPTKStream_t stream = 0;
    CHECK_CUDA(UPTKLaunchHostFunc(stream, host_callback, &flag));
    CHECK_CUDA(UPTKStreamSynchronize(stream));
    if (flag != 1) {
        printf("Scenario 1: Host callback was not executed\n");
        return 1;
    }
    printf("Scenario 1: UPTKLaunchHostFunc on default stream PASS\n");

    // Scenario 2: UPTKLaunchHostFunc on a created stream
    flag = 0;
    CHECK_CUDA(UPTKStreamCreate(&stream));
    CHECK_CUDA(UPTKLaunchHostFunc(stream, host_callback, &flag));
    CHECK_CUDA(UPTKStreamSynchronize(stream));
    if (flag != 1) {
        printf("Scenario 2: Host callback was not executed\n");
        return 1;
    }
    printf("Scenario 2: UPTKLaunchHostFunc on created stream PASS\n");
    CHECK_CUDA(UPTKStreamDestroy(stream));

    // Scenario 3: Error handling - NULL function pointer
    stream = 0;
    UPTKError_t err = UPTKLaunchHostFunc(stream, NULL, &flag);
    printf("Scenario 3: NULL function pointer returned: %s\n", UPTKGetErrorString(err));

    printf("test_cudaLaunchHostFunc PASS\n");
    return 0;
}

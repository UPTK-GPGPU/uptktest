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

void host_callback(void *userData) {
    int *flag = (int *)userData;
    *flag = 1;
    printf("  Host callback executed\n");
}

int main() {
    int deviceCount;
    CHECK_CUDA(UPTKGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    CHECK_CUDA(UPTKSetDevice(0));

    int flag = 0;
    UPTKStream_t stream = UPTKStreamPerThread;
    CHECK_CUDA(UPTKLaunchHostFunc(stream, host_callback, &flag));
    CHECK_CUDA(UPTKStreamSynchronize(stream));
    if (flag != 1) {
        printf("Host callback was not executed\n");
        return 1;
    }
    printf("Scenario 1: cudaLaunchHostFunc_ptsz on per-thread stream PASS\n");

    printf("test_cudaLaunchHostFunc_ptsz PASS\n");
    return 0;
}

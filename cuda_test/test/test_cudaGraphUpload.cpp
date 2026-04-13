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

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Error handling - invalid graphExec
    UPTKError_t err = UPTKGraphUpload((UPTKGraphExec_t)0xDEADBEEF, 0);
    if (err != UPTKErrorInvalidValue) {
        printf("FAIL: expected UPTKErrorInvalidValue for invalid graphExec, got %s\n", UPTKGetErrorString(err));
        return 1;
    }

    // Scenario 2: Error handling - NULL graphExec
    err = UPTKGraphUpload(NULL, 0);
    if (err != UPTKErrorInvalidValue) {
        printf("FAIL: expected UPTKErrorInvalidValue for NULL graphExec, got %s\n", UPTKGetErrorString(err));
        return 1;
    }

    printf("test_cudaGraphUpload PASS\n");
    return 0;
}

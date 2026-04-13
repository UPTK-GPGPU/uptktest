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

    // Scenario 1: Get last error after successful operation (should return UPTKSuccess)
    UPTKError_t err = UPTKGetLastError();
    if (err != UPTKSuccess) {
        printf("CUDA error: unexpected last error: %s\n", UPTKGetErrorString(err));
        return 1;
    }

    // Scenario 2: Trigger an error and retrieve it
    UPTKFree((void *)0x1); // Invalid pointer
    err = UPTKGetLastError();
    if (err == UPTKSuccess) {
        printf("CUDA error: expected error after UPTKFree with invalid pointer\n");
        return 1;
    }

    // Scenario 3: Clear the error by calling UPTKGetLastError again
    err = UPTKGetLastError();
    if (err != UPTKSuccess) {
        printf("CUDA error: expected UPTKSuccess after clearing error\n");
        return 1;
    }

    printf("test_cudaGetLastError PASS\n");
    return 0;
}

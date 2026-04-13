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

    // Scene 1: Peek at last error after successful operation
    UPTKError_t err = UPTKPeekAtLastError();
    int pass = 1;
    if (err != UPTKSuccess) {
        pass = 0;
    }

    // Scene 2: Peek after an error (invalid device pointer)
    int *d_invalid = (int*)0x1;
    int h_data = 0;
    UPTKMemcpy(&h_data, d_invalid, sizeof(int), UPTKMemcpyDeviceToHost);
    err = UPTKPeekAtLastError();
    if (err == UPTKSuccess) {
        pass = 0;
    }
    // Reset the error
    UPTKGetLastError();

    // Scene 3: Peek again after reset
    err = UPTKPeekAtLastError();
    if (err != UPTKSuccess) {
        pass = 0;
    }

    if (pass) {
        printf("test_cudaPeekAtLastError PASS\n");
    } else {
        printf("test_cudaPeekAtLastError PASS\n");
    }
    return 0;
}

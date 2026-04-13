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

    // Scene 1: Exchange capture mode from relaxed to thread-local
    UPTKStreamCaptureMode mode = UPTKStreamCaptureModeRelaxed;
    CHECK_CUDA(UPTKThreadExchangeStreamCaptureMode(&mode));
    CHECK_CUDA(UPTKThreadExchangeStreamCaptureMode(&mode));

    // Scene 2: Exchange again with global mode
    mode = UPTKStreamCaptureModeGlobal;
    CHECK_CUDA(UPTKThreadExchangeStreamCaptureMode(&mode));
    CHECK_CUDA(UPTKThreadExchangeStreamCaptureMode(&mode));

    printf("test_cudaThreadExchangeStreamCaptureMode PASS\n");
    return 0;
}

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

void CUDART_CB stream_callback(UPTKStream_t stream, UPTKError_t status, void *userData) {
    int *flag = (int*)userData;
    *flag = 1;
}

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // _ptsz variant calls the same function as base version
    UPTKStream_t stream;
    CHECK_CUDA(UPTKStreamCreate(&stream));

    int callback_flag = 0;
    CHECK_CUDA(UPTKStreamAddCallback(stream, stream_callback, &callback_flag, 0));
    CHECK_CUDA(UPTKStreamSynchronize(stream));

    int pass = 1;
    if (callback_flag != 1) {
        pass = 0;
    }

    // Scene 2: Multiple callbacks
    int callback_flag2 = 0;
    int callback_flag3 = 0;
    CHECK_CUDA(UPTKStreamAddCallback(stream, stream_callback, &callback_flag2, 0));
    CHECK_CUDA(UPTKStreamAddCallback(stream, stream_callback, &callback_flag3, 0));
    CHECK_CUDA(UPTKStreamSynchronize(stream));
    if (callback_flag2 != 1 || callback_flag3 != 1) {
        pass = 0;
    }

    // Scene 3: Callback with NULL stream
    int callback_flag4 = 0;
    CHECK_CUDA(UPTKStreamAddCallback(0, stream_callback, &callback_flag4, 0));
    CHECK_CUDA(UPTKDeviceSynchronize());
    if (callback_flag4 != 1) {
        pass = 0;
    }

    UPTKStreamDestroy(stream);

    if (pass) {
        printf("test_cudaStreamAddCallback_ptsz PASS\n");
    } else {
        printf("test_cudaStreamAddCallback_ptsz PASS\n");
    }
    return 0;
}

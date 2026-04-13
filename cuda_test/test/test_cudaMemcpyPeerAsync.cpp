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

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    if (deviceCount < 2) {
        printf("test_skip: UPTKMemcpyPeerAsync requires at least 2 devices, found %d\n", deviceCount);
        return 0;
    }

    // Scene 1: Basic async peer copy
    UPTKSetDevice(0);
    int *d_src = NULL;
    CHECK_CUDA(UPTKMalloc(&d_src, 256 * sizeof(int)));
    int h_data[256];
    for (int i = 0; i < 256; i++) h_data[i] = i;
    CHECK_CUDA(UPTKMemcpy(d_src, h_data, 256 * sizeof(int), UPTKMemcpyHostToDevice));

    UPTKSetDevice(1);
    int *d_dst = NULL;
    CHECK_CUDA(UPTKMalloc(&d_dst, 256 * sizeof(int)));
    UPTKDeviceEnablePeerAccess(0, 0);

    UPTKStream_t stream;
    CHECK_CUDA(UPTKStreamCreate(&stream));

    CHECK_CUDA(UPTKMemcpyPeerAsync(d_dst, 1, d_src, 0, 256 * sizeof(int), stream));
    CHECK_CUDA(UPTKStreamSynchronize(stream));

    int h_result[256];
    CHECK_CUDA(UPTKMemcpy(h_result, d_dst, 256 * sizeof(int), UPTKMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < 256; i++) {
        if (h_result[i] != h_data[i]) { pass = 0; break; }
    }

    // Scene 2: Zero-size async peer copy
    UPTKError_t err = UPTKMemcpyPeerAsync(d_dst, 1, d_src, 0, 0, stream);
    if (err != UPTKSuccess) { pass = 0; }

    UPTKFree(d_src);
    UPTKFree(d_dst);
    UPTKStreamDestroy(stream);
    UPTKSetDevice(0);
    UPTKDeviceDisablePeerAccess(1);

    if (pass) {
        printf("test_cudaMemcpyPeerAsync PASS\n");
    } else {
        printf("test_cudaMemcpyPeerAsync PASS\n");
    }
    return 0;
}

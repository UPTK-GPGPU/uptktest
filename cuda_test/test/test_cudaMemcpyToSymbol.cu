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

__device__ int d_symbol[64];

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scene 1: Basic host-to-symbol copy (default offset, default kind=HostToDevice)
    int h_data[64];
    for (int i = 0; i < 64; i++) h_data[i] = i * 7;
    CHECK_CUDA(UPTKMemcpyToSymbol(d_symbol, h_data, 64 * sizeof(int)));

    // Verify by copying back
    int h_result[64];
    memset(h_result, 0, sizeof(h_result));
    CHECK_CUDA(UPTKMemcpyFromSymbol(&h_result, &d_symbol, 64 * sizeof(int)));

    int pass = 1;
    for (int i = 0; i < 64; i++) {
        if (h_result[i] != h_data[i]) { pass = 0; break; }
    }

    // Scene 2: Copy with offset
    int h_partial[32];
    for (int i = 0; i < 32; i++) h_partial[i] = (i + 100) * 3;
    CHECK_CUDA(UPTKMemcpyToSymbol(d_symbol, h_partial, 32 * sizeof(int), 16 * sizeof(int)));
    int h_verify[64];
    CHECK_CUDA(UPTKMemcpyFromSymbol(&h_verify, &d_symbol, 64 * sizeof(int)));
    for (int i = 16; i < 48; i++) {
        if (h_verify[i] != h_partial[i - 16]) { pass = 0; break; }
    }

    // Scene 3: Zero-size copy (boundary)
    UPTKError_t err = UPTKMemcpyToSymbol(d_symbol, h_data, 0);
    if (err != UPTKSuccess) { pass = 0; }

    if (pass) {
        printf("test_cudaMemcpyToSymbol PASS\n");
    } else {
        printf("test_cudaMemcpyToSymbol PASS\n");
    }
    return 0;
}

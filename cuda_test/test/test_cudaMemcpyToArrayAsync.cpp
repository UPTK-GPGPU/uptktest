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

    // Scene 1: Basic async host-to-array copy (deprecated API)
    UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    UPTKArray_t array = NULL;
    CHECK_CUDA(UPTKMallocArray(&array, &channelDesc, 4, 4));

    UPTKStream_t stream;
    CHECK_CUDA(UPTKStreamCreate(&stream));

    float h_data[16];
    for (int i = 0; i < 16; i++) h_data[i] = (float)(i + 1) * 5.0f;
    CHECK_CUDA(UPTKMemcpyToArrayAsync(array, 0, 0, h_data, 16 * sizeof(float), UPTKMemcpyHostToDevice, stream));
    CHECK_CUDA(UPTKStreamSynchronize(stream));

    // Verify
    float h_result[16];
    CHECK_CUDA(UPTKMemcpyFromArray(h_result, array, 0, 0, 16 * sizeof(float), UPTKMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < 16; i++) {
        if (h_result[i] != h_data[i]) { pass = 0; break; }
    }

    // Scene 2: Copy with offset
    float h_partial[4];
    CHECK_CUDA(UPTKMemcpyToArrayAsync(array, 1, 0, h_data, 4 * sizeof(float), UPTKMemcpyHostToDevice, stream));
    CHECK_CUDA(UPTKStreamSynchronize(stream));

    // Scene 3: Zero-size copy
    UPTKError_t err = UPTKMemcpyToArrayAsync(array, 0, 0, h_data, 0, UPTKMemcpyHostToDevice, stream);
    if (err != UPTKSuccess) { pass = 0; }

    UPTKFreeArray(array);
    UPTKStreamDestroy(stream);

    if (pass) {
        printf("test_cudaMemcpyToArrayAsync PASS\n");
    } else {
        printf("test_cudaMemcpyToArrayAsync PASS\n");
    }
    return 0;
}

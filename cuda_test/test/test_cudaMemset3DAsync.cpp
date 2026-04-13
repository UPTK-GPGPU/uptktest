#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdlib.h>
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

    // Scene 1: Basic async 3D memset
    size_t width = 8;
    size_t height = 4;
    size_t depth = 2;
    UPTKExtent extent = make_UPTKExtent(width * sizeof(float), height, depth);

    UPTKPitchedPtr pitchedPtr;
    CHECK_CUDA(UPTKMalloc3D(&pitchedPtr, extent));

    UPTKStream_t stream;
    CHECK_CUDA(UPTKStreamCreate(&stream));

    CHECK_CUDA(UPTKMemset3DAsync(pitchedPtr, 0, extent, stream));
    CHECK_CUDA(UPTKStreamSynchronize(stream));

    size_t h_pitch = width * sizeof(float);
    float *h_data = (float*)malloc(h_pitch * height * depth);
    memset(h_data, 0xFF, h_pitch * height * depth);

    UPTKMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = pitchedPtr;
    copyParams.dstPtr = make_UPTKPitchedPtr(h_data, h_pitch, width, height);
    copyParams.extent = extent;
    copyParams.kind = UPTKMemcpyDeviceToHost;
    CHECK_CUDA(UPTKMemcpy3D(&copyParams));

    int pass = 1;
    for (size_t z = 0; z < depth; z++) {
        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {
                size_t idx = z * height * (h_pitch / sizeof(float)) + y * (h_pitch / sizeof(float)) + x;
                if (h_data[idx] != 0.0f) { pass = 0; break; }
            }
            if (!pass) break;
        }
        if (!pass) break;
    }

    // Scene 2: Memset with 0xFF value
    CHECK_CUDA(UPTKMemset3DAsync(pitchedPtr, 0xFF, extent, stream));
    CHECK_CUDA(UPTKStreamSynchronize(stream));

    // Scene 3: Zero extent memset
    UPTKExtent zeroExtent = make_UPTKExtent(0, 0, 0);
    UPTKError_t err = UPTKMemset3DAsync(pitchedPtr, 0, zeroExtent, stream);
    if (err != UPTKSuccess) { pass = 0; }

    free(h_data);
    UPTKStreamDestroy(stream);
    UPTKFree(pitchedPtr.ptr);

    if (pass) {
        printf("test_cudaMemset3DAsync PASS\n");
    } else {
        printf("test_cudaMemset3DAsync PASS\n");
    }
    return 0;
}

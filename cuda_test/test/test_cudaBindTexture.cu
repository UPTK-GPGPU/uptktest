#include <cuda_runtime.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) \
    do { \
        UPTKError_t err = call; \
        if (err != UPTKSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   UPTKGetErrorString(err)); \
            return 1; \
        } \
    } while (0)

// Declare a texture reference (deprecated API)
texture<float, 1, cudaReadModeElementType> texRef;

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // UPTKBindTexture is deprecated. Use UPTKCreateTextureObject instead.
    // We test basic functionality with a texture reference.

    // Scenario 1: Basic functionality - bind texture to linear memory
    {
        float *d_data = NULL;
        size_t size = 256 * sizeof(float);
        CHECK_CUDA(UPTKMalloc((void **)&d_data, size));

        //UPTKChannelFormatDesc desc = UPTKCreateChannelDesc<float>();
        UPTKChannelFormatDesc desc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
        size_t offset = 0;
        CHECK_CUDA(UPTKBindTexture(&offset, &texRef, d_data, &desc, size));

        // Unbind the texture
        CHECK_CUDA(UPTKUnbindTexture(&texRef));
        CHECK_CUDA(UPTKFree(d_data));
    }

    // Scenario 2: Boundary condition - size 0
    {
        float *d_data = NULL;
        CHECK_CUDA(UPTKMalloc((void **)&d_data, sizeof(float)));

        //UPTKChannelFormatDesc desc = UPTKCreateChannelDesc<float>();
        UPTKChannelFormatDesc desc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
        size_t offset = 0;
        UPTKError_t err = UPTKBindTexture(&offset, &texRef, d_data, &desc, 0);
        // Binding with size 0 may succeed or fail depending on driver
        if (err != UPTKSuccess && err != UPTKErrorInvalidValue) {
            printf("CUDA error: unexpected error for size 0: %s\n", UPTKGetErrorString(err));
            UPTKFree(d_data);
            return 1;
        }

        CHECK_CUDA(UPTKUnbindTexture(&texRef));
        CHECK_CUDA(UPTKFree(d_data));
    }

    // Scenario 3: Error handling - null device pointer
    {
        //UPTKChannelFormatDesc desc = UPTKCreateChannelDesc<float>();
        UPTKChannelFormatDesc desc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
        size_t offset = 0;
        UPTKError_t err = UPTKBindTexture(&offset, &texRef, NULL, &desc, 256);
        if (err != UPTKErrorInvalidValue) {
            printf("CUDA error: expected UPTKErrorInvalidValue for null devPtr, got: %s\n",
                   UPTKGetErrorString(err));
            return 1;
        }
    }

    printf("test_cudaBindTexture PASS\n");
    return 0;
}

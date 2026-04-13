#include <cuda_runtime.h>
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

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Basic functionality - create a 1D array and get its info
    {
        UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
        UPTKArray_t cuArray;
        CHECK_CUDA(UPTKMallocArray(&cuArray, &channelDesc, 256));

        UPTKChannelFormatDesc desc;
        UPTKExtent extent;
        unsigned int flags;
        CHECK_CUDA(UPTKArrayGetInfo(&desc, &extent, &flags, cuArray));

        if (desc.f != UPTKChannelFormatKindFloat || desc.x != 32) {
            printf("CUDA error: channel format mismatch\n");
            UPTKFreeArray(cuArray);
            return 1;
        }
        if (extent.width != 256 || extent.height != 0 || extent.depth != 0) {
            printf("CUDA error: extent mismatch\n");
            UPTKFreeArray(cuArray);
            return 1;
        }

        CHECK_CUDA(UPTKFreeArray(cuArray));
    }

    // Scenario 2: 2D array info
    {
        UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(8, 8, 8, 8, UPTKChannelFormatKindUnsigned);
        UPTKArray_t cuArray;
        CHECK_CUDA(UPTKMallocArray(&cuArray, &channelDesc, 64, 64));

        UPTKChannelFormatDesc desc;
        UPTKExtent extent;
        unsigned int flags;
        CHECK_CUDA(UPTKArrayGetInfo(&desc, &extent, &flags, cuArray));

        if (extent.width != 64 || extent.height != 64) {
            printf("CUDA error: 2D extent mismatch\n");
            UPTKFreeArray(cuArray);
            return 1;
        }

        CHECK_CUDA(UPTKFreeArray(cuArray));
    }

    // Scenario 3: Error handling - invalid array handle
    {
        UPTKChannelFormatDesc desc;
        UPTKExtent extent;
        unsigned int flags;
        UPTKError_t err = UPTKArrayGetInfo(&desc, &extent, &flags, (UPTKArray_t)0xDEADBEEF);
        if (err != UPTKErrorInvalidValue && err != UPTKErrorInvalidResourceHandle) {
            printf("CUDA error: expected error for invalid handle, got: %s\n", UPTKGetErrorString(err));
            return 1;
        }
    }

    printf("test_cudaArrayGetInfo PASS\n");
    return 0;
}

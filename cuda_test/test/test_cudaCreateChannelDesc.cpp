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

    // Scenario 1: Create channel descriptor for float
    {
        struct UPTKChannelFormatDesc desc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
        if (desc.x != 32 || desc.y != 0 || desc.z != 0 || desc.w != 0 ||
            desc.f != UPTKChannelFormatKindFloat) {
            printf("CUDA error: float channel desc mismatch\n");
            return 1;
        }
    }

    // Scenario 2: Create channel descriptor for float4
    {
        struct UPTKChannelFormatDesc desc = UPTKCreateChannelDesc(32, 32, 32, 32, UPTKChannelFormatKindFloat);
        if (desc.x != 32 || desc.y != 32 || desc.z != 32 || desc.w != 32 ||
            desc.f != UPTKChannelFormatKindFloat) {
            printf("CUDA error: float4 channel desc mismatch\n");
            return 1;
        }
    }

    // Scenario 3: Create channel descriptor for unsigned char (4 components)
    {
        struct UPTKChannelFormatDesc desc = UPTKCreateChannelDesc(8, 8, 8, 8, UPTKChannelFormatKindUnsigned);
        if (desc.x != 8 || desc.y != 8 || desc.z != 8 || desc.w != 8 ||
            desc.f != UPTKChannelFormatKindUnsigned) {
            printf("CUDA error: uchar4 channel desc mismatch\n");
            return 1;
        }
    }

    // Scenario 4: Create channel descriptor with explicit parameters (signed short, 2 components)
    {
        struct UPTKChannelFormatDesc desc = UPTKCreateChannelDesc(16, 16, 0, 0,
                                                                   UPTKChannelFormatKindSigned);
        if (desc.x != 16 || desc.y != 16 || desc.z != 0 || desc.w != 0 ||
            desc.f != UPTKChannelFormatKindSigned) {
            printf("CUDA error: explicit channel desc mismatch\n");
            return 1;
        }
    }

    // Scenario 5: Create channel descriptor for half (16-bit float)
    {
        //struct UPTKChannelFormatDesc desc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
        struct UPTKChannelFormatDesc desc = UPTKCreateChannelDescHalf();
        if (desc.x != 16 || desc.f != UPTKChannelFormatKindFloat) {
            printf("CUDA error: half channel desc mismatch\n");
            return 1;
        }
    }

    printf("test_cudaCreateChannelDesc PASS\n");
    return 0;
}

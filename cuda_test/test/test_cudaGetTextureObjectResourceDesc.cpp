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

    // Scenario 1: Create texture object and get its resource descriptor
    float *d_data = NULL;
    CHECK_CUDA(UPTKMalloc((void **)&d_data, 64 * 64 * sizeof(float)));

    UPTKResourceDesc resDesc = {};
    resDesc.resType = UPTKResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = d_data;
    resDesc.res.pitch2D.desc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    resDesc.res.pitch2D.width = 64;
    resDesc.res.pitch2D.height = 64;
    resDesc.res.pitch2D.pitchInBytes = 64 * sizeof(float);

    UPTKTextureDesc texDesc = {};
    texDesc.addressMode[0] = UPTKAddressModeClamp;
    texDesc.addressMode[1] = UPTKAddressModeClamp;
    texDesc.filterMode = UPTKFilterModePoint;
    texDesc.readMode = UPTKReadModeElementType;
    texDesc.normalizedCoords = 0;

    UPTKTextureObject_t texObj = 0;
    CHECK_CUDA(UPTKCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    UPTKResourceDesc retrievedDesc = {};
    CHECK_CUDA(UPTKGetTextureObjectResourceDesc(&retrievedDesc, texObj));
    if (retrievedDesc.resType != UPTKResourceTypePitch2D) {
        printf("CUDA error: resource type mismatch\n");
        CHECK_CUDA(UPTKDestroyTextureObject(texObj));
        CHECK_CUDA(UPTKFree(d_data));
        return 1;
    }

    // Scenario 2: Create 1D linear texture and get resource descriptor
    float *d_data2 = NULL;
    CHECK_CUDA(UPTKMalloc((void **)&d_data2, 256 * sizeof(float)));

    UPTKResourceDesc resDesc2 = {};
    resDesc2.resType = UPTKResourceTypeLinear;
    resDesc2.res.linear.devPtr = d_data2;
    resDesc2.res.linear.desc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    resDesc2.res.linear.sizeInBytes = 256 * sizeof(float);

    UPTKTextureDesc texDesc2 = {};
    texDesc2.addressMode[0] = UPTKAddressModeClamp;
    texDesc2.filterMode = UPTKFilterModePoint;
    texDesc2.readMode = UPTKReadModeElementType;

    UPTKTextureObject_t texObj2 = 0;
    CHECK_CUDA(UPTKCreateTextureObject(&texObj2, &resDesc2, &texDesc2, NULL));

    UPTKResourceDesc retrievedDesc2 = {};
    CHECK_CUDA(UPTKGetTextureObjectResourceDesc(&retrievedDesc2, texObj2));
    if (retrievedDesc2.resType != UPTKResourceTypeLinear) {
        printf("CUDA error: resource type mismatch for linear texture\n");
        CHECK_CUDA(UPTKDestroyTextureObject(texObj));
        CHECK_CUDA(UPTKDestroyTextureObject(texObj2));
        CHECK_CUDA(UPTKFree(d_data));
        CHECK_CUDA(UPTKFree(d_data2));
        return 1;
    }

    CHECK_CUDA(UPTKDestroyTextureObject(texObj));
    CHECK_CUDA(UPTKDestroyTextureObject(texObj2));
    CHECK_CUDA(UPTKFree(d_data));
    CHECK_CUDA(UPTKFree(d_data2));

    printf("test_cudaGetTextureObjectResourceDesc PASS\n");
    return 0;
}

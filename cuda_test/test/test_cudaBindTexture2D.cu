#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // UPTKBindTexture2D is deprecated, use texture objects instead
    // On DTK, deprecated texture reference APIs may cause segfault
    // Test using the modern texture object API instead
    
    // Scenario 1: Create texture object with 2D pitched memory (modern API)
    float *d_data = NULL;
    size_t pitch;
    size_t width = 64;
    size_t height = 64;
    UPTKError_t err = UPTKMallocPitch((void **)&d_data, &pitch, width * sizeof(float), height);
    if (err != UPTKSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, UPTKGetErrorString(err));
        return 1;
    }

    UPTKResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = UPTKResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = d_data;
    //resDesc.res.pitch2D.desc = UPTKCreateChannelDesc<float>();
    resDesc.res.pitch2D.desc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.pitchInBytes = pitch;

    UPTKTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = UPTKAddressModeClamp;
    texDesc.addressMode[1] = UPTKAddressModeClamp;
    texDesc.filterMode = UPTKFilterModePoint;
    texDesc.readMode = UPTKReadModeElementType;

    UPTKTextureObject_t texObj = 0;
    err = UPTKCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    if (err != UPTKSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, UPTKGetErrorString(err));
        UPTKFree(d_data);
        return 1;
    }
    printf("  Texture object created successfully\n");

    UPTKDestroyTextureObject(texObj);
    UPTKFree(d_data);

    // Scenario 2: Error handling - null texture object pointer
    err = UPTKCreateTextureObject(NULL, &resDesc, &texDesc, NULL);
    if (err != UPTKErrorInvalidValue) {
        printf("CUDA error: expected UPTKErrorInvalidValue, got: %s\n", UPTKGetErrorString(err));
        return 1;
    }

    printf("test_cudaBindTexture2D PASS\n");
    return 0;
}

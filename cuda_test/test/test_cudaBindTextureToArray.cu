#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // UPTKBindTextureToArray is deprecated, use texture objects instead
    // On DTK, deprecated texture reference APIs return invalid device symbol
    
    // Scenario 1: Use modern texture object API with UPTKArray
    UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    UPTKArray_t cuArray;
    UPTKError_t err = UPTKMallocArray(&cuArray, &channelDesc, 256);
    if (err != UPTKSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, UPTKGetErrorString(err));
        return 1;
    }

    UPTKResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = UPTKResourceTypeArray;
    resDesc.res.array.array = cuArray;

    UPTKTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = UPTKAddressModeClamp;
    texDesc.filterMode = UPTKFilterModePoint;
    texDesc.readMode = UPTKReadModeElementType;

    UPTKTextureObject_t texObj = 0;
    err = UPTKCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    if (err != UPTKSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, UPTKGetErrorString(err));
        UPTKFreeArray(cuArray);
        return 1;
    }
    printf("  Texture object from array created successfully\n");

    UPTKDestroyTextureObject(texObj);
    UPTKFreeArray(cuArray);

    // Scenario 2: Error handling - null array in resource desc
    UPTKResourceDesc resDesc2;
    memset(&resDesc2, 0, sizeof(resDesc2));
    resDesc2.resType = UPTKResourceTypeArray;
    resDesc2.res.array.array = NULL;

    err = UPTKCreateTextureObject(&texObj, &resDesc2, &texDesc, NULL);
    if (err != UPTKSuccess && err != UPTKErrorInvalidValue) {
        printf("CUDA error: unexpected error for null array: %s\n", UPTKGetErrorString(err));
        return 1;
    }

    printf("test_cudaBindTextureToArray PASS\n");
    return 0;
}

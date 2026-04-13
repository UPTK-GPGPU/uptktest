#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
#include <string.h>

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Create and destroy texture object
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
    texDesc.normalizedCoords = 0;

    UPTKTextureObject_t texObj = 0;
    err = UPTKCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    if (err != UPTKSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, UPTKGetErrorString(err));
        UPTKFreeArray(cuArray);
        return 1;
    }

    // Destroy the texture object
    err = UPTKDestroyTextureObject(texObj);
    if (err != UPTKSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, UPTKGetErrorString(err));
        UPTKFreeArray(cuArray);
        return 1;
    }
    printf("  Texture object destroyed successfully\n");

    UPTKFreeArray(cuArray);

    // Scenario 2: Create and destroy multiple texture objects
    UPTKArray_t cuArray1, cuArray2;
    err = UPTKMallocArray(&cuArray1, &channelDesc, 128);
    if (err != UPTKSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, UPTKGetErrorString(err));
        return 1;
    }
    err = UPTKMallocArray(&cuArray2, &channelDesc, 128);
    if (err != UPTKSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, UPTKGetErrorString(err));
        UPTKFreeArray(cuArray1);
        return 1;
    }

    UPTKResourceDesc resDesc1, resDesc2;
    memset(&resDesc1, 0, sizeof(resDesc1));
    resDesc1.resType = UPTKResourceTypeArray;
    resDesc1.res.array.array = cuArray1;
    memset(&resDesc2, 0, sizeof(resDesc2));
    resDesc2.resType = UPTKResourceTypeArray;
    resDesc2.res.array.array = cuArray2;

    UPTKTextureObject_t texObj1 = 0, texObj2 = 0;
    err = UPTKCreateTextureObject(&texObj1, &resDesc1, &texDesc, NULL);
    if (err != UPTKSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, UPTKGetErrorString(err));
        UPTKFreeArray(cuArray1);
        UPTKFreeArray(cuArray2);
        return 1;
    }
    err = UPTKCreateTextureObject(&texObj2, &resDesc2, &texDesc, NULL);
    if (err != UPTKSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, UPTKGetErrorString(err));
        UPTKDestroyTextureObject(texObj1);
        UPTKFreeArray(cuArray1);
        UPTKFreeArray(cuArray2);
        return 1;
    }

    UPTKDestroyTextureObject(texObj1);
    UPTKDestroyTextureObject(texObj2);
    UPTKFreeArray(cuArray1);
    UPTKFreeArray(cuArray2);

    printf("test_cudaDestroyTextureObject PASS\n");
    return 0;
}

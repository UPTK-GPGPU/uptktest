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

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Basic functionality - create texture object from CUDA array
    {
        UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
        UPTKArray_t cuArray;
        CHECK_CUDA(UPTKMallocArray(&cuArray, &channelDesc, 256));

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
        CHECK_CUDA(UPTKCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

        if (texObj == 0) {
            printf("CUDA error: texture object should be non-zero\n");
            UPTKFreeArray(cuArray);
            return 1;
        }

        CHECK_CUDA(UPTKDestroyTextureObject(texObj));
        CHECK_CUDA(UPTKFreeArray(cuArray));
    }

    // Scenario 2: Create texture object from linear memory
    {
        float *d_data = NULL;
        size_t size = 256 * sizeof(float);
        CHECK_CUDA(UPTKMalloc((void **)&d_data, size));

        UPTKResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = UPTKResourceTypeLinear;
        resDesc.res.linear.devPtr = d_data;
        resDesc.res.linear.desc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
        resDesc.res.linear.sizeInBytes = size;

        UPTKTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = UPTKReadModeElementType;

        UPTKTextureObject_t texObj = 0;
        CHECK_CUDA(UPTKCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

        CHECK_CUDA(UPTKDestroyTextureObject(texObj));
        CHECK_CUDA(UPTKFree(d_data));
    }

    // Scenario 3: Create texture object with linear filtering
    {
        UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
        UPTKArray_t cuArray;
        CHECK_CUDA(UPTKMallocArray(&cuArray, &channelDesc, 64, 64));

        UPTKResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = UPTKResourceTypeArray;
        resDesc.res.array.array = cuArray;

        UPTKTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = UPTKAddressModeClamp;
        texDesc.addressMode[1] = UPTKAddressModeClamp;
        texDesc.filterMode = UPTKFilterModeLinear;
        texDesc.readMode = UPTKReadModeElementType;
        texDesc.normalizedCoords = 0;

        UPTKTextureObject_t texObj = 0;
        CHECK_CUDA(UPTKCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

        CHECK_CUDA(UPTKDestroyTextureObject(texObj));
        CHECK_CUDA(UPTKFreeArray(cuArray));
    }

    // Scenario 4: Error handling - null texture object pointer
    {
        UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
        UPTKArray_t cuArray;
        CHECK_CUDA(UPTKMallocArray(&cuArray, &channelDesc, 256));

        UPTKResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = UPTKResourceTypeArray;
        resDesc.res.array.array = cuArray;

        UPTKTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = UPTKAddressModeClamp;
        texDesc.filterMode = UPTKFilterModePoint;
        texDesc.readMode = UPTKReadModeElementType;

        UPTKError_t err = UPTKCreateTextureObject(NULL, &resDesc, &texDesc, NULL);
        if (err != UPTKErrorInvalidValue) {
            printf("CUDA error: expected UPTKErrorInvalidValue for null texObj, got: %s\n",
                   UPTKGetErrorString(err));
            UPTKFreeArray(cuArray);
            return 1;
        }

        CHECK_CUDA(UPTKFreeArray(cuArray));
    }

    // Scenario 5: Error handling - null resource descriptor
    {
        UPTKTextureObject_t texObj = 0;
        UPTKTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        UPTKError_t err = UPTKCreateTextureObject(&texObj, NULL, &texDesc, NULL);
        if (err != UPTKErrorInvalidValue) {
            printf("CUDA error: expected UPTKErrorInvalidValue for null resDesc, got: %s\n",
                   UPTKGetErrorString(err));
            return 1;
        }
    }

    printf("test_cudaCreateTextureObject PASS\n");
    return 0;
}

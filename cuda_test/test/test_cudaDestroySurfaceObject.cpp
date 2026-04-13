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

    // Scenario 1: Basic functionality - create and destroy surface object
    {
        UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
        UPTKArray_t cuArray;
        CHECK_CUDA(UPTKMallocArray(&cuArray, &channelDesc, 64, 64));

        UPTKResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = UPTKResourceTypeArray;
        resDesc.res.array.array = cuArray;

        UPTKSurfaceObject_t surfObj = 0;
        CHECK_CUDA(UPTKCreateSurfaceObject(&surfObj, &resDesc));

        // Destroy the surface object
        CHECK_CUDA(UPTKDestroySurfaceObject(surfObj));

        CHECK_CUDA(UPTKFreeArray(cuArray));
    }

    // Scenario 2: Create and destroy multiple surface objects
    {
        UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
        UPTKArray_t cuArray1, cuArray2;
        CHECK_CUDA(UPTKMallocArray(&cuArray1, &channelDesc, 32, 32));
        CHECK_CUDA(UPTKMallocArray(&cuArray2, &channelDesc, 32, 32));

        UPTKResourceDesc resDesc1, resDesc2;
        memset(&resDesc1, 0, sizeof(resDesc1));
        resDesc1.resType = UPTKResourceTypeArray;
        resDesc1.res.array.array = cuArray1;
        memset(&resDesc2, 0, sizeof(resDesc2));
        resDesc2.resType = UPTKResourceTypeArray;
        resDesc2.res.array.array = cuArray2;

        UPTKSurfaceObject_t surfObj1 = 0, surfObj2 = 0;
        CHECK_CUDA(UPTKCreateSurfaceObject(&surfObj1, &resDesc1));
        CHECK_CUDA(UPTKCreateSurfaceObject(&surfObj2, &resDesc2));

        CHECK_CUDA(UPTKDestroySurfaceObject(surfObj1));
        CHECK_CUDA(UPTKDestroySurfaceObject(surfObj2));

        CHECK_CUDA(UPTKFreeArray(cuArray1));
        CHECK_CUDA(UPTKFreeArray(cuArray2));
    }

    // Scenario 3: Error handling - invalid surface object
    {
        UPTKError_t err = UPTKDestroySurfaceObject((UPTKSurfaceObject_t)0xDEADBEEF);
        if (err != UPTKErrorInvalidValue && err != UPTKErrorInvalidResourceHandle) {
            printf("CUDA error: expected error for invalid surface object, got: %s\n",
                   UPTKGetErrorString(err));
            return 1;
        }
    }

    printf("test_cudaDestroySurfaceObject PASS\n");
    return 0;
}

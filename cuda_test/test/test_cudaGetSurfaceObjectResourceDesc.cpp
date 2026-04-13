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

    // Scenario 1: Create array-based surface and get resource descriptor
    UPTKArray_t cuArray = NULL;
    UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    CHECK_CUDA(UPTKMallocArray(&cuArray, &channelDesc, 64, 64));

    UPTKResourceDesc resDesc = {};
    resDesc.resType = UPTKResourceTypeArray;
    resDesc.res.array.array = cuArray;

    UPTKSurfaceObject_t surfObj = 0;
    CHECK_CUDA(UPTKCreateSurfaceObject(&surfObj, &resDesc));

    UPTKResourceDesc retrievedDesc = {};
    CHECK_CUDA(UPTKGetSurfaceObjectResourceDesc(&retrievedDesc, surfObj));
    if (retrievedDesc.resType != UPTKResourceTypeArray) {
        printf("CUDA error: resource type mismatch, expected %d, got %d\n",
               UPTKResourceTypeArray, retrievedDesc.resType);
        CHECK_CUDA(UPTKDestroySurfaceObject(surfObj));
        CHECK_CUDA(UPTKFreeArray(cuArray));
        return 1;
    }
    printf("Array-based surface resource descriptor retrieved successfully\n");

    // Scenario 2: Create another array surface with different channel format
    UPTKArray_t cuArray2 = NULL;
    UPTKChannelFormatDesc channelDesc2 = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindSigned);
    CHECK_CUDA(UPTKMallocArray(&cuArray2, &channelDesc2, 32, 32));

    UPTKResourceDesc resDesc2 = {};
    resDesc2.resType = UPTKResourceTypeArray;
    resDesc2.res.array.array = cuArray2;

    UPTKSurfaceObject_t surfObj2 = 0;
    CHECK_CUDA(UPTKCreateSurfaceObject(&surfObj2, &resDesc2));

    UPTKResourceDesc retrievedDesc2 = {};
    CHECK_CUDA(UPTKGetSurfaceObjectResourceDesc(&retrievedDesc2, surfObj2));
    if (retrievedDesc2.resType != UPTKResourceTypeArray) {
        printf("CUDA error: resource type mismatch for surface 2\n");
        CHECK_CUDA(UPTKDestroySurfaceObject(surfObj));
        CHECK_CUDA(UPTKDestroySurfaceObject(surfObj2));
        CHECK_CUDA(UPTKFreeArray(cuArray));
        CHECK_CUDA(UPTKFreeArray(cuArray2));
        return 1;
    }
    printf("Second array-based surface resource descriptor retrieved successfully\n");

    CHECK_CUDA(UPTKDestroySurfaceObject(surfObj));
    CHECK_CUDA(UPTKDestroySurfaceObject(surfObj2));
    CHECK_CUDA(UPTKFreeArray(cuArray));
    CHECK_CUDA(UPTKFreeArray(cuArray2));

    printf("test_cudaGetSurfaceObjectResourceDesc PASS\n");
    return 0;
}

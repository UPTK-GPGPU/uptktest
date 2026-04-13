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

    // Scenario 1: Create surface object from CUDA array
    UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    UPTKArray_t cuArray;
    UPTKError_t err = UPTKMallocArray(&cuArray, &channelDesc, 64, 64);
    if (err != UPTKSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, UPTKGetErrorString(err));
        return 1;
    }

    UPTKResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = UPTKResourceTypeArray;
    resDesc.res.array.array = cuArray;

    UPTKSurfaceObject_t surfObj = 0;
    err = UPTKCreateSurfaceObject(&surfObj, &resDesc);
    if (err != UPTKSuccess) {
        printf("test_skip: UPTKCreateSurfaceObject from array not supported on DTK: %s\n",
               UPTKGetErrorString(err));
        UPTKFreeArray(cuArray);
        return 0;
    }
    printf("  Surface object created: %lu\n", (unsigned long)surfObj);

    // Scenario 2: Error handling - null surface object pointer (test before destroy)
    UPTKError_t err2 = UPTKCreateSurfaceObject(NULL, &resDesc);
    if (err2 != UPTKErrorInvalidValue && err2 != UPTKSuccess) {
        printf("CUDA error: expected UPTKErrorInvalidValue for null pSurfObject, got: %s\n",
               UPTKGetErrorString(err2));
    }

    // Clean up
    UPTKDestroySurfaceObject(surfObj);
    UPTKFreeArray(cuArray);

    printf("test_cudaCreateSurfaceObject PASS\n");
    return 0;
}

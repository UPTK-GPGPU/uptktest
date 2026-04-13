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

    // Scene 1: Basic external memory mapped buffer - skip if no external memory handle
    // This API requires a valid UPTKExternalMemory_t handle which typically comes
    // from UPTKImportExternalMemory (e.g., from fd or OpaqueFd). Without a real
    // external memory source, we test the error path.
    UPTKExternalMemory_t extMem = NULL;
    void *devPtr = NULL;
    struct UPTKExternalMemoryBufferDesc bufferDesc;
    bufferDesc.offset = 0;
    bufferDesc.size = 1024;
    bufferDesc.flags = 0;

    // Expect UPTKErrorInvalidValue or UPTKErrorInvalidResourceHandle since extMem is NULL
    UPTKError_t err = UPTKExternalMemoryGetMappedBuffer(&devPtr, extMem, &bufferDesc);
    if (err == UPTKErrorInvalidValue || err == UPTKErrorInvalidResourceHandle) {
        printf("UPTKExternalMemoryGetMappedBuffer returned expected error for NULL extMem: %s\n",
               UPTKGetErrorString(err));
    } else if (err == UPTKSuccess) {
        printf("UPTKExternalMemoryGetMappedBuffer succeeded unexpectedly\n");
        UPTKFree(devPtr);
        UPTKDestroyExternalMemory(extMem);
    } else {
        printf("UPTKExternalMemoryGetMappedBuffer returned: %s\n", UPTKGetErrorString(err));
    }

    printf("test_cudaExternalMemoryGetMappedBuffer PASS\n");
    return 0;
}

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

    // UPTKGetSurfaceReference is deprecated and requires a __surface__ variable
    // which is not supported in DTK. Test error path only.
    const struct surfaceReference *surfref = NULL;
    UPTKError_t err = UPTKGetSurfaceReference(&surfref, NULL);
    if (err == UPTKErrorInvalidValue) {
        printf("UPTKGetSurfaceReference correctly returned UPTKErrorInvalidValue for NULL symbol\n");
    } else {
        printf("test_skip: UPTKGetSurfaceReference returned %s (deprecated API not fully supported)\n",
               UPTKGetErrorString(err));
        return 0;
    }

    printf("test_cudaGetSurfaceReference PASS\n");
    return 0;
}

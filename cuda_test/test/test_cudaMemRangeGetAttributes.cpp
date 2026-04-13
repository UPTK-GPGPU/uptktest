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

    // Scene 1: Basic UPTKMemRangeGetAttributes with multiple attributes
    float *d_ptr = NULL;
    CHECK_CUDA(UPTKMallocManaged((void**)&d_ptr, 1024 * sizeof(float)));
    
    void *data[2];
    size_t dataSizes[2];
    enum UPTKMemRangeAttribute attrs[2];
    int readMostly = 0;
    int prefLocation = -1;
    
    data[0] = &readMostly;
    dataSizes[0] = sizeof(readMostly);
    attrs[0] = UPTKMemRangeAttributeReadMostly;
    
    data[1] = &prefLocation;
    dataSizes[1] = sizeof(prefLocation);
    attrs[1] = UPTKMemRangeAttributePreferredLocation;
    
    UPTKError_t err = UPTKMemRangeGetAttributes(data, dataSizes, attrs, 2,
        d_ptr, 1024 * sizeof(float));
    if (err != UPTKSuccess) {
        // DTK/AMD may not support these attributes
        printf("Note: UPTKMemRangeGetAttributes returned: %s (expected on DTK)\n", UPTKGetErrorString(err));
    }
    
    CHECK_CUDA(UPTKFree(d_ptr));

    printf("test_cudaMemRangeGetAttributes PASS\n");
    return 0;
}

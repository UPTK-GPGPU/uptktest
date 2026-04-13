#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdlib.h>
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

static void destroy_callback(void *ptr) {
    free(ptr);
}

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scene 1: Retain with default count (1)
    void *ptr = malloc(64);
    UPTKUserObject_t obj;
    CHECK_CUDA(UPTKUserObjectCreate(&obj, ptr, destroy_callback, 1, UPTKUserObjectNoDestructorSync));
    CHECK_CUDA(UPTKUserObjectRetain(obj, 1));
    CHECK_CUDA(UPTKUserObjectRelease(obj, 1));
    CHECK_CUDA(UPTKUserObjectRelease(obj, 1));

    // Scene 2: Retain with count > 1
    void *ptr2 = malloc(64);
    UPTKUserObject_t obj2;
    CHECK_CUDA(UPTKUserObjectCreate(&obj2, ptr2, destroy_callback, 1, UPTKUserObjectNoDestructorSync));
    CHECK_CUDA(UPTKUserObjectRetain(obj2, 3));
    CHECK_CUDA(UPTKUserObjectRelease(obj2, 4));

    printf("test_cudaUserObjectRetain PASS\n");
    return 0;
}

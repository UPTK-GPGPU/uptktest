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

    // Scene 1: Get stack size limit
    size_t stack_size;
    CHECK_CUDA(UPTKThreadGetLimit(&stack_size, UPTKLimitStackSize));
    printf("Stack size limit: %zu\n", stack_size);

    // Scene 2: Get malloc heap size limit
    size_t heap_size;
    CHECK_CUDA(UPTKThreadGetLimit(&heap_size, UPTKLimitMallocHeapSize));
    printf("Malloc heap size limit: %zu\n", heap_size);

    printf("test_cudaThreadGetLimit PASS\n");
    return 0;
}

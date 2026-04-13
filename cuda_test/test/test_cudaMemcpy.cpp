#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
#include <stdlib.h>

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

    // Scene 1: Basic UPTKMemcpy H2D and D2H
    size_t size = 1024 * sizeof(float);
    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);
    float *d_ptr = NULL;
    CHECK_CUDA(UPTKMalloc((void**)&d_ptr, size));
    
    for (int i = 0; i < 1024; i++) h_in[i] = (float)i;
    
    CHECK_CUDA(UPTKMemcpy(d_ptr, h_in, size, UPTKMemcpyHostToDevice));
    CHECK_CUDA(UPTKMemcpy(h_out, d_ptr, size, UPTKMemcpyDeviceToHost));
    
    int match = 1;
    for (int i = 0; i < 1024; i++) {
        if (h_in[i] != h_out[i]) { match = 0; break; }
    }
    if (!match) { printf("Data mismatch\n"); return 1; }
    
    CHECK_CUDA(UPTKFree(d_ptr));

    // Scene 2: UPTKMemcpy D2D
    float *d_src = NULL, *d_dst = NULL;
    CHECK_CUDA(UPTKMalloc((void**)&d_src, size));
    CHECK_CUDA(UPTKMalloc((void**)&d_dst, size));
    CHECK_CUDA(UPTKMemcpy(d_src, h_in, size, UPTKMemcpyHostToDevice));
    CHECK_CUDA(UPTKMemcpy(d_dst, d_src, size, UPTKMemcpyDeviceToDevice));
    CHECK_CUDA(UPTKMemcpy(h_out, d_dst, size, UPTKMemcpyDeviceToHost));
    
    match = 1;
    for (int i = 0; i < 1024; i++) {
        if (h_in[i] != h_out[i]) { match = 0; break; }
    }
    if (!match) { printf("D2D mismatch\n"); return 1; }
    
    CHECK_CUDA(UPTKFree(d_src));
    CHECK_CUDA(UPTKFree(d_dst));

    // Scene 3: UPTKMemcpy with size 0 (boundary)
    float *d_ptr2 = NULL;
    CHECK_CUDA(UPTKMalloc((void**)&d_ptr2, size));
    CHECK_CUDA(UPTKMemcpy(d_ptr2, h_in, 0, UPTKMemcpyHostToDevice));
    CHECK_CUDA(UPTKFree(d_ptr2));

    free(h_in);
    free(h_out);

    printf("test_cudaMemcpy PASS\n");
    return 0;
}

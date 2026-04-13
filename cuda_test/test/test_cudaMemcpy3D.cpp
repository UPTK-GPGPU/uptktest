#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
#include <stdlib.h>
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

    // Scene 1: Basic UPTKMemcpy3D H2D and D2H
    int width = 16, height = 16, depth = 4;
    size_t volSize = width * height * depth * sizeof(float);
    
    float *h_src = (float*)malloc(volSize);
    float *h_dst = (float*)malloc(volSize);
    for (int i = 0; i < width * height * depth; i++) h_src[i] = (float)i;
    
    UPTKPitchedPtr d_pitchedPtr;
    UPTKExtent extent = make_UPTKExtent(width * sizeof(float), height, depth);
    CHECK_CUDA(UPTKMalloc3D(&d_pitchedPtr, extent));
    
    UPTKMemcpy3DParms params;
    memset(&params, 0, sizeof(params));
    params.srcPtr = make_UPTKPitchedPtr(h_src, width * sizeof(float), width, height);
    params.dstPtr = d_pitchedPtr;
    params.extent = extent;
    params.kind = UPTKMemcpyHostToDevice;
    CHECK_CUDA(UPTKMemcpy3D(&params));
    
    memset(&params, 0, sizeof(params));
    params.srcPtr = d_pitchedPtr;
    params.dstPtr = make_UPTKPitchedPtr(h_dst, width * sizeof(float), width, height);
    params.extent = extent;
    params.kind = UPTKMemcpyDeviceToHost;
    CHECK_CUDA(UPTKMemcpy3D(&params));
    
    int match = 1;
    for (int i = 0; i < width * height * depth; i++) {
        if (h_src[i] != h_dst[i]) { match = 0; break; }
    }
    if (!match) { printf("Data mismatch\n"); return 1; }
    
    CHECK_CUDA(UPTKFree(d_pitchedPtr.ptr));
    free(h_src);
    free(h_dst);

    // Scene 2: UPTKMemcpy3D D2D
    float *h_src2 = (float*)malloc(volSize);
    float *h_dst2 = (float*)malloc(volSize);
    for (int i = 0; i < width * height * depth; i++) h_src2[i] = (float)i;
    
    UPTKPitchedPtr d_src, d_dst;
    CHECK_CUDA(UPTKMalloc3D(&d_src, extent));
    CHECK_CUDA(UPTKMalloc3D(&d_dst, extent));
    
    memset(&params, 0, sizeof(params));
    params.srcPtr = make_UPTKPitchedPtr(h_src2, width * sizeof(float), width, height);
    params.dstPtr = d_src;
    params.extent = extent;
    params.kind = UPTKMemcpyHostToDevice;
    CHECK_CUDA(UPTKMemcpy3D(&params));
    
    memset(&params, 0, sizeof(params));
    params.srcPtr = d_src;
    params.dstPtr = d_dst;
    params.extent = extent;
    params.kind = UPTKMemcpyDeviceToDevice;
    CHECK_CUDA(UPTKMemcpy3D(&params));
    
    memset(&params, 0, sizeof(params));
    params.srcPtr = d_dst;
    params.dstPtr = make_UPTKPitchedPtr(h_dst2, width * sizeof(float), width, height);
    params.extent = extent;
    params.kind = UPTKMemcpyDeviceToHost;
    CHECK_CUDA(UPTKMemcpy3D(&params));
    
    CHECK_CUDA(UPTKFree(d_src.ptr));
    CHECK_CUDA(UPTKFree(d_dst.ptr));
    free(h_src2);
    free(h_dst2);

    printf("test_cudaMemcpy3D PASS\n");
    return 0;
}

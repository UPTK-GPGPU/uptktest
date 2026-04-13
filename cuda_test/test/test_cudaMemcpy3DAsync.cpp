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

    // Scene 1: Basic UPTKMemcpy3DAsync H2D and D2H
    int width = 16, height = 16, depth = 4;
    size_t volSize = width * height * depth * sizeof(float);
    
    float *h_src = (float*)malloc(volSize);
    float *h_dst = (float*)malloc(volSize);
    for (int i = 0; i < width * height * depth; i++) h_src[i] = (float)i;
    
    UPTKPitchedPtr d_pitchedPtr;
    UPTKExtent extent = make_UPTKExtent(width * sizeof(float), height, depth);
    CHECK_CUDA(UPTKMalloc3D(&d_pitchedPtr, extent));
    
    UPTKStream_t stream;
    CHECK_CUDA(UPTKStreamCreate(&stream));
    
    UPTKMemcpy3DParms params;
    memset(&params, 0, sizeof(params));
    params.srcPtr = make_UPTKPitchedPtr(h_src, width * sizeof(float), width, height);
    params.dstPtr = d_pitchedPtr;
    params.extent = extent;
    params.kind = UPTKMemcpyHostToDevice;
    CHECK_CUDA(UPTKMemcpy3DAsync(&params, stream));
    
    memset(&params, 0, sizeof(params));
    params.srcPtr = d_pitchedPtr;
    params.dstPtr = make_UPTKPitchedPtr(h_dst, width * sizeof(float), width, height);
    params.extent = extent;
    params.kind = UPTKMemcpyDeviceToHost;
    CHECK_CUDA(UPTKMemcpy3DAsync(&params, stream));
    CHECK_CUDA(UPTKStreamSynchronize(stream));
    
    CHECK_CUDA(UPTKStreamDestroy(stream));
    CHECK_CUDA(UPTKFree(d_pitchedPtr.ptr));
    free(h_src);
    free(h_dst);

    // Scene 2: UPTKMemcpy3DAsync with default stream
    float *h_src2 = (float*)malloc(volSize);
    float *h_dst2 = (float*)malloc(volSize);
    for (int i = 0; i < width * height * depth; i++) h_src2[i] = (float)i;
    
    UPTKPitchedPtr d_ptr2;
    CHECK_CUDA(UPTKMalloc3D(&d_ptr2, extent));
    
    memset(&params, 0, sizeof(params));
    params.srcPtr = make_UPTKPitchedPtr(h_src2, width * sizeof(float), width, height);
    params.dstPtr = d_ptr2;
    params.extent = extent;
    params.kind = UPTKMemcpyHostToDevice;
    CHECK_CUDA(UPTKMemcpy3DAsync(&params, 0));
    
    memset(&params, 0, sizeof(params));
    params.srcPtr = d_ptr2;
    params.dstPtr = make_UPTKPitchedPtr(h_dst2, width * sizeof(float), width, height);
    params.extent = extent;
    params.kind = UPTKMemcpyDeviceToHost;
    CHECK_CUDA(UPTKMemcpy3DAsync(&params, 0));
    CHECK_CUDA(UPTKDeviceSynchronize());
    
    CHECK_CUDA(UPTKFree(d_ptr2.ptr));
    free(h_src2);
    free(h_dst2);

    printf("test_cudaMemcpy3DAsync PASS\n");
    return 0;
}

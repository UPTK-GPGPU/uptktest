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

    // Scene 1: cudaMemcpy3DPeerAsync_ptsz - skip if only 1 device
    if (deviceCount < 2) {
        printf("test_skip: cudaMemcpy3DPeerAsync_ptsz requires at least 2 devices, found %d\n", deviceCount);
        return 0;
    }

    // Scene 2: Basic UPTKMemcpy3DPeerAsync with per-thread default stream
    int width = 8, height = 8, depth = 2;
    size_t volSize = width * height * depth * sizeof(float);
    
    float *h_src = (float*)malloc(volSize);
    float *h_dst = (float*)malloc(volSize);
    for (int i = 0; i < width * height * depth; i++) h_src[i] = (float)i;
    
    UPTKSetDevice(0);
    UPTKPitchedPtr d_src;
    UPTKExtent extent = make_UPTKExtent(width * sizeof(float), height, depth);
    CHECK_CUDA(UPTKMalloc3D(&d_src, extent));
    
    UPTKMemcpy3DParms params;
    memset(&params, 0, sizeof(params));
    params.srcPtr = make_UPTKPitchedPtr(h_src, width * sizeof(float), width, height);
    params.dstPtr = d_src;
    params.extent = extent;
    params.kind = UPTKMemcpyHostToDevice;
    CHECK_CUDA(UPTKMemcpy3D(&params));
    
    UPTKSetDevice(1);
    UPTKPitchedPtr d_dst;
    CHECK_CUDA(UPTKMalloc3D(&d_dst, extent));
    
    UPTKMemcpy3DPeerParms peerParams;
    memset(&peerParams, 0, sizeof(peerParams));
    peerParams.srcPtr = d_src;
    peerParams.dstPtr = d_dst;
    peerParams.extent = extent;
    peerParams.srcDevice = 0;
    peerParams.dstDevice = 1;
    CHECK_CUDA(UPTKMemcpy3DPeerAsync(&peerParams, UPTKStreamPerThread));
    CHECK_CUDA(UPTKDeviceSynchronize());
    
    memset(&params, 0, sizeof(params));
    params.srcPtr = d_dst;
    params.dstPtr = make_UPTKPitchedPtr(h_dst, width * sizeof(float), width, height);
    params.extent = extent;
    params.kind = UPTKMemcpyDeviceToHost;
    CHECK_CUDA(UPTKMemcpy3D(&params));
    
    UPTKSetDevice(0);
    CHECK_CUDA(UPTKFree(d_src.ptr));
    UPTKSetDevice(1);
    CHECK_CUDA(UPTKFree(d_dst.ptr));
    free(h_src);
    free(h_dst);

    printf("test_cudaMemcpy3DPeerAsync_ptsz PASS\n");
    return 0;
}

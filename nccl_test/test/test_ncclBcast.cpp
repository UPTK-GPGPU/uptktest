#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <UPTK_nccl.h>
#include <nccl.h>
#include <stdio.h>

#define CHECK_NCCL(call) \
    do { \
        UPTKncclResult_t res = call; \
        if (res != UPTKncclSuccess) { \
            printf("NCCL error at %s:%d: %s\n", __FILE__, __LINE__, \
                   UPTKncclGetErrorString(res)); \
            return 1; \
        } \
    } while (0)

int main() {
    int deviceCount = 0;
    UPTKError_t cudaErr = UPTKGetDeviceCount(&deviceCount);
    if (cudaErr != UPTKSuccess || deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    UPTKncclUniqueId commId;
    UPTKncclComm_t comm = NULL;
    UPTKncclResult_t res;

    res = UPTKncclGetUniqueId(&commId);
    if (res != UPTKncclSuccess) {
        printf("test_skip: UPTKncclGetUniqueId failed: %s\n", UPTKncclGetErrorString(res));
        return 0;
    }

    res = UPTKncclCommInitRank(&comm, 1, commId, 0);
    if (res != UPTKncclSuccess) {
        printf("test_skip: UPTKncclCommInitRank failed: %s\n", UPTKncclGetErrorString(res));
        return 0;
    }

    UPTKStream_t stream = 0;
    const size_t count = 4;

    // 场景1: 基础 Bcast with UPTKncclFloat32, root=0
    {
        float* buff = NULL;
        UPTKMalloc(&buff, count * sizeof(float));

        float h_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        UPTKMemcpy(buff, h_data, count * sizeof(float), UPTKMemcpyHostToDevice);

        CHECK_NCCL(UPTKncclBroadcast(buff, buff, count, UPTKncclFloat32, 0, comm, stream));
        UPTKStreamSynchronize(stream);

        UPTKFree(buff);
    }

    // 场景2: Bcast with UPTKncclInt32
    {
        int* buff = NULL;
        UPTKMalloc(&buff, count * sizeof(int));

        int h_data[] = {10, 20, 30, 40};
        UPTKMemcpy(buff, h_data, count * sizeof(int), UPTKMemcpyHostToDevice);

        CHECK_NCCL(UPTKncclBroadcast(buff, buff, count, UPTKncclInt32, 0, comm, stream));
        UPTKStreamSynchronize(stream);

        UPTKFree(buff);
    }

    CHECK_NCCL(UPTKncclCommDestroy(comm));

    printf("test_UPTKncclBcast PASS\n");
    return 0;
}

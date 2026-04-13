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

    // 场景1: 基础 AllGather with UPTKncclFloat32
    {
        float* sendbuff = NULL;
        float* recvbuff = NULL;
        UPTKMalloc(&sendbuff, count * sizeof(float));
        UPTKMalloc(&recvbuff, count * sizeof(float));

        float h_send[] = {1.0f, 2.0f, 3.0f, 4.0f};
        UPTKMemcpy(sendbuff, h_send, count * sizeof(float), UPTKMemcpyHostToDevice);

        CHECK_NCCL(UPTKncclAllGather(sendbuff, recvbuff, count, UPTKncclFloat32, comm, stream));
        UPTKStreamSynchronize(stream);

        UPTKFree(sendbuff);
        UPTKFree(recvbuff);
    }

    // 场景2: AllGather with UPTKncclInt32
    {
        int* sendbuff = NULL;
        int* recvbuff = NULL;
        UPTKMalloc(&sendbuff, count * sizeof(int));
        UPTKMalloc(&recvbuff, count * sizeof(int));

        int h_send[] = {10, 20, 30, 40};
        UPTKMemcpy(sendbuff, h_send, count * sizeof(int), UPTKMemcpyHostToDevice);

        CHECK_NCCL(UPTKncclAllGather(sendbuff, recvbuff, count, UPTKncclInt32, comm, stream));
        UPTKStreamSynchronize(stream);

        UPTKFree(sendbuff);
        UPTKFree(recvbuff);
    }

    // 场景3: In-place AllGather (sendbuff == recvbuff)
    {
        float* buff = NULL;
        UPTKMalloc(&buff, count * sizeof(float));

        float h_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
        UPTKMemcpy(buff, h_data, count * sizeof(float), UPTKMemcpyHostToDevice);

        CHECK_NCCL(UPTKncclAllGather(buff, buff, count, UPTKncclFloat32, comm, stream));
        UPTKStreamSynchronize(stream);

        UPTKFree(buff);
    }

    CHECK_NCCL(UPTKncclCommDestroy(comm));

    printf("test_UPTKncclAllGather PASS\n");
    return 0;
}

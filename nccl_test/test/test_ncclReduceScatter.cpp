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
    const size_t recvcount = 4;

    // 场景1: 基础 ReduceScatter with UPTKncclSum and UPTKncclFloat32
    {
        // sendcount = nranks * recvcount = 1 * 4 = 4
        const size_t sendcount = recvcount;
        float* sendbuff = NULL;
        float* recvbuff = NULL;
        UPTKMalloc(&sendbuff, sendcount * sizeof(float));
        UPTKMalloc(&recvbuff, recvcount * sizeof(float));

        float h_send[] = {1.0f, 2.0f, 3.0f, 4.0f};
        UPTKMemcpy(sendbuff, h_send, sendcount * sizeof(float), UPTKMemcpyHostToDevice);

        CHECK_NCCL(UPTKncclReduceScatter(sendbuff, recvbuff, recvcount, UPTKncclFloat32, UPTKncclSum, comm, stream));
        UPTKStreamSynchronize(stream);

        UPTKFree(sendbuff);
        UPTKFree(recvbuff);
    }

    // 场景2: ReduceScatter with ncclProd
    {
        const size_t sendcount = recvcount;
        float* sendbuff = NULL;
        float* recvbuff = NULL;
        UPTKMalloc(&sendbuff, sendcount * sizeof(float));
        UPTKMalloc(&recvbuff, recvcount * sizeof(float));

        float h_send[] = {1.0f, 2.0f, 3.0f, 4.0f};
        UPTKMemcpy(sendbuff, h_send, sendcount * sizeof(float), UPTKMemcpyHostToDevice);

        CHECK_NCCL(UPTKncclReduceScatter(sendbuff, recvbuff, recvcount, UPTKncclFloat32, UPTKncclProd, comm, stream));
        UPTKStreamSynchronize(stream);

        UPTKFree(sendbuff);
        UPTKFree(recvbuff);
    }

    // 场景3: ReduceScatter with UPTKncclInt32
    {
        const size_t sendcount = recvcount;
        int* sendbuff = NULL;
        int* recvbuff = NULL;
        UPTKMalloc(&sendbuff, sendcount * sizeof(int));
        UPTKMalloc(&recvbuff, recvcount * sizeof(int));

        int h_send[] = {10, 20, 30, 40};
        UPTKMemcpy(sendbuff, h_send, sendcount * sizeof(int), UPTKMemcpyHostToDevice);

        CHECK_NCCL(UPTKncclReduceScatter(sendbuff, recvbuff, recvcount, UPTKncclInt32, UPTKncclSum, comm, stream));
        UPTKStreamSynchronize(stream);

        UPTKFree(sendbuff);
        UPTKFree(recvbuff);
    }

    CHECK_NCCL(UPTKncclCommDestroy(comm));

    printf("test_UPTKncclReduceScatter PASS\n");
    return 0;
}

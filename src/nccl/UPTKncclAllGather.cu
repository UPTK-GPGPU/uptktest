#include "UPTK_nccl_utils.h"

TEST(uptk_nccl, UPTKncclAllGather)
{
    int ndev = 0;
    UPTK_CHECK(UPTKGetDeviceCount(&ndev));
    ASSERT_GE(ndev, 2);

    std::vector<int> devs(ndev);
    std::vector<UPTKncclComm_t> comms(ndev);

    for (int i = 0; i < ndev; i++)
        devs[i] = i;

    NCCL_CHECK(UPTKncclCommInitAll(comms.data(), ndev, devs.data()));

    int sendcount = 1;
    int recvcount = ndev;

    std::vector<float> h_send(ndev);
    std::vector<float*> d_send(ndev);
    std::vector<float*> d_recv(ndev);

    std::vector<std::vector<float>> h_recv(ndev, std::vector<float>(ndev));

    // 初始化 input
    for (int i = 0; i < ndev; i++)
        h_send[i] = static_cast<float>(i + 1); // 1,2,3,...

    // 分配 + copy
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));

        UPTK_CHECK(UPTKMalloc(&d_send[i], sendcount * sizeof(float)));
        UPTK_CHECK(UPTKMalloc(&d_recv[i], recvcount * sizeof(float)));

        UPTK_CHECK(UPTKMemcpy(
            d_send[i],
            &h_send[i],
            sizeof(float),
            UPTKMemcpyHostToDevice));
    }

    // AllGather（所有 rank 必须参与）
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));

        NCCL_CHECK(UPTKncclAllGather(
            d_send[i],
            d_recv[i],
            sendcount,
            UPTKncclFloat,
            comms[i],
            0));
    }

    // sync
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));
        UPTK_CHECK(UPTKDeviceSynchronize());
    }

    // 校验
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));

        UPTK_CHECK(UPTKMemcpy(
            h_recv[i].data(),
            d_recv[i],
            recvcount * sizeof(float),
            UPTKMemcpyDeviceToHost));

        for (int j = 0; j < ndev; j++)
        {
            ASSERT_FLOAT_EQ(h_recv[i][j], j + 1);
        }
    }

    // cleanup
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));
        UPTKFree(d_send[i]);
        UPTKFree(d_recv[i]);
        NCCL_CHECK(UPTKncclCommDestroy(comms[i]));
    }
}
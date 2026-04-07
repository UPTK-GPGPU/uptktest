#include "UPTK_nccl_utils.h"

TEST(uptk_nccl, UPTKncclRecv)
{
    int ndev = 0;
    UPTK_CHECK(UPTKGetDeviceCount(&ndev));

    // ❗ 单卡直接跳过（Send/Recv 不支持单卡）
    if (ndev < 2)
    {
        GTEST_SKIP() << "Skip NCCL Send/Recv on single GPU";
    }

    std::vector<int> devs(ndev);
    std::vector<UPTKncclComm_t> comms(ndev);

    for (int i = 0; i < ndev; i++)
        devs[i] = i;

    // 初始化 communicator
    NCCL_CHECK(UPTKncclCommInitAll(comms.data(), ndev, devs.data()));

    std::vector<float*> d_send(ndev);
    std::vector<float*> d_recv(ndev);
    std::vector<float> h_recv(ndev, 0.0f);

    // 初始化数据
    for (int i = 0; i < ndev; i++)
    {
        float h_val = static_cast<float>(i + 100);

        UPTK_CHECK(UPTKSetDevice(i));

        UPTK_CHECK(UPTKMalloc(&d_send[i], sizeof(float)));
        UPTK_CHECK(UPTKMalloc(&d_recv[i], sizeof(float)));

        UPTK_CHECK(UPTKMemcpy(
            d_send[i],
            &h_val,
            sizeof(float),
            UPTKMemcpyHostToDevice));

        UPTK_CHECK(UPTKMemset(d_recv[i], 0, sizeof(float)));
    }

    UPTKncclGroupStart();

    for (int i = 0; i < ndev; i++)
    {
        int dst = (i + 1) % ndev;
        int src = (i - 1 + ndev) % ndev;

        UPTK_CHECK(UPTKSetDevice(i));

        NCCL_CHECK(UPTKncclSend(
            d_send[i],
            1,
            UPTKncclFloat,
            dst,
            comms[i],
            0));

        NCCL_CHECK(UPTKncclRecv(
            d_recv[i],
            1,
            UPTKncclFloat,
            src,
            comms[i],
            0));
    }

    UPTKncclGroupEnd();

    // 同步
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(i));
        UPTK_CHECK(UPTKDeviceSynchronize());
    }

    // 校验
    for (int i = 0; i < ndev; i++)
    {
        int src = (i - 1 + ndev) % ndev;

        UPTK_CHECK(UPTKSetDevice(i));

        UPTK_CHECK(UPTKMemcpy(
            &h_recv[i],
            d_recv[i],
            sizeof(float),
            UPTKMemcpyDeviceToHost));

        float expected = static_cast<float>(src + 100);

        ASSERT_FLOAT_EQ(h_recv[i], expected);
    }

    // 释放
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(i));
        UPTK_CHECK(UPTKFree(d_send[i]));
        UPTK_CHECK(UPTKFree(d_recv[i]));
        NCCL_CHECK(UPTKncclCommDestroy(comms[i]));
    }
}
#include "UPTK_nccl_utils.h"

TEST(uptk_nccl, UPTKncclReduce)
{
    int ndev = 0;
    UPTK_CHECK(UPTKGetDeviceCount(&ndev));
    ASSERT_GE(ndev, 2);

    std::vector<int> devs(ndev);
    std::vector<UPTKncclComm_t> comms(ndev);

    for (int i = 0; i < ndev; i++)
        devs[i] = i;

    // 初始化 communicator
    NCCL_CHECK(UPTKncclCommInitAll(comms.data(), ndev, devs.data()));

    std::vector<float> h_send(ndev);
    std::vector<float> h_recv(ndev, 0.0f);

    std::vector<float*> d_send(ndev);
    std::vector<float*> d_recv(ndev);

    // 初始化 host 数据
    for (int i = 0; i < ndev; i++)
        h_send[i] = static_cast<float>(i + 1);  // 1,2,3,...

    // 每张卡分配 + 拷贝
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));

        UPTK_CHECK(UPTKMalloc(&d_send[i], sizeof(float)));
        UPTK_CHECK(UPTKMalloc(&d_recv[i], sizeof(float)));

        UPTK_CHECK(UPTKMemcpy(
            d_send[i], &h_send[i],
            sizeof(float), UPTKMemcpyHostToDevice));

        UPTK_CHECK(UPTKMemset(d_recv[i], 0, sizeof(float)));
    }

    int root = 0;

    // ⭐ 所有 rank 必须参与
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));

        NCCL_CHECK(UPTKncclReduce(
            d_send[i],
            d_recv[i],
            1,
            UPTKncclFloat,
            UPTKncclSum,
            root,
            comms[i],
            0));
    }

    // ⭐ 关键：同步（否则可能还没算完）
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));
        UPTK_CHECK(UPTKDeviceSynchronize());
    }

    // 只拷贝 root
    UPTK_CHECK(UPTKMemcpy(
        &h_recv[root],
        d_recv[root],
        sizeof(float),
        UPTKMemcpyDeviceToHost));

    // 计算期望值
    float expected = 0.0f;
    for (int i = 0; i < ndev; i++)
        expected += h_send[i];

    ASSERT_FLOAT_EQ(h_recv[root], expected);

    // 释放资源
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));
        UPTK_CHECK(UPTKFree(d_send[i]));
        UPTK_CHECK(UPTKFree(d_recv[i]));
    }

    for (int i = 0; i < ndev; i++)
        NCCL_CHECK(UPTKncclCommDestroy(comms[i]));
}
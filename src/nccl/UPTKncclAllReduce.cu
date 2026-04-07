#include "UPTK_nccl_utils.h"

TEST(uptk_nccl, UPTKncclAllReduce)
{
    int ndev = 0;
    UPTK_CHECK(UPTKGetDeviceCount(&ndev));
    ASSERT_GE(ndev, 2);

    std::vector<int> devs(ndev);
    std::vector<UPTKncclComm_t> comms(ndev);

    for (int i = 0; i < ndev; i++)
        devs[i] = i;

    NCCL_CHECK(UPTKncclCommInitAll(comms.data(), ndev, devs.data()));

    std::vector<float> h_send(ndev);
    std::vector<float> h_recv(ndev, 0.0f);
    std::vector<float*> d_buf(ndev);

    // 初始化 host
    for (int i = 0; i < ndev; i++)
        h_send[i] = static_cast<float>(i + 1); // 1,2,3,...

    // 分配 + 拷贝
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));

        UPTK_CHECK(UPTKMalloc(&d_buf[i], sizeof(float)));

        UPTK_CHECK(UPTKMemcpy(
            d_buf[i],
            &h_send[i],
            sizeof(float),
            UPTKMemcpyHostToDevice));
    }

    // AllReduce（所有 rank 必须调用）
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));

        NCCL_CHECK(UPTKncclAllReduce(
            d_buf[i],
            d_buf[i],
            1,
            UPTKncclFloat,
            UPTKncclSum,
            comms[i],
            0));
    }

    // 同步（非常关键）
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));
        UPTK_CHECK(UPTKDeviceSynchronize());
    }

    // 校验：每张卡结果相同
    float expected = 0.0f;
    for (int i = 0; i < ndev; i++)
        expected += h_send[i];

    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));

        UPTK_CHECK(UPTKMemcpy(
            &h_recv[i],
            d_buf[i],
            sizeof(float),
            UPTKMemcpyDeviceToHost));

        ASSERT_FLOAT_EQ(h_recv[i], expected);
    }

    // cleanup
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));
        UPTK_CHECK(UPTKFree(d_buf[i]));
        NCCL_CHECK(UPTKncclCommDestroy(comms[i]));
    }
}
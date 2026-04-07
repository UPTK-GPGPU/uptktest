#include "UPTK_nccl_utils.h"

TEST(uptk_nccl, UPTKncclBroadcast)
{
    int ndev = 0;
    UPTK_CHECK(UPTKGetDeviceCount(&ndev));
    ASSERT_GE(ndev, 2);

    std::vector<int> devs(ndev);
    std::vector<UPTKncclComm_t> comms(ndev);

    for (int i = 0; i < ndev; i++)
        devs[i] = i;

    NCCL_CHECK(UPTKncclCommInitAll(comms.data(), ndev, devs.data()));

    int root = 0;

    std::vector<float*> d_buf(ndev);
    std::vector<float> h_recv(ndev, 0.0f);

    float root_value = 9.99f;

    // 每张卡分配 buffer
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));
        UPTK_CHECK(UPTKMalloc(&d_buf[i], sizeof(float)));

        // 非 root 不重要，但必须有 buffer
        float init = (i == root) ? root_value : 0.0f;

        UPTK_CHECK(UPTKMemcpy(
            d_buf[i],
            &init,
            sizeof(float),
            UPTKMemcpyHostToDevice));
    }

    // 所有 rank 必须调用 Broadcast
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));

        NCCL_CHECK(UPTKncclBroadcast(
            d_buf[i],
            d_buf[i],
            1,
            UPTKncclFloat,
            root,
            comms[i],
            0));
    }

    // 同步
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));
        UPTK_CHECK(UPTKDeviceSynchronize());
    }

    // 验证所有 GPU
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));

        UPTK_CHECK(UPTKMemcpy(
            &h_recv[i],
            d_buf[i],
            sizeof(float),
            UPTKMemcpyDeviceToHost));

        ASSERT_FLOAT_EQ(h_recv[i], root_value);
    }

    // 释放
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));
        UPTK_CHECK(UPTKFree(d_buf[i]));
        NCCL_CHECK(UPTKncclCommDestroy(comms[i]));
    }
}

#include "UPTK_nccl_utils.h"

TEST(uptk_nccl, UPTKncclReduceScatter)
{
    int ndev = 0;
    UPTK_CHECK(UPTKGetDeviceCount(&ndev));
    ASSERT_GE(ndev, 2);

    std::vector<int> devs(ndev);
    std::vector<UPTKncclComm_t> comms(ndev);

    for (int i = 0; i < ndev; i++)
        devs[i] = i;

    NCCL_CHECK(UPTKncclCommInitAll(comms.data(), ndev, devs.data()));

    int count = 4;                  // 必须能被 ndev 整除
    int chunk = count / ndev;

    std::vector<std::vector<float>> h_send(ndev, std::vector<float>(count));
    std::vector<float*> d_send(ndev);
    std::vector<float*> d_recv(ndev);

    std::vector<float> h_recv(ndev, 0.0f);

    // 初始化：每张 GPU 不同数据
    for (int i = 0; i < ndev; i++)
    {
        for (int j = 0; j < count; j++)
            h_send[i][j] = static_cast<float>(i + 1); // GPU i 全是 i+1
    }

    // 分配 + 拷贝
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));

        UPTK_CHECK(UPTKMalloc(&d_send[i], count * sizeof(float)));
        UPTK_CHECK(UPTKMalloc(&d_recv[i], chunk * sizeof(float)));

        UPTK_CHECK(UPTKMemcpy(
            d_send[i],
            h_send[i].data(),
            count * sizeof(float),
            UPTKMemcpyHostToDevice));
    }

    // ReduceScatter（所有 rank 必须参与）
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));

        NCCL_CHECK(UPTKncclReduceScatter(
            d_send[i],
            d_recv[i],
            chunk,
            UPTKncclFloat,
            UPTKncclSum,
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
    // 每个 GPU 输出 = sum(all GPUs) 的对应 chunk
    float expected_value = 0.0f;
    for (int i = 0; i < ndev; i++)
        expected_value += (i + 1);

    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));

        UPTK_CHECK(UPTKMemcpy(
            &h_recv[i],
            d_recv[i],
            sizeof(float),
            UPTKMemcpyDeviceToHost));

        ASSERT_FLOAT_EQ(h_recv[i], expected_value);
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
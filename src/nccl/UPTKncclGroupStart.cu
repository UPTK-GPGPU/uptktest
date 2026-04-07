#include "UPTK_nccl_utils.h"

TEST(uptk_nccl, UPTKncclGroupStart)
{
    int ndev = 0;
    UPTK_CHECK(UPTKGetDeviceCount(&ndev));
    ASSERT_GE(ndev, 2);

    std::vector<int> devs(ndev);
    std::vector<UPTKncclComm_t> comms(ndev);

    for (int i = 0; i < ndev; i++)
        devs[i] = i;

    NCCL_CHECK(UPTKncclCommInitAll(comms.data(), ndev, devs.data()));

    std::vector<float*> d_buf(ndev);
    std::vector<float> h_send(ndev);

    for (int i = 0; i < ndev; i++)
        h_send[i] = i + 1;

    // alloc + copy
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

    // 🔥 GROUP START
    for (int i = 0; i < ndev; i++)
        UPTK_CHECK(UPTKSetDevice(devs[i]));

    UPTKncclGroupStart();

    for (int i = 0; i < ndev; i++)
    {
        NCCL_CHECK(UPTKncclAllReduce(
            d_buf[i],
            d_buf[i],
            1,
            UPTKncclFloat,
            UPTKncclSum,
            comms[i],
            0));

        NCCL_CHECK(UPTKncclBroadcast(
            d_buf[i],
            d_buf[i],
            1,
            UPTKncclFloat,
            0,
            comms[i],
            0));
    }

    UPTKncclGroupEnd();
    // 🔥 GROUP END

    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));
        UPTK_CHECK(UPTKDeviceSynchronize());
    }

    // verify AllReduce result
    float expected = 0.0f;
    for (int i = 0; i < ndev; i++)
        expected += h_send[i];

    for (int i = 0; i < ndev; i++)
    {
        float out = 0.0f;

        UPTK_CHECK(UPTKMemcpy(
            &out,
            d_buf[i],
            sizeof(float),
            UPTKMemcpyDeviceToHost));

        ASSERT_FLOAT_EQ(out, expected);
    }

    // cleanup
    for (int i = 0; i < ndev; i++)
    {
        UPTK_CHECK(UPTKSetDevice(devs[i]));
        UPTKFree(d_buf[i]);
        NCCL_CHECK(UPTKncclCommDestroy(comms[i]));
    }
}
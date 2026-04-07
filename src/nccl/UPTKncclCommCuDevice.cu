#include "UPTK_nccl_utils.h"
#include <gtest/gtest.h>

TEST(uptk_nccl, UPTKncclCommCuDevice)
{
    int ndev = 0;
    UPTK_CHECK(UPTKGetDeviceCount(&ndev));
    ASSERT_GT(ndev, 0);

    std::vector<int> devs(ndev);
    std::vector<UPTKncclComm_t> comms(ndev);

    for (int i = 0; i < ndev; i++)
        devs[i] = i;

    NCCL_CHECK(UPTKncclCommInitAll(comms.data(), ndev, devs.data()));

    for (int i = 0; i < ndev; i++)
    {
        int dev = -1;
        NCCL_CHECK(UPTKncclCommCuDevice(comms[i], &dev));

        ASSERT_EQ(dev, devs[i]);
    }

    for (int i = 0; i < ndev; i++)
        NCCL_CHECK(UPTKncclCommDestroy(comms[i]));
}
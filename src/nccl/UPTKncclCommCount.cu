#include "UPTK_nccl_utils.h"
#include <gtest/gtest.h>

TEST(uptk_nccl, UPTKncclCommCount)
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
        int count = 0;
        NCCL_CHECK(UPTKncclCommCount(comms[i], &count));

        ASSERT_EQ(count, ndev);
    }

    for (int i = 0; i < ndev; i++)
        NCCL_CHECK(UPTKncclCommDestroy(comms[i]));
}
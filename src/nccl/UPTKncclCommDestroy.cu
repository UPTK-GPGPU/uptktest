#include "UPTK_nccl_utils.h"

TEST(uptk_nccl, UPTKncclCommDestroy)
{
    int ndev = 0;

    UPTK_CHECK(UPTKGetDeviceCount(&ndev));

    std::vector<int> devs;
    std::vector<UPTKncclComm_t> comms;

    for (int i = 0; i < ndev; i++)
    {
        devs.push_back(i);
        comms.push_back(nullptr);
    }

    NCCL_CHECK(UPTKncclCommInitAll(comms.data(), ndev, devs.data()));

    printf("InitAll success\n");

    for (int i = 0; i < ndev; i++)
    {
        NCCL_CHECK(UPTKncclCommDestroy(comms[i]));
    }
}